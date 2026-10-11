"""The Slime trainer on a CPU: the rollout layout, the Megatron arguments, and jobs against a fake Slime process."""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

from honing.backends.slime import (
    SUPPORT,
    SlimeEnvironment,
    SlimeRunError,
    SlimeSettings,
    SlimeTrainer,
    SlimeTrainerSettings,
    hf_snapshot,
    megatron_model_args,
    policy_sample,
    read_samples,
    slime_samples,
    train_args,
    write_samples,
)
from honing.core import (
    CampaignRuleError,
    ContractError,
    DataContract,
    PolicySample,
    TeacherRef,
    TrainingBatch,
    TrainJobSpec,
)

ROOT = Path(__file__).resolve().parents[1]
GRPO_CONTRACT = DataContract("grouped_on_policy_samples", requires_reward=True, requires_logprobs=True)
QWEN2_CONFIG = {
    "model_type": "qwen2",
    "hidden_size": 896,
    "intermediate_size": 4864,
    "num_attention_heads": 14,
    "num_hidden_layers": 24,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "tie_word_embeddings": True,
    "vocab_size": 151936,
}
QWEN3_CONFIG = {
    "model_type": "qwen3",
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "num_attention_heads": 16,
    "num_hidden_layers": 28,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000,
    "tie_word_embeddings": True,
    "vocab_size": 151936,
}


def _sample(group: str, reward: float, prompt: int = 3, response: int = 4) -> PolicySample:
    return PolicySample(
        tokens=tuple(range(100, 100 + prompt + response)),
        loss_mask=(0,) * prompt + (1,) * response,
        reward=reward,
        logprobs=(0.0,) * prompt + tuple(-0.1 * (i + 1) for i in range(response)),
        group=group,
        source_release="r0001",
        source_harness="h1",
    )


def grpo_batch(groups: int = 2, per_group: int = 2) -> TrainingBatch:
    samples = tuple(_sample(f"task-{g}", float(i)) for g in range(groups) for i in range(per_group))
    return TrainingBatch(batch_id="b1", contract=GRPO_CONTRACT, samples=samples)


# ---------------------------------------------------------------- rollout layout


@pytest.mark.unit
def test_samples_round_trip_through_the_slime_layout(tmp_path: Path) -> None:
    batch = grpo_batch()
    rows, layout = slime_samples(batch)
    path = write_samples(tmp_path / "0.json", rows)

    assert read_samples(path) == batch.samples
    assert (layout.prompts, layout.samples_per_prompt) == (2, 2)


@pytest.mark.unit
def test_loss_mask_and_logprobs_cover_only_the_response() -> None:
    rows, _ = slime_samples(grpo_batch(groups=1))
    row = rows[0]

    assert row["response_length"] == 4
    assert row["loss_mask"] == [1, 1, 1, 1]
    assert row["rollout_log_probs"] == pytest.approx([-0.1, -0.2, -0.3, -0.4])
    assert row["status"] == "completed"


@pytest.mark.unit
def test_tool_tokens_inside_a_session_stay_in_the_response_unmasked() -> None:
    # prompt (2) | model turn (2) | tool result (2) | model turn (1)
    sample = PolicySample(tokens=tuple(range(7)), loss_mask=(0, 0, 1, 1, 0, 0, 1), reward=1.0, group="g")
    rows, _ = slime_samples(TrainingBatch(batch_id="b", contract=DataContract("on_policy_samples"), samples=(sample,)))

    assert rows[0]["response_length"] == 5
    assert rows[0]["loss_mask"] == [1, 1, 0, 0, 1]
    assert policy_sample(rows[0]).loss_mask == sample.loss_mask


@pytest.mark.unit
def test_samples_are_written_group_by_group() -> None:
    samples = (_sample("a", 0.0), _sample("b", 1.0), _sample("a", 2.0), _sample("b", 3.0))
    rows, _ = slime_samples(TrainingBatch(batch_id="b", contract=GRPO_CONTRACT, samples=samples))

    assert [row["group_index"] for row in rows] == [0, 0, 1, 1]
    assert [row["reward"] for row in rows] == [0.0, 2.0, 1.0, 3.0]


@pytest.mark.unit
def test_unequal_groups_are_refused() -> None:
    samples = (_sample("a", 0.0), _sample("a", 1.0), _sample("b", 2.0))
    with pytest.raises(ContractError, match="groups of one size"):
        slime_samples(TrainingBatch(batch_id="b", contract=GRPO_CONTRACT, samples=samples))


@pytest.mark.unit
def test_a_sample_with_nothing_to_train_is_refused() -> None:
    sample = PolicySample(tokens=(1, 2), loss_mask=(0, 0), reward=1.0, group="g", logprobs=(0.0, 0.0))
    with pytest.raises(ContractError, match="no trained token"):
        slime_samples(TrainingBatch(batch_id="b", contract=GRPO_CONTRACT, samples=(sample,)))


# ---------------------------------------------------------------- arguments


def _slime_model_args(name: str) -> list[str]:
    """MODEL_ARGS from Slime's own scripts/models/<name>.sh, when a Slime checkout is available."""
    root = os.environ.get("SLIME_SOURCE")
    if root is None or not (Path(root) / "scripts" / "models" / f"{name}.sh").is_file():
        pytest.skip("set SLIME_SOURCE to a Slime checkout to compare with its model scripts")
    text = (Path(root) / "scripts" / "models" / f"{name}.sh").read_text()
    body = text[text.index("(") + 1 : text.rindex(")")]
    return [token.strip('"') for line in body.splitlines() for token in line.split() if not token.startswith("#")]


def _normalized(token: str) -> str:
    try:
        return repr(float(token))
    except ValueError:
        return token


@pytest.mark.unit
def test_qwen2_arguments() -> None:
    args = megatron_model_args(QWEN2_CONFIG)

    assert "--add-qkv-bias" in args and "--qk-layernorm" not in args
    assert args[args.index("--num-query-groups") + 1] == "2"
    assert args[args.index("--rotary-base") + 1] == "1000000"
    assert "--untie-embeddings-and-output-weights" not in args


@pytest.mark.unit
def test_qwen3_arguments_match_slimes_model_script() -> None:
    args = megatron_model_args(QWEN3_CONFIG)

    assert "--qk-layernorm" in args and "--add-qkv-bias" not in args
    assert args[args.index("--kv-channels") + 1] == "128"
    expected = _slime_model_args("qwen3-0.6B")
    assert sorted(map(_normalized, args)) == sorted(map(_normalized, expected))


@pytest.mark.unit
def test_untied_embeddings_are_declared() -> None:
    assert "--untie-embeddings-and-output-weights" in megatron_model_args(
        {**QWEN2_CONFIG, "tie_word_embeddings": False}
    )


@pytest.mark.unit
def test_an_unknown_family_is_refused() -> None:
    with pytest.raises(ContractError, match="no Megatron arguments"):
        megatron_model_args({**QWEN2_CONFIG, "model_type": "qwen3_moe"})


@pytest.mark.unit
def test_grpo_arguments_train_only_on_the_given_rollout(tmp_path: Path) -> None:
    args = train_args(
        algorithm="grpo",
        base=tmp_path / "base",
        rollout_template="r/{rollout_id}.pt",
        save_hf="hf/{rollout_id}",
        save="megatron",
        prompts=4,
        samples_per_prompt=8,
        steps=2,
        gpus=1,
        model_args=("--swiglu",),
        settings=SlimeSettings(),
    )

    def value(name: str) -> str:
        return args[args.index(name) + 1]

    assert "--debug-train-only" in args
    assert value("--load-debug-rollout-data") == "r/{rollout_id}.pt"
    assert (value("--rollout-batch-size"), value("--n-samples-per-prompt"), value("--global-batch-size")) == (
        "4",
        "8",
        "16",
    )
    assert value("--load") == value("--ref-load") == value("--hf-checkpoint") == str(tmp_path / "base")
    assert value("--advantage-estimator") == "grpo"


@pytest.mark.unit
def test_samples_that_do_not_split_into_the_steps_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="equal optimizer steps"):
        train_args(
            algorithm="grpo",
            base=tmp_path,
            rollout_template="r",
            save_hf="h",
            save="m",
            prompts=3,
            samples_per_prompt=1,
            steps=2,
            gpus=1,
            model_args=(),
            settings=SlimeSettings(),
        )


# ---------------------------------------------------------------- the trainer against a fake Slime process


def write_checkpoint(directory: Path, config: dict[str, object] = QWEN2_CONFIG) -> Path:
    """A Hugging Face directory with the config and one small safetensors file."""
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps(config))
    header = json.dumps({"w": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}}).encode()
    (directory / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + bytes(range(16)))
    return directory


def fake_environment(tmp_path: Path) -> SlimeEnvironment:
    launcher = tmp_path / "fake-python"
    launcher.write_text(f"#!/bin/sh\nexec {sys.executable} {ROOT / 'tests' / 'fakes' / 'slime_process.py'} \"$@\"\n")
    launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC)
    return SlimeEnvironment(honing_root=ROOT, python=str(launcher), slime_root="/fake/slime")


def fake_trainer(tmp_path: Path) -> SlimeTrainer:
    return SlimeTrainer(
        SlimeTrainerSettings(environment=fake_environment(tmp_path), work_dir=tmp_path / "runs", lock_gpus=False)
    )


GRPO = TrainJobSpec(algorithm="grpo", rollout="harness", steps=1)


@pytest.mark.unit
def test_a_grpo_job_returns_new_hf_weights_with_a_manifest(tmp_path: Path) -> None:
    base = hf_snapshot(write_checkpoint(tmp_path / "base"))
    result = fake_trainer(tmp_path).train(GRPO, base, grpo_batch())

    trained = Path(result.snapshot.checkpoint.location)
    manifest = json.loads((trained.parents[1] / "checkpoint.json").read_text())
    assert result.snapshot.checkpoint.weight_format == "hf_safetensors"
    assert result.snapshot.checkpoint.checksum != base.checkpoint.checksum
    assert result.snapshot.architecture == base.architecture
    assert manifest["parent"]["checksum"] == base.checkpoint.checksum
    assert manifest["batch_id"] == "b1"
    assert result.metrics["slime_kl_loss"] == pytest.approx(0.003)
    assert result.metrics["rollout_reward_mean"] == pytest.approx(0.5)
    assert "slime_opd_reverse_kl" not in result.metrics


@pytest.mark.unit
def test_each_job_trains_into_a_fresh_directory(tmp_path: Path) -> None:
    base = hf_snapshot(write_checkpoint(tmp_path / "base"))
    trainer = fake_trainer(tmp_path)
    first = trainer.train(GRPO, base, grpo_batch())
    second = trainer.train(GRPO, base, grpo_batch())

    assert first.snapshot.checkpoint.location != second.snapshot.checkpoint.location
    assert Path(first.snapshot.checkpoint.location).is_dir()


@pytest.mark.unit
def test_a_failed_slime_run_names_its_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_SLIME_FAIL", "train")
    base = hf_snapshot(write_checkpoint(tmp_path / "base"))
    with pytest.raises(SlimeRunError, match=r"train\.log"):
        fake_trainer(tmp_path).train(GRPO, base, grpo_batch())


@pytest.mark.unit
def test_unchanged_weights_are_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_SLIME_UNCHANGED", "1")
    base = hf_snapshot(write_checkpoint(tmp_path / "base"))
    with pytest.raises(SlimeRunError, match="unchanged"):
        fake_trainer(tmp_path).train(GRPO, base, grpo_batch())


@pytest.mark.unit
def test_what_the_rules_refuse_never_reaches_slime(tmp_path: Path) -> None:
    base = hf_snapshot(write_checkpoint(tmp_path / "base"))
    trainer = fake_trainer(tmp_path)
    with pytest.raises(CampaignRuleError):
        trainer.train(TrainJobSpec(algorithm="dpo"), base, None)
    with pytest.raises(CampaignRuleError):
        trainer.train(GRPO, base, None)
    assert not (tmp_path / "runs").exists()


@pytest.mark.unit
def test_the_support_takes_language_models_and_never_typed_decisions() -> None:
    assert "typed_decisions" not in SUPPORT.contracts
    assert {name for name, support in SUPPORT.algorithms if support == "native"} == {"sft", "opd", "grpo"}
    assert SUPPORT.support_for("dpo") == "unsupported"
    assert SUPPORT.weight_formats == frozenset({"hf_safetensors"})


# ---------------------------------------------------------------- sft and on-policy distillation


def _args(algorithm: str, tmp_path: Path, teacher: Path | None = None) -> tuple[str, ...]:
    return train_args(
        algorithm=algorithm,
        base=tmp_path / "base",
        rollout_template="r/{rollout_id}.pt",
        save_hf="hf/{rollout_id}",
        save="megatron",
        prompts=4,
        samples_per_prompt=1,
        steps=1,
        gpus=1,
        model_args=(),
        settings=SlimeSettings(),
        teacher=teacher,
    )


def token_batch() -> TrainingBatch:
    samples = tuple(
        PolicySample(tokens=(1, 2, 3, 4 + i), loss_mask=(0, 0, 1, 1)) for i in range(4)
    )  # demonstrations: no reward, no group
    return TrainingBatch(batch_id="sft-1", contract=DataContract("token_sequences"), samples=samples)


def on_policy_batch() -> TrainingBatch:
    samples = tuple(
        PolicySample(tokens=(1, 2, 3, 4 + i), loss_mask=(0, 0, 1, 1), logprobs=(0.0, 0.0, -0.3, -0.2))
        for i in range(4)
    )
    contract = DataContract("on_policy_samples", requires_logprobs=True)
    return TrainingBatch(batch_id="opd-1", contract=contract, samples=samples)


@pytest.mark.unit
def test_sft_arguments_use_slimes_sft_loss_without_advantages(tmp_path: Path) -> None:
    args = _args("sft", tmp_path)

    assert args[args.index("--loss-type") + 1] == "sft_loss"
    assert "--disable-compute-advantages-and-returns" in args
    assert "--use-kl-loss" not in args and "--ref-load" not in args
    assert args[args.index("--lr") + 1] == "1e-05"


@pytest.mark.unit
def test_opd_arguments_load_the_teacher_in_megatron(tmp_path: Path) -> None:
    args = _args("opd", tmp_path, teacher=tmp_path / "teacher")

    assert "--use-opd" in args
    assert args[args.index("--opd-type") + 1] == "megatron"
    assert args[args.index("--opd-teacher-load") + 1] == str(tmp_path / "teacher")
    assert args[args.index("--opd-kl-coef") + 1] == "1.0"


@pytest.mark.unit
def test_opd_without_a_teacher_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="teacher"):
        _args("opd", tmp_path)


@pytest.mark.unit
def test_a_missing_reward_is_written_as_zero_and_read_back_as_none(tmp_path: Path) -> None:
    rows, layout = slime_samples(token_batch())

    assert [row["reward"] for row in rows] == [0.0] * 4
    assert (layout.prompts, layout.samples_per_prompt) == (4, 1)
    assert read_samples(write_samples(tmp_path / "0.json", rows)) == token_batch().samples


SFT = TrainJobSpec(algorithm="sft", steps=1)
OPSD = TrainJobSpec(
    algorithm="opd", rollout="direct", teacher=TeacherRef(kind="lineage", ref="parent", placement="in_process")
)


@pytest.mark.unit
def test_an_sft_job_trains_on_demonstrations(tmp_path: Path) -> None:
    base = hf_snapshot(write_checkpoint(tmp_path / "base"))
    result = fake_trainer(tmp_path).train(SFT, base, token_batch())

    run = Path(result.snapshot.checkpoint.location).parents[1]
    train_args_used = json.loads((run / "job.json").read_text())["train_args"]
    assert result.snapshot.checkpoint.checksum != base.checkpoint.checksum
    assert "sft_loss" in train_args_used
    assert "rollout_reward_mean" not in result.metrics


@pytest.mark.unit
def test_self_distillation_takes_the_parent_checkpoint_as_teacher(tmp_path: Path) -> None:
    base = hf_snapshot(write_checkpoint(tmp_path / "base"))
    result = fake_trainer(tmp_path).train(OPSD, base, on_policy_batch())

    manifest = json.loads((Path(result.snapshot.checkpoint.location).parents[1] / "checkpoint.json").read_text())
    assert manifest["teacher"] == base.checkpoint.location


@pytest.mark.unit
def test_a_named_teacher_comes_from_the_settings(tmp_path: Path) -> None:
    base = hf_snapshot(write_checkpoint(tmp_path / "base"))
    teacher = write_checkpoint(tmp_path / "teacher")
    trainer = SlimeTrainer(
        SlimeTrainerSettings(
            environment=fake_environment(tmp_path),
            work_dir=tmp_path / "runs",
            lock_gpus=False,
            teachers={"r0003": teacher},
        )
    )
    job = TrainJobSpec(
        algorithm="opd", rollout="direct", teacher=TeacherRef(kind="release", ref="r0003", placement="in_process")
    )
    result = trainer.train(job, base, on_policy_batch())

    manifest = json.loads((Path(result.snapshot.checkpoint.location).parents[1] / "checkpoint.json").read_text())
    assert manifest["teacher"] == str(teacher)
    assert result.metrics["slime_opd_reverse_kl"] == pytest.approx(0.04)
    assert result.metrics["slime_teacher_log_probs"] == pytest.approx(-0.5)


@pytest.mark.unit
def test_a_teacher_of_another_architecture_is_refused_before_any_run(tmp_path: Path) -> None:
    base = hf_snapshot(write_checkpoint(tmp_path / "base"))
    teacher = write_checkpoint(tmp_path / "teacher", {**QWEN2_CONFIG, "rope_theta": 10000.0, "rms_norm_eps": 1e-5})
    trainer = SlimeTrainer(
        SlimeTrainerSettings(
            environment=fake_environment(tmp_path),
            work_dir=tmp_path / "runs",
            lock_gpus=False,
            teachers={"other": teacher},
        )
    )
    job = TrainJobSpec(
        algorithm="opd", rollout="direct", teacher=TeacherRef(kind="release", ref="other", placement="in_process")
    )
    with pytest.raises(ContractError, match=r"--norm-epsilon 1e-06/1e-05; --rotary-base 1000000/10000"):
        trainer.train(job, base, on_policy_batch())
    assert not (tmp_path / "runs").exists()


@pytest.mark.unit
def test_a_teacher_slime_cannot_load_is_refused_before_any_run(tmp_path: Path) -> None:
    base = hf_snapshot(write_checkpoint(tmp_path / "base"))
    trainer = fake_trainer(tmp_path)
    served = TrainJobSpec(algorithm="opd", rollout="direct", teacher=TeacherRef(kind="lineage", ref="parent"))
    unknown = TrainJobSpec(
        algorithm="opd", rollout="direct", teacher=TeacherRef(kind="release", ref="r0009", placement="in_process")
    )
    with pytest.raises(ContractError, match="in_process"):
        trainer.train(served, base, on_policy_batch())
    with pytest.raises(ContractError, match="r0009"):
        trainer.train(unknown, base, on_policy_batch())
    assert not (tmp_path / "runs").exists()
