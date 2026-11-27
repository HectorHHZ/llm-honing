"""The Kev trainer's own steps on CPU torch: the warm start from the pinned release, Kev's examples, the update
policy over a frozen base, the adapter and head learning rates under one-cycle, the sft and grpo runs, the saved
layout and manifest, and what it refuses. Kev's package is stood in for by modules around a model of a few hundred
weights (tests/fakes/decision_models.py), and what the trainer sends to ``cuda`` stays on the CPU. The GPU checks
against Kev itself are in integrations/ra2/parity/."""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from safetensors.torch import load_file  # noqa: E402

from honing.backends.jev import kev_trainer  # noqa: E402
from honing.backends.jev.decision_training import group_advantages, laya_question  # noqa: E402
from honing.backends.jev.kev import kev_snapshot, weights_checksum  # noqa: E402
from honing.backends.jev.kev_trainer import KevTrainer, KevTrainerSettings, base_config_of  # noqa: E402
from honing.core import ContractError, ModelSnapshot, TrainingBatch, TrainJobSpec, UpdatePolicy  # noqa: E402
from tests.fakes import decision_models as fake  # noqa: E402

SFT = TrainJobSpec("sft", contract="typed_decisions", steps=3)
GRPO = TrainJobSpec("grpo", rollout="harness", contract="typed_decisions", steps=3)


@pytest.fixture
def machine(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[fake.ProcessState]:
    state = fake.keep_process_state(monkeypatch)
    monkeypatch.setattr(kev_trainer, "gpu_lock", fake.any_gpu)
    fake.install(monkeypatch, fake.kev_package(fake.kev_base_config(tmp_path / "hf-cache" / "qwen")))
    with fake.cuda_on_cpu():
        yield state


def pinned(tmp_path: Path, *, checkout: bool = True, **settings: Any) -> tuple[KevTrainer, ModelSnapshot]:
    """A trainer on a pinned Kev checkout (a git repository unless ``checkout`` is False) and its base release."""
    source = tmp_path / "kev"
    revision = fake.pinned_checkout(source) if checkout else "source-rev"
    pins = fake.kev_pins(tmp_path / "pins.json", revision)
    base = kev_snapshot(fake.kev_checkpoint(tmp_path / "release"), tmp_path / "hf-cache" / "qwen" / "config.json")
    defaults: dict[str, Any] = {"min_validation": 2, "lr": 0.05, "batch_size": 4}
    return KevTrainer(KevTrainerSettings(pins, source, tmp_path / "runs", **{**defaults, **settings})), base


def without_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kev_trainer, "import_kev", lambda source, pins: pins["kevSource"]["revision"])


def manifest_of(checkpoint: str) -> dict[str, Any]:
    return json.loads((Path(checkpoint).parent / "checkpoint.json").read_text())  # type: ignore[no-any-return]


def chosen_probabilities(checkpoint: str, batch: TrainingBatch) -> list[float]:
    """Each decision's chosen-option probability under a checkpoint, computed apart from the trainer."""
    meta = fake.read_meta(checkpoint)
    model = fake.KevDecisionModel(meta.base, fake.WordTokenizer(), "cpu", lora=meta.lora, head_dim=meta.head_dim)
    model.lm.load_state_dict(
        {**model.lm.state_dict(), **load_file(str(Path(checkpoint) / "adapter_model.safetensors"))}
    )
    model.head.load_state_dict(meta.head or {})
    out = []
    for decision in batch.decisions:
        state = "\n".join(line for _, lines in decision.parts for line in lines)
        question = laya_question(decision.questions[0][1])
        options = [f"{k}: {v}" for k, v in question["criteria"].items()]
        record = {"state": state, "questions": [{"instr": question["instructions"], "options": options, "label": 0}]}
        encoded = model.encode(fake.WordTokenizer(), record, max_state=32, max_branch=48, strict=True)
        with torch.no_grad():
            (logits,) = model.forward_batch([encoded])
        out.append(float(logits[0].softmax(-1)[list(question["criteria"]).index(decision.chosen[0][1])]))
    return out


# ---------------------------------------------------------------- sft and grpo


@pytest.mark.functional
def test_sft_continues_the_release_and_saves_kevs_layout_at_temperature_one(
    tmp_path: Path, machine: fake.ProcessState
) -> None:
    trainer, base = pinned(tmp_path)
    batch = fake.teacher_batch()

    result = trainer.train(SFT, base, batch)

    checkpoint = Path(result.snapshot.checkpoint.location)
    base_config = tmp_path / "hf-cache" / "qwen" / "config.json"
    assert result.snapshot == kev_snapshot(checkpoint, base_config) and result.snapshot.checkpoint != base.checkpoint
    assert {p.name for p in checkpoint.iterdir()} == {
        "adapter_model.safetensors", "adapter_config.json", "head.pt", "kev_meta.json", "tokenizer.json"
    }  # fmt: skip
    meta = fake.read_meta(str(checkpoint))
    assert (meta.temperature, meta.weights_dtype, meta.base_revision) == (1.0, "fp32", fake.BASE_REVISION)
    assert meta.extra == {
        "init_source": {"path": base.checkpoint.location, "weightsDtype": "fp32"},
        "trainer": "honing.backends.jev.kev_trainer",
    }
    manifest = manifest_of(str(checkpoint))
    revision = json.loads(trainer._settings.pins_path.read_text())["kevSource"]["revision"]
    assert manifest["kind"] == "trained-kev" and manifest["weightsSha256"] == weights_checksum(checkpoint)
    assert manifest["parent"] == {"repo": "C-Tianyu/Kev", "revision": "release-rev"}
    assert (manifest["sourceRevision"], manifest["batchId"], manifest["updatePolicy"]) == (revision, "teacher", "full")
    assert manifest["settings"]["headLr"] == 0.0 and manifest["settings"]["lr"] == 0.05
    assert (manifest["trainExamples"], manifest["validationExamples"], len(manifest["history"])) == (12, 6, 3)
    losses = [entry["validationLoss"] for entry in manifest["history"]]
    assert result.metrics["validation_loss"] == min(losses) < result.metrics["initial_validation_loss"]
    # The saved weights are the best epoch's: their validation loss, computed apart from the trainer, is the best.
    validation = TrainingBatch("v", fake.DECISIONS, decisions=batch.decisions[12:])
    recomputed = -sum(math.log(p) for p in chosen_probabilities(str(checkpoint), validation)) / 6
    assert recomputed == pytest.approx(result.metrics["validation_loss"], rel=1e-5)


@pytest.mark.functional
def test_grpo_keeps_the_final_weights_and_climbs_the_advantage_weighted_log_probability(
    tmp_path: Path, machine: fake.ProcessState
) -> None:
    trainer, base = pinned(tmp_path, deterministic=True, head_lr=0.1)
    batch = fake.rollout_batch()

    result = trainer.train(GRPO, base, batch)

    manifest = manifest_of(result.snapshot.checkpoint.location)
    assert manifest["algorithm"] == "grpo" and manifest["kind"] == "trained-kev"
    assert (manifest["decisions"], manifest["decisionsWithoutSignal"], manifest["examples"]) == (15, 3, 12)
    assert manifest["packing"] == {"maxInputTokens": 5 + 15, "trimmedRequests": 0}  # state 1 + 4 words, branch 15
    start = chosen_probabilities(base.checkpoint.location, batch)
    end = chosen_probabilities(result.snapshot.checkpoint.location, batch)
    assert manifest["meanBehaviorGap"] == pytest.approx(sum(abs(p - 0.4) for p in start[:12]) / 12, rel=1e-5)
    moves = zip(group_advantages(batch.decisions), start, end, strict=True)
    assert sum(advantage * math.log(after / before) for advantage, before, after in moves) > 0
    assert machine.deterministic == [True]


@pytest.mark.functional
def test_a_head_only_run_leaves_the_adapter_bit_for_bit(tmp_path: Path, machine: fake.ProcessState) -> None:
    trainer, base = pinned(tmp_path)
    job = TrainJobSpec(
        "sft", contract="typed_decisions", update_policy=UpdatePolicy("param_subset", trainable_patterns=(r"^head\.",))
    )

    result = trainer.train(job, base, fake.teacher_batch())

    adapter = "adapter_model.safetensors"
    before = (Path(base.checkpoint.location) / adapter).read_bytes()
    assert (Path(result.snapshot.checkpoint.location) / adapter).read_bytes() == before
    assert result.snapshot.checkpoint.checksum != base.checkpoint.checksum


# ---------------------------------------------------------------- refusals


@pytest.mark.functional
def test_a_lora_job_trains_the_adapter_but_a_pattern_cannot_unfreeze_the_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, machine: fake.ProcessState
) -> None:
    without_git(monkeypatch)
    trainer, base = pinned(tmp_path, checkout=False)
    lora = TrainJobSpec("sft", contract="typed_decisions", update_policy=UpdatePolicy("lora", lora_rank=2))
    assert trainer.train(lora, base, fake.teacher_batch()).snapshot.checkpoint != base.checkpoint
    frozen = UpdatePolicy("param_subset", trainable_patterns=(r"^lm\.base\.",))
    with pytest.raises(ContractError, match="no parameter matches the update policy"):
        trainer.train(
            TrainJobSpec("sft", contract="typed_decisions", update_policy=frozen), base, fake.teacher_batch()
        )


@pytest.mark.functional
def test_a_release_off_the_pin_or_changed_since_its_snapshot_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, machine: fake.ProcessState
) -> None:
    without_git(monkeypatch)
    trainer, base = pinned(tmp_path, checkout=False)
    other = fake.kev_checkpoint(tmp_path / "other", base_revision="another")
    with pytest.raises(ContractError, match=r"Qwen base differs from pins\.json"):
        trainer.train(SFT, kev_snapshot(other, tmp_path / "hf-cache" / "qwen" / "config.json"), fake.teacher_batch())
    (Path(base.checkpoint.location) / "head.pt").write_bytes(b"changed")
    with pytest.raises(ContractError, match="weights differ from its checksum"):
        trainer.train(SFT, base, fake.teacher_batch())
    errors = [json.loads(p.read_text())["error"] for p in sorted((tmp_path / "runs").glob("*/error.json"))]
    assert sorted(errors) == [
        "the base checkpoint's Qwen base differs from pins.json",
        "the base checkpoint's weights differ from its checksum",
    ]


@pytest.mark.functional
def test_a_run_that_changes_no_weight_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, machine: fake.ProcessState
) -> None:
    without_git(monkeypatch)
    trainer, base = pinned(tmp_path, checkout=False, lr=0.0)
    with pytest.raises(ContractError, match="training did not change the weights"):
        trainer.train(SFT, base, fake.teacher_batch())
    with pytest.raises(ContractError, match="training did not change the weights"):
        trainer.train(GRPO, base, fake.rollout_batch())


# ---------------------------------------------------------------- the pieces


def loaded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **settings: Any) -> tuple[KevTrainer, Any]:
    without_git(monkeypatch)
    trainer, base = pinned(tmp_path, checkout=False, **settings)
    return trainer, trainer._load(base)


@pytest.mark.unit
def test_the_warm_start_builds_an_fp32_model_like_the_release_and_resets_its_temperature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, machine: fake.ProcessState
) -> None:
    _, state = loaded(monkeypatch, tmp_path)
    model = state.model
    assert model.options["lora_targets"] == "attention" and model.options["dtype"] == torch.float32
    assert model.device_name == "cuda" and model.lm.checkpointing == {"use_reentrant": False}
    assert model.lm.config.use_cache is False and model.head.temperature == 1.0
    release = Path(tmp_path / "release")
    adapter = load_file(str(release / "adapter_model.safetensors"))
    assert all(torch.equal(model.lm.state_dict()[name], value) for name, value in adapter.items())
    assert (state.meta.weights_dtype, state.revision, state.pin["revision"]) == ("fp32", "source-rev", "release-rev")


@pytest.mark.unit
def test_the_head_learns_at_the_adapter_rate_unless_given_its_own_under_one_cycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, machine: fake.ProcessState
) -> None:
    trainer, state = loaded(monkeypatch, tmp_path, lr=0.05, weight_decay=0.1)
    optimizer, schedule = trainer._optimizer(torch, state.model, steps=7)
    assert [g["max_lr"] for g in optimizer.param_groups] == [0.05, 0.05]
    assert [len(g["params"]) for g in optimizer.param_groups] == [2, 4]  # lora a and b; head state and option
    assert all(g["weight_decay"] == 0.1 for g in optimizer.param_groups) and schedule.total_steps == 7

    own = KevTrainer(dataclasses.replace(trainer._settings, head_lr=0.2))
    optimizer, schedule = own._optimizer(torch, state.model, steps=0)
    assert [g["max_lr"] for g in optimizer.param_groups] == [0.05, 0.2] and schedule.total_steps == 1


@pytest.mark.unit
def test_examples_carry_kevs_encoding_and_hold_out_the_validation_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, machine: fake.ProcessState
) -> None:
    trainer, state = loaded(monkeypatch, tmp_path)
    items, validation, packing = trainer._examples(fake.teacher_batch(3, 1).decisions, state.tok, state.model)
    assert [x["label"] for x in items] == [0, 1, 2] and [x["label"] for x in validation] == [0]
    for item in items:
        assert item["ids"] == item["enc"]["ids"] and item["markers"] == item["enc"]["opt_idx"]
        assert len(item["markers"]) == 3 and item["enc"]["label"] == item["label"]
    assert packing == {"maxInputTokens": 5 + 15, "trimmedRequests": 0}


@pytest.mark.unit
def test_the_base_config_comes_from_the_release_named_qwen_base_in_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, machine: fake.ProcessState
) -> None:
    _, base = pinned(tmp_path, checkout=False)
    assert base_config_of(base) == tmp_path / "hf-cache" / "qwen" / "config.json"
    assert base.architecture.hidden_size == fake.HIDDEN and len(base.architecture.layers) == 2
