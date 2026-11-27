"""The Laya trainer's own steps on CPU torch (honing.backends.jev.trainer): the pinned source, Laya's examples and
their batches, the update policy, the encoder and head learning rates, the sft and grpo runs, the saved layout and
manifest, and what it refuses. Laya's package and tokenizer are stood in for by modules around a model of a few
hundred weights (tests/fakes/decision_models.py), and what the trainer sends to ``cuda`` stays on the CPU. The GPU
checks against Laya itself are in integrations/ra2/parity/."""

from __future__ import annotations

import hashlib
import json
import math
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from safetensors.torch import load_file  # noqa: E402

from honing.backends.jev import trainer as laya_trainer  # noqa: E402
from honing.backends.jev.decision_training import group_advantages, laya_question  # noqa: E402
from honing.backends.jev.laya import HEAD_MAX_LEN, laya_snapshot  # noqa: E402
from honing.backends.jev.trainer import MAX_LEN, JevTrainerSettings, LocalJevTrainer, collate  # noqa: E402
from honing.core import ContractError, ModelSnapshot, TrainingBatch, TrainJobSpec, UpdatePolicy  # noqa: E402
from tests.fakes import decision_models as fake  # noqa: E402

SFT = TrainJobSpec("sft", contract="typed_decisions", steps=4)
GRPO = TrainJobSpec("grpo", rollout="harness", contract="typed_decisions", steps=3)
RECORD = {"repo": "laya/typed-decisions", "revision": "release-rev"}


@pytest.fixture
def machine(monkeypatch: pytest.MonkeyPatch) -> Iterator[fake.ProcessState]:
    state = fake.keep_process_state(monkeypatch)
    monkeypatch.setattr(laya_trainer, "gpu_lock", fake.any_gpu)
    fake.install(monkeypatch, fake.laya_package())
    with fake.cuda_on_cpu():
        yield state


def pinned(tmp_path: Path, *, revision: str | None = None, **settings: Any) -> tuple[LocalJevTrainer, ModelSnapshot]:
    """A trainer on a pinned Laya checkout (a git repository, or ``revision`` when the test fakes git) and its base."""
    source = tmp_path / "laya"
    pins = tmp_path / "pins.json"
    pins.write_text(json.dumps({"layaSource": {"revision": revision or fake.pinned_checkout(source)}}))
    (tmp_path / "model.json").write_text(json.dumps(RECORD))
    base = laya_snapshot(fake.laya_checkpoint(tmp_path / "release"))
    defaults: dict[str, Any] = {"min_validation": 2, "encoder_lr": 0.02, "head_lr": 0.05, "batch_size": 4}
    settings = {**defaults, **settings}
    return (
        LocalJevTrainer(JevTrainerSettings(source, pins, tmp_path / "model.json", tmp_path / "runs", **settings)),
        base,
    )


def git_says(monkeypatch: pytest.MonkeyPatch, revision: str) -> None:
    """``git rev-parse HEAD`` answers ``revision`` without starting git."""
    check_output = lambda command, text: revision + "\n"  # noqa: E731
    monkeypatch.setattr(laya_trainer, "subprocess", types.SimpleNamespace(check_output=check_output))


def manifest_of(checkpoint: str) -> dict[str, Any]:
    return json.loads((Path(checkpoint).parent / "checkpoint.json").read_text())  # type: ignore[no-any-return]


def chosen_probabilities(checkpoint: str, batch: TrainingBatch) -> list[float]:
    """Each decision's chosen-option probability under a checkpoint, computed apart from the trainer."""
    model = fake.TinyLaya(fake.LAYA_ENCODER)
    model.load_state_dict(load_file(str(Path(checkpoint) / "model.safetensors")))
    tok, out = fake.WordTokenizer(), []
    for decision in batch.decisions:
        state = "\n".join(line for _, lines in decision.parts for line in lines)
        question = fake._LayaAgent._to_internal(laya_question(decision.questions[0][1]))
        ids, markers = fake._build_sequence(tok, state, question, MAX_LEN, HEAD_MAX_LEN)
        item = {"ids": ids, "markers": markers, "label": [k for k, _ in fake.QUESTION.options].index("wait")}
        with torch.no_grad():
            logits, _ = model(*collate([item], 0, torch, "cpu")[:5])
        out.append(float(logits.softmax(-1)[0, [k for k, _ in fake.QUESTION.options].index(decision.chosen[0][1])]))
    return out


# ---------------------------------------------------------------- the pieces


@pytest.mark.unit
def test_a_batch_pads_ids_and_option_markers_and_masks_what_padding_added() -> None:
    items = [{"ids": [5, 6, 7], "markers": [1, 2], "label": 1}, {"ids": [8], "markers": [0, 0, 0], "label": 2}]
    ids, attention, positions, mask, qtype, labels = collate(items, 9, torch, "cpu")
    assert ids.tolist() == [[5, 6, 7], [8, 9, 9]] and attention.tolist() == [[1, 1, 1], [1, 0, 0]]
    assert positions.tolist() == [[1, 2, 0], [0, 0, 0]]
    assert mask.tolist() == [[True, True, False], [True, True, True]]
    assert qtype.tolist() == [0, 0] and labels.tolist() == [1, 2]
    assert ids.dtype == positions.dtype == torch.long and mask.dtype == torch.bool


@pytest.mark.unit
def test_the_forward_masks_absent_options_and_keeps_the_heads_auxiliary_output(machine: fake.ProcessState) -> None:
    torch.manual_seed(0)
    model = fake.TinyLaya(fake.LAYA_ENCODER)
    items = [{"ids": [5, 6, 7], "markers": [1, 2], "label": 1}, {"ids": [8, 3], "markers": [0, 1, 1], "label": 2}]

    logits, labels, aux = LocalJevTrainer._forward(torch, model, 0, "cuda")(items)

    assert logits.dtype == torch.float32 and logits.shape == (2, 3) and logits[0, 2] == -1e4
    assert bool((logits[1] > -1e4).all()) and labels.tolist() == [1, 2]
    assert aux.shape == (2,) and aux.requires_grad


@pytest.mark.unit
def test_examples_are_laya_inputs_packed_as_inference_packs_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, machine: fake.ProcessState
) -> None:
    git_says(monkeypatch, "source-rev")
    trainer, _ = pinned(tmp_path, revision="source-rev")
    common = fake.laya_package()["laya.common"]
    helpers = (common.build_sequence, common.render_options)

    items, validation, packing = trainer._examples(
        fake.teacher_batch(3, 1).decisions, fake.WordTokenizer(), fake._LayaAgent, fake.LAYA_QTYPES, helpers
    )

    assert [x["label"] for x in items] == [0, 1, 2] and [x["label"] for x in validation] == [0]
    assert all(x["qtype"] == fake.LAYA_QTYPES["choice"] and len(x["markers"]) == 3 for x in items)
    first = items[0]
    words = [fake.word_id(w) for w in ("Wait", "Build", "Attack")]
    assert [first["ids"][m] for m in first["markers"]] == words
    assert packing == {"maxInputTokens": len(first["ids"]), "trimmedRequests": 0}


@pytest.mark.unit
def test_a_checkpoints_hash_is_its_bytes_sha256(tmp_path: Path) -> None:
    path = tmp_path / "weights"
    path.write_bytes(b"x" * (9 * 1024 * 1024))
    assert laya_trainer._sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- sft


@pytest.mark.functional
def test_sft_keeps_the_best_validation_weights_in_layas_layout_with_raw_probabilities(
    tmp_path: Path, machine: fake.ProcessState
) -> None:
    trainer, base = pinned(tmp_path)
    batch = fake.teacher_batch()

    result = trainer.train(SFT, base, batch)

    checkpoint = Path(result.snapshot.checkpoint.location)
    assert result.snapshot == laya_snapshot(checkpoint) and result.snapshot.checkpoint != base.checkpoint
    weights = load_file(str(checkpoint / "model.safetensors"))
    assert {t.dtype for t in weights.values()} == {torch.bfloat16}
    assert json.loads((checkpoint / "encoder" / "config.json").read_text()) == fake.LAYA_ENCODER
    assert (checkpoint / "tokenizer" / "tokenizer.json").exists()
    agent = json.loads((checkpoint / "rl_agent_config.json").read_text())
    assert agent == {"temperature": [1.0, 1.0, 1.0], "max_len": MAX_LEN, "head_max_len": HEAD_MAX_LEN}
    manifest = manifest_of(str(checkpoint))
    revision = json.loads(trainer._settings.pins_path.read_text())["layaSource"]["revision"]
    assert manifest["kind"] == "trained-laya" and manifest["parent"] == RECORD
    assert manifest["weightsSha256"] == result.snapshot.checkpoint.checksum
    assert (manifest["sourceRevision"], manifest["batchId"], manifest["updatePolicy"]) == (revision, "teacher", "full")
    assert manifest["settings"]["encoderLr"] == 0.02 and manifest["settings"]["headLr"] == 0.05
    assert (manifest["trainExamples"], manifest["validationExamples"], len(manifest["history"])) == (12, 6, 4)
    losses = [entry["validationLoss"] for entry in manifest["history"]]
    assert result.metrics["validation_loss"] == min(losses) < result.metrics["initial_validation_loss"]
    # The saved (bf16) weights are the best epoch's: their validation loss is nearer the best than any other epoch's.
    validation = TrainingBatch("v", fake.DECISIONS, decisions=batch.decisions[12:])
    recomputed = -sum(math.log(p) for p in chosen_probabilities(str(checkpoint), validation)) / 6
    assert min(losses, key=lambda loss: abs(loss - recomputed)) == min(losses)
    assert machine.threads == [4] and machine.deterministic == []


@pytest.mark.functional
def test_the_heads_alone_train_when_the_policy_names_them_and_the_encoder_stays_as_released(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, machine: fake.ProcessState
) -> None:
    git_says(monkeypatch, "source-rev")
    trainer, base = pinned(tmp_path, revision="source-rev")
    job = TrainJobSpec(
        "sft", contract="typed_decisions", update_policy=UpdatePolicy("param_subset", trainable_patterns=(r"^head\.",))
    )

    result = trainer.train(job, base, fake.teacher_batch())

    before = load_file(str(Path(base.checkpoint.location) / "model.safetensors"))
    after = load_file(str(Path(result.snapshot.checkpoint.location) / "model.safetensors"))
    assert all(torch.equal(before[name], after[name]) for name in before if name.startswith("encoder."))
    assert not any(torch.equal(before[name], after[name]) for name in ("head.option.weight", "head.state.weight"))


# ---------------------------------------------------------------- grpo


@pytest.mark.functional
def test_grpo_keeps_the_final_weights_and_climbs_the_advantage_weighted_log_probability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, machine: fake.ProcessState
) -> None:
    git_says(monkeypatch, "source-rev")
    trainer, base = pinned(tmp_path, revision="source-rev", deterministic=True)
    batch = fake.rollout_batch()

    result = trainer.train(GRPO, base, batch)

    checkpoint = result.snapshot.checkpoint.location
    manifest = manifest_of(checkpoint)
    assert manifest["algorithm"] == "grpo" and manifest["parent"] == RECORD
    assert (manifest["decisions"], manifest["decisionsWithoutSignal"], manifest["examples"]) == (15, 3, 12)
    assert manifest["settings"]["lr"] == 0.02 and len(manifest["history"]) == 3
    assert manifest["weightsSha256"] == laya_trainer._sha256(Path(checkpoint) / "model.safetensors")
    start = chosen_probabilities(base.checkpoint.location, batch)
    end = chosen_probabilities(checkpoint, batch)
    moves = zip(group_advantages(batch.decisions), start, end, strict=True)
    assert sum(advantage * math.log(after / before) for advantage, before, after in moves) > 0
    assert machine.deterministic == [True] and machine.threads == [4]


@pytest.mark.functional
@pytest.mark.xfail(
    strict=True,
    reason="LocalJevTrainer._policy_gradient never compares the saved weights with the base, unlike its sft path "
    "and the Kev and NanoJev trainers, so a grpo run that changed nothing is returned as a new checkpoint",
)
def test_a_grpo_run_that_changes_no_weight_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, machine: fake.ProcessState
) -> None:
    git_says(monkeypatch, "source-rev")
    trainer, base = pinned(tmp_path, revision="source-rev", encoder_lr=0.0)
    with pytest.raises(ContractError, match="training did not change the weights"):
        trainer.train(GRPO, base, fake.rollout_batch())


# ---------------------------------------------------------------- refusals


@pytest.mark.functional
@pytest.mark.parametrize("job", [SFT, GRPO], ids=["sft", "grpo"])
def test_a_laya_checkout_off_its_pin_is_refused_before_anything_loads(
    tmp_path: Path, machine: fake.ProcessState, job: TrainJobSpec
) -> None:
    trainer, base = pinned(tmp_path, revision="another-revision")
    fake.pinned_checkout(tmp_path / "laya")
    batch = fake.rollout_batch() if job.algorithm == "grpo" else fake.teacher_batch()
    with pytest.raises(ContractError, match="the Laya source differs from its pin"):
        trainer.train(job, base, batch)
    (record,) = (tmp_path / "runs").glob("train-*/error.json")
    assert json.loads(record.read_text()) == {"error": "the Laya source differs from its pin"}


@pytest.mark.functional
@pytest.mark.parametrize("job", [SFT, GRPO], ids=["sft", "grpo"])
def test_a_base_whose_weights_changed_after_its_snapshot_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, machine: fake.ProcessState, job: TrainJobSpec
) -> None:
    git_says(monkeypatch, "source-rev")
    trainer, base = pinned(tmp_path, revision="source-rev")
    (Path(base.checkpoint.location) / "model.safetensors").write_bytes(b"changed")
    batch = fake.rollout_batch() if job.algorithm == "grpo" else fake.teacher_batch()
    with pytest.raises(ContractError, match="weights differ from its checksum"):
        trainer.train(job, base, batch)


@pytest.mark.functional
@pytest.mark.parametrize("job", [SFT, GRPO], ids=["sft", "grpo"])
def test_a_policy_that_matches_no_parameter_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, machine: fake.ProcessState, job: TrainJobSpec
) -> None:
    git_says(monkeypatch, "source-rev")
    trainer, base = pinned(tmp_path, revision="source-rev")
    nothing = UpdatePolicy("param_subset", trainable_patterns=("^lm_head",))
    narrowed = TrainJobSpec(job.algorithm, rollout=job.rollout, contract="typed_decisions", update_policy=nothing)
    batch = fake.rollout_batch() if job.algorithm == "grpo" else fake.teacher_batch()
    with pytest.raises(ContractError, match="no parameter matches the update policy"):
        trainer.train(narrowed, base, batch)


@pytest.mark.functional
def test_an_sft_run_that_changes_no_weight_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, machine: fake.ProcessState
) -> None:
    git_says(monkeypatch, "source-rev")
    trainer, base = pinned(tmp_path, revision="source-rev", encoder_lr=0.0, head_lr=0.0)
    with pytest.raises(ContractError, match="training did not change the weights"):
        trainer.train(SFT, base, fake.teacher_batch())
