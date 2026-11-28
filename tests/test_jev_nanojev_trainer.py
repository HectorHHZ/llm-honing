"""The NanoJev trainer's own steps on CPU torch: the pinned base, NanoJev's examples, the update policy, the two
learning rates, the sft and grpo runs, the saved layout and manifest, and what it refuses. NanoJev's predictor is
stood in for by a model of a few hundred weights (tests/fakes/decision_models.py), and what the trainer sends to
``cuda`` stays on the CPU. The GPU checks against NanoJev itself are in integrations/ra2/parity/."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from safetensors.torch import load_file  # noqa: E402

from honing.backends.jev import nanojev_trainer  # noqa: E402
from honing.backends.jev.decision_training import group_advantages, laya_question  # noqa: E402
from honing.backends.jev.nanojev import DOWNLOAD_PATTERNS, nanojev_snapshot, payload  # noqa: E402
from honing.backends.jev.nanojev_trainer import NanoJevTrainer, NanoJevTrainerSettings  # noqa: E402
from honing.core import ContractError, ModelSnapshot, TrainingBatch, TrainJobSpec, UpdatePolicy  # noqa: E402
from tests.fakes import decision_models as fake  # noqa: E402

SFT = TrainJobSpec("sft", contract="typed_decisions", steps=4)
GRPO = TrainJobSpec("grpo", rollout="harness", contract="typed_decisions", steps=3)
HEADS = UpdatePolicy("param_subset", trainable_patterns=(r"^(norm|scalar)\.",))


@pytest.fixture
def machine(monkeypatch: pytest.MonkeyPatch) -> Iterator[fake.ProcessState]:
    state = fake.keep_process_state(monkeypatch)
    monkeypatch.setattr(nanojev_trainer, "gpu_lock", fake.any_gpu)
    with fake.cuda_on_cpu():
        yield state


def pinned(tmp_path: Path, *, checkout: bool = True, **settings: Any) -> tuple[NanoJevTrainer, ModelSnapshot]:
    """A trainer on a pinned NanoJev checkout (a git repository unless ``checkout`` is False) and its base release."""
    source = tmp_path / "nanojev"
    if checkout:
        revision = fake.pinned_checkout(source, {"scripts/predict_toy_decisions.py": fake.NANOJEV_PREDICTOR})
    else:
        revision = "source-rev"
    pins = fake.nanojev_pins(tmp_path / "pins.json", revision)
    base = nanojev_snapshot(fake.nanojev_checkpoint(tmp_path / "release"))
    defaults: dict[str, Any] = {"min_validation": 2, "backbone_lr": 0.02, "head_lr": 0.05, "batch_questions": 4}
    trainer = NanoJevTrainer(
        NanoJevTrainerSettings(
            pins_path=pins, source_dir=source, work_dir=tmp_path / "runs", **{**defaults, **settings}
        )
    )
    return trainer, base


def manifest_of(checkpoint: str) -> dict[str, Any]:
    return json.loads((Path(checkpoint).parent / "checkpoint.json").read_text())  # type: ignore[no-any-return]


def chosen_probabilities(checkpoint: str, batch: TrainingBatch) -> list[float]:
    """Each decision's chosen-option probability under a checkpoint, computed apart from the trainer."""
    predictor = fake.DecisionPredictor(checkpoint, 2048, "cpu", "fp32")
    out = []
    for decision in batch.decisions:
        state = "\n".join(line for _, lines in decision.parts for line in lines)
        questions = {key: laya_question(q) for key, q in decision.questions}
        (example,) = fake.prepare_examples(payload(state, questions), predictor.tokenizer, predictor.limit)
        with torch.no_grad():
            logits, _ = predictor.model([example], 0)
        out.append(float(logits.softmax(-1)[0, example["candidate_ids"].index(dict(decision.chosen)["next"])]))
    return out


# ---------------------------------------------------------------- sft


@pytest.mark.functional
def test_sft_keeps_the_best_validation_weights_in_nanojevs_layout_and_names_its_parent(
    tmp_path: Path, machine: fake.ProcessState
) -> None:
    trainer, base = pinned(tmp_path)
    batch = fake.teacher_batch()

    result = trainer.train(SFT, base, batch)

    checkpoint = Path(result.snapshot.checkpoint.location)
    release = Path(base.checkpoint.location)
    assert result.snapshot == nanojev_snapshot(checkpoint) and result.snapshot.checkpoint != base.checkpoint
    for name in ("config.json", "backbone_config/config.json", "tokenizer/tokenizer.json"):
        assert (checkpoint / name).read_bytes() == (release / name).read_bytes(), name
    manifest = manifest_of(str(checkpoint))
    revision = json.loads(trainer._settings.pins_path.read_text())["nanojevSource"]["revision"]
    assert manifest["kind"] == "trained-nanojev" and manifest["path"] == str(checkpoint.resolve())
    assert manifest["weightsSha256"] == manifest["revision"] == result.snapshot.checkpoint.checksum
    assert manifest["parent"] == {"repo": "C-Tianyu/NanoJev", "revision": "release-rev"}
    assert manifest["base"] == {"location": str(release), "checksum": base.checkpoint.checksum}
    assert (manifest["sourceRevision"], manifest["batchId"], manifest["updatePolicy"]) == (revision, "teacher", "full")
    assert (manifest["trainExamples"], manifest["validationExamples"]) == (12, 6)
    assert manifest["packing"]["trimmedRequests"] == 0 and len(manifest["history"]) == 4
    assert manifest["settings"]["epochs"] == 4 and manifest["settings"]["headLr"] == 0.05
    losses = [entry["validationLoss"] for entry in manifest["history"]]
    assert result.metrics["validation_loss"] == min(losses) < result.metrics["initial_validation_loss"]
    # The saved weights are the best epoch's: their validation loss, computed apart from the trainer, is the best.
    validation = TrainingBatch("v", fake.DECISIONS, decisions=batch.decisions[12:])
    recomputed = -sum(math.log(p) for p in chosen_probabilities(str(checkpoint), validation)) / 6
    assert recomputed == pytest.approx(result.metrics["validation_loss"], rel=1e-5)
    assert machine.deterministic == [] and os.environ["HF_HUB_OFFLINE"] == "1"


@pytest.mark.functional
def test_a_param_subset_run_trains_the_heads_and_leaves_the_backbone_bit_for_bit(
    tmp_path: Path, machine: fake.ProcessState
) -> None:
    trainer, base = pinned(tmp_path)
    job = TrainJobSpec("sft", contract="typed_decisions", steps=2, update_policy=HEADS)

    result = trainer.train(job, base, fake.teacher_batch())

    before = load_file(str(Path(base.checkpoint.location) / "best.safetensors"))
    after = load_file(str(Path(result.snapshot.checkpoint.location) / "best.safetensors"))
    assert set(before) == set(after)
    for name in before:
        assert torch.equal(before[name], after[name]) == name.startswith("backbone."), name
    assert manifest_of(result.snapshot.checkpoint.location)["updatePolicy"] == "param_subset"


@pytest.mark.functional
def test_two_runs_with_one_seed_write_the_same_weights_to_separate_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, machine: fake.ProcessState
) -> None:
    monkeypatch.setattr(nanojev_trainer, "import_nanojev", lambda source, pins: (fake.nanojev_upstream(), "rev"))
    trainer, base = pinned(tmp_path, checkout=False)

    first = trainer.train(SFT, base, fake.teacher_batch())
    second = trainer.train(SFT, base, fake.teacher_batch())

    assert first.snapshot.checkpoint.checksum == second.snapshot.checkpoint.checksum
    assert first.snapshot.checkpoint.location != second.snapshot.checkpoint.location


# ---------------------------------------------------------------- grpo


@pytest.mark.functional
def test_grpo_keeps_the_final_weights_and_moves_toward_the_rewarded_choices(
    tmp_path: Path, machine: fake.ProcessState
) -> None:
    trainer, base = pinned(tmp_path, deterministic=True)
    batch = fake.rollout_batch()
    advantages = group_advantages(batch.decisions)

    result = trainer.train(GRPO, base, batch)

    manifest = manifest_of(result.snapshot.checkpoint.location)
    assert manifest["algorithm"] == "grpo" and manifest["kind"] == "trained-nanojev"
    assert (manifest["decisions"], manifest["decisionsWithoutSignal"], manifest["examples"]) == (15, 3, 12)
    assert (manifest["sourceReleases"], manifest["sourceHarnesses"]) == (["r0001"], ["seed"])
    assert len(manifest["history"]) == 3 and manifest["settings"]["clipRatio"] == 0.2
    start = chosen_probabilities(base.checkpoint.location, batch)
    assert manifest["meanBehaviorGap"] == pytest.approx(sum(abs(p - 0.4) for p in start[:12]) / 12, rel=1e-5)
    assert result.metrics["examples"] == 12.0 and result.metrics["decisions_without_signal"] == 3.0
    end = chosen_probabilities(result.snapshot.checkpoint.location, batch)
    moves = list(zip(advantages, start, end, strict=True))
    # What the policy gradient climbs: the advantage-weighted log-probability of the choices made.
    assert sum(advantage * math.log(after / before) for advantage, before, after in moves) > 0.3
    punished = [(before, after) for advantage, before, after in moves if advantage < 0]
    assert sum(after for _, after in punished) < sum(before for before, _ in punished)
    # Deterministic kernels for this job only: the process gets its own setting back after it (#171).
    assert machine.deterministic == [True, False] and "CUBLAS_WORKSPACE_CONFIG" not in os.environ


# ---------------------------------------------------------------- refusals


def error_of(tmp_path: Path) -> str:
    (record,) = (tmp_path / "runs").glob("train-*/error.json")
    assert not list((tmp_path / "runs").glob("train-*/checkpoint.json"))
    return json.loads(record.read_text())["error"]  # type: ignore[no-any-return]


@pytest.mark.functional
def test_a_base_whose_weights_changed_after_its_snapshot_is_refused_and_the_run_records_why(
    tmp_path: Path, machine: fake.ProcessState
) -> None:
    trainer, base = pinned(tmp_path)
    weights = Path(base.checkpoint.location) / "best.safetensors"
    weights.write_bytes(weights.read_bytes() + b" ")
    with pytest.raises(ContractError, match="weights differ from its checksum"):
        trainer.train(SFT, base, fake.teacher_batch())
    assert error_of(tmp_path) == "the base checkpoint's weights differ from its checksum"


@pytest.mark.functional
def test_a_base_on_another_qwen_revision_than_the_pin_is_refused(tmp_path: Path, machine: fake.ProcessState) -> None:
    trainer, _ = pinned(tmp_path)
    other = nanojev_snapshot(fake.nanojev_checkpoint(tmp_path / "other", base_revision="another"))
    with pytest.raises(ContractError, match=r"Qwen base differs from pins\.json"):
        trainer.train(SFT, other, fake.teacher_batch())


@pytest.mark.functional
def test_a_run_that_trains_nothing_or_changes_nothing_is_refused(tmp_path: Path, machine: fake.ProcessState) -> None:
    trainer, base = pinned(tmp_path)
    nothing = TrainJobSpec(
        "sft", contract="typed_decisions", update_policy=UpdatePolicy("param_subset", trainable_patterns=("^lm_head",))
    )
    with pytest.raises(ContractError, match="no parameter matches the update policy"):
        trainer.train(nothing, base, fake.teacher_batch())
    still, _ = pinned(tmp_path / "still", backbone_lr=0.0, head_lr=0.0)
    with pytest.raises(ContractError, match="training did not change the weights"):
        still.train(SFT, base, fake.teacher_batch())
    assert error_of(tmp_path / "still") == "training did not change the weights"


@pytest.mark.functional
def test_too_few_novel_validation_examples_are_refused_before_any_step(
    tmp_path: Path, machine: fake.ProcessState
) -> None:
    trainer, base = pinned(tmp_path, min_validation=7)
    with pytest.raises(ContractError, match="need at least 7 novel validation examples after deduplication, got 6"):
        trainer.train(SFT, base, fake.teacher_batch())


# ---------------------------------------------------------------- the pieces


def loaded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[NanoJevTrainer, Any]:
    monkeypatch.setattr(nanojev_trainer, "import_nanojev", lambda source, pins: (fake.nanojev_upstream(), "rev"))
    trainer, base = pinned(tmp_path, checkout=False)
    return trainer, trainer._load(base)


@pytest.mark.unit
def test_loading_asks_nanojev_for_one_bf16_device_and_checkpointed_backbone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, machine: fake.ProcessState
) -> None:
    _, state = loaded(monkeypatch, tmp_path)
    predictor = state.predictor
    assert (predictor.device_name, predictor.precision, predictor.limit) == ("cuda:0", "bf16", 2048)
    assert predictor.model.backbone.checkpointing == {"use_reentrant": False}
    assert state.pin["revision"] == "release-rev" and state.revision == "rev"


@pytest.mark.unit
def test_examples_are_nanojevs_candidate_paths_identified_by_their_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, machine: fake.ProcessState
) -> None:
    trainer, state = loaded(monkeypatch, tmp_path)
    decisions = fake.teacher_batch(train=3, validation=1).decisions
    repeat = fake.decision(0, "wait", "map:7:repeat")

    items, validation, packing = trainer._examples((*decisions, repeat), state)

    assert [x["label"] for x in items] == [0, 1, 2, 0] and [x["label"] for x in validation] == [0]
    assert all(x["markers"] == [] and x["candidate_ids"] == ["wait", "build", "attack"] for x in items)
    expected = hashlib.sha256(json.dumps(items[0]["leaf_tokens"], separators=(",", ":")).encode()).hexdigest()
    assert items[0]["ids"] == expected and items[3]["ids"] == expected and items[1]["ids"] != expected
    assert packing == {"maxInputTokens": max(len(p) for x in items + validation for p in x["leaf_tokens"]),
                       "trimmedRequests": 0}  # fmt: skip


@pytest.mark.unit
def test_the_backbone_and_the_heads_get_their_own_rates_and_frozen_weights_get_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, machine: fake.ProcessState
) -> None:
    trainer, state = loaded(monkeypatch, tmp_path)
    model = state.predictor.model
    full = trainer._optimizer(torch, model)
    assert [g["lr"] for g in full.param_groups] == [0.02, 0.05]
    assert [len(g["params"]) for g in full.param_groups] == [3, 3]  # embedding, layer weight and bias; norm, scalar
    assert all(g["weight_decay"] == 0.01 for g in full.param_groups)

    trainer._trainable(TrainJobSpec("sft", contract="typed_decisions", update_policy=HEADS), model)
    heads = trainer._optimizer(torch, model)
    assert [g["lr"] for g in heads.param_groups] == [0.05]
    assert {id(p) for p in heads.param_groups[0]["params"]} == {
        id(p) for name, p in model.named_parameters() if not name.startswith("backbone.")
    }


@pytest.mark.unit
def test_the_release_comes_from_the_pinned_revision_with_only_the_files_nanojev_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def snapshot_download(repo: str, *, revision: str, allow_patterns: list[str]) -> str:
        calls.append((repo, revision, allow_patterns))
        return str(tmp_path / "cache")

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download))
    pins = json.loads(fake.nanojev_pins(tmp_path / "pins.json", "source-rev").read_text())

    assert nanojev_trainer.release_path(pins) == tmp_path / "cache"
    assert calls == [("C-Tianyu/NanoJev", "release-rev", DOWNLOAD_PATTERNS)]
