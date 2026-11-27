"""The direction-edit trainer: the direction it reads, the checkpoint it writes, what it refuses, its place in core."""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
safetensors = pytest.importorskip("safetensors.numpy")
pytest.importorskip("torch")

from honing.backends.jev.decision_training import laya_question  # noqa: E402
from honing.backends.jev.direction_edit import (  # noqa: E402
    HEAD_WEIGHT,
    SUPPORT,
    DirectionEditSettings,
    NanoJevDirectionEditTrainer,
    _target_share,
    direction,
    edited_head,
)
from honing.backends.jev.nanojev import nanojev_snapshot, payload  # noqa: E402
from honing.backends.jev.nanojev_serving import trained_manifest  # noqa: E402
from honing.campaign import rules  # noqa: E402
from honing.core import (  # noqa: E402
    ContractError,
    DataContract,
    DecisionQuestion,
    DecisionSample,
    DirectionEditSpec,
    TrainingBatch,
    TrainJobSpec,
    TrainResult,
)
from honing.core.training import requirements_for  # noqa: E402
from tests.fakes import decision_models as fake  # noqa: E402

HIDDEN = 8
RNG = np.random.default_rng(3)


def _inputs(rows: int = 20, question: str = "weapon") -> tuple[np.ndarray, list[dict[str, object]]]:
    """Head inputs whose fire option sits one unit along axis 0 from its hold option, plus fixed noise."""
    rng = np.random.default_rng(11)
    h_in = np.zeros((rows, 9, HIDDEN), dtype=np.float32)
    meta = []
    for i in range(rows):
        base = rng.normal(size=HIDDEN).astype(np.float32)
        h_in[i, 0] = base + np.eye(HIDDEN, dtype=np.float32)[0] + 0.05 * rng.normal(size=HIDDEN)
        h_in[i, 1] = base
        meta.append({"question": question, "options": ["fire", "hold"], "label": 1})
    return h_in, meta


@pytest.mark.unit
def test_the_direction_is_the_mean_target_minus_others_as_a_unit_vector() -> None:
    inputs = _inputs()
    unit, stats = direction(inputs, "weapon", "fire")
    assert unit.shape == (HIDDEN,) and abs(float(np.linalg.norm(unit)) - 1) < 1e-5
    assert unit[0] > 0.95 and stats["questions"] == 20 and 0.9 < stats["margin_per_unit"] < 1.1
    # The label direction points the other way: every label is hold.
    toward_label, _ = direction(inputs, None, "label")
    assert toward_label[0] < -0.95
    with pytest.raises(ContractError, match="no question in the batch offers a direction toward 'left'"):
        direction(inputs, "weapon", "left")
    with pytest.raises(ContractError, match="on question 'motion'"):
        direction(inputs, "motion", "fire")


@pytest.mark.unit
def test_edited_head_keeps_shape_and_dtype() -> None:
    weight = np.zeros((1, HIDDEN), dtype=np.float16)
    unit = np.eye(HIDDEN, dtype=np.float32)[2]
    out = edited_head(weight, unit, 2.5)
    assert out.shape == (1, HIDDEN) and out.dtype == np.float16 and float(out[0, 2]) == 2.5 and float(out.sum()) == 2.5


def _base_checkpoint(root: Path) -> Path:
    """A checkpoint with NanoJev's layout and a tiny head, enough for the trainer and nanojev_snapshot."""
    checkpoint = root / "release"
    (checkpoint / "backbone_config").mkdir(parents=True)
    (checkpoint / "tokenizer").mkdir()
    (checkpoint / "tokenizer" / "tokenizer.json").write_text("{}")
    (checkpoint / "config.json").write_text(json.dumps({"model": "Qwen/Qwen3-0.6B", "set_head": "attention"}))
    (checkpoint / "backbone_config" / "config.json").write_text(
        json.dumps({"hidden_size": HIDDEN, "num_attention_heads": 2, "intermediate_size": 16, "num_hidden_layers": 1})
    )
    state = {
        HEAD_WEIGHT: (0.1 * RNG.normal(size=(1, HIDDEN))).astype(np.float32),
        "scalar.bias": np.zeros((1,), dtype=np.float32),
        "norm.weight": np.ones((HIDDEN,), dtype=np.float32),
        "backbone.embed_tokens.weight": RNG.normal(size=(4, HIDDEN)).astype(np.float32),
    }
    safetensors.save_file(state, str(checkpoint / "best.safetensors"))
    return checkpoint


def _pins(root: Path) -> Path:
    pins = root / "pins.json"
    pins.write_text(
        json.dumps(
            {
                "nanojevSource": {"url": "x", "revision": "source-rev"},
                "nanojevModels": {
                    "nanojev": {
                        "repo": "C-Tianyu/NanoJev",
                        "revision": "release-rev",
                        "base": "b",
                        "baseRevision": "r",
                    }
                },
            }
        )
    )
    return pins


def _batch(rows: int = 20) -> TrainingBatch:
    question = DecisionQuestion(kind="choice", instructions="Fire?", options=(("fire", "Fire"), ("hold", "Hold")))
    decisions = tuple(
        DecisionSample(
            parts=(("essential", (f"state {i}",)),),
            questions=(("weapon", question),),
            chosen=(("weapon", "hold"),),
            behavior_probs=None,
            reward=None,
            group=None,
            source_release=None,
            source_harness="seed",
        )
        for i in range(rows)
    )
    return TrainingBatch("teacher", DataContract("typed_decisions"), decisions=decisions)


def _trainer(tmp_path: Path, target: str = "fire") -> NanoJevDirectionEditTrainer:
    settings = DirectionEditSettings(
        pins_path=_pins(tmp_path),
        source_dir=tmp_path,
        work_dir=tmp_path / "edits",
        question="weapon",
        target=target,
        strength=1.0,
    )
    return NanoJevDirectionEditTrainer(settings, reader=lambda decisions, parent: _inputs(len(decisions)))


@pytest.mark.unit
def test_the_trainer_changes_only_the_scalar_head_and_writes_a_servable_manifest(tmp_path: Path) -> None:
    checkpoint = _base_checkpoint(tmp_path)
    base = nanojev_snapshot(checkpoint)
    trainer = _trainer(tmp_path)
    job = TrainJobSpec("direction_edit", contract="typed_decisions")

    result = trainer.train(job, base, _batch())

    edited = Path(result.snapshot.checkpoint.location)
    assert result.snapshot.checkpoint.checksum != base.checkpoint.checksum
    assert result.snapshot.architecture == base.architecture and result.snapshot.parameters == base.parameters
    before = safetensors.load_file(str(checkpoint / "best.safetensors"))
    after = safetensors.load_file(str(edited / "best.safetensors"))
    assert set(before) == set(after)
    for name in before:
        if name != HEAD_WEIGHT:
            assert np.array_equal(before[name], after[name]), name
    unit, _ = direction(_inputs(), "weapon", "fire")
    assert np.allclose(after[HEAD_WEIGHT], before[HEAD_WEIGHT] + unit, atol=1e-6)
    assert result.metrics["entries_changed"] == HIDDEN and result.metrics["questions"] == 20
    # The edit's whole run is charged as one GPU's time, so a usd budget can price it.
    assert result.gpu_seconds is not None and result.gpu_seconds > 0
    assert result.metrics["target_argmax_after"] == 1.0 and 0.9 < result.metrics["margin_added"] < 1.1
    manifest_path = trained_manifest(edited)
    assert manifest_path is not None
    manifest = json.loads(manifest_path.read_text())
    assert manifest["kind"] == "trained-nanojev" and manifest["weightsSha256"] == result.snapshot.checkpoint.checksum
    assert manifest["parent"] == {"repo": "C-Tianyu/NanoJev", "revision": "release-rev"}
    assert manifest["sourceRevision"] == "source-rev" and manifest["edit"]["tensor"] == HEAD_WEIGHT
    assert (edited / "tokenizer" / "tokenizer.json").exists() and (edited / "backbone_config" / "config.json").exists()
    assert trained_manifest(checkpoint) is None
    # Compare the complete weight file with the pre-refactor NumPy writer.
    expected = dict(before)
    expected[HEAD_WEIGHT] = edited_head(before[HEAD_WEIGHT], unit, 1.0)
    safetensors.save_file(expected, str(tmp_path / "legacy.safetensors"))
    assert (edited / "best.safetensors").read_bytes() == (tmp_path / "legacy.safetensors").read_bytes()


@pytest.mark.unit
def test_the_trainer_refuses_other_jobs_and_empty_batches(tmp_path: Path) -> None:
    base = nanojev_snapshot(_base_checkpoint(tmp_path))
    trainer = _trainer(tmp_path)
    job = TrainJobSpec("direction_edit", contract="typed_decisions")
    with pytest.raises(ContractError, match="direction_edit on typed_decisions only"):
        trainer.train(TrainJobSpec("sft", contract="typed_decisions"), base, _batch())
    with pytest.raises(ContractError, match="needs a typed_decisions batch"):
        trainer.train(job, base, None)
    with pytest.raises(ContractError, match="decisions only"):
        TrainingBatch("empty", DataContract("typed_decisions"))
    with pytest.raises(ContractError, match="no question in the batch offers a direction toward 'left'"):
        _trainer(tmp_path, target="left").train(job, base, _batch())
    assert not list((tmp_path / "edits").glob("*/checkpoint/best.safetensors"))
    with pytest.raises(ValueError, match="strength"):
        DirectionEditSettings(pins_path=tmp_path, source_dir=tmp_path, work_dir=tmp_path, strength=0.0)


@pytest.mark.unit
def test_direction_edit_is_a_decision_algorithm_the_rules_accept() -> None:
    job = TrainJobSpec("direction_edit", contract="typed_decisions")
    requirements = requirements_for(job)
    assert requirements.contract == "typed_decisions"
    assert not (requirements.requires_rollout or requirements.requires_teacher or requirements.requires_reward)
    assert SUPPORT.support_for("direction_edit") == "native" and "typed_decisions" in SUPPORT.contracts
    rules.algorithm_requirements_met(job, SUPPORT)
    with pytest.raises(ContractError, match="does not train on token_sequences"):
        TrainJobSpec("direction_edit", contract="token_sequences")


@pytest.mark.unit
def test_general_nanojev_factory_edits_other_parameters_and_keeps_serving_identity(tmp_path: Path) -> None:
    from honing.backends.jev.weight_edit import weight_edit_trainer
    from honing.e2e.components import PipelineContext

    checkpoint = _base_checkpoint(tmp_path)
    base = nanojev_snapshot(checkpoint)
    context = PipelineContext(root=tmp_path, work_dir=tmp_path / "generic")
    trainer = weight_edit_trainer(
        context,
        pins=str(_pins(tmp_path)),
        edits=[
            {"parameter": "norm.weight", "operation": "scale", "value": 1.5},
            {"parameter": "scalar.bias", "operation": "replace", "value": [0.25]},
        ],
    )
    result = trainer.train(TrainJobSpec("weight_edit"), base, None)
    output = Path(result.snapshot.checkpoint.location)
    before = safetensors.load_file(str(checkpoint / "best.safetensors"))
    after = safetensors.load_file(str(output / "best.safetensors"))
    assert np.array_equal(after["norm.weight"], before["norm.weight"] * 1.5)
    assert after["scalar.bias"].tolist() == [0.25]
    assert np.array_equal(after[HEAD_WEIGHT], before[HEAD_WEIGHT])
    assert nanojev_snapshot(output) == result.snapshot
    manifest = trained_manifest(output)
    assert manifest is not None and json.loads(manifest.read_text())["algorithm"] == "weight_edit"


@pytest.mark.unit
def test_a_job_with_its_own_spec_edits_along_that_direction_and_the_next_job_uses_the_settings(tmp_path: Path) -> None:
    checkpoint = _base_checkpoint(tmp_path)
    base = nanojev_snapshot(checkpoint)
    trainer = _trainer(tmp_path)  # settings: question weapon, target fire, strength 1.0
    before = safetensors.load_file(str(checkpoint / "best.safetensors"))[HEAD_WEIGHT]
    toward_hold = DirectionEditSpec(question="weapon", target="hold", strength=0.5)

    own = trainer.train(
        TrainJobSpec("direction_edit", contract="typed_decisions", direction_edit=toward_hold), base, _batch()
    )
    default = trainer.train(TrainJobSpec("direction_edit", contract="typed_decisions"), base, _batch())

    def head(result: TrainResult) -> np.ndarray:
        return safetensors.load_file(str(Path(result.snapshot.checkpoint.location) / "best.safetensors"))[HEAD_WEIGHT]

    def edit(result: TrainResult) -> dict[str, object]:
        manifest = trained_manifest(Path(result.snapshot.checkpoint.location))
        assert manifest is not None
        return json.loads(manifest.read_text())["edit"]  # type: ignore[no-any-return]

    hold, _ = direction(_inputs(), "weapon", "hold")
    fire, _ = direction(_inputs(), "weapon", "fire")
    assert np.allclose(head(own), before + 0.5 * hold, atol=1e-6)
    assert np.allclose(head(default), before + fire, atol=1e-6)
    assert (edit(own)["target"], edit(own)["strength"]) == ("hold", 0.5)
    assert (edit(default)["target"], edit(default)["strength"]) == ("fire", 1.0)
    with pytest.raises(ContractError, match="no question in the batch offers a direction toward 'fire' on question"):
        other = DirectionEditSpec(question="motion", target="fire", strength=1.0)
        trainer.train(TrainJobSpec("direction_edit", contract="typed_decisions", direction_edit=other), base, _batch())


# ---------------------------------------------------------------- settings, the direction and the plan's refusals


@pytest.mark.unit
@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"strength": float("nan")}, "strength must be a finite, nonzero number"),
        ({"max_questions": 0}, "max_questions and max_length must be positive"),
        ({"max_length": -1}, "max_questions and max_length must be positive"),
        ({"target": ""}, "target must be 'label' or an option id"),
    ],
)
def test_settings_refuse_a_direction_they_cannot_read(
    tmp_path: Path, settings: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        DirectionEditSettings(pins_path=tmp_path, source_dir=tmp_path, work_dir=tmp_path, **settings)  # type: ignore[arg-type]


@pytest.mark.unit
def test_a_question_with_one_option_gives_no_direction_and_coinciding_options_are_refused() -> None:
    h_in, meta = _inputs(rows=4)
    single = [{**m, "options": ["fire"], "label": 0} for m in meta]
    with pytest.raises(ContractError, match="no question in the batch offers a direction toward 'fire'"):
        direction((h_in, single), None, "fire")
    unit, stats = direction((h_in, meta[:2] + single[2:]), None, "fire")
    assert stats["questions"] == 2 and unit[0] > 0.95
    same = np.repeat(h_in[:, :1], 9, axis=1)
    with pytest.raises(ContractError, match="the target and the other options coincide"):
        direction((same, meta), None, "fire")


@pytest.mark.unit
def test_the_target_share_counts_only_the_selected_questions_that_offer_the_target() -> None:
    h_in, meta = _inputs(rows=4)
    meta[3] = {**meta[3], "question": "motion"}
    toward_fire = np.eye(HIDDEN, dtype=np.float32)[0]
    assert _target_share((h_in, meta), toward_fire, "weapon", "fire") == 1.0
    assert _target_share((h_in, meta), -toward_fire, None, "fire") == 0.0
    assert _target_share((h_in, meta), -toward_fire, None, "label") == 1.0  # every label is hold
    assert np.isnan(_target_share((h_in, meta), toward_fire, "weapon", "left"))


@pytest.mark.unit
def test_the_reader_sees_at_most_max_questions_decisions(tmp_path: Path) -> None:
    base = nanojev_snapshot(_base_checkpoint(tmp_path))
    seen: list[int] = []

    def reader(decisions: object, parent: Path) -> tuple[np.ndarray, list[dict[str, object]]]:
        seen.append(len(decisions))  # type: ignore[arg-type]
        return _inputs(len(decisions))  # type: ignore[arg-type]

    settings = DirectionEditSettings(
        pins_path=_pins(tmp_path), source_dir=tmp_path, work_dir=tmp_path / "edits", target="fire", max_questions=5
    )
    result = NanoJevDirectionEditTrainer(settings, reader=reader).train(
        TrainJobSpec("direction_edit", contract="typed_decisions"), base, _batch(20)
    )
    assert seen == [5] and result.metrics["questions"] == 5


@pytest.mark.unit
@pytest.mark.parametrize(
    ("head", "message"),
    [(None, "the base checkpoint has no scalar.weight"), ((1, 4), "scalar.weight has 4 entries, the head's input 8")],
)
def test_a_head_the_direction_does_not_fit_is_refused_and_leaves_no_candidate(
    tmp_path: Path, head: tuple[int, int] | None, message: str
) -> None:
    checkpoint = _base_checkpoint(tmp_path)
    state = safetensors.load_file(str(checkpoint / "best.safetensors"))
    del state[HEAD_WEIGHT]
    if head is not None:
        state[HEAD_WEIGHT] = np.ones(head, dtype=np.float32)
    safetensors.save_file(state, str(checkpoint / "best.safetensors"))
    with pytest.raises(ContractError, match=message):
        _trainer(tmp_path).train(
            TrainJobSpec("direction_edit", contract="typed_decisions"), nanojev_snapshot(checkpoint), _batch()
        )
    assert not list((tmp_path / "edits").glob("*/checkpoint"))


# ---------------------------------------------------------------- the forward pass, on a stand-in NanoJev


@pytest.fixture
def nanojev(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """NanoJev's predictor module and a GPU lock that need no GPU; tensors sent to cuda stay on the CPU."""
    from honing.backends.jev import direction_edit as module
    from honing.backends.jev import laya

    fake.keep_process_state(monkeypatch)
    monkeypatch.setattr(module, "import_nanojev", lambda source, pins: (fake.nanojev_upstream(), "rev"))
    monkeypatch.setattr(laya, "gpu_lock", fake.any_gpu)
    with fake.cuda_on_cpu():
        yield


def _two_question_decisions(rows: int) -> tuple[DecisionSample, ...]:
    """Decisions that ask the three-option question, the second and every fourth also the two-option weapon one."""
    weapon = DecisionQuestion(kind="choice", instructions="Fire?", options=(("fire", "Fire"), ("hold", "Hold")))
    out = []
    for i in range(rows):
        sample = fake.decision(i, ("wait", "build", "attack")[i % 3], f"m:7:{i}")
        if i % 4 == 1:
            sample = dataclasses.replace(
                sample, questions=(*sample.questions, ("weapon", weapon)), chosen=(*sample.chosen, ("weapon", "hold"))
            )
        out.append(sample)
    return tuple(out)


@pytest.mark.unit
def test_the_reader_keeps_the_input_the_scalar_head_reads_for_every_option(tmp_path: Path, nanojev: None) -> None:
    import torch

    checkpoint = fake.nanojev_checkpoint(tmp_path / "release")
    decisions = _two_question_decisions(10)

    h_in, meta = _trainer(tmp_path)._read(decisions, checkpoint)

    assert h_in.shape == (13, 3, fake.HIDDEN) and h_in.dtype == np.float32 and len(meta) == 13
    assert [m["question"] for m in meta[:3]] == ["next", "next", "weapon"]
    assert meta[2] == {"question": "weapon", "options": ["fire", "hold"], "label": 1}
    assert [m["label"] for m in meta if m["question"] == "next"] == [i % 3 for i in range(10)]
    assert not h_in[2, 2].any()  # the two-option question has no third option
    # The kept vectors are exactly what the scalar head scores: weight . h + bias reproduces the model's logits.
    predictor = fake.DecisionPredictor(str(checkpoint), 4096, "cpu", "fp32")
    weight = safetensors.load_file(str(checkpoint / "best.safetensors"))
    examples = [
        example
        for d in decisions
        for example in fake.prepare_examples(
            payload("\n".join(d.parts[0][1]), {k: laya_question(q) for k, q in d.questions}), predictor.tokenizer, 4096
        )
    ]
    with torch.no_grad():
        logits, _ = predictor.model(examples, 0)
    for row, example in enumerate(examples):
        k = len(example["candidate_ids"])
        scores = h_in[row, :k] @ weight[HEAD_WEIGHT][0] + weight["scalar.bias"][0]
        assert np.allclose(scores, logits[row, :k].numpy(), atol=1e-5)
    assert os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["TRANSFORMERS_OFFLINE"] == "1"


@pytest.mark.functional
def test_a_trainer_without_a_reader_edits_along_what_the_model_itself_reads(tmp_path: Path, nanojev: None) -> None:
    checkpoint = fake.nanojev_checkpoint(tmp_path / "release")
    settings = DirectionEditSettings(
        pins_path=_pins(tmp_path), source_dir=tmp_path, work_dir=tmp_path / "edits", question="next", target="build"
    )
    trainer = NanoJevDirectionEditTrainer(settings)
    decisions = _two_question_decisions(10)
    batch = TrainingBatch("teacher", DataContract("typed_decisions"), decisions=decisions)

    result = trainer.train(
        TrainJobSpec("direction_edit", contract="typed_decisions"), nanojev_snapshot(checkpoint), batch
    )

    unit, stats = direction(trainer._read(decisions, checkpoint), "next", "build")
    before = safetensors.load_file(str(checkpoint / "best.safetensors"))[HEAD_WEIGHT]
    after = safetensors.load_file(str(Path(result.snapshot.checkpoint.location) / "best.safetensors"))[HEAD_WEIGHT]
    assert np.allclose(after, before + unit, atol=1e-6)
    assert result.metrics["questions"] == stats["questions"] == 10
    assert result.metrics["margin_added"] == pytest.approx(stats["margin_per_unit"])
    assert 0.0 <= result.metrics["target_argmax_before"] <= result.metrics["target_argmax_after"] <= 1.0
    manifest = trained_manifest(Path(result.snapshot.checkpoint.location))
    assert manifest is not None
    assert json.loads(manifest.read_text())["settings"] == {
        "model": "nanojev",
        "maxQuestions": 1024,
        "maxLength": 4096,
    }
