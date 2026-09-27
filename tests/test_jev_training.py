"""Training a decision model: RA2 demonstrations as typed decisions, the local trainer's refusals, and a
campaign that trains on decisions through whichever trainer it is given. No GPU."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from honing.backends.jev.laya import laya_snapshot, safetensors_parameters
from honing.backends.jev.trainer import JevTrainerSettings, LocalJevTrainer, laya_question, separate_unique
from honing.backends.ra2 import RA2DemonstrationProvider
from honing.campaign import ScriptedScheduler
from honing.core import (
    BackendSupport,
    ContractError,
    DataContract,
    DecisionQuestion,
    Move,
    TrainingBatch,
    TrainJobSpec,
    UpdatePolicy,
)
from tests.fakes.builder import build_campaign
from tests.fakes.data import token_batch
from tests.fakes.ra2.demos import PROTOCOL, write_match

DECISIONS = DataContract("typed_decisions")


# ------------------------------------------------------------------ demonstrations


@pytest.mark.unit
def test_labelled_matches_become_one_batch_of_decisions_in_order(tmp_path: Path) -> None:
    provider = RA2DemonstrationProvider([write_match(tmp_path, 100), write_match(tmp_path, 101)], PROTOCOL)
    batch = provider.next_batch(DECISIONS)
    assert batch is not None and len(batch.decisions) == 6
    first = batch.decisions[0]
    assert first.group == "mp03t4.map:100:red"
    assert dict(first.chosen) == {"g1": "A"} and dict(batch.decisions[1].chosen) == {"g1": "B"}
    assert dict(first.questions)["g1"].options[1] == ("B", "Build a power plant")
    assert dict(first.parts)["essential"] == ("Tick 3. Seed 100.",)
    assert provider.next_batch(DECISIONS) is None


@pytest.mark.unit
def test_a_language_model_contract_gets_nothing(tmp_path: Path) -> None:
    assert (
        RA2DemonstrationProvider([write_match(tmp_path, 100)], PROTOCOL).next_batch(DataContract("token_sequences"))
        is None
    )


@pytest.mark.unit
def test_a_match_outside_the_training_protocol_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="not a training scenario"):
        RA2DemonstrationProvider([write_match(tmp_path, 999)], PROTOCOL).next_batch(DECISIONS)


@pytest.mark.unit
def test_a_model_evaluation_is_not_training_data(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="not labelled training data"):
        RA2DemonstrationProvider([write_match(tmp_path, 100, kind="model-evaluation")], PROTOCOL).next_batch(DECISIONS)


@pytest.mark.unit
def test_a_tampered_source_file_is_refused(tmp_path: Path) -> None:
    match = write_match(tmp_path, 100)
    (match / "source" / "teacher.mjs").write_text("// edited\n")
    with pytest.raises(ContractError, match="differs from its recorded hash"):
        RA2DemonstrationProvider([match], PROTOCOL).next_batch(DECISIONS)


@pytest.mark.unit
def test_a_row_from_another_scenario_is_refused(tmp_path: Path) -> None:
    match = write_match(tmp_path, 100)
    rows = (match / "samples.jsonl").read_text().splitlines()
    wrong = json.loads(rows[0]) | {"seed": 102}
    (match / "samples.jsonl").write_text("\n".join([json.dumps(wrong), *rows[1:]]) + "\n")
    with pytest.raises(ContractError, match="names another scenario"):
        RA2DemonstrationProvider([match], PROTOCOL).next_batch(DECISIONS)


# ------------------------------------------------------------------ the local trainer, before any GPU work


def _trainer(tmp_path: Path) -> LocalJevTrainer:
    return LocalJevTrainer(
        JevTrainerSettings(
            laya_source=tmp_path, pins_path=tmp_path / "pins.json", model_record=tmp_path / "m.json", work_dir=tmp_path
        )
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("job", "message"),
    [
        (TrainJobSpec("sft"), "sft on typed_decisions only"),
        (TrainJobSpec("grpo", rollout="harness", contract="typed_decisions"), "sft on typed_decisions only"),
        (
            TrainJobSpec("sft", contract="typed_decisions", update_policy=UpdatePolicy(kind="lora", lora_rank=8)),
            "cannot apply a lora update",
        ),
    ],
)
def test_the_trainer_refuses_jobs_it_cannot_run(tmp_path: Path, job: TrainJobSpec, message: str) -> None:
    batch = RA2DemonstrationProvider([write_match(tmp_path / "d", 100)], PROTOCOL).next_batch(DECISIONS)
    with pytest.raises(ContractError, match=message):
        _trainer(tmp_path).train(job, None, batch)  # type: ignore[arg-type]


@pytest.mark.unit
def test_the_trainer_refuses_a_missing_or_language_model_batch(tmp_path: Path) -> None:
    job = TrainJobSpec("sft", contract="typed_decisions")
    for batch in (None, token_batch()):
        with pytest.raises(ContractError, match="needs a typed_decisions batch"):
            _trainer(tmp_path).train(job, None, batch)  # type: ignore[arg-type]


@pytest.mark.unit
def test_the_trainer_declares_what_it_can_train() -> None:
    support = LocalJevTrainer.support
    assert support.contracts == frozenset({"typed_decisions"}) and support.support_for("sft") == "native"


@pytest.mark.unit
def test_duplicate_inputs_collapse_and_validation_keeps_only_what_training_never_saw() -> None:
    a = {"ids": [1, 2], "markers": [0], "label": 0}
    b = {"ids": [3], "markers": [0], "label": 1}
    train, valid, audit = separate_unique([a, dict(a), b], [dict(b), {"ids": [9], "markers": [0], "label": 0}])
    assert (len(train), len(valid), audit["removedOverlap"]) == (2, 1, 1)
    with pytest.raises(ContractError, match="Conflicting labels"):
        separate_unique([a, a | {"label": 1}], [])


@pytest.mark.unit
def test_only_choice_questions_reach_the_laya_trainer() -> None:
    choice = DecisionQuestion("choice", "Next?", (("a", "A"), ("b", "B")))
    assert laya_question(choice) == {"type": "choice", "instructions": "Next?", "criteria": {"a": "A", "b": "B"}}
    with pytest.raises(ContractError, match="choice questions only"):
        laya_question(DecisionQuestion("score", "Rate", (("1", "Low"), ("2", "High"))))


@pytest.mark.unit
def test_a_checkpoint_directory_becomes_a_snapshot(tmp_path: Path) -> None:
    (tmp_path / "encoder").mkdir()
    (tmp_path / "encoder/config.json").write_text(
        json.dumps({"hidden_size": 8, "num_attention_heads": 2, "intermediate_size": 16, "num_hidden_layers": 3})
    )
    header = json.dumps(
        {"a": {"dtype": "BF16", "shape": [4, 3], "data_offsets": [0, 24]}, "__metadata__": {"format": "pt"}}
    ).encode()
    (tmp_path / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + bytes(24))
    snapshot = laya_snapshot(tmp_path)
    assert safetensors_parameters(tmp_path / "model.safetensors") == 12 == snapshot.parameters
    assert snapshot.architecture.hidden_size == 8 and len(snapshot.architecture.layers) == 3
    assert snapshot.checkpoint.weight_format == "hf_safetensors" and len(snapshot.checkpoint.checksum) == 64


# ------------------------------------------------------------------ a campaign trains on decisions via any trainer


@pytest.mark.unit
def test_a_campaign_trains_on_ra2_decisions_through_a_swapped_in_trainer(tmp_path: Path) -> None:
    decision_trainer = BackendSupport(
        name="fake-jev",
        architectures="uniform",
        train_numerics=frozenset({"bf16"}),
        weight_formats=frozenset({"hf_safetensors"}),
        algorithms=(("sft", "native"),),
        contracts=frozenset({"typed_decisions"}),
    )
    matches = [write_match(tmp_path / "d", 100), write_match(tmp_path / "d", 102)]
    fixture = build_campaign(
        tmp_path / "c", support=decision_trainer, data=RA2DemonstrationProvider(matches, PROTOCOL)
    )
    job = TrainJobSpec("sft", steps=2, contract="typed_decisions")

    (outcome,) = fixture.campaign.run(ScriptedScheduler([Move(kind="train", train=job)]))

    assert outcome.release is not None and outcome.release.kind == "train"
    (batch,) = fixture.trainer.batches
    assert isinstance(batch, TrainingBatch) and batch.contract.kind == "typed_decisions" and len(batch.decisions) == 6


@pytest.mark.unit
def test_the_gpu_lock_restores_the_visible_devices(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from honing.backends.jev import laya

    def nvidia_smi(command: list[str], text: bool) -> str:
        return "0, GPU-aaa\n1, GPU-bbb\n" if "--query-gpu=index,uuid" in command else ""

    monkeypatch.setattr(laya.subprocess, "check_output", nvidia_smi)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    with laya.gpu_lock("1", tmp_path) as uuid:
        assert uuid == "GPU-bbb" and os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-bbb"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    with laya.gpu_lock("0", tmp_path):
        pass
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
