"""The release chain's ordering guarantee and the gate's refusal to score the wrong weights."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from honing.core import (
    CheckpointRef,
    ContractError,
    EvaluationResult,
    HoneRelease,
    ModelSnapshot,
    ReleaseError,
    SelectionDecision,
    ServingWeights,
    UpdateCandidate,
)
from honing.release import AcceptAll, Gate, NoRegressionSelector, ReleaseChain, WinLossSelector
from tests.fakes.builder import INT4_G128
from tests.fakes.harness import SEED_HARNESS
from tests.fakes.model import parameters_of, small_moe

SELECT = SelectionDecision(outcome="select", policy="test", reason="chosen")
REJECT = SelectionDecision(outcome="reject", policy="test", reason="not chosen")


def snapshot(checksum: str) -> ModelSnapshot:
    architecture = small_moe()
    ref = CheckpointRef(location=f"fake://{checksum}", weight_format="hf_safetensors", checksum=checksum)
    return ModelSnapshot(ref, architecture, parameters_of(architecture))


def started_chain(tmp_path: Path) -> tuple[ReleaseChain, HoneRelease]:
    chain = ReleaseChain(tmp_path)
    initial = chain.initialize(
        UpdateCandidate(kind="init", parent_id=None, master=snapshot("a"), harness=SEED_HARNESS)
    )
    return chain, initial


@pytest.mark.unit
def test_the_commit_record_reaches_disk_before_the_head_moves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chain, initial = started_chain(tmp_path)

    def crash(release_id: str) -> None:
        raise RuntimeError("crashed between writing the record and moving the head")

    monkeypatch.setattr(chain, "_move_head", crash)
    candidate = UpdateCandidate(kind="train", parent_id=initial.release_id, master=snapshot("b"))

    with pytest.raises(RuntimeError):
        chain.commit(candidate, SELECT)

    last = json.loads(chain.record_path.read_text(encoding="utf-8").splitlines()[-1])
    assert (last["op"], last["release_id"]) == ("publish", "r0001")
    assert chain.head() is initial


@pytest.mark.unit
def test_a_candidate_built_on_an_old_head_is_refused(tmp_path: Path) -> None:
    chain, initial = started_chain(tmp_path)
    first = UpdateCandidate(kind="train", parent_id=initial.release_id, master=snapshot("b"))
    chain.commit(first, SELECT)

    with pytest.raises(ReleaseError, match="rebuild it"):
        chain.commit(UpdateCandidate(kind="train", parent_id=initial.release_id, master=snapshot("c")), SELECT)


@pytest.mark.unit
def test_a_rejected_decision_is_never_committed(tmp_path: Path) -> None:
    chain, initial = started_chain(tmp_path)
    with pytest.raises(ReleaseError, match="rejected"):
        chain.commit(UpdateCandidate(kind="train", parent_id=initial.release_id, master=snapshot("b")), REJECT)


@pytest.mark.unit
def test_rollback_to_an_unknown_release_is_refused(tmp_path: Path) -> None:
    chain, _ = started_chain(tmp_path)
    with pytest.raises(ReleaseError, match="no release"):
        chain.rollback("r0042")


@pytest.mark.unit
def test_a_chain_is_initialized_exactly_once(tmp_path: Path) -> None:
    chain = ReleaseChain(tmp_path)
    with pytest.raises(ReleaseError, match="initialize"):
        chain.head()
    chain.initialize(UpdateCandidate(kind="init", parent_id=None, master=snapshot("a")))
    with pytest.raises(ReleaseError, match="already"):
        chain.initialize(UpdateCandidate(kind="init", parent_id=None, master=snapshot("b")))


class MasterOnlyEvaluator:
    """An evaluator that always reports scoring the master, even when the candidate ships serving weights."""

    def evaluate(self, candidate: UpdateCandidate, incumbent: HoneRelease) -> EvaluationResult:
        return EvaluationResult(
            task_ids=("t",), candidate_scores=(1.0,), incumbent_scores=(0.0,), evaluated_variant="master"
        )


@pytest.mark.unit
def test_the_gate_refuses_an_evaluation_of_the_wrong_weights(tmp_path: Path) -> None:
    _, initial = started_chain(tmp_path)
    master = initial.master
    serving = ServingWeights(checkpoint=snapshot("q").checkpoint, scheme=INT4_G128, source=master.checkpoint)
    candidate = UpdateCandidate(kind="quantize", parent_id=initial.release_id, master=master, serving=serving)

    with pytest.raises(ContractError, match="serving"):
        Gate(MasterOnlyEvaluator(), AcceptAll()).judge(candidate, initial)


def evaluation(candidate: tuple[float, ...], incumbent: tuple[float, ...]) -> EvaluationResult:
    task_ids = tuple(f"t{index}" for index in range(len(candidate)))
    return EvaluationResult(
        task_ids=task_ids, candidate_scores=candidate, incumbent_scores=incumbent, evaluated_variant="master"
    )


@pytest.mark.unit
def test_win_loss_rejects_a_tie_on_every_task() -> None:
    candidate = UpdateCandidate(kind="train", parent_id="r0000", master=snapshot("b"))
    assert not WinLossSelector().decide(candidate, evaluation((0.5, 0.5), (0.5, 0.5))).selected
    assert WinLossSelector().decide(candidate, evaluation((0.6, 0.5), (0.5, 0.5))).selected
    assert not WinLossSelector(min_win_margin=1).decide(candidate, evaluation((0.6, 0.5), (0.5, 0.5))).selected


@pytest.mark.unit
def test_no_regression_accepts_a_drop_up_to_its_limit() -> None:
    candidate = UpdateCandidate(kind="prune", parent_id="r0000", master=snapshot("b"))
    selector = NoRegressionSelector(max_regression=0.05)
    assert selector.decide(candidate, evaluation((0.46,), (0.5,))).selected
    assert not selector.decide(candidate, evaluation((0.40,), (0.5,))).selected
