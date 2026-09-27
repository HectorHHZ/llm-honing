"""The duel evaluator (#58): candidate against incumbent in both seats, with no engine or Node."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from honing.backends.ra2 import DuelEvaluator, load_suite
from honing.core import ContractError, HarnessFile, HoneRelease, UpdateCandidate
from honing.release import Gate, WinLossSelector
from tests.test_ra2_backend import FAKES, REPO, SPEC, SUITE_PATH, _endpoints, _snapshot

TASKS = load_suite(SUITE_PATH).tasks("val", limit=2)


def evaluator(tmp_path: Path, **urls: str) -> DuelEvaluator:
    return DuelEvaluator(
        suite=load_suite(SUITE_PATH),
        endpoints=_endpoints(**urls),
        integration_dir=REPO / "integrations/ra2",
        runs_dir=tmp_path / "duels",
        tasks=TASKS,
        command=[sys.executable, str(FAKES / "fake_duel.py")],
    )


def pair(candidate_spec=SPEC, incumbent_spec=SPEC):  # type: ignore[no-untyped-def]
    incumbent = HoneRelease("r0000", UpdateCandidate("init", None, _snapshot("inc"), harness=incumbent_spec), None)
    candidate = UpdateCandidate("train", "r0000", _snapshot("cand"), harness=candidate_spec)
    return candidate, incumbent


@pytest.mark.unit
def test_a_stronger_candidate_wins_every_game_in_both_seats(tmp_path: Path) -> None:
    candidate, incumbent = pair()
    result = evaluator(tmp_path, cand="http://strong:1", inc="http://weak:1").evaluate(candidate, incumbent)
    assert result.task_ids == tuple(f"{t.task_id}@candidate-{seat}" for t in TASKS for seat in ("red", "blue"))
    assert result.candidate_scores == (1.0,) * 4 and result.incumbent_scores == (0.0,) * 4


@pytest.mark.unit
def test_seats_really_swap_between_the_two_games(tmp_path: Path) -> None:
    candidate, incumbent = pair()
    evaluator(tmp_path, cand="http://strong:1", inc="http://weak:1").evaluate(candidate, incumbent)
    import json

    servers = sorted(
        json.loads((d / "result.json").read_text())["arguments"]["red-decision-server"]
        for d in (tmp_path / "duels").iterdir()
        if d.is_dir()
    )
    assert servers == ["http://strong:1"] * 2 + ["http://weak:1"] * 2


@pytest.mark.unit
def test_evenly_matched_models_draw(tmp_path: Path) -> None:
    candidate, incumbent = pair()
    result = evaluator(tmp_path, cand="http://even:1", inc="http://even:2").evaluate(candidate, incumbent)
    assert result.candidate_scores == (0.5,) * 4 == result.incumbent_scores


@pytest.mark.unit
def test_the_gate_selects_on_a_duel(tmp_path: Path) -> None:
    candidate, incumbent = pair()
    gate = Gate(evaluator(tmp_path, cand="http://strong:1", inc="http://weak:1"), default_selector=WinLossSelector())
    _, decision = gate.judge(candidate, incumbent)
    assert decision.selected and decision.metrics["wins"] == 4.0


@pytest.mark.unit
def test_a_duel_refuses_different_question_text(tmp_path: Path) -> None:
    changed = SPEC.with_file("ra2/questions.json", HarnessFile("instructions", '{"x": "y"}'))
    candidate, incumbent = pair(candidate_spec=changed)
    with pytest.raises(ContractError, match="same question text"):
        evaluator(tmp_path, cand="http://strong:1", inc="http://weak:1").evaluate(candidate, incumbent)


@pytest.mark.unit
def test_a_failed_duel_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_DUEL_EXIT", "2")
    candidate, incumbent = pair()
    with pytest.raises(ContractError, match="exited 2"):
        evaluator(tmp_path, cand="http://strong:1", inc="http://weak:1").evaluate(candidate, incumbent)
