"""RA2 as a task environment: the suite, the runner, the grader, and a gate evaluation, with no engine or Node."""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from honing.backends.jev import JevEndpoints
from honing.backends.ra2 import RA2EpisodeRunner, RA2Grader, build_suite, load_suite
from honing.backends.ra2.runner import AGENT
from honing.core import (
    ContractError,
    HarnessFile,
    HarnessSpec,
    HoneRelease,
    PolicyEndpoint,
    PolicyRef,
    UpdateCandidate,
    run_graded,
)
from honing.release import EpisodeEvaluator
from tests.fakes.model import FakeModelSource, FakeWorld

REPO = Path(__file__).resolve().parents[1]
SUITE_PATH = REPO / "integrations/ra2/tasks/suite.jsonl"
FAKES = Path(__file__).parent / "fakes/ra2"
SPEC = HarnessSpec(agent=AGENT)


def _snapshot(checksum: str):  # type: ignore[no-untyped-def]
    snapshot = FakeModelSource(FakeWorld()).materialize(Path("unused"))
    return replace(snapshot, checkpoint=replace(snapshot.checkpoint, checksum=checksum))


def _policy(checksum: str) -> PolicyRef:
    snapshot = _snapshot(checksum)
    return PolicyRef(snapshot.checkpoint, snapshot.architecture, "master")


def _endpoints(**urls: str) -> JevEndpoints:
    return JevEndpoints(
        {name: PolicyEndpoint(url, "tok", "laya", runtime="jev_decision") for name, url in urls.items()}
    )


def _runner(tmp_path: Path, endpoints: JevEndpoints, **kwargs: object) -> RA2EpisodeRunner:
    return RA2EpisodeRunner(
        suite=load_suite(SUITE_PATH),
        endpoints=endpoints,
        integration_dir=REPO / "integrations/ra2",
        runs_dir=tmp_path / "runs",
        command=[sys.executable, str(FAKES / "fake_match.py")],
        **kwargs,  # type: ignore[arg-type]
    )


def _grader(replay: bool = True) -> RA2Grader:
    check = [sys.executable, str(FAKES / "fake_check.py")]
    return RA2Grader(
        suite=load_suite(SUITE_PATH),
        integration_dir=REPO / "integrations/ra2",
        audit_command=[*check, "audit"],
        replay_command=[*check, "replay"] if replay else None,
    )


# ------------------------------------------------------------------ the suite


@pytest.mark.unit
def test_the_committed_suite_is_what_the_protocol_builds() -> None:
    protocol = json.loads((REPO / "integrations/ra2/tasks/protocol.json").read_text())
    committed = [json.loads(line) for line in SUITE_PATH.read_text().splitlines()]
    assert committed == [json.loads(json.dumps(r, sort_keys=True)) for r in build_suite(protocol)]


@pytest.mark.unit
def test_the_suite_splits_follow_the_protocol() -> None:
    suite = load_suite(SUITE_PATH)
    assert (len(suite.tasks("train")), len(suite.tasks("val")), len(suite.tasks("heldout"))) == (256, 24, 132)
    scenario = suite.scenario("ra2-val-mp03t4-s1000-red")
    assert (scenario.map, scenario.seat, scenario.seed, scenario.ticks) == ("mp03t4.map", "red", 1000, 18000)
    assert suite.split_of("ra2-heldout-mp02t2-s20000-blue") == "heldout"


@pytest.mark.unit
def test_an_unknown_task_or_a_bad_suite_line_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="not in the RA2 suite"):
        load_suite(SUITE_PATH).scenario("ra2-val-nowhere")
    bad = tmp_path / "suite.jsonl"
    bad.write_text(json.dumps({"task_id": "x", "prompt": "", "split": "test", "ra2": {}}) + "\n")
    with pytest.raises(ContractError, match=r"suite\.jsonl:1"):
        load_suite(bad)


# ------------------------------------------------------------------ the runner


@pytest.mark.unit
def test_the_runner_plays_the_scenario_against_the_policy_s_server(tmp_path: Path) -> None:
    runner = _runner(tmp_path, _endpoints(a="http://strong:1"))
    task = load_suite(SUITE_PATH).tasks("val")[1]
    outcome = asyncio.run(runner.run(SPEC, task, _policy("a")))
    assert outcome.exit_code == 0 and outcome.task_id == task.task_id and outcome.policy_checksum == "a"
    arguments = json.loads((Path(outcome.workspace_location) / "result.json").read_text())["arguments"]  # type: ignore[arg-type]
    assert arguments["map"] == "mp03t4.map" and arguments["seat"] == "blue" and arguments["seed"] == "1000"
    assert arguments["decision-server"] == "http://strong:1"
    assert json.loads(outcome.final_output)["outcome"] == {"blue": 1, "red": -1}
    assert outcome.transcript_location.endswith("decisions.jsonl")  # type: ignore[union-attr]


@pytest.mark.unit
def test_harness_files_reach_the_match_as_controller_and_questions(tmp_path: Path) -> None:
    runner = _runner(tmp_path, _endpoints(a="http://strong:1"))
    spec = HarnessSpec(
        agent=AGENT,
        files=(
            ("ra2/controller.json", HarnessFile("settings", '{"decisionInterval": 15}')),
            ("ra2/questions.json", HarnessFile("instructions", "{}")),
        ),
    )
    outcome = asyncio.run(runner.run(spec, load_suite(SUITE_PATH).tasks("val")[0], _policy("a")))
    arguments = json.loads((Path(outcome.workspace_location) / "result.json").read_text())["arguments"]  # type: ignore[arg-type]
    assert Path(arguments["controller"]).read_text() == '{"decisionInterval": 15}'
    assert Path(arguments["questions"]).read_text() == "{}"


@pytest.mark.unit
def test_the_runner_refuses_another_agent_or_a_language_model_endpoint(tmp_path: Path) -> None:
    task = load_suite(SUITE_PATH).tasks("val")[0]
    with pytest.raises(ContractError, match="plays agent"):
        asyncio.run(
            _runner(tmp_path, _endpoints(a="http://strong:1")).run(HarnessSpec("claude_code"), task, _policy("a"))
        )

    class LanguageModelEndpoints:
        def resolve(self, policy: PolicyRef) -> PolicyEndpoint:
            return PolicyEndpoint("http://h", "tok", "m")

    runner = _runner(tmp_path, LanguageModelEndpoints())  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="need a Jev decision server"):
        asyncio.run(runner.run(SPEC, task, _policy("a")))


@pytest.mark.unit
def test_the_runner_caps_how_many_matches_run_at_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_MATCH_SECONDS", "0.3")
    runner = _runner(tmp_path, _endpoints(a="http://strong:1"), concurrency=2)
    tasks = load_suite(SUITE_PATH).tasks("val", limit=4)

    async def play() -> list[list[float]]:
        outcomes = await asyncio.gather(*(runner.run(SPEC, task, _policy("a")) for task in tasks))
        return [json.loads((Path(o.workspace_location) / "result.json").read_text())["window"] for o in outcomes]  # type: ignore[arg-type]

    windows = asyncio.run(play())
    peak = max(sum(1 for s, e in windows if s <= t < e) for t, _ in windows)
    assert peak <= 2


# ------------------------------------------------------------------ the grader


@pytest.mark.unit
@pytest.mark.parametrize(("url", "score"), [("http://strong:1", 1.0), ("http://weak:1", 0.0), ("http://even:1", 0.5)])
def test_the_grade_is_the_match_result(tmp_path: Path, url: str, score: float) -> None:
    runner = _runner(tmp_path, _endpoints(a=url))
    task = load_suite(SUITE_PATH).tasks("val")[0]
    record = asyncio.run(run_graded(runner, _grader(), SPEC, task, _policy("a")))
    assert record.score == score
    assert "Credits gained 5200" in (record.feedback or "") and "Commands submitted 40" in (record.feedback or "")


@pytest.mark.unit
def test_a_failed_match_is_not_graded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_MATCH_EXIT", "3")
    runner = _runner(tmp_path, _endpoints(a="http://strong:1"))
    with pytest.raises(ContractError, match="exited 3"):
        asyncio.run(run_graded(runner, _grader(), SPEC, load_suite(SUITE_PATH).tasks("val")[0], _policy("a")))


@pytest.mark.unit
@pytest.mark.parametrize(("marker", "message"), [("audit-fails", "audit failed"), ("replay-fails", "did not verify")])
def test_a_match_that_fails_its_checks_is_not_graded(tmp_path: Path, marker: str, message: str) -> None:
    runner = _runner(tmp_path, _endpoints(a="http://strong:1"))
    task = load_suite(SUITE_PATH).tasks("val")[0]
    outcome = asyncio.run(runner.run(SPEC, task, _policy("a")))
    (Path(outcome.workspace_location) / marker).touch()  # type: ignore[arg-type]
    with pytest.raises(ContractError, match=message):
        asyncio.run(_grader().grade(task, outcome))


# ------------------------------------------------------------------ through the gate's evaluator


@pytest.mark.unit
def test_the_gate_s_evaluator_scores_candidate_and_incumbent_on_the_same_matches(tmp_path: Path) -> None:
    runner = _runner(tmp_path, _endpoints(cand="http://strong:1", inc="http://weak:1"))
    tasks = load_suite(SUITE_PATH).tasks("val", limit=4)
    evaluator = EpisodeEvaluator(runner, _grader(), tasks)
    incumbent = HoneRelease("r0000", UpdateCandidate("init", None, _snapshot("inc"), harness=SPEC), None)
    candidate = UpdateCandidate("train", "r0000", _snapshot("cand"), harness=SPEC)

    evaluation = evaluator.evaluate(candidate, incumbent)

    assert evaluation.task_ids == tuple(t.task_id for t in tasks)
    assert evaluation.candidate_scores == (1.0,) * 4 and evaluation.incumbent_scores == (0.0,) * 4
    assert len(evaluation.records) == 8
