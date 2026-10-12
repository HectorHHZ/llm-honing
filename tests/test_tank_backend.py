"""Tank duels as a task environment: the suite, harness, runner, grader, rollouts, and demonstrations.

The match process is ``tests/fakes/tank_match.py`` here, so nothing needs Node. The tests marked
``integration`` play real rounds through Node and skip when it is missing; CI's e2e job plays them.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from honing.backends.jev import JevEndpoints
from honing.backends.tank import (
    AGENT,
    CONTROLLER_PATH,
    QUESTIONS_PATH,
    TankDemonstrationProvider,
    TankEpisodeRunner,
    TankGrader,
    TankRolloutProvider,
    build_suite,
    check_spec,
    load_suite,
    seed_harness,
)
from honing.backends.tank.harness import question_keys
from honing.backends.tank.rollouts import ROLLOUT_CONTRACT
from honing.core import (
    CampaignRuleError,
    ContractError,
    DataContract,
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
INTEGRATION = REPO / "integrations/tank-duel"
SUITE_PATH = INTEGRATION / "tasks/suite.jsonl"
FAKE_MATCH = Path(__file__).parent / "fakes/tank_match.py"
SEED = seed_harness(INTEGRATION)
NODE = shutil.which("node")


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


def _runner(tmp_path: Path, endpoints: JevEndpoints | None = None, **kwargs: object) -> TankEpisodeRunner:
    return TankEpisodeRunner(
        suite=load_suite(SUITE_PATH),
        endpoints=endpoints or _endpoints(strong="http://strong:1", weak="http://weak:1", even="http://even:1"),
        integration_dir=INTEGRATION,
        runs_dir=tmp_path / "runs",
        command=[sys.executable, str(FAKE_MATCH)],
        **kwargs,  # type: ignore[arg-type]
    )


def _grader() -> TankGrader:
    return TankGrader(suite=load_suite(SUITE_PATH))


def _questions(**changes: object) -> HarnessSpec:
    questions = {**json.loads(SEED.get(QUESTIONS_PATH).content), **changes}  # type: ignore[union-attr]
    return SEED.with_file(QUESTIONS_PATH, HarnessFile("instructions", json.dumps(questions)))


# ------------------------------------------------------------------ the suite and the harness


@pytest.mark.unit
def test_the_committed_suite_is_what_build_suite_writes() -> None:
    committed = [json.loads(line) for line in SUITE_PATH.read_text().splitlines()]
    assert committed == [json.loads(json.dumps(r, sort_keys=True)) for r in build_suite()]


@pytest.mark.unit
def test_every_seed_is_played_from_both_seats_and_splits_do_not_share_seeds() -> None:
    suite = load_suite(SUITE_PATH)
    seeds = {}
    for split in ("train", "val", "heldout"):
        tasks = suite.tasks(split)  # type: ignore[arg-type]
        assert [suite.scenario(t.task_id).seat for t in tasks[:2]] == ["blue", "red"]
        seeds[split] = {suite.scenario(t.task_id).seed for t in tasks}
        assert len(tasks) == 2 * len(seeds[split])
    assert not (seeds["train"] & seeds["val"]) and not (seeds["val"] & seeds["heldout"])
    assert suite.tasks("val", limit=4)[3].task_id == "tank-val-s2002-red"


@pytest.mark.unit
def test_a_suite_line_with_an_unknown_split_or_scenario_is_refused(tmp_path: Path) -> None:
    record = build_suite()[0]
    for bad in ({**record, "split": "dev"}, {**record, "tank": {**record["tank"], "seat": "green"}}):  # type: ignore[dict-item]
        path = tmp_path / "suite.jsonl"
        path.write_text(json.dumps(bad) + "\n")
        with pytest.raises(ContractError, match=re.escape("suite.jsonl:1")):
            load_suite(path)


@pytest.mark.unit
def test_the_seed_harness_is_the_integration_config() -> None:
    assert SEED.agent == AGENT
    assert [(path, file.kind) for path, file in SEED.files] == [
        (CONTROLLER_PATH, "settings"),
        (QUESTIONS_PATH, "instructions"),
    ]
    assert json.loads(SEED.get(CONTROLLER_PATH).content)["strategy"] == "baseline"  # type: ignore[union-attr]
    check_spec(SEED, question_keys(INTEGRATION))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("spec", "message"),
    [
        (HarnessSpec(agent="ra2_decision"), "plays agent 'tank_classic'"),
        (SEED.with_file(CONTROLLER_PATH, HarnessFile("settings", "{")), "is not JSON"),
        (SEED.with_file(CONTROLLER_PATH, HarnessFile("settings", "[]")), "must be a JSON object"),
        (SEED.with_file(CONTROLLER_PATH, HarnessFile("settings", '{"decisionInterval": 50}')), "decisionInterval"),
        (_questions(weapon=""), re.escape("empty ['weapon']")),
        (_questions(extra="x"), re.escape("unknown ['extra']")),
        (
            SEED.with_file(QUESTIONS_PATH, HarnessFile("instructions", '{"motion": "x"}')),
            re.escape("missing ['drive'"),
        ),
    ],
)
def test_a_harness_the_match_cannot_read_is_refused_before_it_runs(spec: HarnessSpec, message: str) -> None:
    with pytest.raises(CampaignRuleError, match=message):
        check_spec(spec, question_keys(INTEGRATION))


# ------------------------------------------------------------------ the runner and the grader


@pytest.mark.unit
def test_a_round_is_played_with_the_scenario_the_harness_and_the_policys_server(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    spec = SEED.with_file("notes/aim.md", HarnessFile("instructions", "Aim first."))
    task = load_suite(SUITE_PATH).tasks("val")[1]
    record = asyncio.run(run_graded(runner, _grader(), spec, task, _policy("strong")))

    assert record.score == 1.0
    assert (record.task_id, record.spec_id, record.policy_checksum) == (task.task_id, spec.spec_id, "strong")
    run = Path(record.outcome.workspace_location)  # type: ignore[arg-type]
    arguments = json.loads((run / "manifest.json").read_text())["arguments"]
    assert {k: arguments[k] for k in ("seed", "seat", "opponent", "seconds", "interval-ms", "decision-server")} == {
        "seed": "2001",
        "seat": "red",
        "opponent": "planner",
        "seconds": "30",
        "interval-ms": "200",
        "decision-server": "http://strong:1",
    }
    assert "sample-seed" not in arguments
    harness = Path(arguments["harness-dir"])
    assert json.loads((harness / "index.json").read_text()) == [
        {"path": CONTROLLER_PATH, "kind": "settings"},
        {"path": QUESTIONS_PATH, "kind": "instructions"},
        {"path": "notes/aim.md", "kind": "instructions"},
    ]
    assert (harness / "notes/aim.md").read_text() == "Aim first."
    assert json.loads(record.outcome.final_output)["winner"] == "red"
    assert record.outcome.transcript_location == str(run / "decisions.jsonl")
    assert "won by a hit after 12.5 s" in (record.feedback or "")


@pytest.mark.unit
@pytest.mark.parametrize(("server", "score"), [("strong", 1.0), ("weak", 0.0), ("even", 0.5)])
def test_a_win_scores_1_a_loss_0_and_running_out_the_clock_half(tmp_path: Path, server: str, score: float) -> None:
    task = load_suite(SUITE_PATH).tasks("val")[0]
    record = asyncio.run(run_graded(_runner(tmp_path), _grader(), SEED, task, _policy(server)))
    assert record.score == score


@pytest.mark.unit
def test_a_sampling_runner_draws_a_fresh_seed_for_every_round(tmp_path: Path) -> None:
    runner = _runner(tmp_path).sampling()
    task = load_suite(SUITE_PATH).tasks("train")[0]

    async def two() -> list[str]:
        outcomes = await asyncio.gather(*(runner.run(SEED, task, _policy("even")) for _ in range(2)))
        return [
            json.loads((Path(o.workspace_location) / "manifest.json").read_text())["sample_seed"]  # type: ignore[arg-type]
            for o in outcomes
        ]

    seeds = asyncio.run(two())
    assert all(isinstance(s, int) for s in seeds) and seeds[0] != seeds[1]


@pytest.mark.unit
def test_a_harness_the_match_refuses_is_a_campaign_rule_error(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("FAKE_TANK_ERROR", "harness")
    task = load_suite(SUITE_PATH).tasks("val")[0]
    with pytest.raises(CampaignRuleError, match="decisionInterval"):
        asyncio.run(_runner(tmp_path).run(SEED, task, _policy("even")))


@pytest.mark.unit
def test_an_over_long_state_scores_0_with_feedback_and_a_crash_is_an_error(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    task = load_suite(SUITE_PATH).tasks("val")[0]
    monkeypatch.setenv("FAKE_TANK_ERROR", "budget")
    record = asyncio.run(run_graded(_runner(tmp_path), _grader(), SEED, task, _policy("even")))
    assert record.score == 0.0 and "Shorten the operating notes" in (record.feedback or "")
    monkeypatch.setenv("FAKE_TANK_ERROR", "crash")
    with pytest.raises(ContractError, match="Wall penetration"):
        asyncio.run(run_graded(_runner(tmp_path), _grader(), SEED, task, _policy("even")))


@pytest.mark.unit
def test_the_grader_refuses_a_round_of_another_scenario_or_a_teacher_round(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    task = load_suite(SUITE_PATH).tasks("val")[0]
    monkeypatch.setenv("FAKE_TANK_SEED", "9")
    with pytest.raises(ContractError, match="but task tank-val-s2001-blue is"):
        asyncio.run(run_graded(_runner(tmp_path), _grader(), SEED, task, _policy("strong")))
    monkeypatch.delenv("FAKE_TANK_SEED")
    outcome = asyncio.run(_runner(tmp_path).run(SEED, task, _policy("strong")))
    manifest = Path(outcome.workspace_location) / "manifest.json"  # type: ignore[arg-type]
    manifest.write_text(json.dumps({**json.loads(manifest.read_text()), "kind": "teacher-demonstration"}))
    with pytest.raises(ContractError, match="'teacher-demonstration'"):
        asyncio.run(_grader().grade(task, outcome))


@pytest.mark.unit
def test_a_jev_server_is_required(tmp_path: Path) -> None:
    from tests.fakes.harness import StaticPolicyEndpoints

    endpoints = StaticPolicyEndpoints({"x": PolicyEndpoint("http://sglang:1", "tok", "m")})
    runner = _runner(tmp_path, endpoints=endpoints)  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="need a Jev decision server"):
        asyncio.run(runner.run(SEED, load_suite(SUITE_PATH).tasks("val")[0], _policy("x")))


@pytest.mark.unit
def test_the_gate_compares_candidate_and_incumbent_on_the_same_rounds(tmp_path: Path) -> None:
    tasks = load_suite(SUITE_PATH).tasks("val", limit=4)
    evaluation = EpisodeEvaluator(_runner(tmp_path), _grader(), tasks).evaluate(
        UpdateCandidate("train", "r0000", _snapshot("strong"), harness=SEED),
        HoneRelease("r0000", UpdateCandidate("init", None, _snapshot("even"), harness=SEED), None),
    )
    assert evaluation.candidate_scores == (1.0,) * 4 and evaluation.incumbent_scores == (0.5,) * 4


# ------------------------------------------------------------------ training data


@pytest.mark.unit
def test_rollouts_are_the_policys_sampled_choices_rewarded_by_the_round(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    tasks = load_suite(SUITE_PATH).tasks("train", limit=2)
    batch = TankRolloutProvider(runner, _grader()).collect(tasks, _policy("strong"), SEED, "r0003", 2)

    assert batch.contract == ROLLOUT_CONTRACT
    assert len(batch.decisions) == 2 * 2 * 2  # tasks x samples x rows per round
    first = batch.decisions[0]
    assert dict(first.chosen) == {"motion": "stop_none"}  # what was executed, not the most probable option
    assert dict(dict(first.behavior_probs)["motion"]) == {"stop_none": 0.25, "forward_none": 0.75}  # type: ignore[arg-type]
    assert {d.reward for d in batch.decisions} == {1.0}
    assert {d.group for d in batch.decisions} == {t.task_id for t in tasks}
    assert {(d.source_release, d.source_harness) for d in batch.decisions} == {("r0003", SEED.spec_id)}
    rounds = [
        json.loads((p / "manifest.json").read_text())
        for p in (tmp_path / "runs").iterdir()
        if p.is_dir() and (p / "manifest.json").exists()
    ]
    assert {m["kind"] for m in rounds} == {"rollout"}
    with pytest.raises(ContractError, match="play under a harness"):
        TankRolloutProvider(runner, _grader()).collect(tasks, _policy("strong"), None, None, 1)


@pytest.mark.unit
def test_demonstrations_label_the_models_questions_with_the_teachers_answers(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    suite = load_suite(SUITE_PATH)
    provider = TankDemonstrationProvider(runner, suite.tasks("train", limit=2), SEED)
    contract = DataContract("typed_decisions")

    batch = provider.next_batch(contract)
    assert batch is not None and len(batch.decisions) == 4
    assert all(d.behavior_probs is None and d.reward is None for d in batch.decisions)
    assert [dict(d.chosen) for d in batch.decisions[:2]] == [{"motion": "stop_none"}, {"weapon": "fire"}]
    assert provider.next_batch(contract) is None  # served once
    assert (
        TankDemonstrationProvider(runner, suite.tasks("train", limit=1), SEED).next_batch(
            DataContract("token_sequences")
        )
        is None
    )
    with pytest.raises(ContractError, match="train tasks only"):
        TankDemonstrationProvider(runner, suite.tasks("val", limit=1), SEED)


# ------------------------------------------------------------------ real rounds through Node


@pytest.mark.integration
@pytest.mark.skipif(NODE is None, reason="needs Node 22")
def test_a_real_round_against_the_stand_in_model_is_graded_and_delivers_every_file(tmp_path: Path) -> None:
    from honing.backends.tank.e2e import TankDeliveryProbe
    from tests.fakes.model import FakeWorld as World
    from tests.fakes.tank import QualityServers

    world = World()
    snapshot = FakeModelSource(world, quality=0.9).materialize(tmp_path)
    policy = PolicyRef(snapshot.checkpoint, snapshot.architecture, "master")
    runner = TankEpisodeRunner(
        suite=load_suite(SUITE_PATH),
        endpoints=QualityServers(world),
        integration_dir=INTEGRATION,
        runs_dir=tmp_path / "runs",
    )
    spec = SEED.with_file("notes/aim.md", HarnessFile("instructions", "Prefer ricochet shots."))
    task = load_suite(SUITE_PATH).tasks("val")[0]
    record = asyncio.run(run_graded(runner, _grader(), spec, task, policy))

    assert record.score in (0.0, 0.5, 1.0)
    assert set(TankDeliveryProbe().delivered(record.outcome, spec).values()) == {True}
    rows = Path(record.outcome.transcript_location).read_text().splitlines()  # type: ignore[arg-type]
    assert rows and json.loads(rows[0])["contextParts"]["essential"][0].startswith("OPERATING NOTES (notes/aim.md)")

    teacher = TankDemonstrationProvider(runner, load_suite(SUITE_PATH).tasks("train", limit=1), SEED)
    batch = teacher.next_batch(DataContract("typed_decisions"))
    assert batch is not None and {k for d in batch.decisions for k, _ in d.questions} == {"motion", "weapon"}
