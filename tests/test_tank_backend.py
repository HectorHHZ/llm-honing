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
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from honing.backends.jev import JevEndpoints
from honing.backends.tank import (
    AGENT,
    CONTROLLER_PATH,
    QUESTIONS_PATH,
    SEED_POOLS,
    TankDemonstrationProvider,
    TankEpisodeRunner,
    TankGrader,
    TankRolloutProvider,
    TankScenario,
    TankSuite,
    build_suite,
    check_spec,
    fresh_seed_gate,
    load_suite,
    pool_tasks,
    seed_harness,
)
from honing.backends.tank.harness import question_keys
from honing.backends.tank.rollouts import ROLLOUT_CONTRACT
from honing.backends.tank.scenarios import DEMONSTRATION_SEEDS, SPLIT_SEEDS
from honing.core import (
    CampaignRuleError,
    ContractError,
    DataContract,
    EpisodeOutcome,
    EvaluationResult,
    Grade,
    HarnessFile,
    HarnessSpec,
    HoneRelease,
    PolicyEndpoint,
    PolicyRef,
    RunnerSupport,
    Task,
    UpdateCandidate,
    run_graded,
)
from honing.release import EpisodeEvaluator, PairedIntervalSelector, ReleaseChain, betting_paired_difference
from tests.fakes.model import FakeModelSource, FakeWorld

REPO = Path(__file__).resolve().parents[1]
INTEGRATION = REPO / "integrations/tank-war"
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
        tasks = suite.heldout_tasks() if split == "heldout" else suite.tasks(split)  # type: ignore[arg-type]
        assert [suite.scenario(t.task_id).seat for t in tasks[:2]] == ["blue", "red"]
        seeds[split] = {suite.scenario(t.task_id).seed for t in tasks}
        assert len(tasks) == 2 * len(seeds[split])
    assert not (seeds["train"] & seeds["val"]) and not (seeds["val"] & seeds["heldout"])
    assert suite.tasks("val", limit=4)[3].task_id == "tank-val-s2002-red"


@pytest.mark.unit
def test_the_seed_pools_are_disjoint_from_the_suite_and_from_each_other() -> None:
    suite_seeds = set().union(*SPLIT_SEEDS.values())
    assert not set(SEED_POOLS["val"]) & suite_seeds and not set(SEED_POOLS["heldout"]) & suite_seeds
    assert not set(SEED_POOLS["val"]) & set(SEED_POOLS["heldout"])
    suite = load_suite(SUITE_PATH)
    task = pool_tasks("val", [100_123])[1]
    assert task.task_id == "tank-val-s100123-red"
    assert (suite.split_of(task.task_id), suite.scenario(task.task_id)) == ("val", TankScenario(100_123, "red"))
    for unknown in ("tank-val-s99999-blue", "tank-val-s0100000-blue", "tank-train-s100000-blue"):
        with pytest.raises(ContractError, match="not in the tank suite"):
            suite.scenario(unknown)


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


@pytest.mark.functional
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
    assert json.loads(arguments["reef-tags"]).keys() == {"episode", "task", "harness", "policy"}
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


@pytest.mark.functional
@pytest.mark.parametrize(("server", "score"), [("strong", 1.0), ("weak", 0.0), ("even", 0.5)])
def test_a_win_scores_1_a_loss_0_and_running_out_the_clock_half(tmp_path: Path, server: str, score: float) -> None:
    task = load_suite(SUITE_PATH).tasks("val")[0]
    record = asyncio.run(run_graded(_runner(tmp_path), _grader(), SEED, task, _policy(server)))
    assert record.score == score


@pytest.mark.functional
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


@pytest.mark.functional
def test_a_harness_the_match_refuses_is_a_campaign_rule_error(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("FAKE_TANK_ERROR", "harness")
    task = load_suite(SUITE_PATH).tasks("val")[0]
    with pytest.raises(CampaignRuleError, match="decisionInterval"):
        asyncio.run(_runner(tmp_path).run(SEED, task, _policy("even")))


@pytest.mark.functional
def test_an_over_long_state_scores_0_with_feedback_and_a_crash_is_an_error(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    task = load_suite(SUITE_PATH).tasks("val")[0]
    monkeypatch.setenv("FAKE_TANK_ERROR", "budget")
    record = asyncio.run(run_graded(_runner(tmp_path), _grader(), SEED, task, _policy("even")))
    assert record.score == 0.0 and "Shorten the operating notes" in (record.feedback or "")
    monkeypatch.setenv("FAKE_TANK_ERROR", "crash")
    with pytest.raises(ContractError, match="Wall penetration"):
        asyncio.run(run_graded(_runner(tmp_path), _grader(), SEED, task, _policy("even")))


@pytest.mark.functional
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


@pytest.mark.functional
def test_the_gate_compares_candidate_and_incumbent_on_the_same_rounds(tmp_path: Path) -> None:
    tasks = load_suite(SUITE_PATH).tasks("val", limit=4)
    evaluation = EpisodeEvaluator(_runner(tmp_path), _grader(), tasks).evaluate(
        UpdateCandidate("train", "r0000", _snapshot("strong"), harness=SEED),
        HoneRelease("r0000", UpdateCandidate("init", None, _snapshot("even"), harness=SEED), None),
    )
    assert evaluation.candidate_scores == (1.0,) * 4 and evaluation.incumbent_scores == (0.5,) * 4


# ------------------------------------------------------------------ the gate's seeds


def _init(checksum: str) -> UpdateCandidate:
    return UpdateCandidate("init", None, _snapshot(checksum), harness=SEED)


def _trained(checksum: str) -> UpdateCandidate:
    return UpdateCandidate("train", "r0000", _snapshot(checksum), harness=SEED)


class _SeatsAgree:
    """Runner and grader in one, with no game: a policy scores ``score(checksum, n)`` on the nth pool seed.

    Both seats of a seed get that score, the strongest correlation two seats can have.
    """

    support = RunnerSupport(agents=frozenset({AGENT}), file_kinds=frozenset({"instructions"}), programmable_loop=False)

    def __init__(self, score: Callable[[str, int], float]) -> None:
        self.score = score
        self.suite = load_suite(SUITE_PATH)

    async def run(self, spec: HarnessSpec, task: Task, policy: PolicyRef) -> EpisodeOutcome:
        return EpisodeOutcome(task.task_id, spec.spec_id, policy.checkpoint.checksum, exit_code=0)

    async def grade(self, task: Task, outcome: EpisodeOutcome) -> Grade:
        seed = self.suite.scenario(task.task_id).seed
        return Grade(self.score(outcome.policy_checksum, seed - SEED_POOLS["val"].start))


@pytest.mark.functional
def test_two_consecutive_gates_draw_disjoint_validation_seeds(tmp_path: Path) -> None:
    suite = load_suite(SUITE_PATH)
    gate = fresh_seed_gate(_runner(tmp_path), _grader(), suite, rounds=4)
    incumbent = HoneRelease("r0000", _init("even"), None)
    played = []
    for checksum in ("strong", "weak"):
        evaluation, decision = gate.judge(_trained(checksum), incumbent)
        seeds = {suite.scenario(record.task_id).seed for record in evaluation.records}
        assert evaluation.task_ids == tuple(f"tank-val-s{seed}" for seed in sorted(seeds))
        assert (decision.metrics["seed_first"], decision.metrics["seed_last"]) == (min(seeds), max(seeds))
        played.append(seeds)
    assert played == [{100_000, 100_001}, {100_002, 100_003}]
    assert not set(SPLIT_SEEDS["val"]) & (played[0] | played[1])


@pytest.mark.functional
def test_a_gate_starts_at_the_configured_first_seed(tmp_path: Path) -> None:
    suite = load_suite(SUITE_PATH)
    incumbent = HoneRelease("r0000", _init("even"), None)

    def first_two_gates(**settings: int) -> list[set[int]]:
        gate = fresh_seed_gate(_runner(tmp_path), _grader(), suite, rounds=4, **settings)
        played = []
        for checksum in ("strong", "weak"):
            evaluation, _ = gate.judge(_trained(checksum), incumbent)
            played.append({suite.scenario(record.task_id).seed for record in evaluation.records})
        return played

    # Two campaigns with the same first seed play the same gates. Another first seed plays others.
    assert first_two_gates(first_seed=150_000) == first_two_gates(first_seed=150_000)
    assert first_two_gates(first_seed=150_000) == [{150_000, 150_001}, {150_002, 150_003}]
    assert first_two_gates() == [{100_000, 100_001}, {100_002, 100_003}]


@pytest.mark.unit
@pytest.mark.parametrize("first_seed", [99_999, 200_000, 2001, True, "150000"])
def test_a_first_seed_outside_the_val_pool_is_refused(tmp_path: Path, first_seed: object) -> None:
    with pytest.raises(ContractError, match="first_seed must be a seed of the val pool, 100000 to 199999"):
        fresh_seed_gate(_runner(tmp_path), _grader(), load_suite(SUITE_PATH), rounds=4, first_seed=first_seed)  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.parametrize(
    "seed_differences", [[0.5] * 6 + [-0.5] * 2, [-0.5] * 2 + [0.5] * 14], ids=["review", "sixteen-seeds"]
)
def test_both_seats_of_a_seed_count_as_one_difference(seed_differences: list[float]) -> None:
    # The second seat repeats the first, so it adds no evidence, and the gate must decide as it would on one
    # difference per seed. The first case is the #197 review's. The second has more seeds and its losses first, so
    # its interval per seed reaches just below zero.
    seeds = len(seed_differences)
    games = _SeatsAgree(lambda policy, seed: 0.5 + seed_differences[seed] if policy == "strong" else 0.5)
    gate = fresh_seed_gate(games, games, load_suite(SUITE_PATH), rounds=2 * seeds)
    candidate = _trained("strong")
    evaluation, decision = gate.judge(candidate, HoneRelease("r0000", _init("even"), None))
    assert evaluation.task_ids == tuple(f"tank-val-s{100_000 + seed}" for seed in range(seeds))
    assert len(evaluation.records) == 4 * seeds  # both seats, under the candidate and under the incumbent
    one_per_seed = betting_paired_difference(
        EvaluationResult(
            task_ids=tuple(f"seed{seed}" for seed in range(seeds)),
            candidate_scores=tuple(0.5 + difference for difference in seed_differences),
            incumbent_scores=(0.5,) * seeds,
            evaluated_variant="master",
        ),
        confidence=0.95,
    )
    assert (decision.metrics["difference_low"], decision.metrics["difference_high"]) == (
        one_per_seed.low,
        one_per_seed.high,
    )
    assert (decision.metrics["tasks"], decision.metrics["rounds"]) == (seeds, 2 * seeds)
    assert not decision.selected
    # The records keep every seat. Counted seat by seat, both cases would be selected.
    candidate_records, incumbent_records = evaluation.records[: 2 * seeds], evaluation.records[2 * seeds :]
    seats = EvaluationResult(
        task_ids=tuple(record.task_id for record in candidate_records),
        candidate_scores=tuple(record.score for record in candidate_records),
        incumbent_scores=tuple(record.score for record in incumbent_records),
        evaluated_variant="master",
    )
    assert PairedIntervalSelector().decide(candidate, seats).selected


@pytest.mark.unit
def test_a_release_records_its_gates_seeds_rounds_played_and_early_stop(tmp_path: Path) -> None:
    games = _SeatsAgree(lambda policy, seed: {"strong": 1.0, "even": 0.5}[policy])
    gate = fresh_seed_gate(games, games, load_suite(SUITE_PATH), rounds=80, looks=2)
    chain = ReleaseChain(tmp_path / "releases")
    initial = chain.initialize(_init("even"))
    # The candidate wins every round of the first block of 20 seeds, so the interim look selects it.
    candidate = _trained("strong")
    _, decision = gate.judge(candidate, initial)
    release = chain.commit(candidate, decision)
    recorded = json.loads(chain.record_path.read_text().splitlines()[-1])["decision"]["metrics"]
    expected = {"seed_first": 100_000.0, "seed_last": 100_019.0, "tasks": 20.0, "rounds": 40.0, "stopped_early": 1.0}
    for metrics in (release.decision.metrics, recorded):  # type: ignore[union-attr]
        assert {name: metrics[name] for name in expected} == expected
    # A tie never clears the interval, so the gate plays both blocks and then rejects.
    _, tie = gate.judge(_trained("even"), initial)
    assert not tie.selected
    assert {name: tie.metrics[name] for name in expected} == {
        "seed_first": 100_020.0,
        "seed_last": 100_059.0,
        "tasks": 40.0,
        "rounds": 80.0,
        "stopped_early": 0.0,
    }


@pytest.mark.unit
def test_no_campaign_code_path_can_request_heldout() -> None:
    suite = load_suite(SUITE_PATH)
    with pytest.raises(ContractError, match="experiment scripts"):
        suite.tasks("heldout")
    with pytest.raises(ContractError, match="experiment scripts"):
        pool_tasks("heldout", [200_000])
    with pytest.raises(ContractError, match="pool must be one of"):
        pool_tasks("train", [1001])
    # Experiment scripts read the suite's heldout tasks and the heldout pool.
    assert len(suite.heldout_tasks()) == 2 * len(SPLIT_SEEDS["heldout"])
    pooled = suite.heldout_tasks([200_000])
    assert [task.task_id for task in pooled] == ["tank-heldout-s200000-blue", "tank-heldout-s200000-red"]
    assert suite.split_of(pooled[0].task_id) == "heldout"
    with pytest.raises(ContractError, match="not in the heldout pool"):
        suite.heldout_tasks([100_000])


@pytest.mark.unit
def test_a_gate_whose_rounds_do_not_split_into_whole_seeds_is_refused(tmp_path: Path) -> None:
    for rounds, looks in ((7, 1), (8, 3), (12, 4)):
        with pytest.raises(ContractError, match="whole seeds"):
            fresh_seed_gate(_runner(tmp_path), _grader(), load_suite(SUITE_PATH), rounds=rounds, looks=looks)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("index", "task_id", "seed"),
    [
        (0, "tank-train-s150000-blue", 150_000),
        (32, "tank-val-s150000-blue", 150_000),
        (48, "tank-heldout-s150000-blue", 150_000),
        # Named as a pool task, the line is what the runner would play for that pool task: seed 2001.
        (32, "tank-val-s100000-blue", 2001),
    ],
    ids=["train", "val", "heldout", "pool-task-id"],
)
def test_a_gate_refuses_a_suite_line_it_could_play(tmp_path: Path, index: int, task_id: str, seed: int) -> None:
    records = build_suite()
    records[index] = {**records[index], "task_id": task_id, "tank": asdict(TankScenario(seed, "blue"))}
    with pytest.raises(ContractError, match=rf"val pool, seeds 100000 to 199999, .*: {task_id} \(seed {seed}\)$"):
        fresh_seed_gate(_runner(tmp_path), _grader(), TankSuite(tuple(records)), rounds=2)


# ------------------------------------------------------------------ training data


@pytest.mark.functional
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


@pytest.mark.functional
def test_demonstrations_label_the_models_questions_with_the_teachers_answers(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    suite = load_suite(SUITE_PATH)
    provider = TankDemonstrationProvider(runner, suite.tasks("train", limit=2), SEED)
    contract = DataContract("typed_decisions")

    batch = provider.next_batch(contract)
    assert batch is not None and len(batch.decisions) == 4
    assert all(d.behavior_probs is None and d.reward is None for d in batch.decisions)
    assert [dict(d.chosen) for d in batch.decisions[:2]] == [{"motion": "stop_none"}, {"weapon": "fire"}]
    assert batch.batch_id == "tank-teacher-demonstrations-1"
    assert [d.group for d in batch.decisions] == ["tank:1001:blue"] * 2 + ["tank:1001:red"] * 2
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


def _teacher_rounds(tmp_path: Path) -> list[tuple[int, str]]:
    """The seed and seat of every teacher round under ``tmp_path``, in the order they were played."""
    manifests = sorted((tmp_path / "runs").glob("*--teacher--*/manifest.json"), key=lambda p: p.stat().st_mtime_ns)
    return [(m["seed"], m["seat"]) for m in (json.loads(path.read_text()) for path in manifests)]


@pytest.mark.functional
def test_every_demonstration_batch_after_the_first_plays_seeds_no_task_uses(tmp_path: Path) -> None:
    runner = _runner(tmp_path, concurrency=1)
    suite = load_suite(SUITE_PATH)
    provider = TankDemonstrationProvider(runner, suite.tasks("train", limit=4), SEED)
    contract = DataContract("typed_decisions")

    batches = [provider.next_batch(contract) for _ in range(3)]

    assert [batch.batch_id for batch in batches if batch is not None] == [
        "tank-teacher-demonstrations-1",
        "tank-teacher-demonstrations-2",
        "tank-teacher-demonstrations-3",
    ]
    fresh = DEMONSTRATION_SEEDS[:4]
    assert sorted(_teacher_rounds(tmp_path)) == sorted(
        (seed, seat) for seed in (1001, 1002, *fresh) for seat in ("blue", "red")
    )
    suite_seeds = {suite.scenario(str(record["task_id"])).seed for record in suite.records}
    assert not suite_seeds & set(DEMONSTRATION_SEEDS[:1000])


@pytest.mark.functional
def test_demonstrations_stop_when_the_fresh_seeds_run_out_and_refuse_a_seed_the_suite_plays(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    tasks = load_suite(SUITE_PATH).tasks("train", limit=2)
    contract = DataContract("typed_decisions")
    provider = TankDemonstrationProvider(runner, tasks, SEED, fresh_seeds=(900_001,))

    assert provider.next_batch(contract) is not None
    assert provider.next_batch(contract) is not None
    assert provider.next_batch(contract) is None
    with pytest.raises(ContractError, match=r"must be in no split; the suite already plays \[2001\]"):
        TankDemonstrationProvider(runner, tasks, SEED, fresh_seeds=(900_001, 2001))


@pytest.mark.unit
def test_every_attempt_tags_its_decisions_with_a_new_episode_and_its_task_harness_and_policy(tmp_path: Path) -> None:
    # The decision server records in Reef under ids derived from the episode tag (#223), so a retried task
    # must not reuse one. The runner is given no release, so it sets no release tag.
    task = load_suite(SUITE_PATH).tasks("val")[1]
    runner = _runner(tmp_path)
    tags = []
    for _ in range(2):
        outcome = asyncio.run(runner.run(SEED, task, _policy("strong")))
        manifest = json.loads((Path(outcome.workspace_location) / "manifest.json").read_text())  # type: ignore[arg-type]
        tags.append(json.loads(manifest["arguments"]["reef-tags"]))
    for tag in tags:
        assert {k: tag[k] for k in ("task", "harness", "policy")} == {
            "task": task.task_id,
            "harness": SEED.spec_id,
            "policy": "strong",
        }
        assert set(tag) == {"episode", "task", "harness", "policy"}
    assert tags[0]["episode"] != tags[1]["episode"]
