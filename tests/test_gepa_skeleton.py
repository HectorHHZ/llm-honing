"""GEPA through the walking skeleton with fakes (#27): one campaign, from the seed until GEPA's budget runs out.

The campaign runs 17 harness moves, rolls back to ``r0002`` while rejected
candidates are parenting new ones, then runs harness moves until GEPA proposes
nothing without running an episode. Each test reads one property of that run
from the commit record, the archive, or the fake runner.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from gepa.utils.stop_condition import MaxMetricCallsStopper

from honing.campaign import ScriptedScheduler
from honing.core import (
    CampaignState,
    EvaluationResult,
    EvolveContext,
    HarnessMoveConfig,
    HarnessProposal,
    Move,
    ProposeResult,
    SelectionDecision,
    Task,
)
from honing.core.state import validate_method_state
from honing.harness.methods.gepa import Archive, GEPAConfig, GEPAMethod
from tests.fakes.builder import CampaignFixture, build_campaign
from tests.fakes.harness import INSTRUCTIONS_PATH, LineEditChatModel, RuleRunner
from tests.gepa_upstream import upstream_replay

FAMILIES = ("sum", "sort", "parse", "count")
VALIDATION = tuple(Task(task_id=family, prompt=f"Validate on {family}.") for family in FAMILIES)
VALIDATION_IDS = tuple(task.task_id for task in VALIDATION)
TRAIN = tuple(
    Task(task_id=f"train-{family}-{n}", prompt=f"Train on {family} {n}.") for family in FAMILIES for n in (0, 1)
)

# Each family has a line that helps only it; "Be brief." and "Show your work." each help
# two families and hurt the other two, so a candidate can lead some tasks and trail others.
# "Show your work." nets zero over the four, so a child that adds it ties its parent's mean.
WEIGHTS = {
    "Add the numbers.": {"sum": 0.25},
    "Order the list.": {"sort": 0.25},
    "Read the date.": {"parse": 0.25},
    "Count the items.": {"count": 0.25},
    "Be brief.": {"sum": 0.125, "count": 0.125, "parse": -0.25, "sort": -0.25},
    "Show your work.": {"parse": 0.125, "sort": 0.125, "sum": -0.125, "count": -0.125},
}
EDITS = (
    ("add", "Be brief."),
    ("add", "Show your work."),
    ("keep", ""),
    ("add", "Add the numbers."),
    ("drop", "Be brief."),
    ("add", "Order the list."),
    ("add", "Read the date."),
    ("add", "Count the items."),
    ("drop", "Show your work."),
)
MINIBATCH = 2
# Chosen so a proposal starts with exactly this many calls spent: the stop's boundary is exercised.
MAX_METRIC_CALLS = 110
PHASE_ONE_MOVES = 17
ROLLBACK_TO = "r0002"
MOVE = Move(
    kind="harness",
    harness=HarnessMoveConfig(tasks=TRAIN, episode_budget=2 * MINIBATCH, method="gepa", label="pre_training"),
)


class SpyGEPAMethod(GEPAMethod):
    """GEPA that remembers, per harness move, the state it started from and the state it settled on."""

    def __init__(self) -> None:
        config = GEPAConfig(max_metric_calls=MAX_METRIC_CALLS, minibatch_size=MINIBATCH)
        super().__init__(LineEditChatModel(EDITS), config, VALIDATION_IDS)
        self.starts: list[Mapping[str, object]] = []
        self.ends: list[Mapping[str, object]] = []

    def propose(self, context: EvolveContext) -> ProposeResult:
        self.starts.append(context.state)
        result = super().propose(context)
        if result.proposal is None:
            self.ends.append(result.state)
        return result

    def record(
        self,
        state: Mapping[str, object],
        proposal: HarnessProposal,
        evaluation: EvaluationResult,
        decision: SelectionDecision,
    ) -> Mapping[str, object]:
        settled = super().record(state, proposal, evaluation, decision)
        self.ends.append(settled)
        return settled


class UntilBudgetSpent:
    """Harness moves until GEPA proposes nothing without running an episode, which here only its budget stop does.

    A scheduler sees the head and the published releases, not the method's
    pending state or its skips, so this one reads the commit record.
    """

    def __init__(self, record_path: Path) -> None:
        self._record_path = record_path

    def next_move(self, state: CampaignState) -> Move:
        last = json.loads(self._record_path.read_text().splitlines()[-1])
        if last["op"] == "skip" and last["episodes_spent"] == 0:
            return Move(kind="stop")
        return MOVE


@dataclass(frozen=True)
class Run:
    fixture: CampaignFixture
    method: SpyGEPAMethod
    records: list[dict[str, object]]
    harness_records: list[dict[str, object]]
    rollback_at: int  # how many harness records precede the rollback
    runs_after_stop: int


def _campaign(work_dir: Path) -> Run:
    method = SpyGEPAMethod()
    fixture = build_campaign(
        work_dir, method=method, make_runner=lambda world: RuleRunner(world, WEIGHTS), gate_tasks=VALIDATION
    )
    campaign = fixture.campaign
    campaign.run(ScriptedScheduler([MOVE] * PHASE_ONE_MOVES))
    campaign.rollback(ROLLBACK_TO)
    campaign.run(UntilBudgetSpent(campaign.chain.record_path), max_steps=200)
    runs = fixture.runner.runs
    # One more move once the budget is spent must not run anything either.
    assert campaign.advance(MOVE).skipped_reason is not None
    runs_after_stop = fixture.runner.runs - runs
    records = [json.loads(line) for line in campaign.chain.record_path.read_text().splitlines()]
    harness_records = [record for record in records if record["op"] in ("publish", "reject", "skip")][1:]
    # Every record between release 0 and the rollback is a harness record.
    rollback_at = [record["op"] for record in records[1:]].index("rollback")
    return Run(fixture, method, records, harness_records, rollback_at, runs_after_stop)


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Run:
    return _campaign(tmp_path_factory.mktemp("gepa-skeleton"))


def _state(record: Mapping[str, object]) -> Mapping[str, object]:
    if record["op"] == "skip":
        return record["method_state"]
    return record["candidate"]["method_state"]


def _archive(record: Mapping[str, object]) -> Archive:
    return Archive.from_state(_state(record))


def _instructions(record: Mapping[str, object]) -> str:
    files = dict(record["candidate"]["harness"]["files"])
    return str(files[INSTRUCTIONS_PATH]["content"])


def _identity(archive: Archive, index: int) -> tuple[object, ...]:
    candidate = archive.candidates[index]
    return candidate.components, candidate.parent, candidate.validation_scores


def _episodes(record: Mapping[str, object]) -> int:
    """The episodes GEPA ran for its proposal: the parent's and child's minibatches."""
    if record["op"] == "skip":
        return int(record["episodes_spent"])
    return int(record["candidate"]["metrics"]["method_episodes"])


def _charge(start: Archive, record: Mapping[str, object]) -> int:
    """What GEPA charges for one harness move: its minibatch episodes, plus validation when the gate ran.

    The gate's validation run costs one call per task for the child and, at the
    seed's first gate only, one per task for the seed (method.py:settle).
    """
    if record["op"] == "skip":
        return _episodes(record)
    seed_unscored = start.candidates[0].validation_scores is None
    return _episodes(record) + len(VALIDATION_IDS) * (2 if seed_unscored else 1)


def _path_totals(run: Run) -> list[tuple[int, int]]:
    """Per harness record, the charges and the minibatch episodes on the path to its state.

    A rollback returns the totals to where they stood when the target release was published.
    """
    at_release: dict[str, tuple[int, int]] = {"r0000": (0, 0)}
    charged = minibatch = 0
    totals = []
    harness = iter(zip(run.harness_records, run.method.starts, strict=True))
    for record in run.records[1:]:
        if record["op"] == "rollback":
            charged, minibatch = at_release[str(record["to"])]
            continue
        _, start = next(harness)
        charged += _charge(Archive.from_state(start), record)
        minibatch += _episodes(record)
        totals.append((charged, minibatch))
        if record["op"] == "publish":
            at_release[str(record["release_id"])] = (charged, minibatch)
    return totals


# ------------------------------------------------------------ #27's definition of done


@pytest.mark.unit
def test_a_campaign_selects_and_rejects(run: Run) -> None:
    ops = [record["op"] for record in run.harness_records]
    assert ops.count("publish") >= 1
    assert ops.count("reject") >= 1
    for record in run.harness_records:
        if record["op"] != "skip":
            assert record["decision"]["policy"] == "gepa"


@pytest.mark.unit
def test_a_rejected_candidate_stays_and_parents_a_later_proposal(run: Run) -> None:
    parented = []
    for position, record in enumerate(run.harness_records):
        if record["op"] != "reject":
            continue
        rejected = _archive(record)
        index = len(rejected.candidates) - 1
        for later in run.harness_records[position + 1 :]:
            archive = _archive(later)
            # The same candidate on the same path: a rollback can reuse an index for another one.
            # Its cursor is left out of the comparison, since it advances when the candidate parents.
            same = len(archive.candidates) > index and _identity(archive, index) == _identity(rejected, index)
            if same and any(candidate.parent == index for candidate in archive.candidates[index + 1 :]):
                parented.append(index)
                break
    assert parented, "no rejected candidate parented a later proposal"


@pytest.mark.unit
def test_method_state_in_every_record_round_trips_without_default(run: Run) -> None:
    assert len(run.method.ends) == len(run.harness_records)
    in_memory = [
        *run.method.starts,
        *run.method.ends,
        *(release.method_state for release in run.fixture.campaign.chain.releases()),
    ]
    for state in in_memory:
        validate_method_state(state)
        assert json.loads(json.dumps(state)) == state
    for record, end in zip(run.harness_records, run.method.ends, strict=True):
        assert _state(record) == json.loads(json.dumps(end))


@pytest.mark.unit
def test_rollback_mid_run_restores_the_archive_and_the_served_harness_together(run: Run) -> None:
    target = run.fixture.campaign.chain.get(ROLLBACK_TO)
    before = Archive.from_state(run.method.ends[run.rollback_at - 1])
    after = run.method.starts[run.rollback_at]

    assert after == target.method_state
    restored = Archive.from_state(after)
    assert len(restored.candidates) < len(before.candidates)
    assert target.harness is not None
    served = dict(restored.candidates[restored.served].components)[INSTRUCTIONS_PATH]
    assert target.harness.get(INSTRUCTIONS_PATH) is not None
    assert target.harness.get(INSTRUCTIONS_PATH).content == served
    # The campaign carried on from the restored archive and published again.
    assert any(record["op"] == "publish" for record in run.harness_records[run.rollback_at :])


@pytest.mark.unit
def test_metric_calls_count_gepas_episodes_on_the_final_path(run: Run) -> None:
    for record, start in zip(run.harness_records, run.method.starts, strict=True):
        assert _archive(record).metric_calls - Archive.from_state(start).metric_calls == _charge(
            Archive.from_state(start), record
        )
    final = _archive(run.harness_records[-1]).metric_calls
    assert final == _path_totals(run)[-1][0]
    # Not the runner's count: the gate re-runs the incumbent, and the rollback discarded a branch.
    assert final < run.fixture.runner.runs


# ------------------------------------------------------------ the rest of the run


@pytest.mark.unit
def test_every_runner_episode_is_in_the_commit_record(run: Run) -> None:
    gates = [record for record in run.harness_records if record["op"] != "skip"]
    recorded = sum(_episodes(record) for record in run.harness_records) + 2 * len(VALIDATION_IDS) * len(gates)
    assert run.fixture.runner.runs == recorded


@pytest.mark.unit
def test_the_campaign_stops_proposing_at_the_budget(run: Run) -> None:
    stopper = MaxMetricCallsStopper(MAX_METRIC_CALLS)
    for record, start in zip(run.harness_records, run.method.starts, strict=True):
        spent = Archive.from_state(start).metric_calls
        stopped = record["op"] == "skip" and _episodes(record) == 0
        assert stopper(SimpleNamespace(total_num_evals=spent)) == stopped
    starts = [Archive.from_state(start).metric_calls for start in run.method.starts]
    assert MAX_METRIC_CALLS in starts, "no proposal started exactly at the budget; the boundary is untested"
    final = _archive(run.harness_records[-1]).metric_calls
    # Upstream checks the budget before an iteration, so the last one may overshoot by one iteration's cost.
    assert MAX_METRIC_CALLS <= final < MAX_METRIC_CALLS + 2 * MINIBATCH + len(VALIDATION_IDS)
    assert run.runs_after_stop == 0


@pytest.mark.unit
def test_skips_are_recorded_with_what_they_spent(run: Run) -> None:
    skips = [
        (record, start)
        for record, start in zip(run.harness_records, run.method.starts, strict=True)
        if record["op"] == "skip"
    ]
    # Budget spent; the child was the parent's own text, so only the parent ran; the child did not improve.
    assert {_episodes(record) for record, _ in skips} == {0, MINIBATCH, 2 * MINIBATCH}
    for record, start in skips:
        assert len(_archive(record).candidates) == len(Archive.from_state(start).candidates)
    # The next proposal starts from the state a skip or rejection recorded, unless a rollback came between.
    for position, record in enumerate(run.harness_records[:-1]):
        if record["op"] != "publish" and position + 1 != run.rollback_at:
            assert run.method.starts[position + 1] == run.method.ends[position]


@pytest.mark.unit
def test_served_and_fronts_match_upstream_at_every_gate(run: Run) -> None:
    gates = ties = 0
    for record, (_, minibatch) in zip(run.harness_records, _path_totals(run), strict=True):
        if record["op"] == "skip":
            continue
        gates += 1
        metrics = record["decision"]["metrics"]
        ties += metrics["gepa_candidate_mean"] == metrics["gepa_served_mean"]
        archive = _archive(record)
        seed = archive.candidates[0].validation_scores
        children = [candidate.validation_scores for candidate in archive.candidates[1:]]
        assert seed is not None and all(scores is not None for scores in children)
        best, best_idx, fronts, evaluations = upstream_replay(VALIDATION_IDS, seed, children)[-1]
        assert archive.served == best == best_idx
        assert archive.fronts() == fronts
        # Upstream counts validation only; the minibatch episodes on the path are the rest.
        assert archive.metric_calls - minibatch == evaluations
        if record["op"] == "publish":
            assert _instructions(record) == dict(archive.candidates[archive.served].components)[INSTRUCTIONS_PATH]
    assert gates >= 3
    # A tie must keep the served candidate, as upstream keeps the earliest best; without one it is untested.
    assert ties >= 1


@pytest.mark.unit
def test_the_fronts_are_non_trivial(run: Run) -> None:
    # Without a candidate that leads a task while another is served, the fronts prove nothing.
    led_elsewhere = [
        record
        for record in run.harness_records
        if record["op"] != "skip"
        and any(_archive(record).served not in front for front in _archive(record).fronts().values())
    ]
    assert led_elsewhere


@pytest.mark.unit
def test_the_same_seed_writes_the_same_commit_record(run: Run, tmp_path: Path) -> None:
    again = _campaign(tmp_path)
    assert again.records == run.records
