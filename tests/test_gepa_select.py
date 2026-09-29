"""GEPA selection and recording (#25): the gate's validation run decides what is served and fills the archive."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest
from gepa.core.adapter import EvaluationBatch
from gepa.core.data_loader import ensure_loader
from gepa.core.engine import GEPAEngine
from gepa.core.result import GEPAResult
from gepa.core.state import GEPAState, ValsetEvaluation, initialize_gepa_state
from gepa.strategies.eval_policy import FullEvaluationPolicy
from gepa.utils.stop_condition import MaxMetricCallsStopper

from honing.core import (
    ContractError,
    EpisodeOutcome,
    EpisodeRecord,
    EvaluationResult,
    EvolveContext,
    Grade,
    HarnessFile,
    HarnessMoveConfig,
    HarnessProposal,
    HarnessSpec,
    Move,
    PolicyRef,
    ProposeResult,
    SelectionDecision,
    Task,
    UpdateCandidate,
)
from honing.core.state import validate_method_state
from honing.harness.methods.gepa import ARCHIVE_SCHEMA_VERSION, Archive, GEPAConfig, GEPAMethod
from tests.fakes.builder import TASKS, build_campaign
from tests.fakes.harness import INSTRUCTIONS_PATH, SEED_HARNESS, FakeChatModel, FakeEpisodeRunner, FakeTaskGrader
from tests.fakes.model import FakeModelSource, FakeWorld

TASK_IDS = tuple(task.task_id for task in TASKS)
WORLD = FakeWorld()
SNAPSHOT = FakeModelSource(WORLD).materialize(Path("unused"))
CHECKSUM = SNAPSHOT.checkpoint.checksum


def _method(max_metric_calls: int = 1000, task_ids: Sequence[str] = TASK_IDS) -> GEPAMethod:
    return GEPAMethod(FakeChatModel([]), GEPAConfig(max_metric_calls=max_metric_calls), task_ids)


def _seed(text: str = "Solve the task.\n") -> HarnessSpec:
    return SEED_HARNESS.with_file(INSTRUCTIONS_PATH, HarnessFile("instructions", text))


def _with_child(state: Mapping[str, object], text: str) -> tuple[Mapping[str, object], HarnessProposal]:
    """The state propose would return with a new child, and its proposal."""
    archive, _ = Archive.from_state(state).add({INSTRUCTIONS_PATH: text}, parent=0)
    return archive.to_state(), HarnessProposal(spec=_seed(text))


def _evaluation(
    candidate: Sequence[float],
    incumbent: Sequence[float],
    *,
    task_ids: Sequence[str] = TASK_IDS,
    checksum: str = CHECKSUM,
    with_records: bool = True,
) -> EvaluationResult:
    def record(task_id: str, score: float) -> EpisodeRecord:
        outcome = EpisodeOutcome(task_id=task_id, spec_id="h-unused", policy_checksum=checksum, exit_code=0)
        return EpisodeRecord(outcome=outcome, grade=Grade(score=score))

    records = (
        tuple(
            record(task_id, score)
            for scores in (candidate, incumbent)
            for task_id, score in zip(task_ids, scores, strict=True)
        )
        if with_records
        else ()
    )
    return EvaluationResult(
        task_ids=tuple(task_ids),
        candidate_scores=tuple(candidate),
        incumbent_scores=tuple(incumbent),
        evaluated_variant="master",
        records=records,
    )


def _judge(
    method: GEPAMethod, state: Mapping[str, object], text: str, evaluation: EvaluationResult
) -> tuple[SelectionDecision, Mapping[str, object]]:
    """Decide on and record one proposed child, as the campaign does (campaign.py:323-327)."""
    state, proposal = _with_child(state, text)
    candidate = UpdateCandidate(
        kind="harness", parent_id="r0000", master=SNAPSHOT, harness=proposal.spec, method_state=state
    )
    decision = method.selector.decide(candidate, evaluation)
    return decision, method.record(state, proposal, evaluation, decision)


def _first_gate(
    candidate: Sequence[float], seed: Sequence[float] = (0.5, 0.5, 0.5)
) -> tuple[SelectionDecision, Archive]:
    method = _method()
    decision, state = _judge(method, method.initial_state(_seed()), "Add the numbers.", _evaluation(candidate, seed))
    return decision, Archive.from_state(state)


# ------------------------------------------------------------ #25 acceptance


@pytest.mark.unit
def test_a_higher_mean_is_selected_and_becomes_the_served_candidate() -> None:
    decision, archive = _first_gate((0.75, 0.5, 0.5))

    assert decision.selected
    assert decision.policy == "gepa"
    assert archive.served == 1
    assert archive.candidates[1].validation_scores == (0.75, 0.5, 0.5)
    assert decision.metrics["gepa_candidate_mean"] == pytest.approx(1.75 / 3)
    assert decision.metrics["gepa_served_mean"] == 0.5
    assert decision.metrics["gepa_archive_size"] == 2.0
    # Fronts are per task: the seed ties the child on "sort" and "parse", so both are on one.
    assert archive.fronts() == {"sum": {1}, "sort": {0, 1}, "parse": {0, 1}}
    assert decision.metrics["gepa_front_size"] == 2.0


@pytest.mark.unit
def test_a_lower_mean_is_rejected_but_stays_on_the_fronts_it_leads_and_can_parent() -> None:
    decision, archive = _first_gate((1.0, 0.0, 0.0))

    assert not decision.selected
    assert archive.served == 0
    assert archive.candidates[1].validation_scores == (1.0, 0.0, 0.0)
    assert archive.fronts() == {"sum": {1}, "sort": {0}, "parse": {0}}
    assert decision.metrics["gepa_front_size"] == 2.0
    parents = set()
    for _ in range(50):
        archive, parent = archive.sample_parent()
        parents.add(parent)
    assert parents == {0, 1}


@pytest.mark.unit
def test_a_tie_on_the_mean_is_rejected_even_when_the_tasks_differ() -> None:
    decision, archive = _first_gate((1.0, 0.0, 0.5))

    assert not decision.selected
    assert archive.served == 0
    assert archive.fronts() == {"sum": {1}, "sort": {0}, "parse": {0, 1}}


# A move's training tasks name the keyword that scores on them, so a child can
# pass its minibatch on one task without leading on the validation mean.
KEYWORDS = {"sum": "add", "sort": "order", "parse": "date"}


class KeywordRunner(FakeEpisodeRunner):
    """Scores 0.25 above the weights' quality on a task whose keyword is in the instructions."""

    async def run(self, spec: HarnessSpec, task: Task, policy: PolicyRef) -> EpisodeOutcome:
        self.runs += 1
        text = " ".join(file.content.lower() for _, file in spec.files if file.kind == "instructions")
        hit = KEYWORDS[task.task_id.removeprefix("train-")] in text
        quality = round(min(1.0, self.world.quality[policy.checkpoint.checksum] + (0.25 if hit else 0.0)), 6)
        return EpisodeOutcome(
            task_id=task.task_id,
            spec_id=spec.spec_id,
            policy_checksum=policy.checkpoint.checksum,
            exit_code=0,
            final_output=f"quality={quality} verified={int(hit)}",
        )


class SpyGEPAMethod(GEPAMethod):
    """Remembers the state each proposal started from."""

    def __init__(self, replies: Sequence[str]) -> None:
        super().__init__(FakeChatModel(replies), GEPAConfig(max_metric_calls=1000, minibatch_size=1), TASK_IDS)
        self.states: list[dict[str, object]] = []

    def propose(self, context: EvolveContext) -> ProposeResult:
        self.states.append(dict(context.state))
        return super().propose(context)


def _move(train: str) -> Move:
    task = Task(task_id=f"train-{train}", prompt=f"Train on {train}.")
    return Move(kind="harness", harness=HarnessMoveConfig(tasks=(task,), episode_budget=10))


def _reply(text: str) -> str:
    return f"```\n{text}\n```"


@pytest.mark.unit
def test_the_campaign_serves_a_selected_child_keeps_a_rejected_one_and_rollback_restores_the_archive(
    tmp_path: Path,
) -> None:
    method = SpyGEPAMethod(
        [_reply("Add the numbers."), _reply("Order the list."), _reply("Parse the date."), _reply("Add it all.")]
    )
    fixture = build_campaign(tmp_path, method=method, make_runner=KeywordRunner)
    campaign = fixture.campaign

    selected = campaign.advance(_move("sum"))
    assert selected.decision is not None and selected.decision.selected
    release = selected.release
    assert release is not None
    archive = Archive.from_state(release.method_state)
    # The served candidate is the harness the chain published.
    served = dict(archive.candidates[archive.served].components)[INSTRUCTIONS_PATH]
    assert release.harness is not None and release.harness.get(INSTRUCTIONS_PATH) == HarnessFile(
        "instructions", served
    )
    assert served == "Add the numbers."
    # 1 parent + 1 child minibatch episode, then the seed's and the child's validation runs.
    assert archive.metric_calls == 2 + 3 + 3

    # Passes its minibatch on "sort", ties the served mean on validation: rejected.
    tied = campaign.advance(_move("sort"))
    assert tied.decision is not None and not tied.decision.selected
    assert tied.decision.metrics["gepa_candidate_mean"] == tied.decision.metrics["gepa_served_mean"]

    campaign.advance(_move("parse"))
    pending = Archive.from_state(method.states[-1])
    assert pending.served == 1
    assert pending.candidates[2].validation_scores == (0.5, 0.75, 0.5)
    assert pending.fronts()["sort"] == {2}
    assert pending.metric_calls == archive.metric_calls + 2 + 3

    campaign.rollback(release.release_id)
    campaign.advance(_move("sum"))
    assert method.states[-1] == dict(release.method_state)


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["train", "prune", "quantize"])
def test_after_a_weight_move_a_gepa_move_is_refused_before_any_episode_runs(tmp_path: Path, kind: str) -> None:
    # Weights are assumed frozen under GEPA: a prune publishes an edited master
    # (campaign.py:237-249) and a quantize publishes serving weights (campaign.py:260-265),
    # and either changes the checksum head.policy() reports (candidate.py:78-81).
    method = SpyGEPAMethod([_reply("Add the numbers."), _reply("Add the numbers and order the list.")])
    fixture = build_campaign(tmp_path, method=method, make_runner=KeywordRunner)
    campaign = fixture.campaign
    first = campaign.advance(_move("sum")).release
    assert first is not None
    changed = campaign.advance(Move(kind=kind))  # type: ignore[arg-type]
    assert changed.release is not None, f"the {kind} move must publish for the test to mean anything"
    assert changed.release.policy().checkpoint.checksum != first.policy().checkpoint.checksum
    runs = fixture.runner.runs

    with pytest.raises(ContractError, match="validation scores were taken under weights"):
        campaign.advance(_move("sort"))
    assert fixture.runner.runs == runs


@pytest.mark.unit
def test_propose_refuses_other_weights_before_drawing_running_or_reflecting() -> None:
    chat = FakeChatModel([_reply("Add the numbers.")])
    method = GEPAMethod(chat, GEPAConfig(max_metric_calls=1000), TASK_IDS)
    _, state = _judge(method, method.initial_state(_seed()), "Add.", _evaluation((0.75, 0.5, 0.5), (0.5,) * 3))
    retrained = FakeModelSource(WORLD, quality=0.6).materialize(Path("unused"))
    assert retrained.checkpoint.checksum != CHECKSUM
    runner = FakeEpisodeRunner(WORLD)
    context = EvolveContext(
        current=_seed("Add."),
        policy=PolicyRef(retrained.checkpoint, retrained.architecture, "master"),
        tasks=TASKS,
        records=(),
        state=state,
        runner=runner,
        grader=FakeTaskGrader(),
        episode_budget=100,
    )

    with pytest.raises(ContractError, match="validation scores were taken under weights"):
        method.propose(context)
    assert runner.runs == 0
    assert chat.conversations == []


@pytest.mark.unit
def test_the_selector_refuses_scores_under_other_weights() -> None:
    method = _method()
    _, state = _judge(method, method.initial_state(_seed()), "Add.", _evaluation((0.75, 0.5, 0.5), (0.5,) * 3))

    with pytest.raises(ContractError, match="validation scores were taken under weights"):
        _judge(method, state, "Order.", _evaluation((0.9, 0.5, 0.5), (0.5,) * 3, checksum="other-weights"))


# ------------------------------------------------------------ the seed, the budget, and what the gate must bring


@pytest.mark.unit
def test_the_seed_is_scored_from_the_first_gate_only_and_later_incumbent_runs_are_ignored() -> None:
    method = _method()
    _, state = _judge(method, method.initial_state(_seed()), "Add.", _evaluation((0.0, 0.0, 0.25), (0.5,) * 3))
    # A later gate re-runs the incumbent and gets 0.9; the stored 0.5 is what is compared.
    decision, state = _judge(method, state, "Order.", _evaluation((0.6, 0.5, 0.5), (0.9,) * 3))

    archive = Archive.from_state(state)
    assert decision.selected
    assert decision.metrics["gepa_served_mean"] == 0.5
    assert archive.candidates[0].validation_scores == (0.5, 0.5, 0.5)
    assert archive.served == 2


@pytest.mark.unit
def test_validation_costs_one_metric_call_per_task_for_the_seed_once_and_for_every_child() -> None:
    method = _method()
    decision, state = _judge(method, method.initial_state(_seed()), "Add.", _evaluation((0.0,) * 3, (0.5,) * 3))
    assert Archive.from_state(state).metric_calls == 6
    assert decision.metrics["gepa_metric_calls"] == 6.0
    decision, state = _judge(method, state, "Order.", _evaluation((1.0,) * 3, (0.5,) * 3))
    assert Archive.from_state(state).metric_calls == 9
    assert decision.metrics["gepa_metric_calls"] == 9.0


@pytest.mark.unit
def test_the_budget_stop_counts_validation_as_upstreams_stopper_does() -> None:
    method = _method(max_metric_calls=6)
    _, state = _judge(method, method.initial_state(_seed()), "Add.", _evaluation((0.75, 0.5, 0.5), (0.5,) * 3))
    runner = FakeEpisodeRunner(WORLD)
    context = EvolveContext(
        current=_seed("Add."),
        policy=PolicyRef(SNAPSHOT.checkpoint, SNAPSHOT.architecture, "master"),
        tasks=TASKS,
        records=(),
        state=state,
        runner=runner,
        grader=FakeTaskGrader(),
        episode_budget=100,
    )

    assert MaxMetricCallsStopper(6)(SimpleNamespace(total_num_evals=Archive.from_state(state).metric_calls))
    assert method.propose(context).proposal is None
    assert runner.runs == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    "task_ids",
    [("sort", "sum", "parse"), ("sum", "sort"), tuple(f"{task_id}@candidate-0" for task_id in TASK_IDS)],
    ids=["reordered", "missing", "duel"],
)
def test_the_gate_must_score_the_archives_validation_tasks_in_order(task_ids: tuple[str, ...]) -> None:
    method = _method()
    scores = (0.5,) * len(task_ids)
    evaluation = _evaluation(scores, scores, task_ids=task_ids)

    with pytest.raises(ContractError, match="validates on"):
        _judge(method, method.initial_state(_seed()), "Add.", evaluation)


@pytest.mark.unit
def test_the_gate_must_bring_its_episode_records() -> None:
    method = _method()
    evaluation = _evaluation((0.75, 0.5, 0.5), (0.5,) * 3, with_records=False)

    with pytest.raises(ContractError, match="episode records"):
        _judge(method, method.initial_state(_seed()), "Add.", evaluation)


@pytest.mark.unit
def test_record_refuses_a_proposal_that_is_not_the_newest_candidate() -> None:
    method = _method()
    state, _ = _with_child(method.initial_state(_seed()), "Add.")
    evaluation = _evaluation((0.75, 0.5, 0.5), (0.5,) * 3)
    decision = SelectionDecision(outcome="select", policy="gepa", reason="test")

    with pytest.raises(ContractError, match="newest candidate"):
        method.record(state, HarnessProposal(spec=_seed("Something else.")), evaluation, decision)


@pytest.mark.unit
def test_the_recorded_state_names_its_weights_and_round_trips_through_json() -> None:
    method = _method()
    _, state = _judge(method, method.initial_state(_seed()), "Add.", _evaluation((0.75, 0.5, 0.5), (0.5,) * 3))

    validate_method_state(state)
    assert state["schema_version"] == ARCHIVE_SCHEMA_VERSION == 3
    assert state["validation_policy"] == CHECKSUM
    loaded = json.loads(json.dumps(state))
    assert Archive.from_state(loaded) == Archive.from_state(state)


# ------------------------------------------------------------ parity with upstream GEPA


def _upstream_replay(
    seed_scores: Sequence[float], children: Sequence[Sequence[float]]
) -> list[tuple[int, int, dict[str, set[int]], int]]:
    """Upstream's state after the seed and after each child: best, GEPAResult.best_idx, fronts, metric calls.

    The seed goes through ``initialize_gepa_state`` and each child through
    ``_evaluate_programs_on_valset`` plus the two calls ``_add_evaluated_program``
    makes to charge and add it (engine.py:282, 291).
    """
    task_ids = TASK_IDS[: len(seed_scores)]
    table = {("seed", task_id): score for task_id, score in zip(task_ids, seed_scores, strict=True)}
    for index, scores in enumerate(children):
        table.update({(f"child {index}", task_id): score for task_id, score in zip(task_ids, scores, strict=True)})

    def batch_evaluate(items: list[tuple[dict[str, str], list[str]]]) -> list[EvaluationBatch]:
        return [
            EvaluationBatch(outputs=[None] * len(batch), scores=[table[(program["text"], t)] for t in batch])
            for program, batch in items
        ]

    engine = SimpleNamespace(
        valset=ensure_loader(list(task_ids)),
        val_evaluation_policy=FullEvaluationPolicy(),
        _batch_evaluate=batch_evaluate,
    )
    seed_evaluation = ValsetEvaluation(
        outputs_by_val_id=dict.fromkeys(range(len(task_ids))),
        scores_by_val_id={index: table[("seed", task_id)] for index, task_id in enumerate(task_ids)},
    )
    state: GEPAState = initialize_gepa_state(
        run_dir=None,
        logger=SimpleNamespace(log=lambda *_: None),
        seed_candidate={"text": "seed"},
        seed_valset_evaluation=seed_evaluation,
    )
    history = []

    def snapshot() -> None:
        fronts = {task_ids[key]: set(members) for key, members in state.get_pareto_front_mapping().items()}
        best = FullEvaluationPolicy().get_best_program(state)
        history.append((best, GEPAResult.from_state(state).best_idx, fronts, state.total_num_evals))

    for index in range(len(children)):
        program = {"text": f"child {index}"}
        ((evaluation, calls),) = GEPAEngine._evaluate_programs_on_valset(engine, [program], state)
        discovered_at = state.total_num_evals
        state.increment_evals(calls)
        state.update_state_with_new_program([0], program, evaluation, None, discovered_at)
        snapshot()
    return history


def _honing_replay(
    seed_scores: Sequence[float], children: Sequence[Sequence[float]], rng: random.Random
) -> list[tuple[int, dict[str, set[int]], int]]:
    task_ids = TASK_IDS[: len(seed_scores)]
    method = _method(task_ids=task_ids)
    state = method.initial_state(_seed("seed"))
    history = []
    for index, scores in enumerate(children):
        # Only the first gate's incumbent side is the seed's validation run; later ones must not matter.
        incumbent = seed_scores if index == 0 else tuple(rng.random() for _ in task_ids)
        evaluation = _evaluation(scores, incumbent, task_ids=task_ids)
        _, state = _judge(method, state, f"child {index}", evaluation)
        archive = Archive.from_state(state)
        history.append((archive.served, archive.fronts(), archive.metric_calls))
    return history


def _histories() -> list[tuple[tuple[float, ...], list[tuple[float, ...]]]]:
    rng = random.Random(25)
    cases = [
        # A tie under plain float addition (3.10, 3.11) that is not one under 3.12's
        # compensated sum or math.fsum: the served candidate must follow upstream's sum/len.
        ((0.7, 0.1, 0.6), [(0.1, 0.9, 0.4)]),
        ((0.5, 0.5, 0.5), [(1.0, 0.0, 0.5), (0.5, 0.5, 0.5), (0.75, 0.5, 0.5), (1.0, 0.25, 0.5)]),
    ]
    for _ in range(40):
        tasks = rng.randint(1, 3)
        grid = rng.random() < 0.5
        draw = (lambda: rng.choice((0.0, 0.5, 1.0))) if grid else (lambda: round(rng.random(), 2))
        seed = tuple(draw() for _ in range(tasks))
        cases.append((seed, [tuple(draw() for _ in range(tasks)) for _ in range(rng.randint(1, 8))]))
    return cases


@pytest.mark.unit
@pytest.mark.parametrize(("seed_scores", "children"), _histories())
def test_served_fronts_and_metric_calls_match_upstream_after_every_validation(
    seed_scores: tuple[float, ...], children: list[tuple[float, ...]]
) -> None:
    upstream = _upstream_replay(seed_scores, children)
    honing = _honing_replay(seed_scores, children, random.Random(0))

    for (best, result_best, upstream_fronts, total_evals), (served, fronts, metric_calls) in zip(
        upstream, honing, strict=True
    ):
        assert served == best == result_best
        assert fronts == upstream_fronts
        assert metric_calls == total_evals
