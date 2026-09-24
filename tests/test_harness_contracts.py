"""The harness contracts from RFC #15: graded episodes, recorded skips, method state rules, and gate statistics."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from honing.campaign import ScriptedScheduler
from honing.core import (
    CampaignRuleError,
    ContractError,
    EpisodeOutcome,
    EvaluationResult,
    EvolveContext,
    HarnessFile,
    HarnessPhaseConfig,
    HarnessProposal,
    HarnessSpec,
    MethodRequirements,
    Move,
    ProposeResult,
    ReleaseError,
    SelectionDecision,
    Task,
    run_graded,
)
from honing.release import GATE_METRIC_NAMES, NoRegressionSelector, WinLossSelector, paired_difference
from honing.release.chain import _json_default
from tests.fakes.builder import TASKS, build_campaign
from tests.fakes.harness import INSTRUCTIONS_PATH, FakeEpisodeRunner, FakeTaskGrader
from tests.fakes.model import FakeWorld


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@dataclass
class MinibatchMethod:
    """Scores the current harness on one task before deciding, the way GEPA checks a minibatch."""

    propose_anything: bool = True
    name: str = "minibatch"
    state_schema_version: int = 1
    requires: MethodRequirements = field(default_factory=lambda: MethodRequirements(file_kinds=frozenset()))
    selector: WinLossSelector = field(default_factory=WinLossSelector)
    contexts: list[EvolveContext] = field(default_factory=list)

    def initial_state(self, seed: HarnessSpec) -> Mapping[str, object]:
        return {"schema_version": self.state_schema_version, "scores": []}

    def propose(self, context: EvolveContext) -> ProposeResult:
        self.contexts.append(context)
        record = asyncio.run(context.episode(context.current, context.tasks[0], context.policy))
        scores = [*list(context.state["scores"]), record.score]  # type: ignore[call-overload]
        state = {**context.state, "scores": scores}
        if not self.propose_anything:
            return ProposeResult(proposal=None, state=state, episodes_spent=1)
        spec = context.current.with_file(INSTRUCTIONS_PATH, HarnessFile("instructions", "Verify the result.\n"))
        return ProposeResult(proposal=HarnessProposal(spec=spec), state=state, episodes_spent=1)

    def record(
        self,
        state: Mapping[str, object],
        proposal: HarnessProposal,
        evaluation: EvaluationResult,
        decision: SelectionDecision,
    ) -> Mapping[str, object]:
        return state


# ------------------------------------------------------------ proposal 1: skips reach the record


@pytest.mark.unit
def test_a_skip_records_its_state_and_the_episodes_it_spent(tmp_path: Path) -> None:
    method = MinibatchMethod(propose_anything=False)
    fixture = build_campaign(tmp_path, method=method)
    head = fixture.campaign.chain.head()

    (outcome,) = fixture.campaign.run(ScriptedScheduler([Move(kind="harness", phase="pre_training")]))

    assert outcome.skipped_reason == "minibatch proposed nothing"
    assert fixture.campaign.chain.head() is head
    skip = _records(fixture.campaign.chain.record_path)[-1]
    assert skip["op"] == "skip"
    assert skip["episodes_spent"] == 1
    assert skip["phase"] == "pre_training"
    assert skip["parent_id"] == head.release_id
    assert skip["method_state"] == {"schema_version": 1, "scores": [0.5]}


@pytest.mark.unit
def test_state_from_a_skip_carries_into_the_next_attempt(tmp_path: Path) -> None:
    method = MinibatchMethod(propose_anything=False)
    fixture = build_campaign(tmp_path, method=method)

    fixture.campaign.run(ScriptedScheduler([Move(kind="harness"), Move(kind="harness")]))

    assert method.contexts[1].state["scores"] == [0.5]


@pytest.mark.unit
def test_rollback_discards_state_from_a_skip(tmp_path: Path) -> None:
    method = MinibatchMethod(propose_anything=False)
    fixture = build_campaign(tmp_path, method=method)
    head = fixture.campaign.chain.head()
    fixture.campaign.run(ScriptedScheduler([Move(kind="harness")]))

    fixture.campaign.rollback(head.release_id)
    fixture.campaign.run(ScriptedScheduler([Move(kind="harness")]))

    assert method.contexts[1].state["scores"] == []


@pytest.mark.unit
def test_a_proposal_carries_the_episodes_its_method_spent(tmp_path: Path) -> None:
    fixture = build_campaign(tmp_path, method=MinibatchMethod())

    (outcome,) = fixture.campaign.run(ScriptedScheduler([Move(kind="harness")]))

    assert outcome.release is not None
    assert outcome.release.metrics["method_episodes"] == 1.0


@pytest.mark.unit
def test_a_skip_on_a_stale_head_is_refused(tmp_path: Path) -> None:
    chain = build_campaign(tmp_path).campaign.chain

    with pytest.raises(ReleaseError):
        chain.record_skip(parent_id="r9999", method="m", reason="r", state={}, episodes_spent=0)


# ------------------------------------------------------------ proposal 2: running and grading are separate


@pytest.mark.unit
def test_a_method_grades_its_own_minibatch_through_the_context(tmp_path: Path) -> None:
    fixture = build_campaign(tmp_path, method=MinibatchMethod(propose_anything=False))
    grades_before = fixture.grader.grades

    fixture.campaign.run(ScriptedScheduler([Move(kind="harness")]))

    assert fixture.grader.grades == grades_before + 1


@pytest.mark.unit
def test_an_outcome_for_a_different_episode_is_refused() -> None:
    world = FakeWorld()
    runner = FakeEpisodeRunner(world)

    class WrongTaskRunner:
        support = runner.support

        async def run(self, spec: HarnessSpec, task: Task, policy: object) -> EpisodeOutcome:
            return EpisodeOutcome(task_id="other", spec_id=spec.spec_id, policy_checksum="x", exit_code=0)

    fixture_policy = build_policy(world)
    with pytest.raises(ContractError, match="asked to run on task"):
        asyncio.run(
            run_graded(WrongTaskRunner(), FakeTaskGrader(), HarnessSpec("claude_code"), TASKS[0], fixture_policy)
        )


def build_policy(world: FakeWorld):  # type: ignore[no-untyped-def]
    from tests.fakes.model import FakeModelSource

    snapshot = FakeModelSource(world).materialize(Path("unused"))
    from honing.core import PolicyRef

    return PolicyRef(snapshot.checkpoint, snapshot.architecture, "master")


# ------------------------------------------------------------ proposal 5: phases and gate statistics


@pytest.mark.unit
def test_a_phase_move_uses_that_phase_s_tasks(tmp_path: Path) -> None:
    method = MinibatchMethod(propose_anything=False)
    phase_tasks = (Task(task_id="phase-only", prompt="x"),)
    fixture = build_campaign(
        tmp_path,
        method=method,
        phases={"post_training": HarnessPhaseConfig(tasks=phase_tasks, max_proposals=1, max_episodes=4)},
    )

    fixture.campaign.run(ScriptedScheduler([Move(kind="harness", phase="post_training"), Move(kind="harness")]))

    assert method.contexts[0].tasks == phase_tasks
    assert method.contexts[1].tasks == TASKS


@pytest.mark.unit
def test_published_and_rejected_candidates_both_carry_the_gate_result(tmp_path: Path) -> None:
    fixture = build_campaign(tmp_path, trainer_gain=-0.1)

    fixture.campaign.run(ScriptedScheduler([Move(kind="train")]))
    fixture.campaign.run(ScriptedScheduler([Move(kind="harness")]))

    records = _records(fixture.campaign.chain.record_path)
    gated = [record for record in records if record["op"] in {"publish", "reject"}]
    rejected, published = (record["candidate"] for record in gated[1:])
    assert [record["op"] for record in gated] == ["publish", "reject", "publish"]
    for candidate in (rejected, published):
        assert set(GATE_METRIC_NAMES) <= set(candidate["metrics"])  # type: ignore[arg-type]
    assert published["metrics"]["gate_tasks"] == float(len(TASKS))  # type: ignore[index]


@pytest.mark.unit
def test_the_bootstrap_interval_is_reproducible_and_brackets_the_mean() -> None:
    evaluation = EvaluationResult(
        task_ids=tuple(f"t{i}" for i in range(8)),
        candidate_scores=(0.9, 0.7, 0.8, 0.6, 0.9, 0.5, 0.7, 0.8),
        incumbent_scores=(0.8, 0.7, 0.6, 0.6, 0.7, 0.6, 0.5, 0.8),
        evaluated_variant="master",
    )

    first, second = paired_difference(evaluation), paired_difference(evaluation)

    assert first == second
    assert first.tasks == 8
    assert first.low <= first.mean <= first.high
    assert first.low < first.high


@pytest.mark.unit
def test_identical_scores_give_an_interval_of_zero_width() -> None:
    evaluation = EvaluationResult(("a", "b"), (0.5, 0.7), (0.5, 0.7), "master")

    difference = paired_difference(evaluation)

    assert (difference.mean, difference.low, difference.high) == (0.0, 0.0, 0.0)


@pytest.mark.unit
@pytest.mark.parametrize("selector", [WinLossSelector(), NoRegressionSelector(0.05)], ids=lambda s: s.name)
def test_selectors_report_the_interval_with_their_decision(selector: object) -> None:
    evaluation = EvaluationResult(("a", "b", "c"), (0.9, 0.6, 0.8), (0.8, 0.6, 0.7), "master")

    decision = selector.decide(None, evaluation)  # type: ignore[attr-defined]

    assert {"tasks", "difference_mean", "difference_low", "difference_high"} <= set(decision.metrics)
    assert decision.metrics["tasks"] == 3.0


# ------------------------------------------------------------ proposal 6: method state and the record


@pytest.mark.unit
def test_state_under_another_schema_is_refused_before_the_method_runs(tmp_path: Path) -> None:
    method = MinibatchMethod()
    fixture = build_campaign(tmp_path, method=method)
    method.state_schema_version = 2

    with pytest.raises(CampaignRuleError) as refused:
        fixture.campaign.run(ScriptedScheduler([Move(kind="harness")]))

    assert refused.value.rule == "method_state_schema_matches"
    assert "version 2" in str(refused.value) and "version 1" in str(refused.value)
    assert method.contexts == []


@pytest.mark.unit
def test_a_method_whose_initial_state_disagrees_with_its_version_is_refused_at_start(tmp_path: Path) -> None:
    @dataclass
    class Mislabeled(MinibatchMethod):
        def initial_state(self, seed: HarnessSpec) -> Mapping[str, object]:
            return {"schema_version": 7}

    with pytest.raises(CampaignRuleError, match="method_state_schema_matches"):
        build_campaign(tmp_path, method=Mislabeled())


@pytest.mark.unit
def test_the_record_refuses_a_type_it_cannot_represent() -> None:
    with pytest.raises(TypeError, match="cannot hold a object"):
        _json_default(object())


@pytest.mark.unit
def test_the_record_writes_a_set_of_scalars_in_a_stable_order() -> None:
    assert _json_default({"b", "a"}) == ["a", "b"]
    assert _json_default(frozenset({3, 1})) == [1, 3]
