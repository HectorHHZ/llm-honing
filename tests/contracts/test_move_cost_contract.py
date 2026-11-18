"""The two core contracts a move's cost rests on (#167): run_graded charges every episode, trainers report GPU time.

Every episode a campaign spends goes through :func:`honing.core.run_graded`, so its charge is what makes the
gate's, a method's and a rollout's episodes countable at all. ``TrainResult.gpu_seconds`` is the only way a
trainer's GPU time reaches the move, and ``None`` must stay distinct from 0 so an unreported job never looks free.
"""

from __future__ import annotations

import asyncio
import math

import pytest

from honing.core import (
    ContractError,
    EpisodeOutcome,
    HarnessSpec,
    PolicyRef,
    Task,
    TrainJobSpec,
    TrainResult,
    charging,
    protocols,
    run_graded,
)
from tests.fakes.builder import TASKS
from tests.fakes.harness import SEED_HARNESS, FakeEpisodeRunner, FakeTaskGrader
from tests.fakes.model import FakeModelSource, FakeTrainer, FakeWorld


class CountingSink:
    """Satisfies :class:`honing.core.CostSink` by counting what it is charged."""

    def __init__(self) -> None:
        self.episodes = 0
        self.chat_calls = 0

    def charge_episode(self) -> None:
        self.episodes += 1

    def charge_chat(self, model: str | None, input_tokens: int, output_tokens: int, cost_usd: float | None) -> None:
        self.chat_calls += 1


class FailingRunner(FakeEpisodeRunner):
    """Raises from every episode, after it has started."""

    async def run(self, spec: HarnessSpec, task: Task, policy: PolicyRef) -> EpisodeOutcome:
        self.runs += 1
        raise RuntimeError("the agent crashed")


class FailedOutcomeRunner(FakeEpisodeRunner):
    """Returns an outcome whose ``failure`` is set, which ``run_graded`` scores 0 without grading."""

    async def run(self, spec: HarnessSpec, task: Task, policy: PolicyRef) -> EpisodeOutcome:
        outcome = await super().run(spec, task, policy)
        return EpisodeOutcome(
            task_id=outcome.task_id,
            spec_id=outcome.spec_id,
            policy_checksum=outcome.policy_checksum,
            exit_code=1,
            failure="the agent exited 1",
        )


def _policy(world: FakeWorld) -> PolicyRef:
    snapshot = FakeModelSource(world).materialize(None)  # type: ignore[arg-type]
    return PolicyRef(snapshot.checkpoint, snapshot.architecture, "master")


@pytest.mark.unit
def test_run_graded_charges_one_episode_per_call_including_concurrent_ones() -> None:
    world = FakeWorld()
    runner, grader, policy = FakeEpisodeRunner(world), FakeTaskGrader(), _policy(world)
    sink = CountingSink()

    async def all_tasks() -> None:
        await asyncio.gather(*(run_graded(runner, grader, SEED_HARNESS, task, policy) for task in TASKS))

    with charging(sink):
        asyncio.run(all_tasks())  # the tasks asyncio starts inherit the sink

    assert sink.episodes == runner.runs == len(TASKS)
    assert sink.chat_calls == 0


@pytest.mark.unit
def test_run_graded_charges_an_episode_that_fails_or_raises() -> None:
    world = FakeWorld()
    policy, sink = _policy(world), CountingSink()

    with charging(sink):
        asyncio.run(run_graded(FailedOutcomeRunner(world), FakeTaskGrader(), SEED_HARNESS, TASKS[0], policy))
        with pytest.raises(RuntimeError, match="crashed"):
            asyncio.run(run_graded(FailingRunner(world), FakeTaskGrader(), SEED_HARNESS, TASKS[0], policy))

    assert sink.episodes == 2


@pytest.mark.unit
def test_run_graded_charges_nothing_without_a_sink_and_the_sink_is_removed_after_the_block() -> None:
    world = FakeWorld()
    runner, grader, policy = FakeEpisodeRunner(world), FakeTaskGrader(), _policy(world)
    sink = CountingSink()

    with charging(sink):
        asyncio.run(run_graded(runner, grader, SEED_HARNESS, TASKS[0], policy))
    asyncio.run(run_graded(runner, grader, SEED_HARNESS, TASKS[0], policy))

    assert runner.runs == 2 and sink.episodes == 1


@pytest.mark.unit
def test_gpu_seconds_defaults_to_unknown_not_zero() -> None:
    snapshot = FakeModelSource(FakeWorld()).materialize(None)  # type: ignore[arg-type]

    assert TrainResult(snapshot=snapshot).gpu_seconds is None
    assert TrainResult(snapshot=snapshot, gpu_seconds=0.0).gpu_seconds == 0.0


@pytest.mark.unit
@pytest.mark.parametrize("seconds", [-1.0, math.nan, math.inf])
def test_gpu_seconds_must_be_a_finite_number_of_seconds_spent(seconds: float) -> None:
    snapshot = FakeModelSource(FakeWorld()).materialize(None)  # type: ignore[arg-type]

    with pytest.raises(ContractError, match="gpu_seconds"):
        TrainResult(snapshot=snapshot, gpu_seconds=seconds)


@pytest.mark.unit
def test_the_trainer_fake_reports_gpu_seconds_through_the_trainer_protocol() -> None:
    world = FakeWorld()
    trainer = FakeTrainer(world, gpu_seconds_per_step=2.0)
    assert isinstance(trainer, protocols.Trainer)
    base = FakeModelSource(world).materialize(None)  # type: ignore[arg-type]

    result = trainer.train(TrainJobSpec(algorithm="sft", steps=3), base, None)

    assert result.gpu_seconds == 6.0
