"""Tank rounds played by the policy being trained, as typed decisions with rewards.

Each round is played with every policy choice drawn from the model's own
probabilities (:meth:`TankEpisodeRunner.sampling`), so repeats of one task
differ and a group has a reward spread to learn from. Every decision in a round
earns that round's score (1 win, 0.5 draw or timeout, 0 loss), is grouped by
task so repeats of one scenario share a baseline, and names the release and
harness it was played under, so a batch cannot mix them.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Sequence
from pathlib import Path

from honing.backends.tank.decisions import decisions_from_log
from honing.backends.tank.grader import TankGrader
from honing.backends.tank.runner import TankEpisodeRunner
from honing.core import ContractError, DataContract, HarnessSpec, PolicyRef, Task, TrainingBatch, run_graded

ROLLOUT_CONTRACT = DataContract("typed_decisions", requires_reward=True, requires_logprobs=True, requires_group=True)


class TankRolloutProvider:
    """Satisfies :class:`~honing.core.RolloutProvider` for tanks: plays tasks with the policy, returns its decisions.

    Each task is played ``samples_per_task`` times by ``runner.sampling()`` and
    graded by ``grader``; a round the grader refuses stops collection.
    """

    def __init__(self, runner: TankEpisodeRunner, grader: TankGrader) -> None:
        self._runner = runner.sampling()
        self._grader = grader

    async def _play(
        self, tasks: Sequence[Task], policy: PolicyRef, harness: HarnessSpec, samples_per_task: int
    ) -> list[tuple[Task, float, Path]]:
        async def one(task: Task) -> tuple[Task, float, Path]:
            record = await run_graded(self._runner, self._grader, harness, task, policy)
            return task, record.score, Path(record.outcome.workspace_location or "")

        return list(await asyncio.gather(*(one(t) for t in tasks for _ in range(samples_per_task))))

    def collect(
        self,
        tasks: Sequence[Task],
        policy: PolicyRef,
        harness: HarnessSpec | None,
        release_id: str | None,
        samples_per_task: int,
    ) -> TrainingBatch:
        if harness is None:
            raise ContractError("tank rollouts play under a harness; the campaign passed none")
        played = asyncio.run(self._play(tasks, policy, harness, samples_per_task))
        decisions = [
            d
            for task, score, run in played
            for d in decisions_from_log(
                run,
                reward=score,
                group=task.task_id,
                release_id=release_id,
                harness_id=harness.spec_id,
                behavior=True,
            )
        ]
        if not decisions:
            raise ContractError("the rollout rounds produced no decisions")
        # Two collections from one release are different batches, so the id names this one.
        batch_id = f"tank-rollout-{release_id or 'unpublished'}-{len(played)}x-{uuid.uuid4().hex[:8]}"
        return TrainingBatch(batch_id, ROLLOUT_CONTRACT, decisions=tuple(decisions))
