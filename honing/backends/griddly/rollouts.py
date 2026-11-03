"""Sample real native matches on training tasks for the existing GRPO trainers."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Sequence
from pathlib import Path

from honing.backends.griddly.decisions import decision_samples
from honing.backends.griddly.grader import GriddlyGrader
from honing.backends.griddly.runner import GriddlyEpisodeRunner
from honing.core import ContractError, DataContract, HarnessSpec, PolicyRef, Task, TrainingBatch, run_graded

ROLLOUT_CONTRACT = DataContract("typed_decisions", requires_reward=True, requires_logprobs=True, requires_group=True)


class GriddlyRolloutProvider:
    def __init__(self, runner: GriddlyEpisodeRunner, grader: GriddlyGrader) -> None:
        self._runner, self._grader = runner.sampling(), grader

    def collect(
        self,
        tasks: Sequence[Task],
        policy: PolicyRef,
        harness: HarnessSpec | None,
        release_id: str | None,
        samples_per_task: int,
    ) -> TrainingBatch:
        if harness is None or type(samples_per_task) is not int or samples_per_task < 2:
            raise ContractError("Griddly rollouts need a harness and at least two repeats per task")
        if not tasks or any(self._runner.suite.split_of(task.task_id) != "train" for task in tasks):
            raise ContractError("Griddly rollouts use train tasks only")

        async def play() -> list[tuple[str, float, Path]]:
            async def one(task: Task) -> tuple[str, float, Path]:
                record = await run_graded(self._runner, self._grader, harness, task, policy)
                return task.task_id, record.score, Path(record.outcome.workspace_location or "")

            return list(await asyncio.gather(*(one(task) for task in tasks for _ in range(samples_per_task))))

        samples = []
        for task_id, reward, run in asyncio.run(play()):
            rows = [json.loads(line) for line in (run / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
            samples.extend(
                decision_samples(
                    rows,
                    group=task_id,
                    reward=reward,
                    release_id=release_id,
                    harness_id=harness.spec_id,
                    behavior=True,
                )
            )
        if not samples:
            raise ContractError("Griddly rollouts produced no model decisions")
        return TrainingBatch("griddly-rollout-" + uuid.uuid4().hex[:16], ROLLOUT_CONTRACT, decisions=tuple(samples))
