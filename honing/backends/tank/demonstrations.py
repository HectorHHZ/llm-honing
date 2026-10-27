"""Planner-teacher demonstrations as typed-decision training data.

The planner teacher (``integrations/tank-war/web/game/teacher.js``) plays
the agent's seat against the planner opponent and answers the same questions,
under the same harness, that the model is asked: the motion with the planner's
best score, and fire only when that motion's fresh shot is verified safe. Its
answers are the labels, so supervised training on them teaches the model to
choose as the planner's geometry does. Only ``train`` tasks are played, and a
round is used only if its manifest says the teacher played it.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Sequence

from honing.backends.tank.decisions import decisions_from_log
from honing.backends.tank.runner import TankEpisodeRunner
from honing.core import ContractError, DataContract, DecisionSample, HarnessSpec, Task, TrainingBatch


class TankDemonstrationProvider:
    """Satisfies :class:`~honing.core.DataProvider`: teacher rounds on the train tasks as one typed-decisions batch.

    ``harness`` is the question text and controller the teacher is asked
    under; the campaign's seed harness is the usual choice. Returns the batch
    once, then ``None``; a contract other than ``typed_decisions`` gets ``None``.
    """

    def __init__(
        self,
        runner: TankEpisodeRunner,
        tasks: Sequence[Task],
        harness: HarnessSpec,
        batch_id: str = "tank-teacher-demonstrations",
    ) -> None:
        for task in tasks:
            if runner.suite.split_of(task.task_id) != "train":
                raise ContractError(f"demonstrations are played on train tasks only, not {task.task_id}")
        runner.check_spec(harness)
        self._runner = runner
        self._tasks = tuple(tasks)
        self._harness = harness
        self._batch_id = batch_id
        self._served = False

    async def _play(self, task: Task) -> list[DecisionSample]:
        scenario = self._runner.suite.scenario(task.task_id)
        name = f"{task.task_id}--teacher--{self._harness.spec_id}--{uuid.uuid4().hex[:8]}"
        directory = self._runner.runs_dir / name
        self._runner.write_harness(self._harness, self._runner.runs_dir / f"{name}.harness")
        arguments = [
            *self._runner.arguments(scenario, directory, self._runner.runs_dir / f"{name}.harness"),
            "--teacher",
        ]
        exit_code = await self._runner.play(arguments, name)
        if exit_code != 0:
            raise ContractError(f"the teacher round for {task.task_id} exited {exit_code}; see {directory}.log")
        if json.loads((directory / "manifest.json").read_text()).get("kind") != "teacher-demonstration":
            raise ContractError(f"{directory} is not a teacher demonstration")
        return decisions_from_log(
            directory,
            reward=None,
            group=None,
            release_id=None,
            harness_id=self._harness.spec_id,
            behavior=False,
        )

    async def _all(self) -> list[DecisionSample]:
        rounds = await asyncio.gather(*(self._play(task) for task in self._tasks))
        return [decision for decisions in rounds for decision in decisions]

    def next_batch(self, contract: DataContract) -> TrainingBatch | None:
        if contract.kind != "typed_decisions" or self._served:
            return None
        decisions = asyncio.run(self._all())
        if not decisions:
            return None
        self._served = True
        return TrainingBatch(self._batch_id, contract, decisions=tuple(decisions))
