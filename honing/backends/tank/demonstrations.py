"""Planner-teacher demonstrations as typed-decision training data.

The planner teacher (``integrations/tank-war/web/game/teacher.js``) plays
the agent's seat against the planner opponent and answers the same questions,
under the same harness, that the model is asked: the motion with the planner's
best score, and fire only when that motion's fresh shot is verified safe. Its
answers are the labels, so supervised training on them teaches the model to
choose as the planner's geometry does. The first batch plays ``train`` tasks.
Every later batch plays new seeds that belong to no split
(:data:`~honing.backends.tank.scenarios.DEMONSTRATION_SEEDS`). A round is
used only if its manifest says the teacher played it.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Sequence
from dataclasses import replace

from honing.backends.tank.decisions import decisions_from_log
from honing.backends.tank.runner import TankEpisodeRunner
from honing.backends.tank.scenarios import DEMONSTRATION_SEEDS, TankScenario
from honing.core import (
    ContractError,
    DataContract,
    DecisionSample,
    HarnessSpec,
    Task,
    TrainingBatch,
    charge_episode,
)


class TankDemonstrationProvider:
    """Satisfies :class:`~honing.core.DataProvider`: teacher rounds as a typed-decisions batch, new on every call.

    ``harness`` is the question text and controller the teacher is asked
    under; the campaign's seed harness is the usual choice. The first batch
    plays ``tasks``. Each later batch plays the same seats and clocks on the
    next unused seeds of ``fresh_seeds``, so a second supervised move does not
    train on the rounds the first one saw. Batch ``n`` is named
    ``<batch_id>-<n>``, counting from 1. A contract other than
    ``typed_decisions`` gets ``None``, and so does a call after the fresh
    seeds run out.

    Every decision's group is ``tank:<seed>:<seat>``, the shape RA2's
    demonstrations use, so a decision-model trainer can hold rounds out for
    validation by seed. Fresh seeds are consecutive, so a pattern on the
    seed's last digit, such as ``^tank:\\d*[05]:``, holds out a share of every batch.
    """

    def __init__(
        self,
        runner: TankEpisodeRunner,
        tasks: Sequence[Task],
        harness: HarnessSpec,
        batch_id: str = "tank-teacher-demonstrations",
        fresh_seeds: Sequence[int] = DEMONSTRATION_SEEDS,
    ) -> None:
        for task in tasks:
            if runner.suite.split_of(task.task_id) != "train":
                raise ContractError(f"demonstrations are played on train tasks only, not {task.task_id}")
        suite_seeds = {runner.suite.scenario(str(record["task_id"])).seed for record in runner.suite.records}
        shared = sorted(seed for seed in suite_seeds if seed in fresh_seeds)
        if shared:
            raise ContractError(f"fresh demonstration seeds must be in no split; the suite already plays {shared}")
        runner.check_spec(harness)
        self._runner = runner
        self._task_ids = tuple(task.task_id for task in tasks)
        self._scenarios = tuple(runner.suite.scenario(task.task_id) for task in tasks)
        self._harness = harness
        self._batch_id = batch_id
        self._fresh_seeds = fresh_seeds
        self._batches = 0

    def _rounds(self, index: int) -> tuple[tuple[str, TankScenario], ...] | None:
        """Batch ``index``'s rounds, each with a label for its run directory; ``None`` when the fresh seeds are spent.

        Batch 0 is the tasks themselves.
        """
        if index == 0:
            return tuple(zip(self._task_ids, self._scenarios, strict=True))
        seeds = list(dict.fromkeys(scenario.seed for scenario in self._scenarios))
        start = (index - 1) * len(seeds)
        if start + len(seeds) > len(self._fresh_seeds):
            return None
        fresh = {seed: self._fresh_seeds[start + offset] for offset, seed in enumerate(seeds)}
        scenarios = [replace(scenario, seed=fresh[scenario.seed]) for scenario in self._scenarios]
        return tuple((f"tank-fresh-s{scenario.seed}-{scenario.seat}", scenario) for scenario in scenarios)

    async def _play(self, label: str, scenario: TankScenario) -> list[DecisionSample]:
        name = f"{label}--teacher--{self._harness.spec_id}--{uuid.uuid4().hex[:8]}"
        directory = self._runner.runs_dir / name
        self._runner.write_harness(self._harness, self._runner.runs_dir / f"{name}.harness")
        arguments = [
            *self._runner.arguments(scenario, directory, self._runner.runs_dir / f"{name}.harness"),
            "--teacher",
        ]
        # A teacher round is played directly, not through run_graded, so it is charged here.
        charge_episode()
        exit_code = await self._runner.play(arguments, name)
        if exit_code != 0:
            raise ContractError(f"the teacher round for {label} exited {exit_code}; see {directory}.log")
        if json.loads((directory / "manifest.json").read_text()).get("kind") != "teacher-demonstration":
            raise ContractError(f"{directory} is not a teacher demonstration")
        return decisions_from_log(
            directory,
            reward=None,
            group=f"tank:{scenario.seed}:{scenario.seat}",
            release_id=None,
            harness_id=self._harness.spec_id,
            behavior=False,
        )

    async def _all(self, rounds: Sequence[tuple[str, TankScenario]]) -> list[DecisionSample]:
        played = await asyncio.gather(*(self._play(label, scenario) for label, scenario in rounds))
        return [decision for decisions in played for decision in decisions]

    def next_batch(self, contract: DataContract) -> TrainingBatch | None:
        if contract.kind != "typed_decisions":
            return None
        rounds = self._rounds(self._batches)
        if rounds is None:
            return None
        decisions = asyncio.run(self._all(rounds))
        if not decisions:
            return None
        self._batches += 1
        return TrainingBatch(f"{self._batch_id}-{self._batches}", contract, decisions=tuple(decisions))
