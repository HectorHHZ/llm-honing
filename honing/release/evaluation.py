"""Evaluate a candidate by running the same episodes under it and under the head."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from honing.core import (
    ContractError,
    EpisodeRecord,
    EpisodeRunner,
    EvaluationResult,
    HarnessSpec,
    HoneRelease,
    PolicyRef,
    Task,
    UpdateCandidate,
    require,
)


def _harness_of(spec: HarnessSpec | None, side: str) -> HarnessSpec:
    if spec is None:
        raise ContractError(f"episode evaluation needs a harness on the {side}; seed one when the campaign starts")
    return spec


class EpisodeEvaluator:
    """Scores a candidate and the head on the same tasks, each under its own harness and weights.

    Both sides run concurrently. ``evaluate`` starts its own event loop, so it
    must be called from synchronous code, not from inside a running loop.
    """

    def __init__(self, runner: EpisodeRunner, tasks: Sequence[Task]) -> None:
        require(len(tasks) > 0, "episode evaluation needs at least one task")
        self._runner = runner
        self._tasks = tuple(tasks)

    def evaluate(self, candidate: UpdateCandidate, incumbent: HoneRelease) -> EvaluationResult:
        candidate_records, incumbent_records = asyncio.run(
            self._run_both(
                _harness_of(candidate.harness, "candidate"),
                candidate.policy(),
                _harness_of(incumbent.harness, "incumbent"),
                incumbent.policy(),
            )
        )
        task_ids = tuple(task.task_id for task in self._tasks)
        for side, records in (("candidate", candidate_records), ("incumbent", incumbent_records)):
            if tuple(record.task_id for record in records) != task_ids:
                raise ContractError(f"the runner returned {side} records out of task order")
        return EvaluationResult(
            task_ids=task_ids,
            candidate_scores=tuple(record.score for record in candidate_records),
            incumbent_scores=tuple(record.score for record in incumbent_records),
            evaluated_variant=candidate.evaluated_variant,
            records=(*candidate_records, *incumbent_records),
        )

    async def _run_both(
        self,
        candidate_spec: HarnessSpec,
        candidate_policy: PolicyRef,
        incumbent_spec: HarnessSpec,
        incumbent_policy: PolicyRef,
    ) -> tuple[list[EpisodeRecord], list[EpisodeRecord]]:
        candidate_runs = asyncio.gather(
            *(self._runner.run(candidate_spec, task, candidate_policy) for task in self._tasks)
        )
        incumbent_runs = asyncio.gather(
            *(self._runner.run(incumbent_spec, task, incumbent_policy) for task in self._tasks)
        )
        candidate_records, incumbent_records = await asyncio.gather(candidate_runs, incumbent_runs)
        return list(candidate_records), list(incumbent_records)
