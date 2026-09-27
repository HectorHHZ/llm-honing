"""RA2 matches played by the policy being trained, as typed decisions with rewards.

Each model call in a match's decision log (one stage of a staged decision) is
one decision: the state that stage saw, its questions, the options the model
chose, and the probability it gave every option. Every decision in a match
earns that match's score (1 win, 0.5 draw or truncation, 0 loss), is grouped by
task so repeats of one scenario share a baseline, and names the release and
harness it was played under, so a batch cannot mix them.

Policy gradient needs the rollouts to be sampled from the policy. Matches played
with argmax choices repeat exactly and give a group no reward spread, and the
trainer skips such groups.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import uuid
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from honing.backends.ra2.grader import RA2Grader
from honing.backends.ra2.runner import RA2EpisodeRunner
from honing.core import (
    ContractError,
    DataContract,
    DecisionQuestion,
    DecisionSample,
    HarnessSpec,
    PolicyRef,
    Task,
    TrainingBatch,
    run_graded,
)

ROLLOUT_CONTRACT = DataContract("typed_decisions", requires_reward=True, requires_logprobs=True, requires_group=True)


def _rows(run: Path) -> Iterator[dict[str, Any]]:
    plain, packed = run / "decisions.jsonl", run / "decisions.jsonl.gz"
    if plain.exists():
        with plain.open() as handle:
            yield from (json.loads(line) for line in handle if line.strip())
    elif packed.exists():
        with gzip.open(packed, "rt") as handle:
            yield from (json.loads(line) for line in handle if line.strip())
    else:
        raise ContractError(f"{run} has no decision log")


def decisions_from_log(
    run: Path, *, reward: float, group: str, release_id: str | None, harness_id: str | None
) -> list[DecisionSample]:
    """Every model call in one match's decision log, as decisions that all earn the match's reward."""
    decisions = []
    for row in _rows(run):
        stages: list[dict[str, Any]] = (row.get("stages") or []) if row.get("decisionMode") == "staged" else [row]
        for stage in stages:
            questions = stage.get("questions") or {}
            if not questions:
                continue
            answers = stage["answers"]
            decisions.append(
                DecisionSample(
                    parts=tuple((name, tuple(lines)) for name, lines in stage["contextParts"].items()),
                    questions=tuple(
                        (
                            key,
                            DecisionQuestion(
                                kind=q["type"], instructions=q["instructions"], options=tuple(q["criteria"].items())
                            ),
                        )
                        for key, q in questions.items()
                    ),
                    chosen=tuple((key, answers[key]["choice"]) for key in questions),
                    behavior_probs=tuple(
                        (
                            key,
                            tuple((option, float(answers[key]["probabilities"][option])) for option in q["criteria"]),
                        )
                        for key, q in questions.items()
                    ),
                    reward=reward,
                    group=group,
                    source_release=release_id,
                    source_harness=harness_id,
                )
            )
    return decisions


class RA2RolloutProvider:
    """Satisfies :class:`~honing.core.RolloutProvider` for RA2: plays tasks with the policy, returns its decisions.

    Each task is played ``samples_per_task`` times through ``runner`` and graded
    by ``grader``; a match the grader refuses stops collection.
    """

    def __init__(self, runner: RA2EpisodeRunner, grader: RA2Grader) -> None:
        self._runner = runner
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
            raise ContractError("RA2 rollouts play under a harness; the campaign passed none")
        played = asyncio.run(self._play(tasks, policy, harness, samples_per_task))
        decisions = [
            d
            for task, score, run in played
            for d in decisions_from_log(
                run, reward=score, group=task.task_id, release_id=release_id, harness_id=harness.spec_id
            )
        ]
        if not decisions:
            raise ContractError("the rollout matches produced no decisions")
        # Two collections from one release are different batches, so the id names this one.
        batch_id = f"ra2-rollout-{release_id or 'unpublished'}-{len(played)}x-{uuid.uuid4().hex[:8]}"
        return TrainingBatch(batch_id, ROLLOUT_CONTRACT, decisions=tuple(decisions))
