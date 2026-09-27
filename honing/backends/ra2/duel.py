"""Judge a candidate by playing it against the release it would replace.

Gating against the Supalosa bot measures progress against one fixed opponent.
:class:`DuelEvaluator` instead plays candidate against incumbent on each
scenario, once in each seat, both served by their own decision servers, with
the same normal view, cadence, and one-command budget. The candidate's score
for a game is its result (1 for a win the engine ended, 0.5 draw or truncation, 0 loss) and the
incumbent's is one minus it, so the gate's paired comparison reads the duel
directly.

Both sides must use the same question text, because a match process loads it
once for both seats, and the default controller settings, which the duel
process uses for both. A duel therefore compares weights under one harness; a
harness change is judged against the bot instead.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

from honing.backends.ra2.runner import CONTROLLER_PATH, QUESTIONS_PATH
from honing.backends.ra2.scenarios import RA2Suite
from honing.core import (
    ContractError,
    EvaluationResult,
    HoneRelease,
    PolicyEndpoints,
    PolicyRef,
    Task,
    UpdateCandidate,
)

SEATS = ("red", "blue")


def result_score(result: Mapping[str, object], seat: str) -> float:
    """1 for a win the engine ended (as #49 counts wins), 0 for a loss, 0.5 otherwise."""
    outcome = result.get("outcome")
    mine = outcome.get(seat, 0) if isinstance(outcome, Mapping) else 0
    if mine == 1 and result.get("ended") is True and not result.get("truncated"):
        return 1.0
    return 0.0 if mine == -1 else 0.5


class DuelEvaluator:
    """Satisfies :class:`~honing.core.Evaluator`: candidate against incumbent, every task, both seats.

    ``command`` starts a duel and defaults to ``node integration/models/duel.mjs``.
    ``verify_command``, when given, runs on each finished duel directory and
    must succeed; a duel that fails, or fails verification, raises.
    """

    def __init__(
        self,
        *,
        suite: RA2Suite,
        endpoints: PolicyEndpoints,
        integration_dir: Path,
        runs_dir: Path,
        tasks: Sequence[Task],
        command: Sequence[str] | None = None,
        verify_command: Sequence[str] | None = None,
        environment: Mapping[str, str] | None = None,
        concurrency: int = 2,
    ) -> None:
        if not tasks:
            raise ContractError("a duel evaluation needs at least one task")
        if concurrency <= 0:
            raise ContractError("concurrency must be positive")
        self._suite = suite
        self._endpoints = endpoints
        self._integration_dir = integration_dir
        self._runs_dir = runs_dir
        self._tasks = tuple(tasks)
        self._command = list(command or ["node", str(integration_dir / "integration/models/duel.mjs")])
        self._verify = list(verify_command) if verify_command else None
        self._environment = dict(environment or {})
        self._concurrency = concurrency

    def _url(self, policy: PolicyRef) -> str:
        endpoint = self._endpoints.resolve(policy)
        if endpoint.runtime != "jev_decision":
            raise ContractError(f"a duel needs Jev decision servers, not a {endpoint.runtime} endpoint")
        return endpoint.base_url

    async def _duel(
        self, task: Task, urls: Mapping[str, str], questions: Path | None, slots: asyncio.Semaphore
    ) -> Mapping[str, object]:
        scenario = self._suite.scenario(task.task_id)
        out = self._runs_dir / f"duel-{task.task_id}-{uuid.uuid4().hex[:8]}"
        arguments = [
            *self._command,
            f"--map={scenario.map}",
            f"--seed={scenario.seed}",
            f"--ticks={scenario.ticks}",
            f"--cadence={scenario.cadence}",
            *(f"--{seat}-decision-server={urls[seat]}" for seat in SEATS),
            f"--out={out}",
            *([f"--questions={questions}"] if questions else []),
        ]
        async with slots:
            with (self._runs_dir / f"{out.name}.log").open("wb") as log:
                process = await asyncio.create_subprocess_exec(
                    *arguments,
                    cwd=self._integration_dir,
                    env={**os.environ, **self._environment},
                    stdout=log,
                    stderr=asyncio.subprocess.STDOUT,
                )
                code = await process.wait()
            if code != 0:
                raise ContractError(f"the duel for {task.task_id} exited {code}; see {out}.log")
            if self._verify is not None:
                check = await asyncio.create_subprocess_exec(
                    *self._verify, str(out), cwd=self._integration_dir, stdout=asyncio.subprocess.DEVNULL
                )
                if await check.wait() != 0:
                    raise ContractError(f"the duel for {task.task_id} failed verification; see {out}")
        result: Mapping[str, object] = json.loads((out / "result.json").read_text())
        return result

    async def _all(self, candidate_url: str, incumbent_url: str, questions: Path | None) -> list[float]:
        slots = asyncio.Semaphore(self._concurrency)
        jobs = []
        for task in self._tasks:
            for seat in SEATS:
                other = "blue" if seat == "red" else "red"
                jobs.append((seat, self._duel(task, {seat: candidate_url, other: incumbent_url}, questions, slots)))
        results = await asyncio.gather(*(job for _, job in jobs))
        return [result_score(r, seat) for (seat, _), r in zip(jobs, results, strict=True)]

    def evaluate(self, candidate: UpdateCandidate, incumbent: HoneRelease) -> EvaluationResult:
        if candidate.harness is None or incumbent.harness is None:
            raise ContractError("a duel plays both sides under a harness")
        mine, theirs = candidate.harness.get(QUESTIONS_PATH), incumbent.harness.get(QUESTIONS_PATH)
        if (mine and mine.content) != (theirs and theirs.content):
            raise ContractError(
                "a duel needs both sides to use the same question text; judge a harness change against the bot"
            )
        # duel.mjs plays both seats with fixed controller settings, the integration's defaults.
        fixed = json.loads((self._integration_dir / "config/controller.json").read_text(encoding="utf-8"))
        for side in (candidate.harness, incumbent.harness):
            controller = side.get(CONTROLLER_PATH)
            if controller is not None and json.loads(controller.content) != fixed:
                raise ContractError(
                    "a duel plays with the default controller settings; judge a controller change against the bot"
                )
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        questions = None
        if mine is not None:
            questions = self._runs_dir / f"questions-{candidate.harness.spec_id}.json"
            questions.write_text(mine.content, encoding="utf-8")
        scores = asyncio.run(self._all(self._url(candidate.policy()), self._url(incumbent.policy()), questions))
        return EvaluationResult(
            task_ids=tuple(f"{task.task_id}@candidate-{seat}" for task in self._tasks for seat in SEATS),
            candidate_scores=tuple(scores),
            incumbent_scores=tuple(1.0 - s for s in scores),
            evaluated_variant=candidate.evaluated_variant,
        )
