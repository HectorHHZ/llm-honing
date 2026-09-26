"""Judge one RA2 match: check it was played honestly, then score its result."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from honing.backends.ra2.scenarios import RA2Suite
from honing.core import ContractError, EpisodeOutcome, Grade, Task

WIN, DRAW, LOSS = 1.0, 0.5, 0.0
CATEGORIES = ("building", "infantry", "vehicle", "aircraft")


def _by_category(value: object) -> str:
    if isinstance(value, Mapping):
        return "/".join(str(value.get(c, 0)) for c in CATEGORIES)
    return str(value)


def feedback_text(result: Mapping[str, object], seat: str, score: float) -> str:
    """A factual summary of the match for a reflective method to read. Never used in the score."""
    verdict = {WIN: "won", DRAW: "drew (no winner)", LOSS: "lost"}[score]
    how = "truncated at the tick limit" if result.get("truncated") else "ended by the engine"
    lines = [f"The {seat} seat {verdict} at tick {result.get('tick')}, {how}."]
    players = result.get("finalPlayers")
    own = next(
        (
            p
            for p in (players if isinstance(players, list) else [])
            if isinstance(p, Mapping) and p.get("name") == seat
        ),
        None,
    )
    if own is not None:
        lines.append(
            f"Credits gained {own.get('credits_gained')}, final credits {own.get('credits')}. "
            f"Built (building/infantry/vehicle/aircraft) {_by_category(own.get('units_built'))}; "
            f"killed {_by_category(own.get('units_killed'))}; lost {_by_category(own.get('units_lost'))}."
        )
    stats = result.get("stats")
    if isinstance(stats, Mapping) and isinstance(stats.get(seat), Mapping):
        mine = stats[seat]
        lines.append(
            f"Commands submitted {mine.get('submitted')}, accepted {mine.get('accepted')}, "
            f"rejected {mine.get('rejected')}, filtered {mine.get('filtered')}."
        )
    return " ".join(lines)


class RA2Grader:
    """Satisfies :class:`~honing.core.TaskGrader` for RA2 tasks.

    A match that failed, or that fails its audit or replay check, is an invalid
    episode, so grading raises instead of scoring it. A valid match scores 1 for
    a win, 0.5 for a draw or a tick-limit truncation, and 0 for a loss.

    ``audit_command`` defaults to ``node integration/laya/audit.mjs``, and
    ``replay_command``, when given, re-plays the replay without any policy and
    must print a report with ``"verified": true``. The run directory is
    appended to the audit command; the replay command gets ``--match <dir>``.
    """

    def __init__(
        self,
        *,
        suite: RA2Suite,
        integration_dir: Path,
        audit_command: Sequence[str] | None = None,
        replay_command: Sequence[str] | None = None,
    ) -> None:
        self._suite = suite
        self._audit = list(audit_command or ["node", str(integration_dir / "integration/laya/audit.mjs")])
        self._replay = list(replay_command) if replay_command else None
        self._cwd = integration_dir

    async def _check(self, arguments: Sequence[str], what: str) -> str:
        process = await asyncio.create_subprocess_exec(
            *arguments, cwd=self._cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
        )
        output, _ = await process.communicate()
        text = output.decode("utf-8", "replace")
        if process.returncode != 0:
            raise ContractError(f"the {what} failed ({process.returncode}): {text[-2000:]}")
        return text

    async def grade(self, task: Task, outcome: EpisodeOutcome) -> Grade:
        scenario = self._suite.scenario(task.task_id)
        if outcome.workspace_location is None:
            raise ContractError(f"the RA2 outcome for {task.task_id} names no run directory")
        run = Path(outcome.workspace_location)
        if outcome.exit_code != 0:
            raise ContractError(f"the match for {task.task_id} exited {outcome.exit_code}; see {run}.log")
        await self._check([*self._audit, str(run)], "audit")
        if self._replay is not None:
            report = await self._check([*self._replay, "--match", str(run)], "replay check")
            if '"verified": true' not in report:
                raise ContractError(f"the replay of {task.task_id} did not verify: {report[-2000:]}")
        result = json.loads((run / "result.json").read_text())
        outcome_by_seat = result.get("outcome") or {}
        mine = outcome_by_seat.get(scenario.seat, 0)
        score = WIN if mine == 1 and not result.get("truncated") else LOSS if mine == -1 else DRAW
        return Grade(score=score, feedback=feedback_text(result, scenario.seat, score))
