"""Judge one tank round: check the run is the task's, then score its result."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from honing.backends.tank.scenarios import TankSuite
from honing.core import ContractError, EpisodeOutcome, Grade, Task

WIN, DRAW, LOSS = 1.0, 0.5, 0.0
# The bridge refuses, rather than truncates, a state that does not fit the model. When the
# harness's notes or wording cause that, the round scores 0 and says why.
OVER_BUDGET = ("did not preserve complete decision context",)
PLAYED_KINDS = frozenset({"evaluation", "rollout"})


def _per_seat(result: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = result.get(key)
    return value if isinstance(value, Mapping) else {}


def feedback_text(result: Mapping[str, object], seat: str, score: float) -> str:
    """A factual summary of the round for a reflective method to read. Never used in the score."""
    other = "red" if seat == "blue" else "blue"
    verdict = {WIN: "won", DRAW: "drew", LOSS: "lost"}[score]
    how = {
        "hit": "by a hit",
        "draw": "with both tanks destroyed",
        "timeout": "at the time limit with both tanks alive",
    }.get(str(result.get("reason")), str(result.get("reason")))
    shots, decisions = _per_seat(result, "shots"), _per_seat(result, "decisions")
    interventions = _per_seat(result, "interventions")
    self_hits = result.get("self_hits")
    self_hits = self_hits if isinstance(self_hits, list) else []
    lines = [
        f"The {seat} tank {verdict} {how} after {result.get('sim_seconds')} s against the planner opponent.",
        f"Shots fired: {seat} {shots.get(seat, 0)}, opponent {shots.get(other, 0)}."
        f" Decisions: {seat} {decisions.get(seat, 0)}, opponent {decisions.get(other, 0)}.",
        f"Geometry or dodge assist overrode the {seat} tank's request {interventions.get(seat, 0)} times.",
    ]
    if seat in self_hits:
        lines.append(f"The {seat} tank was destroyed by its own shell.")
    if other in self_hits:
        lines.append("The opponent was destroyed by its own shell.")
    return " ".join(lines)


class TankGrader:
    """Satisfies :class:`~honing.core.TaskGrader` for tank tasks.

    A round counts only if its manifest is the task's (seed, seat, opponent,
    clock) and was played by a model, not the teacher. A win scores 1, a loss
    0, and a timeout or a double kill 0.5.
    """

    def __init__(self, *, suite: TankSuite) -> None:
        self._suite = suite

    async def grade(self, task: Task, outcome: EpisodeOutcome) -> Grade:
        scenario = self._suite.scenario(task.task_id)
        if outcome.workspace_location is None:
            raise ContractError(f"the tank outcome for {task.task_id} names no run directory")
        run = Path(outcome.workspace_location)
        if outcome.exit_code != 0:
            error_path = run / "error.json"
            message = str(json.loads(error_path.read_text()).get("error", "")) if error_path.exists() else ""
            if any(marker in message for marker in OVER_BUDGET):
                return Grade(
                    score=LOSS,
                    feedback=(
                        f"The round stopped because a decision did not fit the model: {message}."
                        " Shorten the operating notes or the question wording."
                    ),
                )
            raise ContractError(f"the match for {task.task_id} exited {outcome.exit_code}: {message}; see {run}.log")
        manifest = json.loads((run / "manifest.json").read_text())
        expected = {
            "seed": scenario.seed,
            "seat": scenario.seat,
            "opponent": scenario.opponent,
            "seconds": scenario.seconds,
            "interval_ms": scenario.interval_ms,
            "opponent_interval_ms": scenario.opponent_interval_ms,
        }
        played = {key: manifest.get(key) for key in expected}
        if played != expected or manifest.get("kind") not in PLAYED_KINDS:
            raise ContractError(
                f"the run in {run} played {played} as {manifest.get('kind')!r}, but task {task.task_id} is {expected}"
            )
        result = json.loads((run / "result.json").read_text())
        mine = (result.get("outcome") or {}).get(scenario.seat)
        if mine not in (1, 0, -1):
            raise ContractError(f"{run}/result.json has no outcome for the {scenario.seat} seat")
        score = WIN if mine == 1 else LOSS if mine == -1 else DRAW
        return Grade(score=score, feedback=feedback_text(result, scenario.seat, score))
