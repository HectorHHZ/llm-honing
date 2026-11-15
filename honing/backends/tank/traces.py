"""Tank run directories read back as decision traces.

A finished round's ``manifest.json`` records the SHA-256 of
``decisions.jsonl``, ``events.jsonl`` and ``result.json`` under ``outputs``,
and a run whose files do not match is refused before anything is read from
them. Each row of ``decisions.jsonl`` is one decision. The events that follow
it are the round's events up to the next decision, except the ``plan`` events
(the row's ``applied`` already holds the agent's) and the opponent's controller
interventions.

The key events are every death, a tank destroyed by its own shell being a
self-hit, and every missed shot of the agent. One hit ends the round, so a
shell the agent fired more than :data:`SHELL_LIFETIME_SECONDS` before the end
expired without hitting anything. A later one may be the shell that ended it.

This module imports only the standard library and :mod:`honing.core`.
"""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from honing.core import (
    ContractError,
    DecisionQuestion,
    DecisionSample,
    DecisionTrace,
    EpisodeRecord,
    Grade,
    TraceEvent,
)

OUTPUTS = ("decisions.jsonl", "events.jsonl", "result.json")
SEATS = ("blue", "red")
# SHELL_LIFETIME in integrations/tank-war/web/game/engine.js. tests/test_decision_traces.py keeps the two equal.
SHELL_LIFETIME_SECONDS = 4.5
# Every seat's controller writes these, but only the agent's are its decisions' doing.
_ALWAYS_KEPT = frozenset({"shot", "kill", "round"})
# A decision's time is rounded to the millisecond, so an event within half of one belongs to it.
_TIME_TOLERANCE_SECONDS = 0.0005


class TankTraceSource:
    """Satisfies :class:`~honing.core.TraceSource` for rounds the tank match process played.

    A round that exited with an error, or that the runner reported as a
    failure, left no complete run, and has no traces.
    """

    def traces(self, record: EpisodeRecord, *, release_id: str | None = None) -> tuple[DecisionTrace, ...]:
        outcome = record.outcome
        if outcome.failure is not None or outcome.exit_code != 0 or outcome.workspace_location is None:
            return ()
        return load_run(
            Path(outcome.workspace_location), grade=record.grade, release_id=release_id, harness_id=outcome.spec_id
        )


def verify_outputs(run: Path) -> Mapping[str, object]:
    """The run's manifest, once every file it hashes matches. Raises ``ContractError`` otherwise."""
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or sorted(outputs) != sorted(OUTPUTS):
        raise ContractError(
            f"{run}/manifest.json records no SHA-256 of {list(OUTPUTS)}. The round did not finish,"
            " or an older match wrote it"
        )
    for name in OUTPUTS:
        path = run / name
        if not path.is_file():
            raise ContractError(f"{path} is missing, but its manifest records it")
        if hashlib.sha256(path.read_bytes()).hexdigest() != outputs[name]:
            raise ContractError(f"{path} does not match the SHA-256 its manifest records. The run was changed")
    return manifest


def load_run(
    run: Path,
    *,
    grade: Grade | None = None,
    release_id: str | None = None,
    harness_id: str | None = None,
) -> tuple[DecisionTrace, ...]:
    """Every decision of one finished round, with the events that followed each, its outcome and ``grade``."""
    manifest = verify_outputs(run)
    seat = str(manifest["seat"])
    me = SEATS.index(seat)
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    rows = _lines(run / "decisions.jsonl")
    events = _lines(run / "events.jsonl")
    end = next((float(e["time"]) for e in events if e["type"] == "round"), float(result["sim_seconds"]))
    times = [float(row["time"]) for row in rows]
    followed: list[list[TraceEvent]] = [[] for _ in rows]
    for event in events:
        traced = _event(event, me, end)
        if traced is not None and rows:
            index = bisect_right(times, traced.time_seconds + _TIME_TOLERANCE_SECONDS) - 1
            followed[max(index, 0)].append(traced)
    outcome = _outcome(result, seat)
    return tuple(
        DecisionTrace(
            episode_id=run.name,
            step=step,
            time_seconds=times[step],
            decision=decision_from_row(
                row, reward=None, group=None, release_id=release_id, harness_id=harness_id, behavior=True
            ),
            events=tuple(followed[step]),
            constraints=_json(row.get("constraints")),
            applied=_json(row.get("applied")),
            outcome=outcome,
            grade=grade,
        )
        for step, row in enumerate(rows)
    )


def decision_from_row(
    row: Mapping[str, Any],
    *,
    reward: float | None,
    group: str | None,
    release_id: str | None,
    harness_id: str | None,
    behavior: bool,
) -> DecisionSample:
    """One row of ``decisions.jsonl``. ``behavior`` keeps the answers' probabilities as behavior probabilities."""
    questions = row["questions"]
    return DecisionSample(
        parts=tuple((name, tuple(lines)) for name, lines in row["contextParts"].items()),
        questions=tuple(
            (
                key,
                DecisionQuestion(kind=q["type"], instructions=q["instructions"], options=tuple(q["criteria"].items())),
            )
            for key, q in questions.items()
        ),
        chosen=tuple((key, row["chosen"][key]) for key in questions),
        behavior_probs=(
            tuple(
                (
                    key,
                    tuple((option, float(row["answers"][key]["probabilities"][option])) for option in q["criteria"]),
                )
                for key, q in questions.items()
            )
            if behavior
            else None
        ),
        reward=reward,
        group=group,
        source_release=release_id,
        source_harness=harness_id,
    )


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _event(event: Mapping[str, Any], me: int, end: float) -> TraceEvent | None:
    """One game event as a trace event, or ``None`` for one the decisions already hold or the opponent's own."""
    kind = str(event["type"])
    player = event.get("player")
    if kind == "plan" or (kind not in _ALWAYS_KEPT and player != me):
        return None
    time = float(event["time"])
    if kind == "shot":
        missed = player == me and time + SHELL_LIFETIME_SECONDS < end
        return TraceEvent(time, f"{SEATS[event['player']]} fired", "missed_shot" if missed else None)
    if kind == "kill":
        victim = SEATS[event["victim"]]
        if event.get("self"):
            return TraceEvent(time, f"{victim} was destroyed by its own shell", "self_hit")
        return TraceEvent(time, f"{SEATS[event['player']]} destroyed {victim}", "death")
    if kind == "round":
        winner = event.get("winner")
        won = f"{SEATS[winner]} won" if winner in (0, 1) else "no winner"
        return TraceEvent(time, f"the round ended ({event.get('reason')}), {won}")
    details = {key: _rounded(value) for key, value in event.items() if key not in ("type", "time", "round", "player")}
    return TraceEvent(time, f"{kind} for {SEATS[me]}: {_json(details)}")


def _outcome(result: Mapping[str, Any], seat: str) -> str:
    mine = result["outcome"][seat]
    verdict = {1: "won", 0: "drew", -1: "lost"}[mine]
    return f"the {seat} tank {verdict} ({result['reason']}) after {result['sim_seconds']} s"


def _rounded(value: object) -> object:
    """Floats to the millisecond, so an event's numbers stay short."""
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded(item) for item in value]
    return value


def _json(value: object) -> str | None:
    return None if value is None else json.dumps(value, separators=(",", ":"))


__all__ = ["OUTPUTS", "SHELL_LIFETIME_SECONDS", "TankTraceSource", "decision_from_row", "load_run", "verify_outputs"]
