"""Tank run directories read back as decision traces.

A finished round's ``manifest.json`` records the SHA-256 of
``decisions.jsonl``, ``events.jsonl`` and ``result.json`` under ``outputs``,
and a run whose files do not match is refused before anything is read from
them. Each row of ``decisions.jsonl`` is one decision. The events that follow
it are the round's events up to the next decision, except the ``plan`` events
(the row's ``applied`` already holds the agent's) and the opponent's controller
interventions.

A round that logs raw states also writes ``states.jsonl``, one line per agent
decision after a header. Each of that decision's traces then has a
``state_ref`` of ``states.jsonl:<line>``, the 1-based line of its raw state,
which :func:`honing.backends.tank.raw_states.state_at` reads back. A round
recorded before raw states were logged has none.

The key events and their labels: the agent's tank destroyed by the opponent
("agent destroyed") or by its own shell ("agent self-hit"), the opponent's tank
destroyed the same two ways ("opponent destroyed", "opponent self-hit"), and
every missed shot of the agent ("missed shot"). The agent's own death has
priority 0, so a rendering always shows it. The opponent's death and missed
shots have priority 1 and are shown as room allows: winning is not a failure
to explain, and an opponent's self-hit is not the agent's doing.

A shot missed when its shell hit no tank. The round's result settles which
shots those are, since one hit ends the round. When no shell of the agent hit
a tank, every shot the agent fired missed, including those still in flight
when the round ended. When one did, the engine names the shell on the shot and
on the kill, and every other shot missed. A run whose kill by the agent's
shell names no shell was recorded before the engine named shells. It loads
with no shot marked missed, and its outcome says so.

This module imports only the standard library, :mod:`honing.core` and the
decision log's parser (:mod:`honing.backends.tank.decisions`), and importing it
loads none of the runner, the grader, the rollouts or the demonstrations, so
System 2 can read traces without the rest of the backend.
"""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from honing.backends.tank.decisions import decision_from_row
from honing.core import ContractError, DecisionTrace, EpisodeRecord, Grade, TraceEvent

OUTPUTS = ("decisions.jsonl", "events.jsonl", "result.json")
STATES_FILE = "states.jsonl"
SEATS = ("blue", "red")
# The agent's own death is shown whatever the length, and the other key events as room allows.
AGENT_DEATH_PRIORITY = 0
OTHER_KEY_PRIORITY = 1
#: What the outcome adds for a round recorded before the engine named shells.
UNNAMED_SHELLS = "No shot is marked missed: the round was recorded before the engine named shells"
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
            Path(outcome.workspace_location),
            grade=record.grade,
            release_id=release_id,
            harness_id=outcome.spec_id,
            policy_checksum=record.policy_checksum,
        )


def verify_outputs(run: Path, names: Sequence[str] = OUTPUTS) -> Mapping[str, object]:
    """The run's manifest, once it hashes each of ``names`` and each still matches. Raises ``ContractError`` otherwise.

    The manifest may hash more files than ``names``: a round that logs raw states also hashes ``states.jsonl``.
    """
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or not set(names) <= set(outputs):
        raise ContractError(
            f"{run}/manifest.json records no SHA-256 of {list(names)}. The round did not finish,"
            " or an older match wrote it"
        )
    for name in names:
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
    policy_checksum: str | None = None,
) -> tuple[DecisionTrace, ...]:
    """Every decision of one finished round, with the events that followed each, its outcome and ``grade``."""
    manifest = verify_outputs(run)
    seat = str(manifest["seat"])
    me = SEATS.index(seat)
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    rows = _lines(run / "decisions.jsonl")
    events = _lines(run / "events.jsonl")
    hits = _hits(events, me)
    times = [float(row["time"]) for row in rows]
    followed: list[list[TraceEvent]] = [[] for _ in rows]
    for event in events:
        traced = _event(event, me, hits)
        if traced is not None and rows:
            index = bisect_right(times, traced.time_seconds + _TIME_TOLERANCE_SECONDS) - 1
            followed[max(index, 0)].append(traced)
    outcome = _outcome(result, seat) if hits is not None else f"{_outcome(result, seat)}. {UNNAMED_SHELLS}"
    refs = _state_refs(run, manifest)
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
            state_ref=refs.get(row.get("decision")),
            outcome=outcome,
            grade=grade,
            policy_checksum=policy_checksum,
        )
        for step, row in enumerate(rows)
    )


def _state_refs(run: Path, manifest: Mapping[str, object]) -> Mapping[object, str]:
    """``states.jsonl:<line>`` by decision, for a run whose manifest hashes its raw states, or nothing."""
    outputs = manifest["outputs"]
    if not isinstance(outputs, dict) or STATES_FILE not in outputs:
        return {}
    verify_outputs(run, (STATES_FILE,))
    lines = (run / STATES_FILE).read_text(encoding="utf-8").splitlines()
    # The first line is the header. Every decision row names its decision, and so does its state's line.
    return {json.loads(text)["decision"]: f"{STATES_FILE}:{number}" for number, text in enumerate(lines[1:], 2)}


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _hits(events: list[dict[str, Any]], me: int) -> frozenset[str] | None:
    """The agent's shells that hit a tank, named as the engine names them on the shot and the kill.

    ``None`` when a kill by the agent's shell names no shell: the round was
    recorded before the engine named shells, so its missed shots cannot be
    told from the hit.
    """
    kills = [event for event in events if event["type"] == "kill" and event["player"] == me]
    if any("shell" not in kill for kill in kills):
        return None
    return frozenset(str(kill["shell"]) for kill in kills)


def _event(event: Mapping[str, Any], me: int, hits: frozenset[str] | None) -> TraceEvent | None:
    """One game event as a trace event, or ``None`` for one the decisions already hold or the opponent's own."""
    kind = str(event["type"])
    player = event.get("player")
    if kind == "plan" or (kind not in _ALWAYS_KEPT and player != me):
        return None
    time = float(event["time"])
    if kind == "shot":
        text = f"{SEATS[event['player']]} fired"
        if player == me and hits is not None and event.get("shell") not in hits:
            return TraceEvent(time, text, "missed shot", OTHER_KEY_PRIORITY)
        return TraceEvent(time, text)
    if kind == "kill":
        victim = SEATS[event["victim"]]
        mine = event["victim"] == me
        whose, priority = ("agent", AGENT_DEATH_PRIORITY) if mine else ("opponent", OTHER_KEY_PRIORITY)
        if event.get("self"):
            return TraceEvent(time, f"{victim} was destroyed by its own shell", f"{whose} self-hit", priority)
        return TraceEvent(time, f"{SEATS[event['player']]} destroyed {victim}", f"{whose} destroyed", priority)
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


__all__ = [
    "AGENT_DEATH_PRIORITY",
    "OTHER_KEY_PRIORITY",
    "OUTPUTS",
    "UNNAMED_SHELLS",
    "TankTraceSource",
    "load_run",
    "verify_outputs",
]
