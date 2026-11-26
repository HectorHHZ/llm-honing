"""Schedulers that decide a campaign's next move, and resuming one from the commit record."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from honing.campaign import rules
from honing.core import SCHEDULER_STATE_VERSION_KEY, CampaignState, CoAdaptScheduler, ContractError, Move
from honing.release import read_scheduler_state


class ScriptedScheduler:
    """Replays a fixed list of moves, then stops.

    For tests and reproducible experiments. A policy-driven scheduler that
    reads the state, such as one that prunes when utilization is stable,
    replaces it in a real campaign. Its state is its place in the script and
    a digest of the script's moves, their kinds and labels in order, so a
    restored scheduler refuses another script, even one of the same length.
    """

    name = "scripted"
    state_schema_version = 2

    def __init__(self, moves: Sequence[Move]) -> None:
        self._moves = tuple(moves)
        self._next = 0
        self._script = _script_digest(self._moves)

    def next_move(self, state: CampaignState) -> Move:
        if self._next >= len(self._moves):
            return Move(kind="stop")
        move = self._moves[self._next]
        self._next += 1
        return move

    def export_state(self) -> Mapping[str, object]:
        return {
            SCHEDULER_STATE_VERSION_KEY: self.state_schema_version,
            "next": self._next,
            "moves": len(self._moves),
            "script": self._script,
        }

    def restore_state(self, state: Mapping[str, object]) -> None:
        moves, position = state.get("moves"), state.get("next")
        if moves != len(self._moves):
            raise ContractError(f"the state is for a script of {moves} moves, but this one has {len(self._moves)}")
        if state.get("script") != self._script:
            raise ContractError(
                "the state is for another script of the same length: its moves' kinds or labels differ from this one's"
            )
        if isinstance(position, bool) or not isinstance(position, int) or not 0 <= position <= len(self._moves):
            raise ContractError(f"the state's place in the script must be 0 to {len(self._moves)}, got {position!r}")
        self._next = position


def _script_digest(moves: Sequence[Move]) -> str:
    """A digest of each move's kind and label, in order. The label is a harness move's, or ``None``."""
    script = [[move.kind, move.harness.label if move.harness is not None else None] for move in moves]
    return hashlib.sha256(json.dumps(script).encode()).hexdigest()[:16]


def restore_scheduler(scheduler: CoAdaptScheduler, record_path: Path) -> bool:
    """Restore ``scheduler`` from the last state a commit record holds, and say whether it held one.

    State written by another scheduler, or under another schema version, is
    refused before the scheduler reads it.
    """
    recorded = read_scheduler_state(record_path)
    if recorded is None:
        return False
    written_by, state = recorded
    rules.scheduler_state_schema_matches(state, written_by, scheduler.name, scheduler.state_schema_version)
    scheduler.restore_state(state)
    return True
