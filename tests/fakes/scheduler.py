"""A scheduler that reads the campaign's history and keeps what it learned in its own state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, get_args

from honing.core import CampaignState, Move, MoveKind


class HistoryScheduler:
    """Replays a script, and remembers every step it has read as rejected.

    ``seen`` holds each state the campaign handed it, for a test to read. Its
    own state is its place in the script and the rejected steps, so a restored
    scheduler continues from both even when its new campaign has no history.
    """

    name = "history"
    state_schema_version = 1

    def __init__(self, moves: Sequence[Move]) -> None:
        self._moves = tuple(moves)
        self._next = 0
        self._rejected: list[int] = []
        self.seen: list[CampaignState] = []

    @property
    def rejected(self) -> tuple[int, ...]:
        return tuple(self._rejected)

    def next_move(self, state: CampaignState) -> Move:
        self.seen.append(state)
        for record in state.history:
            if record.result == "rejected" and record.step not in self._rejected:
                self._rejected.append(record.step)
        if self._next >= len(self._moves):
            return Move(kind="stop")
        move = self._moves[self._next]
        self._next += 1
        return move

    def export_state(self) -> Mapping[str, object]:
        return {"schema_version": self.state_schema_version, "next": self._next, "rejected": list(self._rejected)}

    def restore_state(self, state: Mapping[str, object]) -> None:
        position, rejected = state["next"], state["rejected"]
        assert isinstance(position, int) and isinstance(rejected, list)
        self._next, self._rejected = position, [int(step) for step in rejected]


def history_scheduler(context: Any, kinds: Sequence[str]) -> HistoryScheduler:
    """A pipeline factory: the moves are named by kind, each with the campaign's defaults."""
    for kind in kinds:
        if kind not in get_args(MoveKind):
            raise ValueError(f"unknown move kind {kind!r}")
    return HistoryScheduler([Move(kind=kind) for kind in kinds])  # type: ignore[arg-type]
