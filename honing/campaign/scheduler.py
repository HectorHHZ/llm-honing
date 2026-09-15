"""Schedulers that decide a campaign's next move."""

from __future__ import annotations

from collections.abc import Sequence

from honing.core import CampaignState, Move


class ScriptedScheduler:
    """Replays a fixed list of moves, then stops.

    For tests and reproducible experiments. A policy-driven scheduler that
    reads the state, such as one that prunes when utilization is stable,
    replaces it in a real campaign.
    """

    def __init__(self, moves: Sequence[Move]) -> None:
        self._moves = tuple(moves)
        self._next = 0

    def next_move(self, state: CampaignState) -> Move:
        if self._next >= len(self._moves):
            return Move(kind="stop")
        move = self._moves[self._next]
        self._next += 1
        return move
