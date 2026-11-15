"""Decision traces: every decision a model made in an episode, read back with what followed it.

What explains a decision model's failures is in its decisions: the state it
read, the options with the probability it gave each, its choice, what the
controller changed, and what happened next. A :class:`DecisionTrace` holds one
decision with all of that and the episode's outcome and grade, so a reflective
method and a scheduler read the same shape whichever source recorded the
episode. A :class:`~honing.core.protocols.TraceSource` reads them back.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, get_args

from honing.core.data import DecisionSample
from honing.core.errors import require, require_choice
from honing.core.harness import Grade

KeyEvent = Literal["death", "self_hit", "missed_shot"]


@dataclass(frozen=True)
class TraceEvent:
    """Something that happened in an episode, in the words of the source that read it.

    ``key`` marks an event a reader looks around first: a death, a self-hit
    (a death the agent caused itself), or a missed shot. It is ``None`` for an
    ordinary event.
    """

    time_seconds: float
    text: str
    key: KeyEvent | None = None

    def __post_init__(self) -> None:
        require(
            math.isfinite(self.time_seconds) and self.time_seconds >= 0.0,
            f"a trace event's time must be finite and >= 0, got {self.time_seconds}",
        )
        require(bool(self.text), "a trace event needs a description")
        if self.key is not None:
            require_choice(self.key, get_args(KeyEvent), "TraceEvent.key")


@dataclass(frozen=True)
class DecisionTrace:
    """One decision of an episode as the model saw and made it, and what followed.

    ``decision`` holds the state as the model read it (the rendered parts), the
    questions with their options, the probability the model gave every option
    (``behavior_probs``, always present here), and the policy's choice. Its
    ``source_release`` and ``source_harness`` name the release and the harness
    the episode ran under, when the source knows them.

    ``step`` counts the episode's decisions from 0, and ``time_seconds`` is the
    episode clock when the decision was asked. ``state_ref`` points at the raw
    state behind the rendered parts, for a source that records one.
    ``constraints`` and ``applied`` are, as JSON text, what the controller
    allowed and what it executed. ``events`` are what happened from this
    decision until the next one, in time order. ``outcome`` and ``grade`` are
    the episode's, the same on each of its decisions.
    """

    episode_id: str
    step: int
    time_seconds: float
    decision: DecisionSample
    events: tuple[TraceEvent, ...] = ()
    constraints: str | None = None
    applied: str | None = None
    state_ref: str | None = None
    outcome: str = ""
    grade: Grade | None = None

    def __post_init__(self) -> None:
        require(bool(self.episode_id), "DecisionTrace.episode_id must be set")
        require(self.step >= 0, f"a decision's step must be >= 0, got {self.step}")
        require(
            math.isfinite(self.time_seconds) and self.time_seconds >= 0.0,
            f"a decision's time must be finite and >= 0, got {self.time_seconds}",
        )
        require(self.decision.behavior_probs is not None, "a decision trace keeps the probability of every option")
        times = [event.time_seconds for event in self.events]
        require(times == sorted(times), "a decision's events are in time order")

    @property
    def release_id(self) -> str | None:
        return self.decision.source_release

    @property
    def harness_id(self) -> str | None:
        return self.decision.source_harness
