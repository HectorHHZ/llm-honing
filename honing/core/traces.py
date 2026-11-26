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

from honing.core.data import DecisionSample
from honing.core.errors import require
from honing.core.harness import Grade


@dataclass(frozen=True)
class TraceEvent:
    """Something that happened in an episode, in the words of the source that read it.

    A key event is one a reader looks around first, such as the agent's death
    or a missed shot. It has a ``label`` the source chooses, which names the
    kind of event, and a ``priority``: under a length limit, events with a
    lower number are kept longer, and 0 marks the events a reader must see. An
    ordinary event has neither.
    """

    time_seconds: float
    text: str
    label: str | None = None
    priority: int | None = None

    def __post_init__(self) -> None:
        require(
            math.isfinite(self.time_seconds) and self.time_seconds >= 0.0,
            f"a trace event's time must be finite and >= 0, got {self.time_seconds}",
        )
        require(bool(self.text), "a trace event needs a description")
        require(
            (self.label is None) == (self.priority is None),
            "a key event has both a label and a priority, and an ordinary event has neither",
        )
        if self.label is not None:
            require(bool(self.label.strip()), "a key event's label must not be blank")
        if self.priority is not None:
            require(self.priority >= 0, f"a key event's priority must be >= 0, got {self.priority}")

    @property
    def key(self) -> bool:
        """Whether this is a key event."""
        return self.label is not None


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
    the episode's, the same on each of its decisions, and so is
    ``policy_checksum``, the checksum of the weights that made the decisions,
    which names the policy when the release is not known.
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
    policy_checksum: str | None = None

    def __post_init__(self) -> None:
        require(bool(self.episode_id), "DecisionTrace.episode_id must be set")
        require(self.policy_checksum is None or bool(self.policy_checksum), "a policy checksum must not be empty")
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
