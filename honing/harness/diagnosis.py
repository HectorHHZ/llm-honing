"""System 2's diagnosis of one episode: the part of the decision system it blames for the loss.

The diagnoser (#172) reads a whole episode and gives one cause for it:
``harness``, ``weights`` or ``controller``, or ``uncertain`` when it abstains.
It may also point at the decisions where the failure shows, and name the
option System 2 would have chosen there. :class:`Diagnosis` is that record,
with the episode it is about and the release and harness the episode ran
under. Code that acts on a diagnosis, such as the corrected-decision samples
in :mod:`honing.backends.tank.corrected`, reads it. It imports only
:mod:`honing.core`, so the diagnoser can write it without a game backend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, get_args

from honing.core import require, require_choice

Cause = Literal["harness", "weights", "controller", "uncertain"]


@dataclass(frozen=True)
class DiagnosedDecision:
    """A decision a diagnosis points at, and the option System 2 would have chosen, when it names one.

    ``step`` counts the episode's decisions from 0, as a
    :class:`~honing.core.DecisionTrace`'s does, and ``question`` is the key of
    one question asked there.
    """

    step: int
    question: str
    option: str | None = None

    def __post_init__(self) -> None:
        require(self.step >= 0, f"a diagnosed decision's step must be >= 0, got {self.step}")
        require(bool(self.question), "a diagnosed decision names its question")
        require(self.option is None or bool(self.option), "a diagnosed decision's option must not be empty")


@dataclass(frozen=True)
class Diagnosis:
    """One episode's diagnosis: its cause, how sure the diagnoser is, and why.

    ``release_id`` and ``harness_id`` name the release and the harness the
    episode ran under, when the caller knows them. A weights diagnosis is
    about those weights only, so the samples built from it carry both
    (design principle P4).
    """

    episode_id: str
    cause: Cause
    confidence: float
    explanation: str
    release_id: str | None = None
    harness_id: str | None = None
    decisions: tuple[DiagnosedDecision, ...] = ()

    def __post_init__(self) -> None:
        require(bool(self.episode_id), "Diagnosis.episode_id must be set")
        require_choice(self.cause, get_args(Cause), "Diagnosis.cause")
        require(
            math.isfinite(self.confidence) and 0.0 <= self.confidence <= 1.0,
            f"a diagnosis's confidence is between 0 and 1, got {self.confidence}",
        )
        require(self.release_id is None or bool(self.release_id), "a diagnosis's release id must not be empty")
        require(self.harness_id is None or bool(self.harness_id), "a diagnosis's harness id must not be empty")


__all__ = ["Cause", "DiagnosedDecision", "Diagnosis"]
