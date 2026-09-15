"""The moves a campaign makes, and the state a scheduler chooses the next one from."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

from honing.core.compression import CompressionBudget
from honing.core.errors import require, require_choice
from honing.core.numerics import QuantScheme
from honing.core.release import HoneRelease
from honing.core.training import RecoverySpec, TrainJobSpec

MoveKind = Literal["train", "prune", "quantize", "harness", "stop"]


@dataclass(frozen=True)
class Move:
    """One step a scheduler asks for.

    The optional fields override the campaign's defaults for this move only, and
    each belongs to exactly one kind of move.
    """

    kind: MoveKind
    train: TrainJobSpec | None = None
    budget: CompressionBudget | None = None
    recovery: RecoverySpec | None = None
    scheme: QuantScheme | None = None

    def __post_init__(self) -> None:
        require_choice(self.kind, get_args(MoveKind), "Move.kind")
        require(self.train is None or self.kind == "train", "a train job belongs to a train move")
        require(
            (self.budget is None and self.recovery is None) or self.kind == "prune",
            "a budget and a recovery belong to a prune move",
        )
        require(self.scheme is None or self.kind == "quantize", "a quantization scheme belongs to a quantize move")


@dataclass(frozen=True)
class CampaignState:
    """What a scheduler sees: the current head, every published release, and the step count."""

    head: HoneRelease
    releases: tuple[HoneRelease, ...]
    step: int
