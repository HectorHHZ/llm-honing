"""The moves a campaign makes, and the state a scheduler chooses the next one from."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, get_args

from honing.core.compression import CompressionBudget
from honing.core.cost import MoveCost
from honing.core.errors import require, require_choice
from honing.core.harness import HarnessMoveConfig
from honing.core.numerics import QuantScheme
from honing.core.release import HoneRelease
from honing.core.training import RecoverySpec, TrainJobSpec

MoveKind = Literal["train", "prune", "quantize", "harness", "stop"]
StepResult = Literal["published", "rejected", "skipped"]


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
    harness: HarnessMoveConfig | None = None

    def __post_init__(self) -> None:
        require_choice(self.kind, get_args(MoveKind), "Move.kind")
        require(self.train is None or self.kind == "train", "a train job belongs to a train move")
        require(
            (self.budget is None and self.recovery is None) or self.kind == "prune",
            "a budget and a recovery belong to a prune move",
        )
        require(self.scheme is None or self.kind == "quantize", "a quantization scheme belongs to a quantize move")
        require(self.harness is None or self.kind == "harness", "a harness config belongs to a harness move")


@dataclass(frozen=True)
class StepRecord:
    """One move the campaign advanced, as a scheduler reads it back: published, rejected, or skipped.

    ``step`` is the move's index in the campaign's history. ``parent_id`` is
    the head the move was built on, and ``release_id`` the release it
    published, set only when it was published. ``reason`` is the gate's reason,
    or why a harness method proposed nothing. ``gate_metrics`` are the
    evaluation's numbers that every gated candidate records (the ``gate_*``
    metrics). A skipped move reached no gate, so it has none.

    ``cost`` is what the move spent, the same :class:`~honing.core.MoveCost`
    its commit record line holds, whatever its result: a skipped move still
    spent its method's episodes. An amount that could not be known is ``None``
    inside it, never 0. A campaign always sets it; ``None`` is left for a
    record built by hand.
    """

    step: int
    move: Move
    result: StepResult
    parent_id: str
    reason: str
    label: str | None = None
    release_id: str | None = None
    gate_metrics: Mapping[str, float] = field(default_factory=dict)
    cost: MoveCost | None = None

    def __post_init__(self) -> None:
        require(self.step >= 0, f"StepRecord.step must be >= 0, got {self.step}")
        require_choice(self.result, get_args(StepResult), "StepRecord.result")
        require(self.move.kind != "stop", "a stop is not a step, since the campaign advances no move for it")
        require(
            (self.release_id is not None) == (self.result == "published"),
            "a step names a release exactly when it was published",
        )
        require(self.result != "skipped" or not self.gate_metrics, "a skipped step reached no gate")
        require(all(math.isfinite(value) for value in self.gate_metrics.values()), "gate metrics must be finite")
        require(
            self.cost is None or isinstance(self.cost, MoveCost),
            f"StepRecord.cost must be a MoveCost or None, got {self.cost!r}",
        )


@dataclass(frozen=True)
class CampaignState:
    """What a scheduler sees when it chooses the next move.

    ``head`` and ``releases`` are the published releases. ``history`` is every
    move the campaign advanced, in order, including the rejected and skipped
    ones, and ``step`` is its length. The scheduler's own state is not here:
    the scheduler holds it.
    """

    head: HoneRelease
    releases: tuple[HoneRelease, ...]
    step: int
    history: tuple[StepRecord, ...] = ()
