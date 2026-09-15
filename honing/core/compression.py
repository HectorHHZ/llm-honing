"""What a compression step does: what it removes or quantizes, why, and what follows it.

Pruning is three independent stages, each with its own registry, so methods
combine instead of multiplying:

- an ``ImportanceScorer`` reads a signal (telemetry, calibration data, weights,
  gradients) and returns an :class:`ImportanceMap`;
- a ``PruneSelector`` turns that map and a :class:`CompressionBudget` into a
  :class:`PrunePlan`;
- a ``ModelEditor`` applies the plan to the master weights.

Recovery is not part of this module. It is training, so a step names a
:class:`~honing.core.training.RecoverySpec` and the Train plane runs it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, get_args

from honing.core.errors import require, require_choice
from honing.core.numerics import QuantScheme
from honing.core.structure import ArchitectureSpec, ParamMask, StructureAddress
from honing.core.training import RecoverySpec
from honing.core.weights import CheckpointRef

SignalKind = Literal["telemetry", "calibration", "weights", "gradients"]
SettleMode = Literal["after_recovery", "per_stage"]
TransformKind = Literal["prune", "quantize"]


@dataclass(frozen=True)
class CostDelta:
    """The estimated change a plan makes. Negative values are savings."""

    parameters: int
    memory_bytes: int = 0


@dataclass(frozen=True)
class CompressionBudget:
    """How much a step may remove, and the floor it must never cross."""

    keep_ratio: float
    min_experts_per_layer: int = 1
    min_layers: int = 1

    def __post_init__(self) -> None:
        require(0.0 < self.keep_ratio <= 1.0, f"keep_ratio must be in (0, 1], got {self.keep_ratio}")
        require(self.min_experts_per_layer >= 1, "min_experts_per_layer must be >= 1")
        require(self.min_layers >= 1, "min_layers must be >= 1")


@dataclass(frozen=True)
class ImportanceMap:
    """A score per structure, and the signal the scores were computed from."""

    signal: SignalKind
    scores: tuple[tuple[StructureAddress, float], ...]

    def __post_init__(self) -> None:
        require_choice(self.signal, get_args(SignalKind), "ImportanceMap.signal")
        addresses = [address for address, _ in self.scores]
        require(len(set(addresses)) == len(addresses), "an importance map scores each structure once")
        require(all(math.isfinite(score) for _, score in self.scores), "importance scores must be finite")

    def as_dict(self) -> dict[StructureAddress, float]:
        return dict(self.scores)


@dataclass(frozen=True)
class PrunePlan:
    """A declarative pruning step: what to remove, the shape that remains, and the scores behind it.

    The plan is stored on the release it produces, so every cut a model has
    taken can be read back from its release chain.
    """

    removals: tuple[StructureAddress, ...]
    before: ArchitectureSpec
    after: ArchitectureSpec
    importance: ImportanceMap
    estimated: CostDelta

    def __post_init__(self) -> None:
        require(len(self.removals) > 0, "a prune plan must remove something")
        require(len(set(self.removals)) == len(self.removals), "a prune plan names each removal once")
        masks_only = all(isinstance(address, ParamMask) for address in self.removals)
        if masks_only:
            require(self.after == self.before, "a mask-only plan keeps the architecture unchanged")
        else:
            require(self.after != self.before, "a structural plan must change the architecture")

    @property
    def changes_shape(self) -> bool:
        return self.after != self.before


@dataclass(frozen=True)
class QuantizePlan:
    """Build serving weights from ``source`` under ``scheme``. The master is left as it is."""

    scheme: QuantScheme
    source: CheckpointRef


@dataclass(frozen=True)
class TransformManifest:
    """The record of one transform a release applied."""

    kind: TransformKind
    plan: PrunePlan | QuantizePlan

    def __post_init__(self) -> None:
        require_choice(self.kind, get_args(TransformKind), "TransformManifest.kind")
        expected = "prune" if isinstance(self.plan, PrunePlan) else "quantize"
        require(self.kind == expected, f"a {type(self.plan).__name__} is recorded as kind {expected!r}")

    @classmethod
    def of(cls, plan: PrunePlan | QuantizePlan) -> TransformManifest:
        return cls(kind="prune" if isinstance(plan, PrunePlan) else "quantize", plan=plan)


@dataclass(frozen=True)
class CompressionStep:
    """One transform, and whether the gate judges it before or after recovery.

    ``after_recovery`` is the default because a freshly pruned model almost
    always regresses; judging it before recovery would reject good cuts.
    ``per_stage`` publishes the raw cut and leaves recovery to a later train
    move, for measuring what a cut costs on its own.
    """

    transform: PrunePlan | QuantizePlan
    recovery: RecoverySpec | None = None
    settle: SettleMode = "after_recovery"

    def __post_init__(self) -> None:
        require_choice(self.settle, get_args(SettleMode), "CompressionStep.settle")
        require(
            self.recovery is None or isinstance(self.transform, PrunePlan),
            "recovery follows a prune; quantization-aware training is a train move with its own numerics",
        )
