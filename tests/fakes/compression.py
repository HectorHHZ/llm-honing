"""Compression fakes: expert-count telemetry, a uniform top-k expert selection, and cost-only edits."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace

from honing.core import (
    ArchitectureSpec,
    CompressionBudget,
    ContractError,
    CostDelta,
    Expert,
    ImportanceMap,
    ModelSnapshot,
    PrunePlan,
    QuantizePlan,
    ServingWeights,
    StructureAddress,
    UtilizationTelemetry,
)
from tests.fakes.model import FakeWorld, parameters_of


def skewed_expert_telemetry(architecture: ArchitectureSpec) -> UtilizationTelemetry:
    """Deterministic, uneven routing counts, different in every layer."""
    counts = tuple(
        (Expert(layer, expert), (expert * 7 + layer * 3) % 11 + 1)
        for layer, shape in enumerate(architecture.layers)
        for expert in range(shape.num_experts)
    )
    return UtilizationTelemetry(sample_count=1_000, expert_activations=counts)


class FixedTelemetryStore:
    def __init__(self, telemetry: UtilizationTelemetry | None) -> None:
        self._telemetry = telemetry

    def observe(self, stats: UtilizationTelemetry) -> None:
        self._telemetry = stats

    def window(self) -> UtilizationTelemetry | None:
        return self._telemetry


class ActivationCountScorer:
    """Scores each expert by how often it was routed to."""

    signal = "telemetry"

    def score(self, model: ModelSnapshot, telemetry: UtilizationTelemetry | None) -> ImportanceMap:
        if telemetry is None:
            raise ContractError("activation-count scoring needs a telemetry window")
        experts = [
            Expert(layer, expert)
            for layer, shape in enumerate(model.architecture.layers)
            for expert in range(shape.num_experts)
        ]
        return ImportanceMap(
            signal="telemetry", scores=tuple((expert, float(telemetry.activation(expert))) for expert in experts)
        )


def _least_used(scores: Mapping[StructureAddress, float], layer: int, count: int, keep: int) -> list[Expert]:
    ranked = sorted(range(count), key=lambda expert: (-scores.get(Expert(layer, expert), 0.0), expert))
    return [Expert(layer, expert) for expert in sorted(ranked[keep:])]


class KeepTopKExpertSelector:
    """Keeps the same number of experts in every layer: the most-used ones in each."""

    def select(self, importance: ImportanceMap, model: ModelSnapshot, budget: CompressionBudget) -> PrunePlan:
        scores = importance.as_dict()
        before = model.architecture
        removals: list[Expert] = []
        layers = []
        for index, shape in enumerate(before.layers):
            keep = max(budget.min_experts_per_layer, math.ceil(shape.num_experts * budget.keep_ratio))
            removals.extend(_least_used(scores, index, shape.num_experts, keep))
            layers.append(replace(shape, num_experts=keep))
        after = replace(before, layers=tuple(layers))
        return PrunePlan(
            removals=tuple(removals),
            before=before,
            after=after,
            importance=importance,
            estimated=CostDelta(parameters=parameters_of(after) - model.parameters),
        )


class UnevenExpertSelector:
    """Keeps a different number of experts per layer, which a uniform-only backend must refuse."""

    def select(self, importance: ImportanceMap, model: ModelSnapshot, budget: CompressionBudget) -> PrunePlan:
        scores = importance.as_dict()
        before = model.architecture
        removals: list[Expert] = []
        layers = []
        for index, shape in enumerate(before.layers):
            keep = max(budget.min_experts_per_layer, shape.num_experts - 1 - index)
            removals.extend(_least_used(scores, index, shape.num_experts, keep))
            layers.append(replace(shape, num_experts=keep))
        after = replace(before, layers=tuple(layers))
        return PrunePlan(
            removals=tuple(removals),
            before=before,
            after=after,
            importance=importance,
            estimated=CostDelta(parameters=parameters_of(after) - model.parameters),
        )


class UniformExpertEditor:
    """Applies a plan by recomputing size and lowering quality in proportion to what was cut."""

    def __init__(self, world: FakeWorld, cost_per_removed_fraction: float = 0.3) -> None:
        self.world = world
        self.cost = cost_per_removed_fraction
        self.applied: list[PrunePlan] = []

    def apply(self, model: ModelSnapshot, plan: PrunePlan) -> ModelSnapshot:
        self.applied.append(plan)
        removed_fraction = 1 - parameters_of(plan.after) / model.parameters
        quality = self.world.quality[model.checkpoint.checksum] - self.cost * removed_fraction
        return self.world.snapshot(plan.after, quality)


class FakeQuantizer:
    """Serving weights lose ``cost`` quality, or nothing when the master was trained for the scheme."""

    def __init__(self, world: FakeWorld, cost: float = 0.02) -> None:
        self.world = world
        self.cost = cost
        self.quantized: list[QuantizePlan] = []

    def quantize(self, model: ModelSnapshot, plan: QuantizePlan) -> ServingWeights:
        self.quantized.append(plan)
        trained_for = self.world.trained_scheme.get(model.checkpoint.checksum)
        cost = 0.0 if trained_for == plan.scheme else self.cost
        checkpoint = self.world.checkpoint(self.world.quality[model.checkpoint.checksum] - cost)
        return ServingWeights(checkpoint=checkpoint, scheme=plan.scheme, source=model.checkpoint)
