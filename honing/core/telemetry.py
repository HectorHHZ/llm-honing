"""Observed capacity usage: the statistics compression scores structures by.

Telemetry is aggregated per window rather than stored per request, because the
per-token counts that matter here would swamp a per-request record store.
Rollout and serving use the same engines, so a window can be filled before a
serving plane exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from honing.core.errors import require
from honing.core.structure import Expert


@dataclass(frozen=True)
class UtilizationTelemetry:
    """Routing and activation statistics over one window of traffic."""

    sample_count: int
    expert_activations: tuple[tuple[Expert, int], ...] = ()
    layer_activation_norms: tuple[tuple[int, float], ...] = ()

    def __post_init__(self) -> None:
        require(self.sample_count > 0, f"a telemetry window needs samples, got {self.sample_count}")
        require(all(count >= 0 for _, count in self.expert_activations), "activation counts must be >= 0")
        require(all(math.isfinite(norm) for _, norm in self.layer_activation_norms), "norms must be finite")

    def activation(self, expert: Expert) -> int:
        """How often ``expert`` was routed to in this window; 0 when it never was."""
        return dict(self.expert_activations).get(expert, 0)
