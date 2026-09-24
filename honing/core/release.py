"""A published version, restored as a whole.

Rolling back weights while keeping a harness evolved for those weights, or
restoring a master while keeping serving weights quantized from another master,
produces a combination no gate ever evaluated. So a release is the entire
accepted candidate, and a rollback restores all of it: master and serving
weights, architecture, the transforms applied, the harness, and the evolution
method's state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from honing.core.candidate import CandidateKind, PolicyRef, SelectionDecision, UpdateCandidate
from honing.core.compression import TransformManifest
from honing.core.errors import require
from honing.core.harness import HarnessSpec
from honing.core.numerics import NumericsPolicy
from honing.core.structure import ArchitectureSpec
from honing.core.weights import MASTER_WEIGHT_FORMAT, ModelSnapshot, ServingWeights


@dataclass(frozen=True)
class HoneRelease:
    """One accepted candidate with its id and the decision that published it."""

    release_id: str
    candidate: UpdateCandidate
    decision: SelectionDecision | None

    def __post_init__(self) -> None:
        require(bool(self.release_id), "HoneRelease.release_id must be set")
        weight_format = self.candidate.master.checkpoint.weight_format
        require(
            weight_format == MASTER_WEIGHT_FORMAT,
            f"a release stores master weights as {MASTER_WEIGHT_FORMAT}, got {weight_format}",
        )
        require(
            (self.candidate.kind == "init") == (self.decision is None),
            "only the initial release is published without a gate decision",
        )
        require(self.decision is None or self.decision.selected, "a rejected candidate is never a release")

    @property
    def kind(self) -> CandidateKind:
        return self.candidate.kind

    @property
    def parent_id(self) -> str | None:
        return self.candidate.parent_id

    @property
    def master(self) -> ModelSnapshot:
        return self.candidate.master

    @property
    def serving(self) -> ServingWeights | None:
        return self.candidate.serving

    @property
    def architecture(self) -> ArchitectureSpec:
        return self.candidate.master.architecture

    @property
    def transforms(self) -> tuple[TransformManifest, ...]:
        return self.candidate.transforms

    @property
    def harness(self) -> HarnessSpec | None:
        return self.candidate.harness

    @property
    def trained_numerics(self) -> NumericsPolicy:
        return self.candidate.trained_numerics

    @property
    def method_state(self) -> Mapping[str, object]:
        return self.candidate.method_state

    @property
    def metrics(self) -> Mapping[str, float]:
        return self.candidate.metrics

    @property
    def move_label(self) -> str | None:
        return self.candidate.move_label

    def policy(self) -> PolicyRef:
        return self.candidate.policy()
