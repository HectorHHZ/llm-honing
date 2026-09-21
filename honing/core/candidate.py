"""The one candidate type every plane produces, and the gate's verdict on it.

Weight training, pruning, quantization, and harness evolution all produce an
:class:`UpdateCandidate`. One gate judges it and one commit protocol publishes
it, so a bad prune is a rejected candidate or a rolled-back release, not a new
failure mode with its own machinery.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from statistics import fmean
from typing import Literal, get_args

from honing.core.compression import TransformManifest
from honing.core.errors import require, require_choice
from honing.core.harness import EpisodeRecord, HarnessSpec
from honing.core.numerics import NumericsPolicy
from honing.core.structure import ArchitectureSpec
from honing.core.weights import CheckpointRef, ModelSnapshot, ServingWeights

CandidateKind = Literal["init", "train", "prune", "quantize", "harness"]
PolicyVariant = Literal["master", "serving"]
Outcome = Literal["select", "reject"]


@dataclass(frozen=True)
class PolicyRef:
    """The exact weights an episode or a rollout runs against."""

    checkpoint: CheckpointRef
    architecture: ArchitectureSpec
    variant: PolicyVariant

    def __post_init__(self) -> None:
        require_choice(self.variant, get_args(PolicyVariant), "PolicyRef.variant")


@dataclass(frozen=True)
class UpdateCandidate:
    """A proposed next release.

    ``trained_numerics`` records the precision the master was last trained in,
    so a later quantize step can be checked against it. ``method_state`` is the
    harness evolution method's own state, published and rolled back with the
    rest of the release.
    """

    kind: CandidateKind
    parent_id: str | None
    master: ModelSnapshot
    serving: ServingWeights | None = None
    transforms: tuple[TransformManifest, ...] = ()
    harness: HarnessSpec | None = None
    trained_numerics: NumericsPolicy = field(default_factory=NumericsPolicy)
    method_state: Mapping[str, object] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_choice(self.kind, get_args(CandidateKind), "UpdateCandidate.kind")
        require((self.kind == "init") == (self.parent_id is None), "only the initial candidate has no parent")
        require(all(math.isfinite(value) for value in self.metrics.values()), "candidate metrics must be finite")

    @property
    def evaluated_variant(self) -> PolicyVariant:
        """The weights the gate must score: the serving weights when they exist, since those ship."""
        return "master" if self.serving is None else "serving"

    def policy(self) -> PolicyRef:
        if self.serving is None:
            return PolicyRef(self.master.checkpoint, self.master.architecture, "master")
        return PolicyRef(self.serving.checkpoint, self.master.architecture, "serving")


@dataclass(frozen=True)
class EvaluationResult:
    """Per-task scores for a candidate and the release it would replace, on the same tasks.

    The scores stay per task rather than averaged, because selection policies
    such as Pareto fronts need the vector, not the mean.
    """

    task_ids: tuple[str, ...]
    candidate_scores: tuple[float, ...]
    incumbent_scores: tuple[float, ...]
    evaluated_variant: PolicyVariant
    records: tuple[EpisodeRecord, ...] = ()

    def __post_init__(self) -> None:
        require(len(self.task_ids) > 0, "an evaluation needs at least one task")
        require(len(set(self.task_ids)) == len(self.task_ids), "an evaluation scores each task once")
        require(
            len(self.candidate_scores) == len(self.task_ids) == len(self.incumbent_scores),
            "an evaluation has one candidate and one incumbent score per task",
        )
        scores = (*self.candidate_scores, *self.incumbent_scores)
        require(all(math.isfinite(score) for score in scores), "evaluation scores must be finite")
        require_choice(self.evaluated_variant, get_args(PolicyVariant), "EvaluationResult.evaluated_variant")

    @property
    def candidate_mean(self) -> float:
        return fmean(self.candidate_scores)

    @property
    def incumbent_mean(self) -> float:
        return fmean(self.incumbent_scores)


@dataclass(frozen=True)
class SelectionDecision:
    """Whether a candidate is published, which policy decided, and why."""

    outcome: Outcome
    policy: str
    reason: str
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_choice(self.outcome, get_args(Outcome), "SelectionDecision.outcome")
        require(bool(self.policy), "SelectionDecision.policy must name the policy")

    @property
    def selected(self) -> bool:
        return self.outcome == "select"
