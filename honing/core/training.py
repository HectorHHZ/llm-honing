"""What a training job is: algorithm, update policy, numerics, rollout source, and teacher.

The five are independent choices. :data:`ALGORITHM_REQUIREMENTS` states what each
algorithm needs from the others and which executor runs it. The campaign checks
a job against it before calling any trainer, so an impossible combination fails
at the start of a step rather than hours into it.

Executors:

``slime``
    SFT, PPO and GRPO, and on-policy distillation run natively; offline logit
    distillation and GKD run through Slime's custom-loss hook.
``local``
    The preference trainer that runs DPO without Slime, because Slime's data
    flow is built around rollouts and has no preference-pair path; and the
    decision-model trainer, because a Jev-style model with a scoring head is not
    a Megatron model.

An algorithm has a default data contract. ``sft`` and ``grpo`` can also train a
decision model on ``typed_decisions`` (:data:`DECISION_REQUIREMENTS`); a job
opts in with ``TrainJobSpec.contract``, and :func:`requirements_for` returns the
requirements that apply to it.

On-policy *self*-distillation is ``opd`` with a teacher of kind ``lineage``: the
student distills from an earlier release of itself.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Literal, get_args

from honing.core.data import ContractKind, DataContract
from honing.core.errors import ContractError, require, require_choice
from honing.core.numerics import NumericsPolicy
from honing.core.structure import ParamMask
from honing.core.weights import ModelSnapshot

AlgorithmName = Literal["sft", "distill", "opd", "gkd", "ppo", "grpo", "dpo"]
UpdatePolicyKind = Literal["full", "lora", "param_subset", "sparse_mask"]
RolloutKind = Literal["none", "direct", "harness"]
TeacherKind = Literal["lineage", "release", "external"]
TeacherPlacement = Literal["engine", "in_process"]
Executor = Literal["slime", "local"]


@dataclass(frozen=True)
class AlgorithmRequirements:
    """What one algorithm needs, and who runs it."""

    contract: ContractKind
    requires_rollout: bool
    requires_teacher: bool
    executor: Executor
    requires_reward: bool = False
    requires_logprobs: bool = False
    requires_group: bool = False

    def data_contract(self) -> DataContract:
        return DataContract(
            kind=self.contract,
            requires_reward=self.requires_reward,
            requires_logprobs=self.requires_logprobs,
            requires_group=self.requires_group,
        )


_REQUIREMENTS: dict[AlgorithmName, AlgorithmRequirements] = {
    "sft": AlgorithmRequirements("token_sequences", requires_rollout=False, requires_teacher=False, executor="slime"),
    "distill": AlgorithmRequirements(
        "token_sequences", requires_rollout=False, requires_teacher=True, executor="slime"
    ),
    "opd": AlgorithmRequirements(
        "on_policy_samples", requires_rollout=True, requires_teacher=True, executor="slime", requires_logprobs=True
    ),
    "gkd": AlgorithmRequirements(
        "on_policy_samples", requires_rollout=True, requires_teacher=True, executor="slime", requires_logprobs=True
    ),
    "ppo": AlgorithmRequirements(
        "on_policy_samples",
        requires_rollout=True,
        requires_teacher=False,
        executor="slime",
        requires_reward=True,
        requires_logprobs=True,
    ),
    "grpo": AlgorithmRequirements(
        "grouped_on_policy_samples",
        requires_rollout=True,
        requires_teacher=False,
        executor="slime",
        requires_reward=True,
        requires_logprobs=True,
    ),
    "dpo": AlgorithmRequirements("preference_pairs", requires_rollout=False, requires_teacher=False, executor="local"),
}

ALGORITHM_REQUIREMENTS: Final[Mapping[AlgorithmName, AlgorithmRequirements]] = MappingProxyType(_REQUIREMENTS)

# Training a decision model. Supervised decisions need no rollout; policy
# gradient over option distributions needs each decision's reward, behavior
# probabilities, and a group to compare within, for example the seeds of one scenario.
_DECISION_REQUIREMENTS: dict[AlgorithmName, AlgorithmRequirements] = {
    "sft": AlgorithmRequirements("typed_decisions", requires_rollout=False, requires_teacher=False, executor="local"),
    "grpo": AlgorithmRequirements(
        "typed_decisions",
        requires_rollout=True,
        requires_teacher=False,
        executor="local",
        requires_reward=True,
        requires_logprobs=True,
        requires_group=True,
    ),
}

DECISION_REQUIREMENTS: Final[Mapping[AlgorithmName, AlgorithmRequirements]] = MappingProxyType(_DECISION_REQUIREMENTS)


def _compile_pattern(pattern: str) -> None:
    try:
        re.compile(pattern)
    except re.error as error:
        raise ContractError(f"invalid trainable parameter pattern {pattern!r}: {error}") from error


@dataclass(frozen=True)
class UpdatePolicy:
    """Which parameters a training job may change.

    ``param_subset`` selects parameters by name pattern, which Slime supports
    natively; ``sparse_mask`` selects elements, using the same
    :class:`~honing.core.structure.ParamMask` a pruning plan uses.
    """

    kind: UpdatePolicyKind = "full"
    lora_rank: int | None = None
    lora_targets: tuple[str, ...] = ()
    trainable_patterns: tuple[str, ...] = ()
    masks: tuple[ParamMask, ...] = ()

    def __post_init__(self) -> None:
        require_choice(self.kind, get_args(UpdatePolicyKind), "UpdatePolicy.kind")
        require((self.kind == "lora") == (self.lora_rank is not None), "lora_rank is set exactly when kind is 'lora'")
        if self.lora_rank is not None:
            require(self.lora_rank > 0, f"lora_rank must be > 0, got {self.lora_rank}")
        require(not self.lora_targets or self.kind == "lora", "lora_targets apply only to kind 'lora'")
        require(
            (self.kind == "param_subset") == bool(self.trainable_patterns),
            "trainable_patterns are set exactly when kind is 'param_subset'",
        )
        for pattern in self.trainable_patterns:
            _compile_pattern(pattern)
        require((self.kind == "sparse_mask") == bool(self.masks), "masks are set exactly when kind is 'sparse_mask'")


@dataclass(frozen=True)
class TeacherRef:
    """The model a distillation job learns from, and where it runs.

    ``engine`` serves the teacher from its own inference engine and needs only a
    compatible tokenizer. ``in_process`` loads it beside the student and needs
    the same architecture, so it cannot score for a student whose shape a
    compression step just changed.
    """

    kind: TeacherKind
    ref: str
    placement: TeacherPlacement = "engine"

    def __post_init__(self) -> None:
        require_choice(self.kind, get_args(TeacherKind), "TeacherRef.kind")
        require_choice(self.placement, get_args(TeacherPlacement), "TeacherRef.placement")
        require(bool(self.ref), "TeacherRef.ref must name a release, 'parent', or an endpoint")
        require(
            not (self.kind == "external" and self.placement == "in_process"),
            "an external teacher is reached through its engine, never loaded in process",
        )


@dataclass(frozen=True)
class TrainJobSpec:
    """One training job. Construction checks shape; campaign rules check the combination."""

    algorithm: AlgorithmName
    update_policy: UpdatePolicy = field(default_factory=UpdatePolicy)
    numerics: NumericsPolicy = field(default_factory=NumericsPolicy)
    rollout: RolloutKind = "none"
    teacher: TeacherRef | None = None
    steps: int = 1
    contract: ContractKind | None = None

    def __post_init__(self) -> None:
        require_choice(self.algorithm, get_args(AlgorithmName), "TrainJobSpec.algorithm")
        require_choice(self.rollout, get_args(RolloutKind), "TrainJobSpec.rollout")
        require(self.steps > 0, f"steps must be > 0, got {self.steps}")
        if self.contract is not None:
            require_choice(self.contract, get_args(ContractKind), "TrainJobSpec.contract")
            defaults = ALGORITHM_REQUIREMENTS[self.algorithm].contract
            decisions = DECISION_REQUIREMENTS.get(self.algorithm)
            require(
                self.contract == defaults or (decisions is not None and self.contract == decisions.contract),
                f"{self.algorithm} does not train on {self.contract}",
            )


def requirements_for(job: TrainJobSpec) -> AlgorithmRequirements:
    """What this job's algorithm needs on the data contract the job trains on."""
    if job.contract == "typed_decisions":
        return DECISION_REQUIREMENTS[job.algorithm]
    return ALGORITHM_REQUIREMENTS[job.algorithm]


@dataclass(frozen=True)
class RecoverySpec:
    """The training that follows a compression step, so the gate judges the recovered model."""

    algorithm: AlgorithmName
    steps: int
    teacher: TeacherRef | None = None
    update_policy: UpdatePolicy = field(default_factory=UpdatePolicy)

    def __post_init__(self) -> None:
        require_choice(self.algorithm, get_args(AlgorithmName), "RecoverySpec.algorithm")
        require(self.steps > 0, f"steps must be > 0, got {self.steps}")

    def as_job(self, numerics: NumericsPolicy) -> TrainJobSpec:
        """The job that runs this recovery, keeping the precision the weights were trained in."""
        requirements = ALGORITHM_REQUIREMENTS[self.algorithm]  # recovery trains the pruned language model
        return TrainJobSpec(
            algorithm=self.algorithm,
            update_policy=self.update_policy,
            numerics=numerics,
            rollout="direct" if requirements.requires_rollout else "none",
            teacher=self.teacher,
            steps=self.steps,
        )


@dataclass(frozen=True)
class TrainResult:
    """The weights a training job produced, always in the master format."""

    snapshot: ModelSnapshot
    metrics: Mapping[str, float] = field(default_factory=dict)
