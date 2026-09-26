"""The ordering rules a campaign checks before a step touches any weights.

Each rule is a function that returns nothing when the step may proceed and
raises :class:`~honing.core.CampaignRuleError`, carrying its own name, when it
may not. Rules are checked before training, editing, or quantizing starts, so a
refused step costs no compute and leaves no partial output behind.
"""

from __future__ import annotations

from collections.abc import Mapping

from honing.core import (
    ALGORITHM_REQUIREMENTS,
    MASTER_WEIGHT_FORMAT,
    ArchitectureSpec,
    BackendSupport,
    CampaignRuleError,
    CompressionStep,
    HarnessSpec,
    HoneRelease,
    MethodRequirements,
    NumericsPolicy,
    PrunePlan,
    QuantScheme,
    RunnerSupport,
    TrainingBatch,
    TrainJobSpec,
    UpdateCandidate,
    method_state_version,
)


def architecture_supported(architecture: ArchitectureSpec, support: BackendSupport) -> None:
    """A shape the backend cannot build is refused before any weight is edited."""
    if support.architectures == "uniform" and not architecture.uniform:
        shapes = len(set(architecture.layers))
        raise CampaignRuleError(
            "architecture_supported",
            f"{support.name} builds uniform architectures only; this one has {shapes} different layer shapes",
        )


def master_is_hf_safetensors(candidate: UpdateCandidate) -> None:
    """Releases keep master weights in one format that every editor and trainer reads."""
    weight_format = candidate.master.checkpoint.weight_format
    if weight_format != MASTER_WEIGHT_FORMAT:
        raise CampaignRuleError(
            "master_is_hf_safetensors",
            f"master weights must be {MASTER_WEIGHT_FORMAT}, got {weight_format}; "
            "convert back after the backend job",
        )


def serving_built_from_master(candidate: UpdateCandidate) -> None:
    """Serving weights must have been quantized from the master they ship with.

    A train or prune step changes the master, so it must drop the old serving
    weights; a later quantize step rebuilds them.
    """
    serving = candidate.serving
    if serving is not None and serving.source != candidate.master.checkpoint:
        raise CampaignRuleError(
            "serving_built_from_master",
            f"serving weights were quantized from {serving.source.checksum}, "
            f"not from this candidate's master {candidate.master.checkpoint.checksum}",
        )


def qat_scheme_matches_export(trained: NumericsPolicy, export: QuantScheme) -> None:
    """Weights trained to tolerate one quantization scheme are exported under that scheme only."""
    if trained.mode == "fake_quant_qat" and trained.scheme != export:
        raise CampaignRuleError(
            "qat_scheme_matches_export",
            f"the master was trained for {trained.scheme}, but the export uses {export}",
        )


def recovery_teacher_fits_new_shape(step: CompressionStep) -> None:
    """After a shape change, the pre-cut teacher must be served from its own engine.

    An in-process teacher shares the student's model definition (Slime's
    in-process teacher mode requires the same architecture), so it cannot score
    for a student whose shape the plan just changed.
    """
    plan, recovery = step.transform, step.recovery
    if not isinstance(plan, PrunePlan) or recovery is None or recovery.teacher is None:
        return
    if plan.changes_shape and recovery.teacher.placement == "in_process":
        raise CampaignRuleError(
            "recovery_teacher_fits_new_shape",
            "the plan changes the architecture, so the recovery teacher must be served by an engine",
        )


def algorithm_requirements_met(job: TrainJobSpec, support: BackendSupport) -> None:
    """The job's rollout source, teacher, and precision are what its algorithm and backend need.

    This reads the job alone, so it runs before any batch is drawn or any
    weight is edited. :func:`batch_meets_contract` checks the batch once it exists.
    """
    requirements = ALGORITHM_REQUIREMENTS[job.algorithm]
    problems = []
    if support.support_for(job.algorithm) == "unsupported":
        problems.append(f"{support.name} cannot run {job.algorithm}")
    if requirements.requires_rollout and job.rollout == "none":
        problems.append(f"{job.algorithm} learns from its own samples and needs a rollout source")
    if not requirements.requires_rollout and job.rollout != "none":
        problems.append(f"{job.algorithm} trains on fixed data; rollout {job.rollout!r} would be ignored")
    if requirements.requires_teacher and job.teacher is None:
        problems.append(f"{job.algorithm} needs a teacher")
    if not requirements.requires_teacher and job.teacher is not None:
        problems.append(f"{job.algorithm} takes no teacher")
    if job.numerics.mode not in support.train_numerics:
        problems.append(f"{support.name} cannot train in {job.numerics.mode}")
    if problems:
        raise CampaignRuleError("algorithm_requirements_met", "; ".join(problems))


def batch_meets_contract(job: TrainJobSpec, batch: TrainingBatch | None) -> None:
    """A training job starts only with a batch of the kind its algorithm consumes."""
    contract = ALGORITHM_REQUIREMENTS[job.algorithm].contract
    if batch is None:
        raise CampaignRuleError(
            "batch_meets_contract", f"no batch meets the {contract} contract {job.algorithm} needs"
        )
    if batch.contract.kind != contract:
        raise CampaignRuleError(
            "batch_meets_contract", f"{job.algorithm} needs a {contract} batch, got {batch.contract.kind}"
        )


def batch_from_one_release(batch: TrainingBatch, head: HoneRelease) -> None:
    """Samples earned under different weights or different harnesses are not comparable.

    Offline samples name no release or harness and are always accepted.
    """
    foreign_releases = batch.source_releases - {head.release_id}
    head_harness = frozenset() if head.harness is None else frozenset({head.harness.spec_id})
    foreign_harnesses = batch.source_harnesses - head_harness
    if foreign_releases or foreign_harnesses:
        raise CampaignRuleError(
            "batch_from_one_release",
            f"batch {batch.batch_id} mixes releases {sorted(batch.source_releases)} and harnesses "
            f"{sorted(batch.source_harnesses)}; the head is {head.release_id}",
        )


def method_supported_by_runner(requires: MethodRequirements, support: RunnerSupport) -> None:
    """A method that edits what the runner cannot run is refused when the campaign starts."""
    problems = []
    missing_kinds = requires.file_kinds - support.file_kinds
    if missing_kinds:
        problems.append(f"the runner cannot run harness files of kind {sorted(missing_kinds)}")
    if requires.programmable_loop and not support.programmable_loop:
        problems.append("the method rewrites the agent loop, but the runner's agents have fixed loops")
    if requires.agents and not requires.agents & support.agents:
        problems.append(f"the method targets {sorted(requires.agents)}; the runner has {sorted(support.agents)}")
    if problems:
        raise CampaignRuleError("method_supported_by_runner", "; ".join(problems))


def spec_runnable(spec: HarnessSpec, support: RunnerSupport) -> None:
    """A proposed harness must be one the runner can actually run."""
    problems = []
    if spec.agent not in support.agents:
        problems.append(f"the runner has no agent {spec.agent!r}")
    unsupported = spec.kinds - support.file_kinds
    if unsupported:
        problems.append(f"the runner cannot run harness files of kind {sorted(unsupported)}")
    if problems:
        raise CampaignRuleError("spec_runnable", "; ".join(problems))


def method_state_schema_matches(state: Mapping[str, object], method: str, expected: int) -> None:
    """State written under another schema is refused before the method reads it.

    A rollback can hand an old release's state to a newer method, whose archive
    has a different shape. Refusing here names both versions; letting the
    method read it would fail somewhere inside the method, or not at all.
    """
    found = method_state_version(state)
    if found != expected:
        raise CampaignRuleError(
            "method_state_schema_matches",
            f"{method} reads method state schema version {expected}, but the head's state is version {found}",
        )
