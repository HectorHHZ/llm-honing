"""What any trainer must do to plug into Honing, checked against the campaign's own rules.

A training backend (Slime, the local Jev trainers, ...) enters Honing as a
:class:`~honing.core.Trainer` with its :class:`~honing.core.BackendSupport`.
:func:`check_trainer` puts it through what a campaign relies on and returns one
:class:`TrainerCheck` per case, so a new backend can be tested before it meets
a campaign:

- ``support``: the trainer declares a ``BackendSupport``, the base model's
  architecture is one it supports, and every job to try is one the campaign's
  rules (``algorithm_requirements_met``) would let reach it;
- ``train:<algorithm>``: each job runs on a batch drawn the way the campaign draws
  it (``DataProvider`` for fixed data, ``RolloutProvider`` for on-policy
  algorithms, checked by ``batch_meets_contract`` and ``batch_from_one_release``),
  and returns a ``TrainResult`` the gate can take: new weights (a checksum other
  than the base's), Hugging Face safetensors (the release's master format), a
  supported architecture, and finite metrics;
- ``fresh``: training again does not overwrite the first result, which a release
  may already point at;
- ``refuses``: the trainer refuses a missing batch and an algorithm it does not
  declare, rather than training something else.

``scripts/e2e/check_trainer.py`` runs these on the trainer a pipeline config names.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import get_args

from honing.campaign import rules
from honing.core import (
    AlgorithmName,
    DataProvider,
    HarnessSpec,
    HoneRelease,
    ModelSnapshot,
    RolloutProvider,
    Task,
    TeacherRef,
    Trainer,
    TrainingBatch,
    TrainJobSpec,
    TrainResult,
    UpdateCandidate,
    requirements_for,
)

MASTER_FORMAT = "hf_safetensors"


@dataclass(frozen=True)
class TrainerCheck:
    """One case: whether it passed, and what was seen."""

    name: str
    passed: bool
    detail: str


def _batch(
    job: TrainJobSpec,
    data: DataProvider,
    rollout: RolloutProvider | None,
    head: HoneRelease,
    tasks: Sequence[Task],
    samples_per_task: int,
) -> TrainingBatch:
    """The batch the campaign would hand this job, checked by the campaign's rules."""
    requirements = requirements_for(job)
    if requirements.requires_rollout:
        if rollout is None:
            raise ValueError(f"{job.algorithm} learns from rollouts, but the config names no rollout component")
        batch = rollout.collect(
            tasks, head.policy(), head.harness if job.rollout == "harness" else None, head.release_id, samples_per_task
        )
    else:
        found = data.next_batch(requirements.data_contract())
        if found is None:
            raise ValueError(f"the data component has no {requirements.contract} batch for {job.algorithm}")
        batch = found
    rules.batch_meets_contract(job, batch)
    rules.batch_from_one_release(batch, head)
    return batch


def _result_problems(result: object, base: ModelSnapshot, trainer: Trainer) -> list[str]:
    if not isinstance(result, TrainResult):
        return [f"train returned a {type(result).__name__}, not a TrainResult"]
    problems = []
    snapshot = result.snapshot
    if snapshot.checkpoint.checksum == base.checkpoint.checksum:
        problems.append("the weights did not change (same checksum as the base)")
    if snapshot.checkpoint.weight_format != MASTER_FORMAT:
        problems.append(f"the weights are {snapshot.checkpoint.weight_format}, not {MASTER_FORMAT}")
    try:
        rules.architecture_supported(snapshot.architecture, trainer.support)
    except Exception as error:  # the campaign would refuse this result
        problems.append(str(error))
    bad = sorted(name for name, value in result.metrics.items() if not math.isfinite(value))
    if bad:
        problems.append(f"metrics that are not finite: {bad}")
    return problems


def _undeclared(trainer: Trainer) -> TrainJobSpec | None:
    """A job for an algorithm the trainer does not declare, shaped so its own fields are valid."""
    for algorithm in get_args(AlgorithmName):
        if trainer.support.support_for(algorithm) != "unsupported":
            continue
        probe = TrainJobSpec(algorithm=algorithm)
        requirements = requirements_for(probe)
        try:
            return TrainJobSpec(
                algorithm=algorithm,
                rollout="direct" if requirements.requires_rollout else "none",
                teacher=TeacherRef(kind="lineage", ref="parent") if requirements.requires_teacher else None,
            )
        except Exception:  # a combination the core refuses to build; try the next algorithm
            continue
    return None


def check_trainer(
    trainer: Trainer,
    *,
    base: ModelSnapshot,
    jobs: Sequence[TrainJobSpec],
    data: DataProvider,
    rollout: RolloutProvider | None,
    tasks: Sequence[Task],
    harness: HarnessSpec,
    samples_per_task: int = 2,
) -> list[TrainerCheck]:
    """Run every case; a case that raises fails with the error as its detail, and later cases still run."""
    if not jobs:
        raise ValueError("give at least one job to try")
    checks: list[TrainerCheck] = []
    head = HoneRelease("r0000", UpdateCandidate("init", None, base, harness=harness), None)

    def case(name: str, body: Callable[[], tuple[bool, str]]) -> bool:
        try:
            passed, detail = body()
        except Exception as error:  # a conformance case reports, rather than stops on, a trainer's failure
            passed, detail = False, f"{type(error).__name__}: {error}"
        checks.append(TrainerCheck(name, passed, detail))
        return passed

    def support() -> tuple[bool, str]:
        rules.architecture_supported(base.architecture, trainer.support)
        for job in jobs:
            rules.algorithm_requirements_met(job, trainer.support)
        declared = sorted(name for name, how in trainer.support.algorithms if how != "unsupported")
        return True, f"{trainer.support.name}: algorithms {declared}, contracts {sorted(trainer.support.contracts)}"

    if not case("support", support):
        return checks

    results: dict[int, tuple[TrainJobSpec, TrainingBatch, TrainResult]] = {}
    for index, job in enumerate(jobs):

        def train(job: TrainJobSpec = job, index: int = index) -> tuple[bool, str]:
            batch = _batch(job, data, rollout, head, tasks, samples_per_task)
            result = trainer.train(job, base, batch)
            problems = _result_problems(result, base, trainer)
            if problems:
                return False, "; ".join(problems)
            results[index] = (job, batch, result)
            source = "rollouts" if requirements_for(job).requires_rollout else "fixed data"
            return True, f"{source} batch {batch.batch_id} -> {result.snapshot.checkpoint.checksum[:16]}"

        case(f"train:{job.algorithm}", train)

    def fresh() -> tuple[bool, str]:
        if not results:
            return False, "no job trained, so there is nothing to train again"
        job, batch, first = next(iter(results.values()))
        second = trainer.train(job, base, batch)
        a, b = first.snapshot.checkpoint, second.snapshot.checkpoint
        if a.location == b.location and a.checksum != b.checksum:
            return False, f"the second run wrote different weights to the first run's location {a.location}"
        same = "the same weights" if a.checksum == b.checksum else "new weights"
        return True, f"a second {job.algorithm} run left the first in place and produced {same}"

    case("fresh", fresh)

    def refuses() -> tuple[bool, str]:
        if not results:
            return False, "no job trained, so there is no valid batch to misuse"
        job, batch, _ = next(iter(results.values()))
        refused = []
        try:
            trainer.train(job, base, None)
        except Exception as error:  # the expected outcome: the trainer refuses
            refused.append(f"no batch ({type(error).__name__})")
        else:
            return False, f"{job.algorithm} trained with no batch"
        undeclared = _undeclared(trainer)
        if undeclared is not None:
            try:
                trainer.train(replace(undeclared, steps=job.steps), base, batch)
            except Exception as error:  # the expected outcome
                refused.append(f"undeclared {undeclared.algorithm} ({type(error).__name__})")
            else:
                return False, f"it trained {undeclared.algorithm}, which its BackendSupport does not declare"
        return True, "refused " + " and ".join(refused)

    case("refuses", refuses)
    return checks


__all__ = ["TrainerCheck", "check_trainer"]
