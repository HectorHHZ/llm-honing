"""A stand-in for SlimeTrainer, and a rollout provider for it, in the all-planes pipeline (``configs/e2e/slime.json``).

It declares the real :data:`honing.backends.slime.SUPPORT`, refuses what the
campaign's rules refuse, and converts every batch into Slime's rollout layout
(:func:`honing.backends.slime.slime_samples`) before training the fake model, so
a batch the real trainer could not write fails here too. The fake world's
checkpoints are not Hugging Face directories, so the real trainer cannot run in
this pipeline; ``scripts/e2e/slime_train_step.py`` runs it on a GPU.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from honing.backends.slime import SUPPORT, slime_samples
from honing.campaign import rules
from honing.core import (
    ContractError,
    HarnessSpec,
    ModelSnapshot,
    PolicyRef,
    Task,
    TrainingBatch,
    TrainJobSpec,
    TrainResult,
)
from honing.e2e.components import PipelineContext
from tests.fakes.data import FakeRolloutProvider
from tests.fakes.model import FakeTrainer
from tests.fakes.pipeline import _world


class SlimeStandIn:
    """Satisfies Trainer with Slime's declared support, checking each batch against Slime's rollout layout."""

    def __init__(self, inner: FakeTrainer) -> None:
        self._inner = inner
        self.support = SUPPORT
        self.layouts: list[tuple[int, int]] = []

    def train(self, job: TrainJobSpec, base: ModelSnapshot, batch: TrainingBatch | None) -> TrainResult:
        rules.algorithm_requirements_met(job, self.support)
        rules.batch_meets_contract(job, batch)
        if batch is None:
            raise ContractError(f"{job.algorithm} needs a batch")
        _, layout = slime_samples(batch, rewarded=job.algorithm == "grpo")
        self.layouts.append((layout.prompts, layout.samples_per_prompt))
        return self._inner.train(job, base, batch)


def trainer(context: PipelineContext, gain: float = 0.1) -> SlimeStandIn:
    return SlimeStandIn(FakeTrainer(_world(context), support=SUPPORT, gain=gain))


class PrivilegedRolloutProvider:
    """Satisfies RolloutProvider: the fake rollouts, each also scored by the policy under privileged context.

    On-policy self-distillation needs the teacher's log-probability of every sampled token; a real provider
    scores each response again with the answer or a hint added to the prompt. Here the teacher is a little
    more confident than the sampler on every trained token.
    """

    def __init__(self, inner: FakeRolloutProvider) -> None:
        self._inner = inner

    def collect(
        self,
        tasks: Sequence[Task],
        policy: PolicyRef,
        harness: HarnessSpec | None,
        release_id: str | None,
        samples_per_task: int,
    ) -> TrainingBatch:
        batch = self._inner.collect(tasks, policy, harness, release_id, samples_per_task)
        samples = tuple(
            replace(
                sample,
                teacher_logprobs=tuple(
                    (value + 0.1) * mask for value, mask in zip(sample.logprobs or (), sample.loss_mask, strict=True)
                ),
            )
            for sample in batch.samples
        )
        contract = replace(batch.contract, requires_teacher_logprobs=True)
        return replace(batch, contract=contract, samples=samples)


def rollout(context: PipelineContext) -> PrivilegedRolloutProvider:
    return PrivilegedRolloutProvider(FakeRolloutProvider())
