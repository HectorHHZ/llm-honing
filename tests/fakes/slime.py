"""A stand-in for SlimeTrainer in the all-planes pipeline (``configs/e2e/slime_grpo.json``).

It declares the real :data:`honing.backends.slime.SUPPORT`, refuses what the
campaign's rules refuse, and converts every batch into Slime's rollout layout
(:func:`honing.backends.slime.slime_samples`) before training the fake model, so
a batch the real trainer could not write fails here too. The fake world's
checkpoints are not Hugging Face directories, so the real trainer cannot run in
this pipeline; ``scripts/e2e/slime_grpo_step.py`` runs it on a GPU.
"""

from __future__ import annotations

from honing.backends.slime import SUPPORT, slime_samples
from honing.campaign import rules
from honing.core import ContractError, ModelSnapshot, TrainingBatch, TrainJobSpec, TrainResult
from honing.e2e.components import PipelineContext
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
        _, layout = slime_samples(batch)
        self.layouts.append((layout.prompts, layout.samples_per_prompt))
        return self._inner.train(job, base, batch)


def trainer(context: PipelineContext, gain: float = 0.1) -> SlimeStandIn:
    return SlimeStandIn(FakeTrainer(_world(context), support=SUPPORT, gain=gain))
