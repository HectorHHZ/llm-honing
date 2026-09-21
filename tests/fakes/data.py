"""Data fakes: a provider that holds one fixed batch per contract kind."""

from __future__ import annotations

from collections.abc import Sequence

from honing.core import (
    ContractKind,
    DataContract,
    HarnessSpec,
    PolicyRef,
    PolicySample,
    PreferencePair,
    Task,
    TrainingBatch,
)


class ListDatasetProvider:
    """Returns the batch it holds for a contract's kind, or ``None`` when it holds none."""

    def __init__(self, batches: Sequence[TrainingBatch]) -> None:
        self._batches = {batch.contract.kind: batch for batch in batches}
        self.requests: list[DataContract] = []

    def next_batch(self, contract: DataContract) -> TrainingBatch | None:
        self.requests.append(contract)
        return self._batches.get(contract.kind)


class FakeRolloutProvider:
    """Samples a few rewarded, grouped sequences per task.

    Each sample is stamped with the release it was collected under, unless
    ``stamp_release`` overrides it, which is how a test builds a batch that
    claims to come from a different release.
    """

    def __init__(self, kind: ContractKind = "grouped_on_policy_samples", stamp_release: str | None = None) -> None:
        self.kind = kind
        self.stamp_release = stamp_release
        self.collected = 0

    def collect(
        self,
        tasks: Sequence[Task],
        policy: PolicyRef,
        harness: HarnessSpec | None,
        release_id: str | None,
        samples_per_task: int,
    ) -> TrainingBatch:
        self.collected += 1
        source_release = self.stamp_release if self.stamp_release is not None else release_id
        samples = tuple(
            PolicySample(
                tokens=(1, 2),
                loss_mask=(0, 1),
                reward=float(index),
                logprobs=(0.0, -0.5),
                group=task.task_id,
                source_release=source_release,
                source_harness=None if harness is None else harness.spec_id,
            )
            for task in tasks
            for index in range(samples_per_task)
        )
        contract = DataContract(kind=self.kind, requires_reward=True, requires_logprobs=True)
        return TrainingBatch(batch_id=f"rollout-{policy.checkpoint.checksum}", contract=contract, samples=samples)


def token_batch(batch_id: str = "corpus-0") -> TrainingBatch:
    samples = tuple(PolicySample(tokens=(1, 2, 3, index), loss_mask=(0, 1, 1, 1)) for index in range(4))
    return TrainingBatch(batch_id=batch_id, contract=DataContract(kind="token_sequences"), samples=samples)


def preference_batch(batch_id: str = "preferences-0") -> TrainingBatch:
    pair = PreferencePair(prompt=(1, 2), chosen=(3, 4), rejected=(3, 5))
    return TrainingBatch(batch_id=batch_id, contract=DataContract(kind="preference_pairs"), pairs=(pair,))
