"""Training data as the Train plane receives it: one batch type for every source.

An offline dataset and online feedback both produce a :class:`TrainingBatch`,
and the trainer cannot tell which one fed it. The difference is confined to the
:class:`DataContract` each algorithm declares and a provider must meet.

Samples may name the release and harness that produced them. Offline data names
neither; rollout data names both, and a batch that mixes releases or harnesses
is refused by ``honing.campaign.rules.batch_from_one_release``, because rewards
earned under different policies or different harnesses are not comparable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, get_args

from honing.core.errors import require, require_choice

ContractKind = Literal["token_sequences", "on_policy_samples", "grouped_on_policy_samples", "preference_pairs"]


@dataclass(frozen=True)
class DataContract:
    """What an algorithm needs from a batch."""

    kind: ContractKind
    requires_reward: bool = False
    requires_logprobs: bool = False

    def __post_init__(self) -> None:
        require_choice(self.kind, get_args(ContractKind), "DataContract.kind")


@dataclass(frozen=True)
class PolicySample:
    """One token sequence with its loss mask, and whatever the algorithm scores it by."""

    tokens: tuple[int, ...]
    loss_mask: tuple[int, ...]
    reward: float | None = None
    logprobs: tuple[float, ...] | None = None
    group: str | None = None
    source_release: str | None = None
    source_harness: str | None = None

    def __post_init__(self) -> None:
        require(len(self.tokens) > 0, "a sample needs at least one token")
        require(len(self.loss_mask) == len(self.tokens), "loss_mask must have one entry per token")
        require(set(self.loss_mask) <= {0, 1}, "loss_mask entries must be 0 or 1")
        if self.logprobs is not None:
            require(len(self.logprobs) == len(self.tokens), "logprobs must have one entry per token")
        if self.reward is not None:
            require(math.isfinite(self.reward), "reward must be finite")


@dataclass(frozen=True)
class PreferencePair:
    """A prompt with a preferred and a dispreferred response."""

    prompt: tuple[int, ...]
    chosen: tuple[int, ...]
    rejected: tuple[int, ...]

    def __post_init__(self) -> None:
        require(len(self.chosen) > 0 and len(self.rejected) > 0, "a preference pair needs both responses")
        require(self.chosen != self.rejected, "a preference pair needs two different responses")


@dataclass(frozen=True)
class TrainingBatch:
    """The unit a training job consumes."""

    batch_id: str
    contract: DataContract
    samples: tuple[PolicySample, ...] = ()
    pairs: tuple[PreferencePair, ...] = ()

    def __post_init__(self) -> None:
        require(bool(self.batch_id), "TrainingBatch.batch_id must be set")
        if self.contract.kind == "preference_pairs":
            require(len(self.pairs) > 0 and not self.samples, "a preference batch carries pairs and no samples")
            return
        require(len(self.samples) > 0 and not self.pairs, f"a {self.contract.kind} batch carries samples and no pairs")
        if self.contract.requires_reward:
            require(all(sample.reward is not None for sample in self.samples), "every sample needs a reward")
        if self.contract.requires_logprobs:
            require(all(sample.logprobs is not None for sample in self.samples), "every sample needs logprobs")
        if self.contract.kind == "grouped_on_policy_samples":
            require(all(sample.group is not None for sample in self.samples), "every grouped sample needs a group")

    @property
    def source_releases(self) -> frozenset[str]:
        """The releases the samples name. Unattributed (offline) samples contribute nothing."""
        return frozenset(sample.source_release for sample in self.samples if sample.source_release is not None)

    @property
    def source_harnesses(self) -> frozenset[str]:
        """The harness specs the samples name. Unattributed samples contribute nothing."""
        return frozenset(sample.source_harness for sample in self.samples if sample.source_harness is not None)
