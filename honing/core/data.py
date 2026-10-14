"""Training data as the Train plane receives it: one batch type for every source.

An offline dataset and online feedback both produce a :class:`TrainingBatch`,
and the trainer cannot tell which one fed it. The difference is confined to the
:class:`DataContract` each algorithm declares and a provider must meet.

Samples may name the release and harness that produced them. Offline data names
neither; rollout data names both, and a batch that mixes releases or harnesses
is refused by ``honing.campaign.rules.batch_from_one_release``, because rewards
earned under different policies or different harnesses are not comparable.

A decision model (a Jev-style System One model) learns from typed decisions, not
token sequences: the state, the questions asked about it, the options each
offered, and the option chosen. :class:`DecisionSample` carries one decision, and
a ``typed_decisions`` batch carries them, so data collection is the same whichever
trainer consumes it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, get_args

from honing.core.errors import require, require_choice

ContractKind = Literal[
    "token_sequences", "on_policy_samples", "grouped_on_policy_samples", "preference_pairs", "typed_decisions"
]
DecisionKind = Literal["choice", "score", "noul"]
PROBABILITY_TOLERANCE = 1e-3


@dataclass(frozen=True)
class DataContract:
    """What an algorithm needs from a batch.

    For a ``typed_decisions`` batch, ``requires_logprobs`` asks for each
    decision's behavior probabilities, the decision model's analogue of the
    log-probabilities a language model's rollout carries.
    ``requires_teacher_logprobs`` asks for a teacher's log-probabilities of
    every sampled token, scored when the rollout was collected, for on-policy
    distillation whose teacher is not loaded beside the student.
    """

    kind: ContractKind
    requires_reward: bool = False
    requires_logprobs: bool = False
    requires_group: bool = False
    requires_teacher_logprobs: bool = False

    def __post_init__(self) -> None:
        require_choice(self.kind, get_args(ContractKind), "DataContract.kind")


@dataclass(frozen=True)
class PolicySample:
    """One token sequence with its loss mask, and whatever the algorithm scores it by.

    ``logprobs`` are the sampling policy's log-probabilities of each token.
    ``teacher_logprobs`` are a teacher's log-probabilities of the same tokens,
    for on-policy distillation; in on-policy self-distillation the teacher is
    the same model shown privileged context (the answer, or a hint) that the
    student's prompt leaves out. Positions the loss mask excludes carry 0.0 in both.
    """

    tokens: tuple[int, ...]
    loss_mask: tuple[int, ...]
    reward: float | None = None
    logprobs: tuple[float, ...] | None = None
    group: str | None = None
    source_release: str | None = None
    source_harness: str | None = None
    teacher_logprobs: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        require(len(self.tokens) > 0, "a sample needs at least one token")
        require(len(self.loss_mask) == len(self.tokens), "loss_mask must have one entry per token")
        require(set(self.loss_mask) <= {0, 1}, "loss_mask entries must be 0 or 1")
        if self.logprobs is not None:
            require(len(self.logprobs) == len(self.tokens), "logprobs must have one entry per token")
        if self.teacher_logprobs is not None:
            require(len(self.teacher_logprobs) == len(self.tokens), "teacher_logprobs must have one entry per token")
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
class DecisionQuestion:
    """One typed question about a state, with the options it offers.

    ``kind`` follows the three question forms a System One model answers:
    ``choice`` picks one option, ``score`` rates against levels, and ``noul``
    is a yes-or-no evaluation. ``options`` pairs each option's key with its label.
    """

    kind: DecisionKind
    instructions: str
    options: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        require_choice(self.kind, get_args(DecisionKind), "DecisionQuestion.kind")
        require(bool(self.instructions), "a decision question needs instructions")
        keys = [key for key, _ in self.options]
        require(len(keys) >= 2, "a decision question offers at least two options")
        require(len(set(keys)) == len(keys), "a decision question names each option once")
        require(all(key and label for key, label in self.options), "every option needs a key and a label")

    @property
    def option_keys(self) -> tuple[str, ...]:
        return tuple(key for key, _ in self.options)


@dataclass(frozen=True)
class DecisionSample:
    """One decision a model made, or was taught to make, about one state.

    ``parts`` is the state as named sections of text lines. Every question is
    answered in ``chosen``. ``behavior_probs``, when present, is the probability
    the acting model gave every option of every question; policy-gradient
    training needs it, supervised training does not. ``reward`` is what the
    decision earned, for example the game's outcome.
    """

    parts: tuple[tuple[str, tuple[str, ...]], ...]
    questions: tuple[tuple[str, DecisionQuestion], ...]
    chosen: tuple[tuple[str, str], ...]
    behavior_probs: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] | None = None
    reward: float | None = None
    group: str | None = None
    source_release: str | None = None
    source_harness: str | None = None

    def __post_init__(self) -> None:
        sections = [name for name, _ in self.parts]
        require(len(set(sections)) == len(sections), "a decision's state names each section once")
        keys = [key for key, _ in self.questions]
        require(len(keys) > 0, "a decision answers at least one question")
        require(len(set(keys)) == len(keys), "a decision names each question once")
        by_key = dict(self.questions)
        chosen = dict(self.chosen)
        require(len(chosen) == len(self.chosen), "a decision chooses once per question")
        require(set(chosen) == set(keys), "a decision chooses an option for every question it asks")
        for key, option in chosen.items():
            require(option in by_key[key].option_keys, f"question {key!r} offers no option {option!r} to choose")
        if self.behavior_probs is not None:
            probs = dict(self.behavior_probs)
            require(set(probs) == set(keys), "behavior probabilities cover every question")
            for key, distribution in probs.items():
                values = dict(distribution)
                require(
                    set(values) == set(by_key[key].option_keys),
                    f"behavior probabilities for {key!r} cover exactly its options",
                )
                require(
                    all(math.isfinite(p) and 0.0 <= p <= 1.0 for p in values.values()),
                    f"behavior probabilities for {key!r} are in [0, 1]",
                )
                require(
                    abs(sum(values.values()) - 1.0) <= PROBABILITY_TOLERANCE,
                    f"behavior probabilities for {key!r} sum to 1",
                )
        if self.reward is not None:
            require(math.isfinite(self.reward), "reward must be finite")


@dataclass(frozen=True)
class TrainingBatch:
    """The unit a training job consumes."""

    batch_id: str
    contract: DataContract
    samples: tuple[PolicySample, ...] = ()
    pairs: tuple[PreferencePair, ...] = ()
    decisions: tuple[DecisionSample, ...] = ()

    def __post_init__(self) -> None:
        require(bool(self.batch_id), "TrainingBatch.batch_id must be set")
        kind = self.contract.kind
        if kind == "preference_pairs":
            require(
                len(self.pairs) > 0 and not self.samples and not self.decisions,
                "a preference batch carries pairs only",
            )
            return
        if kind == "typed_decisions":
            require(
                len(self.decisions) > 0 and not self.samples and not self.pairs,
                "a typed_decisions batch carries decisions only",
            )
            if self.contract.requires_reward:
                require(all(d.reward is not None for d in self.decisions), "every decision needs a reward")
            if self.contract.requires_logprobs:
                require(
                    all(d.behavior_probs is not None for d in self.decisions),
                    "every decision needs its behavior probabilities",
                )
            if self.contract.requires_group:
                require(all(d.group is not None for d in self.decisions), "every decision needs a group")
            return
        require(
            len(self.samples) > 0 and not self.pairs and not self.decisions,
            f"a {kind} batch carries samples only",
        )
        if self.contract.requires_reward:
            require(all(sample.reward is not None for sample in self.samples), "every sample needs a reward")
        if self.contract.requires_logprobs:
            require(all(sample.logprobs is not None for sample in self.samples), "every sample needs logprobs")
        if self.contract.requires_teacher_logprobs:
            require(
                all(sample.teacher_logprobs is not None for sample in self.samples),
                "every sample needs teacher_logprobs",
            )
        if kind == "grouped_on_policy_samples" or self.contract.requires_group:
            require(all(sample.group is not None for sample in self.samples), "every grouped sample needs a group")

    @property
    def source_releases(self) -> frozenset[str]:
        """The releases the samples name. Unattributed (offline) samples contribute nothing."""
        items: tuple[PolicySample | DecisionSample, ...] = (*self.samples, *self.decisions)
        return frozenset(item.source_release for item in items if item.source_release is not None)

    @property
    def source_harnesses(self) -> frozenset[str]:
        """The harness specs the samples name. Unattributed samples contribute nothing."""
        items: tuple[PolicySample | DecisionSample, ...] = (*self.samples, *self.decisions)
        return frozenset(item.source_harness for item in items if item.source_harness is not None)
