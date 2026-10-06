"""Rollout samples in the layout Slime's train-only path reads.

Slime trains on rollouts it did not generate with ``--load-debug-rollout-data``:
a ``torch.save`` file per rollout id holding ``{"rollout_id": i, "samples":
[Sample.to_dict(), ...]}`` (``slime/ray/rollout.py`` at the pinned commit). A
:class:`~honing.core.PolicySample` maps onto one Slime ``Sample`` with two
differences in layout:

- Slime splits a sequence into a prompt and a response. Its ``loss_mask`` and
  ``rollout_log_probs`` cover only the response, ``response_length`` tokens at
  the end of ``tokens``. A ``PolicySample`` masks every token. The prompt here is
  everything before the first trained token, so the tokens of a multi-turn
  session that the model did not write (tool results, user turns) stay inside
  the response with a mask of 0, as Slime's own agent rollouts record them.
- Slime's GRPO normalizes rewards over ``n_samples_per_prompt`` consecutive
  samples. Samples are therefore written group by group, and every group must
  have the same size; with unequal groups Slime would normalize over the whole
  batch instead.

Honing never imports torch. :func:`write_samples` writes the sample dicts as
JSON, and ``honing/backends/slime/worker.py``, run inside Slime's image, saves
them with ``torch.save``. Only these files and checkpoint paths cross between
the two environments.

The log-probabilities of prompt tokens are not part of Slime's layout. Reading a
file back gives 0.0 for them, so a ``PolicySample`` round-trips exactly when its
untrained positions carry 0.0, which is what a rollout provider should record.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from honing.core import ContractError, PolicySample, TrainingBatch

COMPLETED = "completed"


@dataclass(frozen=True)
class RolloutLayout:
    """How a batch's samples fill one Slime rollout: ``prompts`` groups of ``samples_per_prompt``."""

    prompts: int
    samples_per_prompt: int

    @property
    def samples(self) -> int:
        return self.prompts * self.samples_per_prompt


def _prompt_length(sample: PolicySample) -> int:
    if 1 not in sample.loss_mask:
        raise ContractError("a sample with no trained token gives Slime nothing to learn from")
    return sample.loss_mask.index(1)


def grouped(samples: Sequence[PolicySample]) -> tuple[tuple[PolicySample, ...], ...]:
    """The samples in groups of equal size, in the order each group first appears.

    A sample without a group is its own group, as Slime treats SFT data.
    """
    groups: dict[str, list[PolicySample]] = {}
    for index, sample in enumerate(samples):
        key = sample.group if sample.group is not None else f"\0{index}"
        groups.setdefault(key, []).append(sample)
    sizes = {len(members) for members in groups.values()}
    if len(sizes) != 1:
        counts = {key: len(members) for key, members in groups.items()}
        raise ContractError(f"Slime normalizes rewards over groups of one size; the batch has {counts}")
    return tuple(tuple(members) for members in groups.values())


def slime_samples(batch: TrainingBatch, *, rewarded: bool = True) -> tuple[list[dict[str, Any]], RolloutLayout]:
    """The batch's samples as Slime sample dicts, group by group, and the rollout layout they fill.

    With ``rewarded=False`` (SFT, on-policy distillation) every sample is its own group and its reward is
    written as 0.0, so Slime adds no task advantage even when the batch carries groups and rewards.
    """
    if not batch.samples:
        raise ContractError(f"Slime trains on token samples; batch {batch.batch_id} has none")
    groups = grouped(batch.samples) if rewarded else tuple((sample,) for sample in batch.samples)
    rows: list[dict[str, Any]] = []
    for group_index, members in enumerate(groups):
        for sample in members:
            start = _prompt_length(sample)
            # Slime post-processes every sample's reward, even for SFT, so a missing reward is written as 0.0
            # and the original is kept in the metadata for the round trip.
            metadata: dict[str, Any] = {"batch_id": batch.batch_id, "reward": sample.reward}
            if sample.group is not None:
                metadata["group"] = sample.group
            if sample.source_release is not None:
                metadata["source_release"] = sample.source_release
            if sample.source_harness is not None:
                metadata["source_harness"] = sample.source_harness
            rows.append(
                {
                    "group_index": group_index,
                    "index": len(rows),
                    "tokens": list(sample.tokens),
                    "response_length": len(sample.tokens) - start,
                    "loss_mask": list(sample.loss_mask[start:]),
                    "reward": sample.reward if rewarded and sample.reward is not None else 0.0,
                    "rollout_log_probs": None if sample.logprobs is None else list(sample.logprobs[start:]),
                    "status": COMPLETED,
                    "metadata": metadata,
                }
            )
    return rows, RolloutLayout(prompts=len(groups), samples_per_prompt=len(groups[0]))


def policy_sample(row: Mapping[str, Any]) -> PolicySample:
    """One Slime sample dict read back as a PolicySample; prompt positions get mask 0 and log-probability 0.0."""
    tokens = tuple(int(token) for token in row["tokens"])
    response_length = int(row["response_length"])
    prompt = len(tokens) - response_length
    if response_length <= 0 or prompt < 0:
        raise ContractError(
            f"sample {row.get('index')} has response length {response_length} for {len(tokens)} tokens"
        )
    mask = row.get("loss_mask")
    response_mask = [1] * response_length if mask is None else [int(value) for value in mask]
    logprobs = row.get("rollout_log_probs")
    metadata = row.get("metadata") or {}
    return PolicySample(
        tokens=tokens,
        loss_mask=(0,) * prompt + tuple(response_mask),
        reward=_reward(row, metadata),
        logprobs=None if logprobs is None else (0.0,) * prompt + tuple(float(value) for value in logprobs),
        group=metadata.get("group"),
        source_release=metadata.get("source_release"),
        source_harness=metadata.get("source_harness"),
    )


def _reward(row: Mapping[str, Any], metadata: Mapping[str, Any]) -> float | None:
    value = metadata["reward"] if "reward" in metadata else row.get("reward")
    return None if value is None else float(value)


def write_samples(path: Path, rows: Sequence[Mapping[str, Any]], rollout_id: int = 0) -> Path:
    """Write one rollout's sample dicts as JSON, for the worker to save in Slime's format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"rollout_id": rollout_id, "samples": list(rows)}) + "\n", encoding="utf-8")
    return path


def read_samples(path: Path) -> tuple[PolicySample, ...]:
    """Read a rollout written by :func:`write_samples` (or saved from it as JSON) back as PolicySamples."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(policy_sample(row) for row in payload["samples"])
