"""What every local decision-model trainer shares: the examples, the validation split, the losses and the loops.

The Laya (:mod:`honing.backends.jev.trainer`), NanoJev
(:mod:`honing.backends.jev.nanojev_trainer`) and Kev
(:mod:`honing.backends.jev.kev_trainer`) trainers differ only in how they load
a checkpoint, encode a question, build an optimizer and save weights. The rest
is here, so the three train the same way.

Examples
    Every question of every decision is one example, encoded exactly as
    inference encodes it (:func:`decision_examples`). Identical inputs are
    deduplicated and must agree on their label; examples from validation groups
    are held out and must not also appear in training (:func:`separate_unique`,
    :func:`deduplicated_split`).

``sft``
    Cross-entropy over the option logits, gradients clipped at 1.0, and the
    weights saved whenever the validation loss improves, so the checkpoint with
    the best validation loss is kept (:func:`supervised_epochs`).

``grpo``
    Policy gradient over the option distributions with a group baseline. Each
    decision was made by the policy being trained (on-policy rollouts) and
    earned a reward, for example its match's result. Decisions in the same group
    (repeats of one scenario) share a baseline: a decision's advantage is its
    reward minus the group's mean reward, divided by the group's standard
    deviation (:func:`group_advantages`). A group whose rewards are all equal
    teaches nothing and is skipped.

    The update maximizes the clipped surrogate ``min(r * A, clip(r, 1 - eps, 1 + eps) * A)``
    per question, where ``r`` is the chosen option's probability under the current
    weights divided by its probability under the weights the update started from
    (:func:`policy_gradient_epochs`). Those starting probabilities are recomputed
    from the model before the first step (:func:`record_start`), not read from the
    rollout: the rollout's probabilities come from inference, which may apply
    temperature calibration that training does not, and would bias ``r``. The
    rollout's probabilities are reported for comparison. The final weights are kept.

The loops take the model, its optimizer, and a :data:`Forward` that turns a
chunk of examples into option logits. They run on whichever device the logits
are on, so the CPU tests exercise them with a small model.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from honing.core import ContractError, DecisionSample, TrainingBatch

Example = dict[str, Any]
Forward = Callable[[Sequence[Example]], tuple[Any, Any, Any]]
"""A chunk of examples to its option logits (float, ``[N, K]``, absent options at -1e4), the chosen options'
indices (``[N]``), and an auxiliary model output that must stay in the graph, added to the loss with weight
zero (Laya's head returns one), or None."""
Pack = Callable[[dict[str, list[str]], dict[str, dict[str, Any]]], tuple[Any, Mapping[str, Mapping[str, Any]], Any]]
"""A decision's state sections and questions to the packed state, a per-question token audit carrying
``inputTokens``, and what packing trimmed, as inference packs them."""
Encode = Callable[[Any, dict[str, dict[str, Any]], dict[str, str]], Iterable[tuple[str, Example]]]
"""A packed state, its questions and the chosen options to ``(question key, example)`` pairs. An example holds
the ``ids`` and ``markers`` that identify its input (:func:`identity`) and its ``label``, the chosen option's
index."""


def laya_question(question: Any) -> dict[str, Any]:
    """A DecisionQuestion as the question object a decision model reads (Laya's format, which the others share)."""
    if question.kind != "choice":
        raise ContractError(f"the decision-model trainers handle choice questions only, not {question.kind}")
    return {"type": "choice", "instructions": question.instructions, "criteria": dict(question.options)}


def identity(item: Mapping[str, Any]) -> str:
    """One encoded model input: its token ids and option markers."""
    value = json.dumps([item["ids"], item["markers"]], separators=(",", ":"))
    return hashlib.sha256(value.encode()).hexdigest()


def separate_unique(
    training: Sequence[dict[str, Any]], validation: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Deduplicate inputs within each split, and keep only validation inputs training never saw."""

    def unique(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            key = identity(row)
            if key in result and result[key]["label"] != row["label"]:
                raise ContractError("Conflicting labels for the same complete model input")
            result.setdefault(key, row)
        return result

    train, valid = unique(training), unique(validation)
    overlap = set(train) & set(valid)
    for key in overlap:
        if train[key]["label"] != valid[key]["label"]:
            raise ContractError("Conflicting labels across episode splits")
    novel = [row for key, row in valid.items() if key not in train]
    audit = {
        "rawTraining": len(training),
        "rawValidation": len(validation),
        "uniqueTraining": len(train),
        "uniqueValidation": len(valid),
        "removedOverlap": len(overlap),
        "novelValidation": len(novel),
    }
    return list(train.values()), novel, audit


def deduplicated_split(
    items: Sequence[Example], validation: Sequence[Example], min_validation: int
) -> tuple[list[Example], list[Example], dict[str, int]]:
    """:func:`separate_unique`, refusing a split with no training examples or too few novel validation ones."""
    train, novel, audit = separate_unique(items, validation)
    if not train or len(novel) < min_validation:
        raise ContractError(
            f"need at least {min_validation} novel validation examples after deduplication, got {len(novel)}"
        )
    return train, novel, audit


def group_advantages(decisions: Sequence[DecisionSample], eps: float = 1e-6) -> list[float]:
    """Each decision's reward minus its group's mean, over the group's standard deviation; 0 when all agree."""
    rewards: dict[str, list[float]] = defaultdict(list)
    for decision in decisions:
        if decision.group is None or decision.reward is None:
            raise ContractError("policy gradient needs a group and a reward on every decision")
        rewards[decision.group].append(decision.reward)
    stats = {}
    for group, values in rewards.items():
        mean = statistics.fmean(values)
        spread = statistics.pstdev(values) if len(values) > 1 else 0.0
        stats[group] = (mean, spread)
    advantages = []
    for decision in decisions:
        mean, spread = stats[decision.group]  # type: ignore[index]
        advantages.append(0.0 if spread <= eps else (decision.reward - mean) / spread)  # type: ignore[operator]
    if not all(math.isfinite(a) for a in advantages):
        raise ContractError("nonfinite advantage")
    return advantages


def decision_examples(
    decisions: Sequence[DecisionSample],
    *,
    pack: Pack,
    encode: Encode,
    validation_groups: str,
    advantages: Sequence[float] | None = None,
) -> tuple[list[Example], list[Example], dict[str, int]]:
    """Every question of every decision as one example: training examples, validation examples, packing audit.

    Without ``advantages`` (``sft``), decisions whose group matches
    ``validation_groups`` go to validation. With them (``grpo``), every
    example trains, decisions with a zero advantage are skipped, and each
    example also carries its decision's ``advantage`` and the rollout's
    ``behavior`` probability of the chosen option (None when the rollout did
    not record one); a batch that leaves nothing to learn from is refused.
    """
    held_out = re.compile(validation_groups)
    items: list[Example] = []
    validation: list[Example] = []
    packing = {"maxInputTokens": 0, "trimmedRequests": 0}
    for index, decision in enumerate(decisions):
        if advantages is not None and advantages[index] == 0.0:
            continue
        parts = {name: list(lines) for name, lines in decision.parts}
        questions = {key: laya_question(q) for key, q in decision.questions}
        state, audit, trimmed = pack(parts, questions)
        packing["maxInputTokens"] = max(packing["maxInputTokens"], max(v["inputTokens"] for v in audit.values()))
        packing["trimmedRequests"] += bool(trimmed)
        chosen = dict(decision.chosen)
        behavior = dict(decision.behavior_probs or ())
        target = (
            validation
            if advantages is None and decision.group is not None and held_out.search(decision.group)
            else items
        )
        for key, item in encode(state, questions, chosen):
            if advantages is not None:
                item["advantage"] = advantages[index]
                item["behavior"] = dict(behavior.get(key, ())).get(chosen[key])
            target.append(item)
    if advantages is not None and not items:
        raise ContractError("every group's rewards are equal; there is nothing to learn from this batch")
    return items, validation, packing


def _step(torch: Any, model: Any, optimizer: Any, schedule: Any, loss: Any) -> None:
    """Backpropagate, clip the trainable parameters' gradients at 1.0, and step the optimizer and schedule."""
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    optimizer.step()
    if schedule is not None:
        schedule.step()


@dataclass(frozen=True)
class Supervised:
    """What a supervised run measured: validation loss and accuracy before training, the best validation loss,
    and one entry per epoch."""

    initial: tuple[float, float]
    best: float
    history: list[dict[str, Any]]

    def record(
        self, items: Sequence[Example], validation: Sequence[Example], overlap: Mapping[str, int]
    ) -> tuple[dict[str, Any], dict[str, float]]:
        """The run's fields for the checkpoint manifest, and its TrainResult metrics."""
        manifest = {
            "dataOverlapAudit": dict(overlap),
            "trainExamples": len(items),
            "validationExamples": len(validation),
            "initialValidation": self.initial,
        }
        metrics = {
            "initial_validation_loss": self.initial[0],
            "validation_loss": self.best,
            "final_validation_accuracy": self.history[-1]["validationAccuracy"],
            "train_examples": float(len(items)),
            "validation_examples": float(len(validation)),
        }
        return manifest, metrics


def supervised_epochs(
    torch: Any,
    model: Any,
    optimizer: Any,
    items: list[Example],
    validation: Sequence[Example],
    *,
    forward: Forward,
    epochs: int,
    batch_size: int,
    seed: int,
    save: Callable[[], None],
    schedule: Any = None,
) -> Supervised:
    """``sft``: ``epochs`` passes of cross-entropy over ``items``, calling ``save`` whenever validation improves.

    ``items`` is shuffled in place, each epoch with its own seed. ``trainLoss``
    is the mean over the epoch's examples.
    """

    def evaluate() -> tuple[float, float]:
        model.eval()
        losses: list[float] = []
        correct = 0
        with torch.no_grad():
            for start in range(0, len(validation), batch_size):
                logits, labels, _ = forward(validation[start : start + batch_size])
                losses.extend(torch.nn.functional.cross_entropy(logits, labels, reduction="none").tolist())
                correct += int((logits.argmax(-1) == labels).sum())
        return sum(losses) / len(losses), correct / len(validation)

    initial = evaluate()
    best, history = float("inf"), []
    for epoch in range(epochs):
        random.Random(seed + epoch).shuffle(items)
        model.train()
        total, count = 0.0, 0
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits, labels, aux = forward(chunk)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            if aux is not None:
                loss = loss + 0 * aux.float().sum()
            if not torch.isfinite(loss):
                raise ContractError("nonfinite training loss")
            _step(torch, model, optimizer, schedule, loss)
            total += float(loss.detach()) * len(chunk)
            count += len(chunk)
        val, accuracy = evaluate()
        history.append(
            {"epoch": epoch + 1, "trainLoss": total / count, "validationLoss": val, "validationAccuracy": accuracy}
        )
        if val < best:
            best = val
            save()
    return Supervised(initial, best, history)


def chosen_log_probs(torch: Any, logits: Any, labels: Any) -> Any:
    """The log-probability of each example's chosen option."""
    return torch.log_softmax(logits, dim=-1).gather(1, labels[:, None]).squeeze(1)


def record_start(
    torch: Any, model: Any, examples: Sequence[Example], *, forward: Forward, batch_size: int
) -> float | None:
    """Store each example's chosen-option log-probability under the starting weights as ``old``.

    Returns the mean absolute gap between that probability and the rollout's
    behavior probability, over the examples that recorded one, or None.
    """
    model.eval()
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            logits, labels, _ = forward(chunk)
            for item, value in zip(chunk, chosen_log_probs(torch, logits, labels).tolist(), strict=True):
                item["old"] = value
    gaps = [abs(math.exp(x["old"]) - x["behavior"]) for x in examples if x["behavior"] is not None]
    return statistics.fmean(gaps) if gaps else None


def policy_gradient_epochs(
    torch: Any,
    model: Any,
    optimizer: Any,
    examples: list[Example],
    *,
    forward: Forward,
    epochs: int,
    batch_size: int,
    seed: int,
    clip_ratio: float,
    schedule: Any = None,
) -> list[dict[str, Any]]:
    """``grpo``: ``epochs`` passes of the clipped surrogate over ``examples``, after :func:`record_start`.

    ``examples`` is shuffled in place, each epoch with its own seed. Each
    epoch's entry holds the mean loss, clipped fraction, approximate KL to the
    starting weights, and probability ratio over its examples.
    """
    eps = clip_ratio
    history = []
    for epoch in range(epochs):
        random.Random(seed + epoch).shuffle(examples)
        model.train()
        totals = {"loss": 0.0, "clipped": 0.0, "kl": 0.0, "ratio": 0.0}
        count = 0
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits, labels, aux = forward(chunk)
            log_probs = chosen_log_probs(torch, logits, labels)
            old = torch.tensor([x["old"] for x in chunk], device=log_probs.device)
            advantage = torch.tensor([x["advantage"] for x in chunk], device=log_probs.device)
            ratio = torch.exp(log_probs - old)
            surrogate = torch.minimum(ratio * advantage, torch.clamp(ratio, 1 - eps, 1 + eps) * advantage)
            loss = -surrogate.mean()
            if aux is not None:
                loss = loss + 0 * aux.float().sum()
            if not torch.isfinite(loss):
                raise ContractError("nonfinite policy-gradient loss")
            _step(torch, model, optimizer, schedule, loss)
            n = len(chunk)
            totals["loss"] += float(loss.detach()) * n
            totals["clipped"] += float(((ratio - 1).abs() > eps).float().sum())
            totals["kl"] += float((old - log_probs.detach()).sum())
            totals["ratio"] += float(ratio.detach().sum())
            count += n
        history.append({"epoch": epoch + 1, **{k: v / count for k, v in totals.items()}})
    return history


def policy_gradient_record(
    batch: TrainingBatch,
    advantages: Sequence[float],
    examples: Sequence[Example],
    history: Sequence[Mapping[str, Any]],
    behavior_gap: float | None,
) -> tuple[dict[str, Any], dict[str, float]]:
    """A ``grpo`` run's fields for the checkpoint manifest, and its TrainResult metrics."""
    skipped = sum(1 for a in advantages if a == 0.0)
    final = history[-1]
    manifest = {
        "algorithm": "grpo",
        "sourceReleases": sorted(batch.source_releases),
        "sourceHarnesses": sorted(batch.source_harnesses),
        "decisions": len(batch.decisions),
        "decisionsWithoutSignal": skipped,
        "examples": len(examples),
        "meanBehaviorGap": behavior_gap,
    }
    metrics = {
        "policy_loss": final["loss"],
        "approx_kl": final["kl"],
        "clipped_fraction": final["clipped"],
        "examples": float(len(examples)),
        "decisions_without_signal": float(skipped),
    }
    return manifest, metrics
