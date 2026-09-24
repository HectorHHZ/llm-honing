"""How far apart a candidate and the incumbent are, task by task, and how sure the gate can be.

A gate that compares two means cannot tell a real difference from noise once
the difference gets small, which is where a long training run spends most of
its steps. Comparing each task's pair of scores and bootstrapping the mean of
the differences gives the gate an interval to report next to the mean.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import fmean

from honing.core import EvaluationResult, require

GATE_METRIC_NAMES = (
    "gate_tasks",
    "gate_candidate_mean",
    "gate_incumbent_mean",
    "gate_difference_low",
    "gate_difference_high",
)


@dataclass(frozen=True)
class PairedDifference:
    """The mean of candidate minus incumbent over paired tasks, with a bootstrap interval."""

    tasks: int
    mean: float
    low: float
    high: float


def paired_difference(
    evaluation: EvaluationResult, *, resamples: int = 1000, confidence: float = 0.95, seed: int = 0
) -> PairedDifference:
    """Bootstrap the mean per-task difference. The same inputs always give the same interval."""
    require(resamples > 0, f"resamples must be > 0, got {resamples}")
    require(0.0 < confidence < 1.0, f"confidence must be in (0, 1), got {confidence}")
    differences = [
        candidate - incumbent
        for candidate, incumbent in zip(evaluation.candidate_scores, evaluation.incumbent_scores, strict=True)
    ]
    count = len(differences)
    rng = random.Random(seed)
    means = sorted(fmean(rng.choices(differences, k=count)) for _ in range(resamples))
    tail = (1.0 - confidence) / 2.0
    low_index = min(resamples - 1, int(tail * resamples))
    high_index = max(0, min(resamples - 1, int((1.0 - tail) * resamples) - 1))
    return PairedDifference(tasks=count, mean=fmean(differences), low=means[low_index], high=means[high_index])


def gate_metrics(evaluation: EvaluationResult) -> dict[str, float]:
    """What every gated release records about the evaluation that settled it."""
    difference = paired_difference(evaluation)
    return {
        "gate_tasks": float(difference.tasks),
        "gate_candidate_mean": evaluation.candidate_mean,
        "gate_incumbent_mean": evaluation.incumbent_mean,
        "gate_difference_low": difference.low,
        "gate_difference_high": difference.high,
    }
