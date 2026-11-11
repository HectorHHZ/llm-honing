"""How far apart a candidate and the incumbent are, task by task, and how sure the gate can be.

A gate that compares two means cannot tell a real difference from noise once
the difference gets small, which is where a long training run spends most of
its steps. Comparing each task's pair of scores and bootstrapping the mean of
the differences gives the gate an interval to report next to the mean. A
selector that decides on an interval uses the normal interval instead, because
a sequential gate needs tail levels that a thousand resamples cannot resolve.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import sqrt
from statistics import NormalDist, fmean, stdev

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
    """The mean of candidate minus incumbent over paired tasks, with an interval on that mean."""

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


def normal_paired_difference(evaluation: EvaluationResult, *, confidence: float) -> PairedDifference:
    """The mean per-task difference plus or minus ``z`` standard errors, two-sided at ``confidence``.

    Every score is in ``[0, 1]``, so each difference is in ``[-1, 1]`` and the
    mean of a few dozen of them is close to normal. With fewer tasks the
    interval is too narrow, as any large-sample interval is.
    """
    require(0.0 < confidence < 1.0, f"confidence must be in (0, 1), got {confidence}")
    require(len(evaluation.task_ids) >= 2, "a normal interval needs at least two tasks to estimate the spread")
    differences = [
        candidate - incumbent
        for candidate, incumbent in zip(evaluation.candidate_scores, evaluation.incumbent_scores, strict=True)
    ]
    mean = fmean(differences)
    z = NormalDist().inv_cdf((1.0 + confidence) / 2.0)
    half_width = z * stdev(differences, mean) / sqrt(len(differences))
    return PairedDifference(tasks=len(differences), mean=mean, low=mean - half_width, high=mean + half_width)


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
