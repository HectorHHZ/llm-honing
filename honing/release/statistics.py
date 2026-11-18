"""How far apart a candidate and the incumbent are, task by task, and how sure the gate can be.

A gate that compares two means cannot tell a real difference from noise once
the difference gets small, which is where a long training run spends most of
its steps. Comparing each task's pair of scores and bootstrapping the mean of
the differences gives the gate an interval to report next to the mean. A
selector that decides on an interval uses the betting interval instead. Its
error bound holds at any number of tasks and for any differences in
``[-1, 1]``, and it reaches the small tail levels a sequential gate's looks
need, which a thousand resamples cannot resolve.
"""

from __future__ import annotations

import math
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

#: The largest share of the capital a betting interval stakes on one task. Below 1, no task can empty it.
#: Waudby-Smith and Ramdas suggest 1/2 or 3/4. With a paired standard deviation of 0.3, 3/4 needs about 10%
#: fewer rounds than 1/2 for the same power, and at 0.5 or more the stakes rarely reach either cap.
MAX_BET = 0.75
#: How far below the exact end of a betting interval the reported end may be.
BOUND_TOLERANCE = 1e-6


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


def betting_paired_difference(evaluation: EvaluationResult, *, confidence: float) -> PairedDifference:
    """An interval on the mean per-task difference that holds at ``confidence`` for any differences in ``[-1, 1]``.

    Every score is in ``[0, 1]``, so every difference is in ``[-1, 1]``, and
    that bound is all the interval assumes. Each end is a betting test
    (Waudby-Smith and Ramdas, 2023). To test a mean ``m``, a gambler starts
    with one unit and on task ``i`` stakes a share ``b`` of the capital on the
    difference being above ``m``, which multiplies the capital by
    ``1 + b (d_i - m)``. ``b`` depends only on the tasks before ``i``, so if the
    true mean is ``m`` or less, each factor's expectation given the tasks
    before it is at most 1. Ville's inequality then bounds the chance that the
    capital ever reaches ``2 / (1 - confidence)`` by ``(1 - confidence) / 2``.
    Every ``m`` whose capital reaches that level is ruled out, and the low end
    is the largest such ``m``. The high end bets the other way. No normal
    approximation is involved, so the bound holds for skewed differences and
    for a handful of tasks alike.

    The tasks must be independent, in an order fixed before they are scored,
    and their number must be fixed in advance, because the bets are tuned to
    it. Equal differences on every task give an interval whose width depends
    only on how many tasks there are.
    """
    require(0.0 < confidence < 1.0, f"confidence must be in (0, 1), got {confidence}")
    differences = [
        candidate - incumbent
        for candidate, incumbent in zip(evaluation.candidate_scores, evaluation.incumbent_scores, strict=True)
    ]
    log_level = math.log(2.0 / (1.0 - confidence))
    bets = _bets(differences, log_level)
    low = _low_end(differences, bets, log_level)
    high = -_low_end([-difference for difference in differences], bets, log_level)
    return PairedDifference(tasks=len(differences), mean=fmean(differences), low=low, high=high)


def _bets(differences: list[float], log_level: float) -> list[float]:
    """Each task's stake before the cap, from the tasks before it only.

    ``sqrt(2 log_level / (n s2))``, with ``s2`` the spread seen so far, makes
    the capital grow fastest at the mean the bound should just exclude after
    ``n`` tasks (Waudby-Smith and Ramdas's fixed-time choice). ``s2`` starts as
    one imagined task at a spread of 1, the largest a difference in ``[-1, 1]``
    can have, so the first stakes are small.
    """
    bets = []
    total, squares = 0.0, 1.0
    for count, difference in enumerate(differences, start=1):
        bets.append(math.sqrt(2.0 * log_level * count / (len(differences) * squares)))
        total += difference
        squares += (difference - total / (count + 1)) ** 2
    return bets


def _low_end(differences: list[float], bets: list[float], log_level: float) -> float:
    """The largest mean the bets above it rule out, less at most ``BOUND_TOLERANCE``, or -1 if they rule out none.

    The capital falls as the tested mean rises, so the end is the one root of
    the log capital less ``log_level`` on ``[-1, 1]``. The Illinois variant of
    regula falsi finds it in about seven evaluations where bisection needs 21.
    It keeps the root bracketed and returns the side the bets rule out, so the
    end it returns is never above the exact one.
    """
    low, high = -1.0, 1.0
    excess_low = _log_capital(differences, bets, low) - log_level
    if excess_low < 0.0:
        return low
    excess_high = _log_capital(differences, bets, high) - log_level
    kept = 0
    while high - low > BOUND_TOLERANCE:
        middle = (low * excess_high - high * excess_low) / (excess_high - excess_low)
        if not low < middle < high:
            middle = (low + high) / 2.0
        excess = _log_capital(differences, bets, middle) - log_level
        # Halving the value at an end kept twice in a row moves the next try toward it, so both ends close in.
        if excess >= 0.0:
            low, excess_low = middle, excess
            excess_high = excess_high / 2.0 if kept == 1 else excess_high
            kept = 1
        else:
            high, excess_high = middle, excess
            excess_low = excess_low / 2.0 if kept == -1 else excess_low
            kept = -1
    return low


def _log_capital(differences: list[float], bets: list[float], mean: float) -> float:
    """The log of the capital from betting that the differences are above ``mean``.

    A difference can fall ``1 + mean`` below ``mean``, so a stake of at most
    ``MAX_BET / (1 + mean)`` never loses more than ``MAX_BET`` of the capital.
    """
    cap = MAX_BET / (1.0 + mean) if mean > -1.0 else math.inf
    return sum(
        math.log1p(min(bet, cap) * (difference - mean)) for difference, bet in zip(differences, bets, strict=True)
    )


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
