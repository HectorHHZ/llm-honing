"""How far apart a candidate and the incumbent are, task by task, and how sure the gate can be.

A gate that compares two means cannot tell a real difference from noise once
the difference gets small, which is where a long training run spends most of
its steps. Comparing each task's pair of scores and bootstrapping the mean of
the differences gives the gate an interval to report next to the mean. A
selector that decides on an interval uses the betting interval instead. Its
error bound holds for any differences in ``[-1, 1]`` and at every number of
tasks at once, so a gate can look after each block of tasks at the full
level.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from itertools import accumulate, chain, repeat
from operator import mul, sub, truediv
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
    differences = _differences(evaluation)
    count = len(differences)
    rng = random.Random(seed)
    means = sorted(fmean(rng.choices(differences, k=count)) for _ in range(resamples))
    tail = (1.0 - confidence) / 2.0
    low_index = min(resamples - 1, int(tail * resamples))
    high_index = max(0, min(resamples - 1, int((1.0 - tail) * resamples) - 1))
    return PairedDifference(tasks=count, mean=fmean(differences), low=means[low_index], high=means[high_index])


def betting_paired_difference(
    evaluation: EvaluationResult, *, confidence: float, planned: int | None = None
) -> PairedDifference:
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
    Every ``m`` whose capital reached that level after any task so far is
    ruled out, and the low end is the largest such ``m``. The high end bets
    the other way. No normal approximation is involved, so the bound holds for
    skewed differences and for a handful of tasks alike.

    The bound covers the capital's running maximum, so it holds at every
    number of tasks at once. A gate may compute the interval after each block
    of tasks and stop at the first that decides, at the full ``confidence``,
    as long as each look continues the same bets on the same tasks in the
    same order. The stakes are tuned to ``planned``, the number of tasks the
    gate plays if no look stops it, which defaults to the tasks evaluated.
    ``planned`` and the tasks' order must be fixed before any task is scored,
    and the tasks must be independent. Equal differences on every task give
    an interval whose width depends only on how many tasks there are and how
    many were planned.
    """
    require(0.0 < confidence < 1.0, f"confidence must be in (0, 1), got {confidence}")
    differences = _differences(evaluation)
    tasks = len(differences)
    planned = tasks if planned is None else planned
    require(
        planned >= tasks,
        f"the bets were tuned to {planned} planned tasks, but the evaluation has {tasks}",
    )
    log_level = math.log(2.0 / (1.0 - confidence))
    bets = _bets(differences, log_level, planned)
    low = _low_end(differences, bets, log_level)
    high = -_low_end([-difference for difference in differences], bets, log_level)
    return PairedDifference(tasks=tasks, mean=fmean(differences), low=low, high=high)


def _differences(evaluation: EvaluationResult) -> list[float]:
    """Each task's candidate score less its incumbent score, refused outside ``[-1, 1]``.

    Both intervals assume that range, which scores in ``[0, 1]`` give. A
    ``Grade`` checks its score, but an ``EvaluationResult`` built by hand does
    not, and outside the range a stake can lose more than the whole capital.
    """
    differences = list(map(sub, evaluation.candidate_scores, evaluation.incumbent_scores))
    if min(differences) < -1.0 or max(differences) > 1.0:
        for task_id, candidate, incumbent, difference in zip(
            evaluation.task_ids, evaluation.candidate_scores, evaluation.incumbent_scores, differences, strict=True
        ):
            require(
                -1.0 <= difference <= 1.0,
                f"task {task_id} scores {candidate} against {incumbent}, a difference of {difference:+g}. "
                "The interval needs every difference in [-1, 1], which scores in [0, 1] give",
            )
    return differences


def _bets(differences: list[float], log_level: float, planned: int) -> list[float]:
    """Each task's stake before the cap, from the tasks before it only.

    ``sqrt(2 log_level / (planned s2))``, with ``s2`` the spread seen so far,
    makes the capital grow fastest at the mean the bound should just exclude
    after ``planned`` tasks (Waudby-Smith and Ramdas's fixed-time choice).
    ``s2`` starts as one imagined task at a spread of 1, the largest a
    difference in ``[-1, 1]`` can have, so the first stakes are small. Like
    ``_log_capital_path``, it runs in builtins alone.
    """
    tasks = len(differences)
    means = map(truediv, accumulate(differences), range(2, tasks + 2))
    deviations = map(sub, differences, means)
    squares = accumulate(chain((1.0,), map(pow, deviations, repeat(2))))
    numerators = map(mul, repeat(2.0 * log_level), range(1, tasks + 1))
    return list(map(math.sqrt, map(truediv, numerators, map(mul, repeat(planned), squares))))


def _low_end(differences: list[float], bets: list[float], log_level: float) -> float:
    """The largest mean the bets above it rule out, less at most ``BOUND_TOLERANCE``, or -1 if they rule out none.

    Every task's factor falls as the tested mean ``m`` rises. With the stake
    capped at ``MAX_BET / (1 + m)`` the factor is
    ``1 + MAX_BET (d - m) / (1 + m)``, which falls too, since ``d`` is at least
    -1. So the log capital after every task falls as ``m`` rises, and so does
    its running maximum, and the end is the one root of that maximum less
    ``log_level`` on ``[-1, 1]``.

    The maximum bends wherever the task it peaks at changes, and above the
    root it flattens out at the first task's factor, so regula falsi on it
    would crawl. Each step instead interpolates the log capital after the task
    where the capital peaks at the bracket's low side, which is smooth. The
    Illinois variant halves the value at an end kept twice in a row, so both
    ends close in, and the root takes about eight evaluations where bisection
    needs 21. The running maximum decides which side each try replaces, so
    the end returned is never above the exact one.
    """
    low, high = -1.0, 1.0
    path_low = _log_capital_path(differences, bets, low)
    peak = max(path_low)
    if peak < log_level:
        return low
    task = path_low.index(peak)
    path_high = _log_capital_path(differences, bets, high)
    scale_low = scale_high = 1.0
    kept = 0
    while high - low > BOUND_TOLERANCE:
        excess_low = (path_low[task] - log_level) * scale_low
        excess_high = (path_high[task] - log_level) * scale_high
        middle = (low * excess_high - high * excess_low) / (excess_high - excess_low)
        if not low < middle < high:
            middle = (low + high) / 2.0
        path = _log_capital_path(differences, bets, middle)
        peak = max(path)
        if peak >= log_level:
            low, path_low, task = middle, path, path.index(peak)
            scale_low, scale_high = 1.0, (scale_high / 2.0 if kept == 1 else scale_high)
            kept = 1
        else:
            high, path_high = middle, path
            scale_low, scale_high = (scale_low / 2.0 if kept == -1 else scale_low), 1.0
            kept = -1
    return low


def _log_capital_path(differences: list[float], bets: list[float], mean: float) -> list[float]:
    """The log of the capital after each task, from betting that the differences are above ``mean``.

    A difference can fall ``1 + mean`` below ``mean``, so a stake of at most
    ``MAX_BET / (1 + mean)`` never loses more than ``MAX_BET`` of the capital.

    It runs in builtins alone, with no Python frame per task, because the
    simulation tests evaluate it millions of times and coverage tracing slows
    a generator here about fourfold.
    """
    cap = MAX_BET / (1.0 + mean) if mean > -1.0 else math.inf
    stakes = map(min, bets, repeat(cap))
    shifts = map(sub, differences, repeat(mean))
    return list(accumulate(map(math.log1p, map(mul, stakes, shifts))))


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
