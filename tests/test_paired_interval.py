"""The paired interval selector: its decisions, its looks, and its error rates on simulated gates.

A simulated gate draws per-round differences with a chosen mean and standard
deviation on tank's score steps and runs the selector on them, in one look or
in blocks. Every simulation is seeded, so each rate is the same on every run.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping

import pytest

from honing.core import CheckpointRef, ContractError, EvaluationResult, ModelSnapshot, UpdateCandidate
from honing.release import PairedIntervalSelector, normal_paired_difference
from tests.fakes.model import parameters_of, small_moe

#: The improvement in round score the gate must detect (#166).
EFFECT = 0.10
GATES = 2000
#: Paired standard deviations to plan for until #165 measures tank's.
PAIRED_SDS = (0.3, 0.5, 0.7)
TASK_IDS = tuple(f"t{index}" for index in range(1000))


def _candidate() -> UpdateCandidate:
    architecture = small_moe()
    ref = CheckpointRef(location="fake://b", weight_format="hf_safetensors", checksum="b")
    return UpdateCandidate(
        kind="train", parent_id="r0000", master=ModelSnapshot(ref, architecture, parameters_of(architecture))
    )


CANDIDATE = _candidate()


def _evaluation(differences: list[float]) -> EvaluationResult:
    """Scores in [0, 1] whose per-task differences are ``differences``."""
    return EvaluationResult(
        task_ids=TASK_IDS[: len(differences)],
        candidate_scores=tuple(0.5 + difference / 2 for difference in differences),
        incumbent_scores=tuple(0.5 - difference / 2 for difference in differences),
        evaluated_variant="master",
    )


def _three_point(mean: float, sd: float) -> dict[float, float]:
    """Differences of +step, 0 and -step with this mean and standard deviation.

    The step is 1, a win against a loss, for a standard deviation of 0.4 or
    more. Below that it is 0.5, a draw against a win or a loss, because whole
    wins and losses cannot give a mean of 0.10 with a spread under 0.3.
    """
    step = 1.0 if sd >= 0.4 else 0.5
    second_moment = (sd * sd + mean * mean) / (step * step)
    return {step: (second_moment + mean / step) / 2, -step: (second_moment - mean / step) / 2, 0.0: 1 - second_moment}


def _select_rate(
    selector: PairedIntervalSelector, looks: int, rounds: int, distribution: Mapping[float, float], seed: int
) -> tuple[float, float]:
    """The share of simulated gates that select, and the mean number of rounds they play."""
    values, weights = list(distribution), list(distribution.values())
    rng = random.Random(seed)
    block = rounds // looks
    selected = played = 0
    for _ in range(GATES):
        differences = rng.choices(values, weights, k=rounds)
        for look in range(1, looks + 1):
            decision = selector.look(CANDIDATE, _evaluation(differences[: look * block]), final=look == looks)
            if decision is not None:
                selected += decision.selected
                played += look * block
                break
    return selected / GATES, played / GATES


# ---------------------------------------------------------------- decisions


@pytest.mark.unit
def test_the_normal_interval_is_the_mean_plus_or_minus_z_standard_errors() -> None:
    interval = normal_paired_difference(_evaluation([1.0, 0.0, 1.0, 0.0]), confidence=0.95)
    half_width = 1.959964 * math.sqrt(1 / 3) / 2
    assert interval.tasks == 4
    assert interval.mean == pytest.approx(0.5)
    assert (interval.low, interval.high) == pytest.approx((0.5 - half_width, 0.5 + half_width))
    with pytest.raises(ContractError, match="two tasks"):
        normal_paired_difference(_evaluation([1.0]), confidence=0.95)


@pytest.mark.unit
def test_it_selects_only_when_the_whole_interval_is_above_min_effect() -> None:
    mostly_wins = _evaluation([1.0] * 30 + [0.0] * 10)
    decision = PairedIntervalSelector().decide(CANDIDATE, mostly_wins)
    assert decision.selected
    assert decision.metrics["difference_low"] > 0.0
    assert decision.metrics["interval_confidence"] == 0.95
    assert decision.metrics["tasks"] == 40.0
    # The mean, 0.75, is above 0.7, but the interval reaches down to 0.61.
    assert not PairedIntervalSelector(min_effect=0.7).decide(CANDIDATE, mostly_wins).selected
    assert not PairedIntervalSelector().decide(CANDIDATE, _evaluation([0.0] * 40)).selected


@pytest.mark.unit
def test_an_interim_look_stops_only_when_the_interval_clears_min_effect() -> None:
    selector = PairedIntervalSelector(looks=4)
    mixed = _evaluation([1.0, -1.0, 0.0, 0.5] * 10)
    assert selector.look(CANDIDATE, mixed, final=False) is None
    last = selector.look(CANDIDATE, mixed, final=True)
    assert last is not None and not last.selected

    clear_win = selector.look(CANDIDATE, _evaluation([1.0] * 36 + [-1.0] * 4), final=False)
    assert clear_win is not None and clear_win.selected
    assert clear_win.metrics["interval_confidence"] == pytest.approx(1 - 0.1 * 0.05 / 3)

    clear_loss = selector.look(CANDIDATE, _evaluation([-1.0] * 36 + [1.0] * 4), final=False)
    assert clear_loss is not None and not clear_loss.selected


@pytest.mark.unit
def test_it_refuses_settings_and_looks_that_cannot_work() -> None:
    with pytest.raises(ContractError, match="confidence"):
        PairedIntervalSelector(confidence=1.0)
    with pytest.raises(ContractError, match="looks"):
        PairedIntervalSelector(looks=0)
    with pytest.raises(ContractError, match="min_effect"):
        PairedIntervalSelector(min_effect=math.nan)
    with pytest.raises(ContractError, match="no interim looks"):
        PairedIntervalSelector().look(CANDIDATE, _evaluation([1.0, 0.0]), final=False)
    with pytest.raises(ContractError, match="effect must be above"):
        PairedIntervalSelector(min_effect=0.1).rounds_needed(paired_sd=0.5, effect=0.1)


@pytest.mark.unit
def test_the_gate_size_follows_the_normal_approximation() -> None:
    # ((1.960 + 0.842) * 0.5 / 0.1) ** 2 + 1.960 ** 2 / 2 = 198.1. Four looks use 2.005 at the final one,
    # which gives 204.5, rounded up to whole blocks of 52.
    assert PairedIntervalSelector().rounds_needed(paired_sd=0.5, effect=EFFECT) == 199
    assert PairedIntervalSelector(looks=4).rounds_needed(paired_sd=0.5, effect=EFFECT) == 208


# ---------------------------------------------------------------- error rates over 2,000 simulated gates


@pytest.mark.unit
@pytest.mark.parametrize("looks", [1, 4])
@pytest.mark.parametrize("paired_sd", PAIRED_SDS)
def test_a_candidate_with_no_effect_is_selected_at_most_five_percent_of_the_time(paired_sd: float, looks: int) -> None:
    selector = PairedIntervalSelector(looks=looks)
    rounds = selector.rounds_needed(paired_sd, EFFECT)
    rate, _ = _select_rate(selector, looks, rounds, _three_point(0.0, paired_sd), seed=0)
    assert rate <= 0.05


@pytest.mark.unit
@pytest.mark.parametrize("looks", [1, 4])
@pytest.mark.parametrize("paired_sd", PAIRED_SDS)
def test_a_candidate_better_by_the_effect_is_selected_at_least_80_percent_of_the_time(
    paired_sd: float, looks: int
) -> None:
    selector = PairedIntervalSelector(looks=looks)
    rounds = selector.rounds_needed(paired_sd, EFFECT)
    rate, played = _select_rate(selector, looks, rounds, _three_point(EFFECT, paired_sd), seed=1)
    assert rate >= 0.80
    if looks > 1:
        assert played < rounds, "a sequential gate stops early on some candidates this good"


@pytest.mark.unit
@pytest.mark.parametrize("looks", [1, 4])
def test_a_skewed_candidate_with_no_effect_is_selected_at_most_five_percent_of_the_time(looks: int) -> None:
    # Half-point gains in 20% of rounds paid for by whole-round losses in 10%: mean 0, skewed toward the
    # losses. A sample that misses the rare losses looks like a clear gain, so a normal interval selects
    # this shape more often than a symmetric one.
    skewed = {0.5: 0.2, -1.0: 0.1, 0.0: 0.7}
    selector = PairedIntervalSelector(looks=looks)
    rounds = selector.rounds_needed(math.sqrt(0.15), EFFECT)
    rate, _ = _select_rate(selector, looks, rounds, skewed, seed=2)
    assert rate <= 0.05
