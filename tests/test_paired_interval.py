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
from honing.release import PairedDifference, PairedIntervalSelector, betting_paired_difference
from tests.fakes.model import parameters_of, small_moe

#: The improvement in round score the gate must detect (#166).
EFFECT = 0.10
GATES = 2000
#: The most often a candidate with no effect may be selected: the selector's bound, (1 - 0.95) / 2, plus three
#: Monte Carlo standard errors over ``GATES`` gates, so a selector that just meets its bound fails one seed in 700.
FALSE_ACCEPT_LIMIT = 0.025 + 3 * math.sqrt(0.025 * 0.975 / GATES)
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
def test_the_betting_interval_needs_no_spread_and_holds_from_one_task() -> None:
    # Nineteen equal differences have no spread. A normal interval would have zero width and select at once.
    tied = betting_paired_difference(_evaluation([0.1] * 19), confidence=0.95)
    assert (tied.tasks, tied.mean) == (19, pytest.approx(0.1))
    assert (tied.low, tied.high) == pytest.approx((-0.147470, 0.305272), abs=1e-5)
    assert betting_paired_difference(_evaluation([1.0]), confidence=0.95) == PairedDifference(1, 1.0, -1.0, 1.0)
    # Eight straight wins are enough to clear zero, and the interval stays inside [-1, 1].
    sweep = betting_paired_difference(_evaluation([1.0] * 8), confidence=0.95)
    assert (sweep.low, sweep.high) == pytest.approx((0.122895, 1.0), abs=1e-5)
    nothing = betting_paired_difference(_evaluation([0.0] * 40), confidence=0.95)
    assert (nothing.low, nothing.high) == pytest.approx((-0.116478, 0.116478), abs=1e-5)
    with pytest.raises(ContractError, match="confidence"):
        betting_paired_difference(_evaluation([1.0]), confidence=1.0)


@pytest.mark.unit
def test_it_selects_only_when_the_whole_interval_is_above_min_effect() -> None:
    mostly_wins = _evaluation([1.0] * 30 + [0.0] * 10)
    decision = PairedIntervalSelector().decide(CANDIDATE, mostly_wins)
    assert decision.selected
    assert decision.metrics["difference_low"] > 0.0
    assert decision.metrics["interval_confidence"] == 0.95
    assert decision.metrics["tasks"] == 40.0
    # The mean, 0.75, is above 0.7, but the interval reaches down to 0.50.
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
    for min_effect in (math.nan, -1.0, 1.0):
        with pytest.raises(ContractError, match="min_effect"):
            PairedIntervalSelector(min_effect=min_effect)
    with pytest.raises(ContractError, match="no interim looks"):
        PairedIntervalSelector().look(CANDIDATE, _evaluation([1.0, 0.0]), final=False)
    with pytest.raises(ContractError, match="effect must be above"):
        PairedIntervalSelector(min_effect=0.1).rounds_needed(paired_sd=0.5, effect=0.1)


@pytest.mark.unit
def test_the_gate_size_is_where_the_capital_is_expected_to_clear_the_final_level() -> None:
    # At its final look a four-look gate's capital must reach 2 / (0.9 * 0.05) where a fixed gate's must reach
    # 2 / 0.05, and the count rounds up to whole blocks of 84.
    assert PairedIntervalSelector().rounds_needed(paired_sd=0.5, effect=EFFECT) == 327
    assert PairedIntervalSelector(looks=4).rounds_needed(paired_sd=0.5, effect=EFFECT) == 336


@pytest.mark.unit
def test_a_planned_gate_can_play_every_look_and_select() -> None:
    # A large effect on a small spread needs few rounds, but every block still gets at least two.
    selector = PairedIntervalSelector(looks=4)
    rounds = selector.rounds_needed(paired_sd=0.1, effect=0.5)
    assert rounds >= 2 * 4 and rounds % 4 == 0
    differences = [0.4, 0.6] * (rounds // 2)
    block = rounds // 4
    for look in range(1, 4):
        assert selector.look(CANDIDATE, _evaluation(differences[: look * block]), final=False) is None
    last = selector.look(CANDIDATE, _evaluation(differences), final=True)
    assert last is not None and last.selected


# ---------------------------------------------------------------- error rates over 2,000 simulated gates


@pytest.mark.unit
@pytest.mark.parametrize("looks", [1, 4])
@pytest.mark.parametrize("paired_sd", PAIRED_SDS)
def test_a_candidate_with_no_effect_is_selected_at_most_at_the_bound(paired_sd: float, looks: int) -> None:
    selector = PairedIntervalSelector(looks=looks)
    rounds = selector.rounds_needed(paired_sd, EFFECT)
    rate, _ = _select_rate(selector, looks, rounds, _three_point(0.0, paired_sd), seed=0)
    assert rate <= FALSE_ACCEPT_LIMIT


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
def test_a_skewed_candidate_with_no_effect_is_selected_at_most_at_the_bound(looks: int) -> None:
    # Half-point gains in 20% of rounds paid for by whole-round losses in 10%: mean 0, skewed toward the
    # losses. A sample that misses the rare losses looks like a clear gain.
    skewed = {0.5: 0.2, -1.0: 0.1, 0.0: 0.7}
    selector = PairedIntervalSelector(looks=looks)
    rounds = selector.rounds_needed(math.sqrt(0.15), EFFECT)
    rate, _ = _select_rate(selector, looks, rounds, skewed, seed=2)
    assert rate <= FALSE_ACCEPT_LIMIT


@pytest.mark.unit
@pytest.mark.parametrize("looks", [1, 4])
def test_a_candidate_whose_rare_losses_a_block_can_miss_is_selected_at_most_at_the_bound(looks: int) -> None:
    # +0.1 in 90% of rounds and -0.9 in 10%: mean 0, standard deviation 0.3. A block of 19 misses every loss
    # 13.5% of the time and then has no spread at all, which made a normal interval's four-look gate select
    # 16% of these candidates with 76 rounds. 76 is what that interval planned for this spread, and the
    # betting interval plans more.
    rare_losses = {0.1: 0.9, -0.9: 0.1}
    selector = PairedIntervalSelector(looks=looks)
    for rounds in (76, selector.rounds_needed(0.3, EFFECT)):
        rate, _ = _select_rate(selector, looks, rounds, rare_losses, seed=0)
        assert rate <= FALSE_ACCEPT_LIMIT
