"""The paired interval selector: its decisions, its looks, the gate's blocks, and its error rates on simulated gates.

A simulated gate draws per-round differences with a chosen mean and standard
deviation on tank's score steps and runs the selector on them, in one look or
in blocks. Every simulation is seeded, so each rate is the same on every run.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping
from pathlib import Path

import pytest

from honing.core import (
    CheckpointRef,
    ContractError,
    EvaluationResult,
    HoneRelease,
    ModelSnapshot,
    Move,
    Task,
    UpdateCandidate,
)
from honing.release import (
    Gate,
    PairedDifference,
    PairedIntervalSelector,
    betting_paired_difference,
    paired_difference,
)
from tests.fakes.builder import TASKS, build_campaign
from tests.fakes.harness import FakeEpisodeRunner
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
INCUMBENT = HoneRelease("r0000", UpdateCandidate(kind="init", parent_id=None, master=CANDIDATE.master), None)


def _evaluation(differences: list[float], first: int = 0) -> EvaluationResult:
    """Scores in [0, 1] whose per-task differences are ``differences``, on tasks from number ``first``."""
    return EvaluationResult(
        task_ids=TASK_IDS[first : first + len(differences)],
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
            evaluation = _evaluation(differences[: look * block])
            decision = selector.look(CANDIDATE, evaluation, final=look == looks, planned=rounds)
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
    # Stakes tuned to more tasks are smaller, so the same 40 ties give a wider interval.
    planned = betting_paired_difference(_evaluation([0.0] * 40), confidence=0.95, planned=160)
    assert planned.low < nothing.low and planned.high > nothing.high
    with pytest.raises(ContractError, match="confidence"):
        betting_paired_difference(_evaluation([1.0]), confidence=1.0)
    with pytest.raises(ContractError, match="tuned to 39 planned tasks, but the evaluation has 40"):
        betting_paired_difference(_evaluation([0.0] * 40), confidence=0.95, planned=39)


@pytest.mark.unit
def test_the_interval_keeps_the_highest_capital_the_bets_reached() -> None:
    # Ten straight wins clear zero. Ten losses after them bring the mean back to 0 and the capital back below the
    # level, but Ville's bound covers the capital's running maximum, so the low end keeps what the wins reached.
    # A candidate with no effect wins ten straight rounds less than 0.1% of the time.
    rose = betting_paired_difference(_evaluation([1.0] * 10), confidence=0.95, planned=20)
    fell = betting_paired_difference(_evaluation([1.0] * 10 + [-1.0] * 10), confidence=0.95)
    assert rose.low > 0.25 and fell.mean == 0.0 and fell.low == pytest.approx(rose.low, abs=1e-6)
    assert PairedIntervalSelector().decide(CANDIDATE, _evaluation([1.0] * 10 + [-1.0] * 10)).selected
    # The same twenty differences, alternating, never lift the capital that far.
    assert betting_paired_difference(_evaluation([1.0, -1.0] * 10), confidence=0.95).low < 0.0

    # Sixty losses after the wins push the high end below the low end. No single mean fits, so the gate rejects.
    crossed = PairedIntervalSelector().decide(CANDIDATE, _evaluation([1.0] * 10 + [-1.0] * 60))
    assert crossed.metrics["difference_low"] > crossed.metrics["difference_high"]
    assert not crossed.selected and crossed.reason.endswith("Its ends crossed, so no single mean fits the differences")


@pytest.mark.unit
@pytest.mark.parametrize("scores", [(1.7, 0.0), (0.0, 1.2)])
def test_a_difference_outside_minus_one_to_one_is_refused(scores: tuple[float, float]) -> None:
    # A Grade keeps scores in [0, 1], but an EvaluationResult built by hand does not.
    evaluation = EvaluationResult(
        task_ids=("t0", "t1"),
        candidate_scores=(0.5, scores[0]),
        incumbent_scores=(0.5, scores[1]),
        evaluated_variant="master",
    )
    with pytest.raises(ContractError, match=r"task t1 scores .* a difference of [+-]1\.\d+\. The interval needs"):
        betting_paired_difference(evaluation, confidence=0.95)
    with pytest.raises(ContractError, match="task t1 scores"):
        paired_difference(evaluation)


@pytest.mark.unit
def test_it_selects_only_when_the_whole_interval_is_above_min_effect() -> None:
    mostly_wins = _evaluation([1.0, 1.0, 1.0, 0.0] * 10)
    decision = PairedIntervalSelector().decide(CANDIDATE, mostly_wins)
    assert decision.selected
    assert decision.metrics["difference_low"] > 0.0
    assert decision.metrics["interval_confidence"] == 0.95
    assert (decision.metrics["tasks"], decision.metrics["planned_tasks"]) == (40.0, 40.0)
    # The mean, 0.75, is above 0.7, but the interval reaches down to 0.52.
    assert not PairedIntervalSelector(min_effect=0.7).decide(CANDIDATE, mostly_wins).selected
    assert not PairedIntervalSelector().decide(CANDIDATE, _evaluation([0.0] * 40)).selected


@pytest.mark.unit
def test_an_interim_look_stops_only_when_the_interval_clears_min_effect() -> None:
    # The first of four looks at 40 rounds each.
    selector = PairedIntervalSelector(looks=4)
    mixed = _evaluation([1.0, -1.0, 0.0, 0.5] * 10)
    assert selector.look(CANDIDATE, mixed, final=False, planned=160) is None
    last = selector.look(CANDIDATE, mixed, final=True)
    assert last is not None and not last.selected

    clear_win = selector.look(CANDIDATE, _evaluation([1.0] * 36 + [-1.0] * 4), final=False, planned=160)
    assert clear_win is not None and clear_win.selected
    # Every look reads the same betting process, so each uses the full level.
    assert (clear_win.metrics["interval_confidence"], clear_win.metrics["planned_tasks"]) == (0.95, 160.0)

    clear_loss = selector.look(CANDIDATE, _evaluation([-1.0] * 36 + [1.0] * 4), final=False, planned=160)
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
        PairedIntervalSelector().look(CANDIDATE, _evaluation([1.0, 0.0]), final=False, planned=4)
    with pytest.raises(ContractError, match="an interim look needs the planned rounds"):
        PairedIntervalSelector(looks=2).look(CANDIDATE, _evaluation([1.0, 0.0]), final=False)
    with pytest.raises(ContractError, match="effect must be above"):
        PairedIntervalSelector(min_effect=0.1).rounds_needed(paired_sd=0.5, effect=0.1)


@pytest.mark.unit
def test_the_gate_size_is_where_the_capital_is_likely_to_have_reached_the_level() -> None:
    # Every look bets at the full level, so a four-look gate needs only the fixed gate's count rounded up to whole
    # blocks of 70.
    assert PairedIntervalSelector().rounds_needed(paired_sd=0.5, effect=EFFECT) == 279
    assert PairedIntervalSelector(looks=4).rounds_needed(paired_sd=0.5, effect=EFFECT) == 280


@pytest.mark.unit
def test_a_planned_gate_can_play_every_look_and_select() -> None:
    # A large effect on a small spread needs few rounds, but every block still gets at least two.
    selector = PairedIntervalSelector(looks=4)
    rounds = selector.rounds_needed(paired_sd=0.1, effect=0.5)
    assert rounds >= 2 * 4 and rounds % 4 == 0
    differences = [0.4, 0.6] * (rounds // 2)
    block = rounds // 4
    decisions = [
        selector.look(CANDIDATE, _evaluation(differences[: look * block]), final=look == 4, planned=rounds)
        for look in range(1, 5)
    ]
    # The third look already selects, and so would the last.
    assert [decision and decision.selected for decision in decisions] == [None, None, True, True]


# ---------------------------------------------------------------- the gate's blocks and harness moves


class _Blocks:
    """An evaluator that plays the next ``block`` differences on new tasks at every call, or the first again."""

    def __init__(self, differences: list[float], block: int, *, replay: bool = False) -> None:
        self.differences, self.block, self.replay, self.calls = differences, block, replay, 0

    def evaluate(self, candidate: UpdateCandidate, incumbent: HoneRelease) -> EvaluationResult:
        first = 0 if self.replay else self.calls * self.block
        self.calls += 1
        return _evaluation(self.differences[first : first + self.block], first)


class _Sizes:
    """An evaluator that plays blocks of these sizes on new tasks, every one a tie."""

    def __init__(self, sizes: list[int]) -> None:
        self.sizes, self.played = sizes, 0

    def evaluate(self, candidate: UpdateCandidate, incumbent: HoneRelease) -> EvaluationResult:
        size = self.sizes.pop(0)
        self.played += size
        return _evaluation([0.0] * size, self.played - size)


@pytest.mark.unit
def test_a_sequential_gate_plays_blocks_until_a_look_decides() -> None:
    selector = PairedIntervalSelector(looks=4)
    # 19 wins and a loss clear zero at the first look of a gate that plans 80 rounds.
    clear = _Blocks([1.0] * 19 + [-1.0] + [0.0] * 60, block=20)
    evaluation, decision = Gate(clear, selector).judge(CANDIDATE, INCUMBENT)
    assert (clear.calls, evaluation.task_ids) == (1, TASK_IDS[:20])
    assert decision.selected and decision.metrics["stopped_early"] == 1.0

    mixed = _Blocks([1.0, -1.0, 0.0, 0.5] * 10, block=10)
    evaluation, decision = Gate(mixed, selector).judge(CANDIDATE, INCUMBENT)
    assert (mixed.calls, evaluation.task_ids) == (4, TASK_IDS[:40])
    assert not decision.selected and decision.metrics["stopped_early"] == 0.0

    # An evaluator that plays the same tasks again would count their rounds twice.
    with pytest.raises(ContractError, match="scores each task once"):
        Gate(_Blocks([1.0, -1.0] * 5, block=10, replay=True), selector).judge(CANDIDATE, INCUMBENT)

    # The bets are tuned to four blocks of the first one's size, so blocks that add up to more are refused.
    with pytest.raises(ContractError, match="tuned to 40 planned tasks, but the evaluation has 55"):
        Gate(_Sizes([10, 15, 15, 15]), selector).judge(CANDIDATE, INCUMBENT)


@pytest.mark.unit
def test_a_harness_move_with_a_method_evaluator_must_pass_both_selectors(tmp_path: Path) -> None:
    # The verify instruction adds 0.5 to every task's score. The method's win-loss selector decides on its
    # own two tasks, and the gate's interval selector on twelve, whose interval is [+0.01, +0.70].
    gate_tasks = tuple(Task(task_id=f"gate{index}", prompt="Answer.") for index in range(12))
    for min_effect, published in ((0.0, True), (0.1, False)):
        fixture = build_campaign(
            tmp_path / str(min_effect),
            make_runner=lambda world: FakeEpisodeRunner(world, instruction_bonus=0.5),
            gate_tasks=gate_tasks,
            harness_selector=PairedIntervalSelector(min_effect),
            method_tasks=TASKS[:2],
        )
        outcome = fixture.campaign.advance(Move(kind="harness"))
        assert (outcome.release is not None) is published
        assert outcome.decision is not None and outcome.decision.policy == "win_loss+paired_interval"
        assert outcome.evaluation is not None and len(outcome.evaluation.task_ids) == len(gate_tasks)
        record = json.loads(fixture.campaign.chain.record_path.read_text().splitlines()[-1])
        assert record["candidate"]["method_state"]["accepted"] == int(published)
        # The gate ran, so the history's gate metrics are its twelve tasks.
        (step,) = fixture.campaign.state().history
        assert (step.metrics_from, step.gate_metrics["gate_tasks"]) == ("gate", 12.0)

    # A candidate the method rejects never reaches the gate.
    fixture = build_campaign(
        tmp_path / "tie",
        harness_selector=PairedIntervalSelector(),
        method_tasks=TASKS[:2],
        make_runner=lambda world: FakeEpisodeRunner(world, instruction_bonus=0.0),
    )
    outcome = fixture.campaign.advance(Move(kind="harness"))
    assert outcome.decision is not None and outcome.decision.policy == "win_loss" and outcome.release is None
    assert fixture.runner.runs == 2 * 2  # two tasks, under the candidate and the incumbent
    # The history's gate metrics then come from the method's two tasks, and it says so.
    (step,) = fixture.campaign.state().history
    assert (step.result, step.metrics_from, step.gate_metrics["gate_tasks"]) == ("rejected", "method", 2.0)


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
