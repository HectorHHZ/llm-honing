"""Selection policies the gate applies to an evaluated candidate."""

from __future__ import annotations

import math
from statistics import NormalDist

from honing.core import EvaluationResult, SelectionDecision, UpdateCandidate, require
from honing.release.statistics import MAX_BET, PairedDifference, betting_paired_difference, paired_difference

#: The share of ``1 - confidence`` a sequential gate's interim looks spend, split evenly between them. It is
#: small so the final look's interval stays close to a fixed gate's, and interim looks stop only on clear results.
INTERIM_SHARE = 0.1


def _interval_metrics(evaluation: EvaluationResult) -> dict[str, float]:
    difference = paired_difference(evaluation)
    return {
        "tasks": float(difference.tasks),
        "difference_mean": difference.mean,
        "difference_low": difference.low,
        "difference_high": difference.high,
    }


class WinLossSelector:
    """Select when the candidate wins more tasks than it loses, by more than ``min_win_margin``.

    Suits changes expected to improve the model: training steps and harness
    edits. A tie on every task is a rejection. The decision reports the
    bootstrap interval on the mean per-task difference alongside the counts.
    """

    name = "win_loss"

    def __init__(self, min_win_margin: int = 0) -> None:
        require(min_win_margin >= 0, f"min_win_margin must be >= 0, got {min_win_margin}")
        self._min_win_margin = min_win_margin

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        pairs = list(zip(evaluation.candidate_scores, evaluation.incumbent_scores, strict=True))
        wins = sum(candidate_score > incumbent_score for candidate_score, incumbent_score in pairs)
        losses = sum(candidate_score < incumbent_score for candidate_score, incumbent_score in pairs)
        selected = wins - losses > self._min_win_margin
        return SelectionDecision(
            outcome="select" if selected else "reject",
            policy=self.name,
            reason=f"{wins} wins, {losses} losses, {len(pairs) - wins - losses} ties",
            metrics={"wins": float(wins), "losses": float(losses), **_interval_metrics(evaluation)},
        )


class NoRegressionSelector:
    """Select when the mean score drops by no more than ``max_regression``.

    Suits changes made for cost rather than quality: a prune or a quantization
    earns its place by being cheaper, so it need not win, only not lose much.
    The regression is the mean of the per-task differences, which equals the
    difference of the means; the decision also reports its bootstrap interval.
    """

    name = "no_regression"

    def __init__(self, max_regression: float) -> None:
        require(max_regression >= 0.0, f"max_regression must be >= 0, got {max_regression}")
        self._max_regression = max_regression

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        interval = _interval_metrics(evaluation)
        regression = -interval["difference_mean"]
        selected = regression <= self._max_regression
        return SelectionDecision(
            outcome="select" if selected else "reject",
            policy=self.name,
            reason=f"mean score changed by {-regression:+.4f}; the limit is -{self._max_regression:.4f}",
            metrics={"regression": regression, **interval},
        )


class PairedIntervalSelector:
    """Select when the interval on the mean per-task difference lies above ``min_effect``.

    The interval is :func:`~honing.release.statistics.betting_paired_difference`,
    two-sided at ``confidence``. A candidate whose true mean difference is at
    most ``min_effect`` is selected with probability at most
    ``(1 - confidence) / 2``, which is 2.5% at the default 95%. The bound
    assumes only that the per-task differences are independent and lie in
    ``[-1, 1]``, which every pair of scores in ``[0, 1]`` does. It holds for
    skewed differences, for a block whose differences are all equal, and for
    a handful of tasks.

    With ``looks > 1`` the :class:`~honing.release.Gate` plays its rounds in
    ``looks`` equal blocks and calls :meth:`look` after each one, so it can
    stop once the result is clear. The decision records ``stopped_early``.
    Every look is another chance for noise to clear the bar, so the looks split
    ``1 - confidence`` between them. The interim looks share a tenth of it
    evenly and the final look keeps the rest. The chance that any look's
    interval misses the true mean is at most the sum of the parts, so the bound
    above holds across all the looks. This is a Haybittle-Peto design with a
    Bonferroni split.

    A bound that holds for every distribution costs rounds. To detect 0.10
    with 80% power at a paired standard deviation of 0.5, a fixed gate needs
    327 rounds and a four-look gate 336, where a normal interval needed 199
    and 208. The normal interval selected a candidate with no effect up to 16%
    of the time on skewed differences. In simulation this one selected at most
    0.6% on every shape the tests try. :meth:`rounds_needed` sizes a gate for
    this interval.
    """

    name = "paired_interval"

    def __init__(self, min_effect: float = 0.0, confidence: float = 0.95, looks: int = 1) -> None:
        require(
            -1.0 < min_effect < 1.0,
            f"min_effect must be in (-1, 1), where differences of scores lie, got {min_effect}",
        )
        require(0.0 < confidence < 1.0, f"confidence must be in (0, 1), got {confidence}")
        require(looks >= 1, f"looks must be >= 1, got {looks}")
        self._min_effect = min_effect
        self._confidence = confidence
        self._looks = looks

    @property
    def looks(self) -> int:
        return self._looks

    def look(
        self, candidate: UpdateCandidate, evaluation: EvaluationResult, *, final: bool
    ) -> SelectionDecision | None:
        """Decide on the rounds played so far, or return None to play the next block.

        An interim look selects when the interval is above ``min_effect``,
        rejects when it is below, and otherwise returns None. The final look
        always decides. A gate makes at most ``looks - 1`` interim looks, or the
        bound in the class docstring no longer holds.
        """
        if final:
            return self.decide(candidate, evaluation)
        require(self._looks > 1, "a selector with one look has no interim looks")
        confidence = self._look_confidence(final=False)
        interval = betting_paired_difference(evaluation, confidence=confidence)
        if interval.low > self._min_effect or interval.high < self._min_effect:
            return self._decision(interval, confidence, stopped_early=True)
        return None

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        """The final look. A gate that plays all its rounds at once calls only this."""
        confidence = self._look_confidence(final=True)
        interval = betting_paired_difference(evaluation, confidence=confidence)
        return self._decision(interval, confidence, stopped_early=False)

    def rounds_needed(self, paired_sd: float, effect: float, power: float = 0.8) -> int:
        """Paired rounds for the final look to select a true mean difference of ``effect`` with ``power``.

        ``paired_sd`` is the standard deviation of one round's difference. The
        low end clears ``min_effect`` when the log capital of betting above it
        reaches the final look's level, ``log(2 / alpha)``. With stake ``b``, a
        round adds ``log(1 + b (d - min_effect))``, which averages about
        ``b g - b ** 2 (paired_sd ** 2 + g ** 2) / 2`` with a standard deviation
        of about ``b paired_sd``, where ``g`` is ``effect - min_effect``. The
        count is the smallest at which the total's mean, less ``z(power)`` of
        its standard deviations, reaches the level, at the stake the interval
        uses once it has seen the spread. Simulated power at that count is 80
        to 83% for spreads of 0.3 to 0.7 and an effect of 0.10. It is rounded
        up to whole blocks of at least two rounds. Interim looks add a little
        power, since they can select early.
        """
        require(paired_sd > 0.0, f"paired_sd must be > 0, got {paired_sd}")
        require(effect > self._min_effect, f"effect must be above min_effect {self._min_effect}, got {effect}")
        require(0.0 < power < 1.0, f"power must be in (0, 1), got {power}")
        level = math.log(2.0 / (1.0 - self._look_confidence(final=True)))
        gap = effect - self._min_effect
        z_power = NormalDist().inv_cdf(power)

        def enough(rounds: int) -> bool:
            bet = min(math.sqrt(2.0 * level / (rounds * paired_sd**2)), MAX_BET / (1.0 + self._min_effect))
            growth = rounds * (bet * gap - bet**2 * (paired_sd**2 + gap**2) / 2.0)
            return growth - z_power * bet * paired_sd * math.sqrt(rounds) >= level

        # Enough rounds stay enough with more, so double past the count and bisect back to it.
        short, rounds = 0, 1
        while not enough(rounds):
            short, rounds = rounds, rounds * 2
        while rounds - short > 1:
            middle = (short + rounds) // 2
            short, rounds = (short, middle) if enough(middle) else (middle, rounds)
        blocks = max(math.ceil(rounds / self._looks), 2)
        return blocks * self._looks

    def _look_confidence(self, *, final: bool) -> float:
        """One look's interval level. The levels' miss chances add up to ``1 - confidence``."""
        if self._looks == 1:
            return self._confidence
        alpha = 1.0 - self._confidence
        if final:
            return 1.0 - (1.0 - INTERIM_SHARE) * alpha
        return 1.0 - INTERIM_SHARE * alpha / (self._looks - 1)

    def _decision(self, interval: PairedDifference, confidence: float, *, stopped_early: bool) -> SelectionDecision:
        selected = interval.low > self._min_effect
        return SelectionDecision(
            outcome="select" if selected else "reject",
            policy=self.name,
            reason=(
                f"the {confidence:.2%} interval on the mean difference over {interval.tasks} tasks is"
                f" [{interval.low:+.4f}, {interval.high:+.4f}], and selecting needs it above {self._min_effect:+.4f}"
            ),
            metrics={
                "tasks": float(interval.tasks),
                "difference_mean": interval.mean,
                "difference_low": interval.low,
                "difference_high": interval.high,
                "interval_confidence": confidence,
                "min_effect": self._min_effect,
                "stopped_early": float(stopped_early),
            },
        )


class AcceptAll:
    """Select every candidate. For methods whose next evaluation is the real judge."""

    name = "accept_all"

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        return SelectionDecision(outcome="select", policy=self.name, reason="every candidate is accepted")
