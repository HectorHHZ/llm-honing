"""Selection policies the gate applies to an evaluated candidate."""

from __future__ import annotations

import math
from statistics import NormalDist

from honing.core import EvaluationResult, SelectionDecision, UpdateCandidate, require
from honing.release.statistics import PairedDifference, normal_paired_difference, paired_difference

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

    The interval is :func:`~honing.release.statistics.normal_paired_difference`,
    two-sided at ``confidence``. A candidate whose true mean difference is at
    most ``min_effect`` is selected with probability at most
    ``(1 - confidence) / 2``, which is 2.5% at the default 95%, up to the error
    of the normal approximation. Skew toward rare large losses is where that
    error is largest, because a sample that misses the losses looks like a
    clear gain. In simulation, with half-point gains in 10% of rounds and
    whole-round losses in 5%, a fixed gate selected at 6.7% with 61 rounds,
    5.0% with 100 and 4.1% with 200.

    With ``looks > 1`` the gate plays its rounds in ``looks`` equal blocks and
    calls :meth:`look` after each one, so it can stop once the result is clear.
    Every look is another chance for noise to clear the bar, so the looks split
    ``1 - confidence`` between them. The interim looks share a tenth of it
    evenly and the final look keeps the rest. The chance that any look's
    interval misses the true mean is at most the sum of the parts, so the bound
    above holds across all the looks. This is a Haybittle-Peto design with a
    Bonferroni split. With four looks an interim look stops only on a
    difference of 3.1 standard errors, and the final interval uses 2.0 where a
    fixed gate uses 1.96. The sequential gate needs about 3% more rounds for
    the same power, before rounding up to whole blocks, and stops early on
    candidates that are clearly better or clearly worse.

    A confidence sequence for bounded scores, such as the predictable plug-in
    empirical-Bernstein sequence of Waudby-Smith and Ramdas (2023), would also
    control repeated looks, and would allow a look after every round. Tuned to
    400 rounds at a paired standard deviation of 0.5, it is about 1.8 times as
    wide as this interval there. In simulation it needed 450 rounds for 80%
    power to detect 0.10, where this design needs 208. The gate looks only
    between blocks whose number it knows in advance, so it does not need that
    freedom.
    """

    name = "paired_interval"

    def __init__(self, min_effect: float = 0.0, confidence: float = 0.95, looks: int = 1) -> None:
        require(math.isfinite(min_effect), f"min_effect must be finite, got {min_effect}")
        require(0.0 < confidence < 1.0, f"confidence must be in (0, 1), got {confidence}")
        require(looks >= 1, f"looks must be >= 1, got {looks}")
        self._min_effect = min_effect
        self._confidence = confidence
        self._looks = looks

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
        interval = normal_paired_difference(evaluation, confidence=confidence)
        if interval.low > self._min_effect or interval.high < self._min_effect:
            return self._decision(interval, confidence)
        return None

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        """The final look. A gate that plays all its rounds at once calls only this."""
        confidence = self._look_confidence(final=True)
        return self._decision(normal_paired_difference(evaluation, confidence=confidence), confidence)

    def rounds_needed(self, paired_sd: float, effect: float, power: float = 0.8) -> int:
        """Paired rounds for the final look to select a true mean difference of ``effect`` with ``power``.

        ``paired_sd`` is the standard deviation of one round's difference. The
        count is the normal approximation plus ``z ** 2 / 2``, where ``z`` is the
        final look's quantile, rounded up to whole blocks. The extra rounds are
        Guenther's (1981) allowance for estimating the standard deviation from
        the same rounds. Without them a fixed gate's simulated power falls about
        0.3 points short. Interim looks add a little power, since they can
        select early, and almost never reject a candidate this good.
        """
        require(paired_sd > 0.0, f"paired_sd must be > 0, got {paired_sd}")
        require(effect > self._min_effect, f"effect must be above min_effect {self._min_effect}, got {effect}")
        require(0.0 < power < 1.0, f"power must be in (0, 1), got {power}")
        z = NormalDist().inv_cdf
        final_z = z((1.0 + self._look_confidence(final=True)) / 2.0)
        normal = ((final_z + z(power)) * paired_sd / (effect - self._min_effect)) ** 2
        rounds = math.ceil(normal + final_z**2 / 2.0)
        return math.ceil(rounds / self._looks) * self._looks

    def _look_confidence(self, *, final: bool) -> float:
        """One look's interval level. The levels' miss chances add up to ``1 - confidence``."""
        if self._looks == 1:
            return self._confidence
        alpha = 1.0 - self._confidence
        if final:
            return 1.0 - (1.0 - INTERIM_SHARE) * alpha
        return 1.0 - INTERIM_SHARE * alpha / (self._looks - 1)

    def _decision(self, interval: PairedDifference, confidence: float) -> SelectionDecision:
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
            },
        )


class AcceptAll:
    """Select every candidate. For methods whose next evaluation is the real judge."""

    name = "accept_all"

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        return SelectionDecision(outcome="select", policy=self.name, reason="every candidate is accepted")
