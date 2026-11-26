"""Selection policies the gate applies to an evaluated candidate."""

from __future__ import annotations

import math
from statistics import NormalDist

from honing.core import EvaluationResult, SelectionDecision, UpdateCandidate, require
from honing.release.statistics import MAX_BET, PairedDifference, betting_paired_difference, paired_difference

#: How far a random walk seen after whole steps overshoots a level a Brownian motion just touches, in standard
#: deviations of one step: ``-zeta(1/2) / sqrt(2 pi)``, Siegmund's correction for discrete monitoring.
OVERSHOOT = 0.5826


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
    Every look continues one betting process, with stakes tuned to the planned
    total of rounds, and the interval's bound covers the capital's running
    maximum over every round. So each look uses the full ``confidence``, and
    the bound above holds across all of them.

    A bound that holds for every distribution costs rounds. To detect 0.10
    with 80% power at a paired standard deviation of 0.5, a gate needs 279
    rounds, where a normal interval needed 199. A four-look gate plans 280
    and plays 205 on average. The normal interval selected a candidate with
    no effect up to 16% of the time on skewed differences. In simulation this
    one selected at most 2.0% on every shape the tests try. Testing the final
    capital alone selected at most 0.6%, but needed a sixth more rounds.
    :meth:`rounds_needed` sizes a gate for this interval.
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
        self, candidate: UpdateCandidate, evaluation: EvaluationResult, *, final: bool, planned: int | None = None
    ) -> SelectionDecision | None:
        """Decide on the rounds played so far, or return None to play the next block.

        ``planned`` is the number of rounds the gate plays if no look stops it,
        fixed before the first round. Every look tunes the stakes to it, so all
        of a gate's looks read one betting process, and an interim look needs
        it. An interim look selects when the interval is above ``min_effect``,
        rejects when it is below, and otherwise returns None. The final look
        always decides. The bound holds however many looks a gate makes.
        """
        if final:
            return self.decide(candidate, evaluation, planned=planned)
        require(self._looks > 1, "a selector with one look has no interim looks")
        require(planned is not None, "an interim look needs the planned rounds, to which every look tunes its stakes")
        interval = betting_paired_difference(evaluation, confidence=self._confidence, planned=planned)
        if interval.low > self._min_effect or interval.high < self._min_effect:
            return self._decision(interval, planned, stopped_early=True)
        return None

    def decide(
        self, candidate: UpdateCandidate, evaluation: EvaluationResult, *, planned: int | None = None
    ) -> SelectionDecision:
        """The final look. ``planned`` defaults to the rounds evaluated, for a gate that plays them all at once."""
        interval = betting_paired_difference(evaluation, confidence=self._confidence, planned=planned)
        return self._decision(interval, planned, stopped_early=False)

    def rounds_needed(self, paired_sd: float, effect: float, power: float = 0.8) -> int:
        """Paired rounds for a gate to select a true mean difference of ``effect`` with ``power``.

        ``paired_sd`` is the standard deviation of one round's difference. The
        low end clears ``min_effect`` once the log capital of betting above it
        has reached ``log(2 / (1 - confidence))`` after some round. With stake
        ``b``, a round adds ``log(1 + b (d - min_effect))``, which averages
        about ``b g - b ** 2 (paired_sd ** 2 + g ** 2) / 2`` with a standard
        deviation of about ``b paired_sd``, where ``g`` is
        ``effect - min_effect`` and ``b`` is the stake the interval uses once
        it has seen the spread, tuned to the planned rounds. The count is the
        smallest at which a Brownian motion with that drift and spread reaches
        the level within it with probability ``power``
        (:func:`_reach_chance`). The capital is seen only after whole rounds,
        and a Brownian motion can cross the level between them, so the level
        is raised by ``OVERSHOOT`` of a round's standard deviation. The count
        is whole blocks of at least two rounds. Simulated power at it is 81
        to 82% for spreads of 0.3 to 0.7 and an effect of 0.10, with one look
        or four.
        """
        require(paired_sd > 0.0, f"paired_sd must be > 0, got {paired_sd}")
        require(effect > self._min_effect, f"effect must be above min_effect {self._min_effect}, got {effect}")
        require(0.0 < power < 1.0, f"power must be in (0, 1), got {power}")
        level = math.log(2.0 / (1.0 - self._confidence))
        gap = effect - self._min_effect
        cap = MAX_BET / (1.0 + self._min_effect)

        def enough(blocks: int) -> bool:
            rounds = blocks * self._looks
            bet = min(math.sqrt(2.0 * level / (rounds * paired_sd**2)), cap)
            drift = bet * gap - bet**2 * (paired_sd**2 + gap**2) / 2.0
            spread = bet * paired_sd
            return _reach_chance(drift, spread, level + OVERSHOOT * spread, rounds) >= power

        # Enough blocks stay enough with more, so double past the count and bisect back to it.
        short, blocks = 1, 2
        while not enough(blocks):
            short, blocks = blocks, blocks * 2
        while blocks - short > 1:
            middle = (short + blocks) // 2
            short, blocks = (short, middle) if enough(middle) else (middle, blocks)
        return blocks * self._looks

    def _decision(self, interval: PairedDifference, planned: int | None, *, stopped_early: bool) -> SelectionDecision:
        """Select when the low end clears ``min_effect``, unless the ends crossed.

        Each end keeps the furthest the capital reached, so tasks whose
        differences change with their order, such as ten wins and then sixty
        losses, can leave the low end above the high end. No single mean fits
        such tasks, which breaks the bound's assumption, so the selector
        rejects. With independent tasks this happens rarely, and rejecting
        never raises the false-accept rate.
        """
        crossed = interval.low > interval.high
        selected = interval.low > self._min_effect and not crossed
        planned = interval.tasks if planned is None else planned
        return SelectionDecision(
            outcome="select" if selected else "reject",
            policy=self.name,
            reason=(
                f"the {self._confidence * 100:g}% interval on the mean difference over {interval.tasks} of {planned}"
                f" planned tasks is [{interval.low:+.4f}, {interval.high:+.4f}], and selecting needs it above"
                f" {self._min_effect:+.4f}"
                + (". Its ends crossed, so no single mean fits the differences" if crossed else "")
            ),
            metrics={
                "tasks": float(interval.tasks),
                "planned_tasks": float(planned),
                "difference_mean": interval.mean,
                "difference_low": interval.low,
                "difference_high": interval.high,
                "interval_confidence": self._confidence,
                "min_effect": self._min_effect,
                "stopped_early": float(stopped_early),
            },
        )


def _reach_chance(drift: float, spread: float, level: float, steps: int) -> float:
    """The chance a Brownian motion with this drift and spread per step reaches ``level`` within ``steps``.

    It is ``Phi(a) + exp(2 drift level / spread ** 2) Phi(-c)``, where ``a``
    and ``c`` are ``drift steps - level`` and ``drift steps + level`` over
    ``spread sqrt(steps)``. On a rising walk the exponential can overflow
    while ``Phi(-c)`` underflows, so the second term is computed as
    ``pdf(a) Phi(-c) / pdf(c)``, which is the same value. Past ``c = 30``,
    where both parts of that ratio near underflow, ``1 / c`` is within 0.2%
    of it.
    """
    scale = spread * math.sqrt(steps)
    a = (drift * steps - level) / scale
    c = (drift * steps + level) / scale
    normal = NormalDist()
    if drift <= 0.0:
        return normal.cdf(a) + math.exp(2.0 * drift * level / spread**2) * normal.cdf(-c)
    ratio = 1.0 / c if c > 30.0 else math.erfc(c / math.sqrt(2.0)) / 2.0 / normal.pdf(c)
    return normal.cdf(a) + normal.pdf(a) * ratio


class AcceptAll:
    """Select every candidate. For methods whose next evaluation is the real judge."""

    name = "accept_all"

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        return SelectionDecision(outcome="select", policy=self.name, reason="every candidate is accepted")
