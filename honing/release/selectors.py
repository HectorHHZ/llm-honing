"""Selection policies the gate applies to an evaluated candidate."""

from __future__ import annotations

from honing.core import EvaluationResult, SelectionDecision, UpdateCandidate, require


class WinLossSelector:
    """Select when the candidate wins more tasks than it loses, by more than ``min_win_margin``.

    Suits changes expected to improve the model: training steps and harness
    edits. A tie on every task is a rejection.
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
            metrics={"wins": float(wins), "losses": float(losses)},
        )


class NoRegressionSelector:
    """Select when the mean score drops by no more than ``max_regression``.

    Suits changes made for cost rather than quality: a prune or a quantization
    earns its place by being cheaper, so it need not win, only not lose much.
    """

    name = "no_regression"

    def __init__(self, max_regression: float) -> None:
        require(max_regression >= 0.0, f"max_regression must be >= 0, got {max_regression}")
        self._max_regression = max_regression

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        regression = evaluation.incumbent_mean - evaluation.candidate_mean
        selected = regression <= self._max_regression
        return SelectionDecision(
            outcome="select" if selected else "reject",
            policy=self.name,
            reason=f"mean score changed by {-regression:+.4f}; the limit is -{self._max_regression:.4f}",
            metrics={"regression": regression},
        )


class AcceptAll:
    """Select every candidate. For methods whose next evaluation is the real judge."""

    name = "accept_all"

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        return SelectionDecision(outcome="select", policy=self.name, reason="every candidate is accepted")
