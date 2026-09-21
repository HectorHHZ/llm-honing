"""The one gate every candidate passes: evaluate against the head, then select."""

from __future__ import annotations

from collections.abc import Mapping

from honing.core import (
    CandidateKind,
    CandidateSelector,
    ContractError,
    EvaluationResult,
    Evaluator,
    HoneRelease,
    ReleaseError,
    SelectionDecision,
    UpdateCandidate,
)


class Gate:
    """Judges candidates of every kind with one evaluator and a selector per kind.

    The gate refuses an evaluation that scored the wrong weights. A candidate
    with serving weights must be scored on them, because they are what ships;
    scoring the master instead would hide exactly the regression quantization
    can cause.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        default_selector: CandidateSelector,
        selectors: Mapping[CandidateKind, CandidateSelector] | None = None,
    ) -> None:
        self._evaluator = evaluator
        self._default_selector = default_selector
        self._selectors = dict(selectors or {})

    def selector_for(self, kind: CandidateKind) -> CandidateSelector:
        return self._selectors.get(kind, self._default_selector)

    def judge(
        self, candidate: UpdateCandidate, incumbent: HoneRelease, selector: CandidateSelector | None = None
    ) -> tuple[EvaluationResult, SelectionDecision]:
        """Evaluate ``candidate`` against ``incumbent`` and decide. ``selector`` overrides the per-kind one."""
        if candidate.parent_id != incumbent.release_id:
            raise ReleaseError(
                f"candidate was built on {candidate.parent_id} but is being judged against {incumbent.release_id}"
            )
        evaluation = self._evaluator.evaluate(candidate, incumbent)
        if evaluation.evaluated_variant != candidate.evaluated_variant:
            raise ContractError(
                f"the evaluator scored the {evaluation.evaluated_variant} weights of a candidate that ships "
                f"its {candidate.evaluated_variant} weights"
            )
        chosen = selector if selector is not None else self.selector_for(candidate.kind)
        return evaluation, chosen.decide(candidate, evaluation)
