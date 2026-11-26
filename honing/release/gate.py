"""The one gate every candidate passes: evaluate against the head, then select."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace

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
from honing.release.selectors import PairedIntervalSelector


class Gate:
    """Judges candidates of every kind with one evaluator and a selector per kind.

    The gate refuses an evaluation that scored the wrong weights. A candidate
    with serving weights must be scored on them, because they are what ships;
    scoring the master instead would hide exactly the regression quantization
    can cause.

    A :class:`~honing.release.PairedIntervalSelector` with more than one look
    makes the gate sequential. The gate calls the evaluator once per block and
    asks the selector to look after each block, and stops at the first look
    that decides. Every look tunes its bets to ``looks`` blocks the size of
    the first, and a gate whose blocks add up to more is refused. Every call
    must play tasks no earlier call played, as
    :class:`~honing.backends.tank.FreshSeedEvaluator` does. An evaluator that
    plays the same tasks again is refused at the second block, since its
    rounds would count twice. ``metrics``, when given, adds what it returns
    for the evaluation to every decision, such as the seeds a gate played.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        default_selector: CandidateSelector,
        selectors: Mapping[CandidateKind, CandidateSelector] | None = None,
        metrics: Callable[[EvaluationResult], Mapping[str, float]] | None = None,
    ) -> None:
        self._evaluator = evaluator
        self._default_selector = default_selector
        self._selectors = dict(selectors or {})
        self._metrics = metrics

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
        chosen = selector if selector is not None else self.selector_for(candidate.kind)
        evaluation = self._evaluate(candidate, incumbent)
        if isinstance(chosen, PairedIntervalSelector):
            evaluation, decision = self._look_by_block(chosen, candidate, incumbent, evaluation)
        else:
            decision = chosen.decide(candidate, evaluation)
        if self._metrics is not None:
            decision = replace(decision, metrics={**decision.metrics, **self._metrics(evaluation)})
        return evaluation, decision

    def _look_by_block(
        self,
        selector: PairedIntervalSelector,
        candidate: UpdateCandidate,
        incumbent: HoneRelease,
        evaluation: EvaluationResult,
    ) -> tuple[EvaluationResult, SelectionDecision]:
        """Look after each block, every look planning ``looks`` blocks the size of the first."""
        planned = selector.looks * len(evaluation.task_ids)
        for _ in range(selector.looks - 1):
            decision = selector.look(candidate, evaluation, final=False, planned=planned)
            if decision is not None:
                return evaluation, decision
            evaluation = _joined(evaluation, self._evaluate(candidate, incumbent))
        return evaluation, selector.decide(candidate, evaluation, planned=planned)

    def _evaluate(self, candidate: UpdateCandidate, incumbent: HoneRelease) -> EvaluationResult:
        evaluation = self._evaluator.evaluate(candidate, incumbent)
        if evaluation.evaluated_variant != candidate.evaluated_variant:
            raise ContractError(
                f"the evaluator scored the {evaluation.evaluated_variant} weights of a candidate that ships "
                f"its {candidate.evaluated_variant} weights"
            )
        return evaluation


def _joined(first: EvaluationResult, second: EvaluationResult) -> EvaluationResult:
    """Two blocks of one gate as one evaluation. ``EvaluationResult`` refuses a task scored in both."""
    return EvaluationResult(
        task_ids=(*first.task_ids, *second.task_ids),
        candidate_scores=(*first.candidate_scores, *second.candidate_scores),
        incumbent_scores=(*first.incumbent_scores, *second.incumbent_scores),
        evaluated_variant=first.evaluated_variant,
        records=(*first.records, *second.records),
    )
