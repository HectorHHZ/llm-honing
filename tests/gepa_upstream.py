"""Replays a history of validation scores through upstream GEPA, for parity tests."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

from gepa.core.adapter import EvaluationBatch
from gepa.core.data_loader import ensure_loader
from gepa.core.engine import GEPAEngine
from gepa.core.result import GEPAResult
from gepa.core.state import GEPAState, ValsetEvaluation, initialize_gepa_state
from gepa.strategies.eval_policy import FullEvaluationPolicy


def upstream_replay(
    task_ids: Sequence[str], seed_scores: Sequence[float], children: Sequence[Sequence[float]]
) -> list[tuple[int, int, dict[str, set[int]], int]]:
    """Upstream's state after each child: best, GEPAResult.best_idx, fronts, metric calls.

    The seed goes through ``initialize_gepa_state`` and each child through
    ``_evaluate_programs_on_valset`` plus the two calls ``_add_evaluated_program``
    makes to charge and add it (engine.py:282, 291).
    """
    task_ids = tuple(task_ids)
    table = {("seed", task_id): score for task_id, score in zip(task_ids, seed_scores, strict=True)}
    for index, scores in enumerate(children):
        table.update({(f"child {index}", task_id): score for task_id, score in zip(task_ids, scores, strict=True)})

    def batch_evaluate(items: list[tuple[dict[str, str], list[str]]]) -> list[EvaluationBatch]:
        return [
            EvaluationBatch(outputs=[None] * len(batch), scores=[table[(program["text"], t)] for t in batch])
            for program, batch in items
        ]

    engine = SimpleNamespace(
        valset=ensure_loader(list(task_ids)),
        val_evaluation_policy=FullEvaluationPolicy(),
        _batch_evaluate=batch_evaluate,
    )
    seed_evaluation = ValsetEvaluation(
        outputs_by_val_id=dict.fromkeys(range(len(task_ids))),
        scores_by_val_id={index: table[("seed", task_id)] for index, task_id in enumerate(task_ids)},
    )
    state: GEPAState = initialize_gepa_state(
        run_dir=None,
        logger=SimpleNamespace(log=lambda *_: None),
        seed_candidate={"text": "seed"},
        seed_valset_evaluation=seed_evaluation,
    )
    history = []

    def snapshot() -> None:
        fronts = {task_ids[key]: set(members) for key, members in state.get_pareto_front_mapping().items()}
        best = FullEvaluationPolicy().get_best_program(state)
        history.append((best, GEPAResult.from_state(state).best_idx, fronts, state.total_num_evals))

    for index in range(len(children)):
        program = {"text": f"child {index}"}
        ((evaluation, calls),) = GEPAEngine._evaluate_programs_on_valset(engine, [program], state)
        discovered_at = state.total_num_evals
        state.increment_evals(calls)
        state.update_state_with_new_program([0], program, evaluation, None, discovered_at)
        snapshot()
    return history
