"""The tank gate: the paired interval selector on validation seeds no earlier gate played.

A loop that reads gate verdicts can overfit rounds it sees at every step, so
each gate draws new seeds from the ``val`` pool
(:data:`~honing.backends.tank.scenarios.SEED_POOLS`). The suite file's ``val``
tasks stay fixed for a method with its own validation set, as GEPA has for its
Pareto front (:class:`~honing.campaign.HarnessComponents`). A suite with a
task the pool reaches is refused (:func:`refuse_pool_overlap`), so no gate
plays a seed the suite trains, selects or reports on.
"""

from __future__ import annotations

from statistics import fmean

from honing.backends.tank.scenarios import SEED_POOLS, TankSuite, pool_tasks
from honing.core import EpisodeRunner, EvaluationResult, HoneRelease, TaskGrader, UpdateCandidate, require
from honing.release import EpisodeEvaluator, Gate, PairedIntervalSelector


def refuse_pool_overlap(suite: TankSuite) -> None:
    """Refuse a suite with a task the gate could play: on a ``val`` pool seed in any split, or named as a pool task."""
    pool = SEED_POOLS["val"]
    overlap = suite.pool_overlap("val")
    require(
        not overlap,
        f"the fresh-seed gate draws its rounds from the val pool, seeds {pool.start} to {pool.stop - 1}, "
        "so it could replay the seeds of these suite tasks: "
        + ", ".join(f"{task_id} (seed {seed})" for task_id, seed in overlap.items()),
    )


class FreshSeedEvaluator:
    """Satisfies :class:`~honing.core.Evaluator`, playing seeds no earlier call played.

    Each call takes the next ``rounds // 2`` seeds of the ``val`` pool and
    plays each from both seats, under the candidate and under the incumbent.
    Both seats of a seed play the same maze, so their results can move
    together, and counting them as two independent tasks would narrow the
    interval as if the second seat were a new maze. So the evaluation has one
    task per seed, ``tank-val-s<seed>``, scored by the mean of its two seats,
    and keeps every seat's episode record for audit.

    A gate that plays in blocks calls it once per block, so one gate's seeds
    run from a first seed to a last with none skipped, and the next gate
    starts after them. Seeds are taken before their rounds are played, so a
    gate that fails partway never hands its seeds to the next. The count lives
    in memory: one evaluator serves one campaign.

    ``first_seed`` is where the first gate starts, the start of the pool by
    default. Campaigns that start at the same seed play the same gates, which
    pairs two arms of an experiment. Repeats of one arm start elsewhere.
    """

    def __init__(
        self, runner: EpisodeRunner, grader: TaskGrader, suite: TankSuite, rounds: int, first_seed: int | None = None
    ) -> None:
        require(
            rounds > 0 and rounds % 2 == 0,
            f"each seed is played from both seats, so rounds must be even, got {rounds}",
        )
        pool = SEED_POOLS["val"]
        first = pool.start if first_seed is None else first_seed
        require(
            not isinstance(first, bool) and isinstance(first, int) and first in pool,
            f"first_seed must be a seed of the val pool, {pool.start} to {pool.stop - 1}, got {first_seed!r}",
        )
        refuse_pool_overlap(suite)
        self._runner = runner
        self._grader = grader
        self._suite = suite
        self._rounds = rounds
        self._next_seed = first

    def evaluate(self, candidate: UpdateCandidate, incumbent: HoneRelease) -> EvaluationResult:
        seeds = range(self._next_seed, self._next_seed + self._rounds // 2)
        require(seeds.stop <= SEED_POOLS["val"].stop, "the val seed pool is used up")
        self._next_seed = seeds.stop
        seats = EpisodeEvaluator(self._runner, self._grader, pool_tasks("val", seeds)).evaluate(candidate, incumbent)

        # pool_tasks lists both seats of a seed together, and the evaluator keeps that order.
        def per_seed(scores: tuple[float, ...]) -> tuple[float, ...]:
            return tuple(fmean(scores[index : index + 2]) for index in range(0, len(scores), 2))

        return EvaluationResult(
            task_ids=tuple(f"tank-val-s{seed}" for seed in seeds),
            candidate_scores=per_seed(seats.candidate_scores),
            incumbent_scores=per_seed(seats.incumbent_scores),
            evaluated_variant=seats.evaluated_variant,
            records=seats.records,
        )

    def metrics(self, evaluation: EvaluationResult) -> dict[str, float]:
        """The first and last seed a gate played, and its rounds. It played every seed between, from both seats."""
        seeds = sorted({self._suite.scenario(record.task_id).seed for record in evaluation.records})
        require(len(seeds) == seeds[-1] - seeds[0] + 1, f"the gate's seeds {seeds} are not one run of seeds")
        return {"seed_first": float(seeds[0]), "seed_last": float(seeds[-1]), "rounds": float(2 * len(seeds))}


def fresh_seed_gate(
    runner: EpisodeRunner,
    grader: TaskGrader,
    suite: TankSuite,
    *,
    rounds: int = 64,
    looks: int = 1,
    min_effect: float = 0.0,
    confidence: float = 0.95,
    first_seed: int | None = None,
) -> Gate:
    """``rounds`` fresh rounds in ``looks`` blocks, settled by :class:`~honing.release.PairedIntervalSelector`.

    The selector sees one difference per seed (:class:`FreshSeedEvaluator`),
    so ``rounds`` is twice the gate's sample size. The defaults are
    placeholders. #165 measures the spread of tank rounds, and
    :meth:`~honing.release.PairedIntervalSelector.rounds_needed`, given the
    standard deviation of a seed's mean difference, returns the seeds a gate
    needs. Until then the gate plays 64 rounds (32 seeds) in one look. The
    direction edit's gate measured a seed standard deviation of 0.42, at which
    that is enough for 80% power only on a gain near 0.32, and a
    ``min_effect`` of 0 selects any gain whose interval clears zero.
    ``first_seed`` is the val pool seed the first gate starts at
    (:class:`FreshSeedEvaluator`).
    """
    selector = PairedIntervalSelector(min_effect=min_effect, confidence=confidence, looks=looks)
    require(
        rounds % (2 * looks) == 0,
        f"{rounds} rounds do not split into {looks} blocks of whole seeds, each played from both seats",
    )
    evaluator = FreshSeedEvaluator(runner, grader, suite, rounds // looks, first_seed)
    return Gate(evaluator, default_selector=selector, metrics=evaluator.metrics)
