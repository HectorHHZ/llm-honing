"""A campaign: builds a candidate for each move, checks the rules, and settles it through the gate."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from honing.campaign import rules
from honing.core import (
    ALGORITHM_REQUIREMENTS,
    CampaignState,
    CandidateSelector,
    CoAdaptScheduler,
    CompressionBudget,
    CompressionStep,
    ContractError,
    DataProvider,
    EpisodeRecord,
    EpisodeRunner,
    EvaluationResult,
    EvolveContext,
    EvolveMethod,
    HarnessSpec,
    HoneRelease,
    ImportanceScorer,
    ModelEditor,
    ModelSnapshot,
    ModelSource,
    Move,
    PolicyRef,
    PruneSelector,
    QuantizePlan,
    Quantizer,
    QuantScheme,
    RecoverySpec,
    RolloutProvider,
    SelectionDecision,
    Task,
    TelemetryStore,
    Trainer,
    TrainingBatch,
    TrainJobSpec,
    TransformManifest,
    UpdateCandidate,
    require,
)
from honing.release import Gate, ReleaseChain


@dataclass(frozen=True)
class CompressionComponents:
    """The Compression plane a campaign uses."""

    scorer: ImportanceScorer
    selector: PruneSelector
    editor: ModelEditor
    quantizer: Quantizer
    telemetry: TelemetryStore


@dataclass(frozen=True)
class HarnessComponents:
    """The Harness plane a campaign uses."""

    runner: EpisodeRunner
    method: EvolveMethod
    tasks: tuple[Task, ...]
    episode_budget: int = 0
    samples_per_task: int = 1

    def __post_init__(self) -> None:
        require(len(self.tasks) > 0, "a campaign needs at least one task")
        require(self.samples_per_task > 0, "samples_per_task must be > 0")


@dataclass(frozen=True)
class CampaignDefaults:
    """What a move uses when it does not say otherwise."""

    train: TrainJobSpec
    budget: CompressionBudget
    recovery: RecoverySpec | None
    scheme: QuantScheme


@dataclass(frozen=True)
class StepOutcome:
    """What one move produced. ``release`` is set only when the candidate was published."""

    move: Move
    decision: SelectionDecision | None
    evaluation: EvaluationResult | None
    release: HoneRelease | None
    skipped_reason: str | None = None


def _size_metrics(snapshot: ModelSnapshot) -> dict[str, float]:
    return {"parameters": float(snapshot.parameters)}


def _policy_of(snapshot: ModelSnapshot) -> PolicyRef:
    return PolicyRef(checkpoint=snapshot.checkpoint, architecture=snapshot.architecture, variant="master")


class Campaign:
    """Runs moves against one release chain.

    Every move builds an :class:`~honing.core.UpdateCandidate`, passes the rules
    in :mod:`honing.campaign.rules`, and is settled by the one gate. The harness
    method's state is carried across rejected steps in memory and published
    with the next accepted release, whatever its kind.
    """

    def __init__(
        self,
        *,
        chain: ReleaseChain,
        gate: Gate,
        trainer: Trainer,
        data: DataProvider,
        compression: CompressionComponents,
        harness: HarnessComponents,
        defaults: CampaignDefaults,
        rollout: RolloutProvider | None = None,
    ) -> None:
        self._chain = chain
        self._gate = gate
        self._trainer = trainer
        self._data = data
        self._compression = compression
        self._harness = harness
        self._defaults = defaults
        self._rollout = rollout
        self._pending_method_state: Mapping[str, object] | None = None
        self._recent_records: tuple[EpisodeRecord, ...] = ()

    @property
    def chain(self) -> ReleaseChain:
        return self._chain

    def start(self, source: ModelSource, work_dir: Path, harness_seed: HarnessSpec) -> HoneRelease:
        """Check the planes fit together, materialize the initial model, and publish it as release 0."""
        method = self._harness.method
        rules.method_supported_by_runner(method.requires, self._harness.runner.support)
        rules.spec_runnable(harness_seed, self._harness.runner.support)
        snapshot = source.materialize(work_dir)
        rules.architecture_supported(snapshot.architecture, self._trainer.support)
        candidate = UpdateCandidate(
            kind="init",
            parent_id=None,
            master=snapshot,
            harness=harness_seed,
            method_state=method.initial_state(harness_seed),
            metrics=_size_metrics(snapshot),
        )
        rules.master_is_hf_safetensors(candidate)
        return self._chain.initialize(candidate)

    def run(self, scheduler: CoAdaptScheduler, *, max_steps: int = 1000) -> tuple[StepOutcome, ...]:
        """Ask ``scheduler`` for moves until it stops or ``max_steps`` is reached."""
        require(max_steps > 0, f"max_steps must be > 0, got {max_steps}")
        outcomes: list[StepOutcome] = []
        for step in range(max_steps):
            state = CampaignState(head=self._chain.head(), releases=self._chain.releases(), step=step)
            move = scheduler.next_move(state)
            if move.kind == "stop":
                break
            outcomes.append(self.advance(move))
        return tuple(outcomes)

    def advance(self, move: Move) -> StepOutcome:
        """Build, check, and settle the candidate for one move."""
        head = self._chain.head()
        if move.kind == "harness":
            return self._advance_harness(move, head)
        if move.kind == "train":
            candidate = self._train_candidate(move.train if move.train is not None else self._defaults.train, head)
        elif move.kind == "prune":
            budget = move.budget if move.budget is not None else self._defaults.budget
            recovery = move.recovery if move.recovery is not None else self._defaults.recovery
            candidate = self._prune_candidate(budget, recovery, head)
        elif move.kind == "quantize":
            candidate = self._quantize_candidate(
                move.scheme if move.scheme is not None else self._defaults.scheme, head
            )
        else:
            raise ContractError(f"a campaign cannot advance a {move.kind!r} move")
        return self._settle(move, candidate, head, selector=None)

    def rollback(self, release_id: str) -> HoneRelease:
        """Restore an earlier release as the head. Method state not yet published is discarded."""
        self._pending_method_state = None
        return self._chain.rollback(release_id)

    # ------------------------------------------------------------ weight candidates

    def _train_candidate(self, job: TrainJobSpec, head: HoneRelease) -> UpdateCandidate:
        self._check_job(job)
        batch = self._batch_for(job, head, head.policy(), release_id=head.release_id)
        rules.batch_meets_contract(job, batch)
        result = self._trainer.train(job, head.master, batch)
        rules.architecture_supported(result.snapshot.architecture, self._trainer.support)
        return UpdateCandidate(
            kind="train",
            parent_id=head.release_id,
            master=result.snapshot,
            harness=head.harness,
            trained_numerics=job.numerics,
            method_state=self._method_state(head),
            metrics={**_size_metrics(result.snapshot), **result.metrics},
        )

    def _prune_candidate(
        self, budget: CompressionBudget, recovery: RecoverySpec | None, head: HoneRelease
    ) -> UpdateCandidate:
        compression = self._compression
        importance = compression.scorer.score(head.master, compression.telemetry.window())
        plan = compression.selector.select(importance, head.master, budget)
        step = CompressionStep(transform=plan, recovery=recovery)
        recovery_job = None
        if step.recovery is not None and step.settle == "after_recovery":
            recovery_job = step.recovery.as_job(head.trained_numerics)
        # Everything that can be checked without the edited weights is checked
        # here, so a refused plan never reaches the editor.
        rules.architecture_supported(plan.after, self._trainer.support)
        rules.recovery_teacher_fits_new_shape(step)
        if recovery_job is not None:
            self._check_job(recovery_job)

        snapshot = compression.editor.apply(head.master, plan)
        metrics = {"estimated_parameter_change": float(plan.estimated.parameters)}
        if recovery_job is not None:
            # The edited weights are not published, so their rollout samples name no release.
            batch = self._batch_for(recovery_job, head, _policy_of(snapshot), release_id=None)
            rules.batch_meets_contract(recovery_job, batch)
            result = self._trainer.train(recovery_job, snapshot, batch)
            snapshot = result.snapshot
            metrics.update(result.metrics)
        return UpdateCandidate(
            kind="prune",
            parent_id=head.release_id,
            master=snapshot,
            transforms=(TransformManifest.of(plan),),
            harness=head.harness,
            trained_numerics=head.trained_numerics,
            method_state=self._method_state(head),
            metrics={**_size_metrics(snapshot), **metrics},
        )

    def _quantize_candidate(self, scheme: QuantScheme, head: HoneRelease) -> UpdateCandidate:
        rules.qat_scheme_matches_export(head.trained_numerics, scheme)
        plan = QuantizePlan(scheme=scheme, source=head.master.checkpoint)
        serving = self._compression.quantizer.quantize(head.master, plan)
        return UpdateCandidate(
            kind="quantize",
            parent_id=head.release_id,
            master=head.master,
            serving=serving,
            transforms=(TransformManifest.of(plan),),
            harness=head.harness,
            trained_numerics=head.trained_numerics,
            method_state=self._method_state(head),
            metrics=_size_metrics(head.master),
        )

    # ------------------------------------------------------------ harness candidates

    def _advance_harness(self, move: Move, head: HoneRelease) -> StepOutcome:
        method, runner = self._harness.method, self._harness.runner
        if head.harness is None:
            raise ContractError("the head has no harness to evolve; start the campaign with a harness seed")
        state = self._method_state(head)
        context = EvolveContext(
            current=head.harness,
            policy=head.policy(),
            tasks=self._harness.tasks,
            records=self._recent_records,
            state=state,
            runner=runner,
            episode_budget=self._harness.episode_budget,
        )
        proposal = method.propose(context)
        if proposal is None:
            return StepOutcome(
                move=move,
                decision=None,
                evaluation=None,
                release=None,
                skipped_reason=f"{method.name} proposed nothing",
            )
        rules.spec_runnable(proposal.spec, runner.support)
        candidate = UpdateCandidate(
            kind="harness",
            parent_id=head.release_id,
            master=head.master,
            serving=head.serving,
            harness=proposal.spec,
            trained_numerics=head.trained_numerics,
            method_state=state,
            metrics=_size_metrics(head.master),
        )
        self._check_candidate(candidate)
        evaluation, decision = self._gate.judge(candidate, head, selector=method.selector)
        self._recent_records = evaluation.records
        # The method learns from every verdict, so its new state is attached
        # whether the candidate is published or rejected.
        settled = replace(candidate, method_state=method.record(state, proposal, evaluation, decision))
        return self._publish_or_reject(move, settled, evaluation, decision)

    # ------------------------------------------------------------ shared

    def _settle(
        self, move: Move, candidate: UpdateCandidate, head: HoneRelease, selector: CandidateSelector | None
    ) -> StepOutcome:
        self._check_candidate(candidate)
        evaluation, decision = self._gate.judge(candidate, head, selector=selector)
        return self._publish_or_reject(move, candidate, evaluation, decision)

    def _publish_or_reject(
        self, move: Move, candidate: UpdateCandidate, evaluation: EvaluationResult, decision: SelectionDecision
    ) -> StepOutcome:
        if decision.selected:
            release = self._chain.commit(candidate, decision)
            self._pending_method_state = None
            return StepOutcome(move=move, decision=decision, evaluation=evaluation, release=release)
        self._chain.record_rejection(candidate, decision)
        self._pending_method_state = candidate.method_state
        return StepOutcome(move=move, decision=decision, evaluation=evaluation, release=None)

    @staticmethod
    def _check_candidate(candidate: UpdateCandidate) -> None:
        rules.master_is_hf_safetensors(candidate)
        rules.serving_built_from_master(candidate)

    def _check_job(self, job: TrainJobSpec) -> None:
        rules.algorithm_requirements_met(job, self._trainer.support)
        if ALGORITHM_REQUIREMENTS[job.algorithm].requires_rollout and self._rollout is None:
            raise ContractError(f"{job.algorithm} learns from rollouts, but the campaign has no rollout provider")

    def _method_state(self, head: HoneRelease) -> Mapping[str, object]:
        return self._pending_method_state if self._pending_method_state is not None else head.method_state

    def _batch_for(
        self, job: TrainJobSpec, head: HoneRelease, policy: PolicyRef, *, release_id: str | None
    ) -> TrainingBatch | None:
        requirements = ALGORITHM_REQUIREMENTS[job.algorithm]
        if not requirements.requires_rollout:
            return self._data.next_batch(requirements.data_contract())
        if self._rollout is None:
            raise ContractError(f"{job.algorithm} learns from rollouts, but the campaign has no rollout provider")
        batch = self._rollout.collect(
            self._harness.tasks,
            policy,
            head.harness if job.rollout == "harness" else None,
            release_id,
            self._harness.samples_per_task,
        )
        rules.batch_from_one_release(batch, head)
        return batch
