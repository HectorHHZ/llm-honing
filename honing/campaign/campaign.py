"""A campaign: builds a candidate for each move, checks the rules, and settles it through the gate."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from honing.campaign import rules
from honing.campaign.cost import CampaignBudget, CostMeter
from honing.core import (
    CampaignState,
    CandidateSelector,
    CoAdaptScheduler,
    CompressionBudget,
    CompressionStep,
    ContractError,
    DataProvider,
    EpisodePurpose,
    EpisodeRecord,
    EpisodeRunner,
    EvaluationResult,
    Evaluator,
    EvolveContext,
    EvolveMethod,
    HarnessMoveConfig,
    HarnessSpec,
    HoneRelease,
    ImportanceScorer,
    ModelEditor,
    ModelSnapshot,
    ModelSource,
    Move,
    MoveCost,
    PolicyRef,
    PruneSelector,
    QuantizePlan,
    Quantizer,
    QuantScheme,
    RecoverySpec,
    RolloutProvider,
    SelectionDecision,
    StepRecord,
    StepResult,
    Task,
    TaskGrader,
    TelemetryStore,
    Trainer,
    TrainingBatch,
    TrainJobSpec,
    TransformManifest,
    UpdateCandidate,
    charging,
    require,
    requirements_for,
)
from honing.release import Gate, ReleaseChain, gate_metrics


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
    """The Harness plane a campaign uses. A move's :class:`~honing.core.HarnessMoveConfig` overrides it.

    A method's selector replaces the gate's for harness candidates. With
    ``method_evaluator`` set, both must select instead: the method's selector
    decides on that evaluator's run, and :meth:`~honing.core.EvolveMethod.record`
    sees that run, so a method such as GEPA keeps its own fixed validation
    tasks. A candidate it selects then goes through the gate with the gate's
    own selector, on the gate's evaluator, which may draw fresh tasks.
    """

    runner: EpisodeRunner
    grader: TaskGrader
    method: EvolveMethod
    tasks: tuple[Task, ...]
    episode_budget: int = 0
    samples_per_task: int = 1
    method_evaluator: Evaluator | None = None

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
    """What one move produced. ``release`` is set only when the candidate was published.

    ``cost`` is what the move spent, whatever its verdict; a campaign always sets it.
    """

    move: Move
    decision: SelectionDecision | None
    evaluation: EvaluationResult | None
    release: HoneRelease | None
    skipped_reason: str | None = None
    cost: MoveCost | None = None


def _size_metrics(snapshot: ModelSnapshot) -> dict[str, float]:
    return {"parameters": float(snapshot.parameters)}


def _policy_of(snapshot: ModelSnapshot) -> PolicyRef:
    return PolicyRef(checkpoint=snapshot.checkpoint, architecture=snapshot.architecture, variant="master")


def _both(method: SelectionDecision, gate: SelectionDecision) -> SelectionDecision:
    """The gate's verdict on a candidate the method selected, with both reasons and both sets of metrics."""
    return SelectionDecision(
        outcome=gate.outcome,
        policy=f"{method.policy}+{gate.policy}",
        reason=f"{method.policy}: {method.reason}; {gate.policy}: {gate.reason}",
        metrics={**method.metrics, **gate.metrics},
    )


def _step_record(step: int, parent_id: str, outcome: StepOutcome) -> StepRecord:
    decision, release, evaluation = outcome.decision, outcome.release, outcome.evaluation
    result: StepResult
    if decision is None:
        result, reason = "skipped", outcome.skipped_reason or ""
    else:
        result, reason = ("published" if release is not None else "rejected"), decision.reason
    return StepRecord(
        step=step,
        move=outcome.move,
        result=result,
        parent_id=parent_id,
        reason=reason,
        label=outcome.move.harness.label if outcome.move.harness is not None else None,
        release_id=release.release_id if release is not None else None,
        # The same numbers the settled candidate's record holds: the bootstrap is seeded.
        gate_metrics=gate_metrics(evaluation) if evaluation is not None else {},
        # The cost its commit record line holds, so a scheduler can weigh a move's gain against what it spent.
        cost=outcome.cost,
    )


class Campaign:
    """Runs moves against one release chain.

    Every move builds an :class:`~honing.core.UpdateCandidate`, passes the rules
    in :mod:`honing.campaign.rules`, and is settled by the one gate. The harness
    method's state is carried across rejected and skipped steps in memory and
    published with the next accepted release, whatever its kind. A skipped
    step is still written to the commit record, with the episodes it spent.

    Every move it advances, published, rejected, or skipped, is added to the
    history a scheduler reads from :meth:`state`. When a scheduler chooses the
    move (:meth:`step`), its state is written into every commit record line
    from then on, and neither a rejection nor a rollback changes it.

    Every move's :class:`~honing.core.MoveCost` is written to its commit record
    (published, rejected or skipped) and its known parts to the candidate's
    ``cost_*`` metrics. With a ``budget``, :meth:`step` admits no move once the
    spend reaches the limit; a move already admitted runs to the end, so the
    spend can overshoot the limit by at most one move's cost. Under a dollar
    budget, a move whose dollars are unknown is recorded and then stops the
    campaign the same way, because it can no longer tell what it has spent;
    :meth:`budget_report` names what could not be priced.
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
        budget: CampaignBudget | None = None,
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
        self._history: list[StepRecord] = []
        self._budget = budget
        self._spent = MoveCost()
        self._meter: CostMeter | None = None
        self._stopped: Literal["limit", "unknown_cost"] | None = None
        self._stop_reason: str | None = None

    @property
    def chain(self) -> ReleaseChain:
        return self._chain

    def state(self) -> CampaignState:
        """What a scheduler sees now: the head, the releases, and every move advanced so far."""
        return CampaignState(
            head=self._chain.head(),
            releases=self._chain.releases(),
            step=len(self._history),
            history=tuple(self._history),
        )

    @property
    def budget(self) -> CampaignBudget | None:
        """The budget the campaign keeps to, or ``None`` when it has none."""
        return self._budget

    @property
    def spent(self) -> MoveCost:
        """The sum of every move's cost so far."""
        return self._spent

    def admits_move(self) -> bool:
        """Whether the budget leaves room for another move; always true without a budget.

        Once it returns false the campaign counts as stopped by its budget.
        """
        if self._budget is None:
            return True
        if self._stopped == "unknown_cost":
            return False
        spent = self._budget.spent(self._spent)
        if spent is not None and spent < self._budget.limit:
            return True
        self._stopped = "limit"
        self._stop_reason = f"spent {spent} of the {self._budget.limit} {self._budget.unit} budget"
        return False

    def budget_report(self) -> dict[str, object] | None:
        """The budget's unit and limit, what was spent in that unit, the overshoot, and what stopped the run.

        ``stopped`` is ``None`` while the budget admits moves, ``"limit"`` once the
        spend reached the limit, and ``"unknown_cost"`` once a move's dollars could
        not be priced under a dollar budget; ``reason`` says which.
        """
        if self._budget is None:
            return None
        spent = self._budget.spent(self._spent)
        return {
            "unit": self._budget.unit,
            "limit": self._budget.limit,
            "spent": spent,
            "overshoot": None if spent is None else max(0.0, spent - self._budget.limit),
            "stopped": self._stopped,
            "reason": self._stop_reason,
        }

    def start(self, source: ModelSource, work_dir: Path, harness_seed: HarnessSpec) -> HoneRelease:
        """Check the planes fit together, materialize the initial model, and publish it as release 0."""
        method = self._harness.method
        rules.method_supported_by_runner(method.requires, self._harness.runner.support)
        rules.spec_runnable(harness_seed, self._harness.runner.support)
        initial_state = method.initial_state(harness_seed)
        rules.method_state_schema_matches(initial_state, method.name, method.state_schema_version)
        snapshot = source.materialize(work_dir)
        rules.architecture_supported(snapshot.architecture, self._trainer.support)
        candidate = UpdateCandidate(
            kind="init",
            parent_id=None,
            master=snapshot,
            harness=harness_seed,
            method_state=initial_state,
            metrics=_size_metrics(snapshot),
        )
        rules.master_is_hf_safetensors(candidate)
        return self._chain.initialize(candidate)

    def run(self, scheduler: CoAdaptScheduler, *, max_steps: int = 1000) -> tuple[StepOutcome, ...]:
        """Ask ``scheduler`` for moves until it stops or ``max_steps`` is reached."""
        require(max_steps > 0, f"max_steps must be > 0, got {max_steps}")
        outcomes: list[StepOutcome] = []
        for _ in range(max_steps):
            outcome = self.step(scheduler)
            if outcome is None:
                break
            outcomes.append(outcome)
        return tuple(outcomes)

    def step(self, scheduler: CoAdaptScheduler) -> StepOutcome | None:
        """Ask ``scheduler`` for the next move and advance it. Returns ``None`` when the scheduler stops.

        It also returns ``None``, without asking the scheduler, once the budget
        admits no more moves (:meth:`admits_move`), so the scheduler's state
        never records a choice for a move that does not run.
        :meth:`budget_report` tells the two stops apart.

        The scheduler's state is exported right after its choice, before the
        move's result is known, checked against its declared schema, and
        written into every record line from here on. The scheduler reads the
        result in the history at its next choice. The history is only in
        memory, so a restart resumes the scheduler from its last choice
        without the last move's result (see
        :class:`~honing.core.CoAdaptScheduler`).
        """
        if not self.admits_move():
            return None
        move = scheduler.next_move(self.state())
        exported = scheduler.export_state()
        # Refused now, not at a restart hours later when the state could no longer be restored.
        rules.scheduler_state_schema_matches(
            exported, written_by=scheduler.name, scheduler=scheduler.name, expected=scheduler.state_schema_version
        )
        self._chain.set_scheduler_state(scheduler.name, exported)
        if move.kind == "stop":
            return None
        return self.advance(move)

    def advance(self, move: Move) -> StepOutcome:
        """Build, check, and settle the candidate for one move, charge what it spends, and add it to the history."""
        parent_id = self._chain.head().release_id
        meter = CostMeter(self._budget)
        self._meter = meter
        try:
            with charging(meter):
                outcome = self._advance(move)
        finally:
            self._meter = None
        assert outcome.cost is not None
        self._spent = self._spent.plus(outcome.cost)
        if self._budget is not None and self._budget.unit == "usd" and outcome.cost.usd is None:
            # The move is already recorded; stopping here keeps an unpriced spend from passing as free.
            unknown = [
                f"chat model {model} returned no price; set budget.chat_usd_per_million_tokens[{model}]"
                for model in meter.unpriced_models
            ]
            if outcome.cost.gpu_seconds is None:
                unknown.append(
                    f"the trainer {type(self._trainer).__name__} reported no gpu_seconds; it must set "
                    "TrainResult.gpu_seconds"
                )
            self._stopped = "unknown_cost"
            self._stop_reason = f"the {move.kind} move's dollars are unknown: " + "; ".join(unknown)
        self._history.append(_step_record(len(self._history), parent_id, outcome))
        return outcome

    def _advance(self, move: Move) -> StepOutcome:
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
        """Restore an earlier release as the head. Method state not yet published is discarded.

        The scheduler's state is kept: it is the scheduler's memory, not part of a release.
        """
        self._pending_method_state = None
        return self._chain.rollback(release_id)

    # ------------------------------------------------------------ weight candidates

    def _train_candidate(self, job: TrainJobSpec, head: HoneRelease) -> UpdateCandidate:
        self._check_job(job)
        batch = self._batch_for(job, head, head.policy(), release_id=head.release_id)
        rules.batch_meets_contract(job, batch)
        result = self._trainer.train(job, head.master, batch)
        self._metering().charge_gpu_seconds(result.gpu_seconds)
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
            self._metering().charge_gpu_seconds(result.gpu_seconds)
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
        harness = self._harness
        method, runner = harness.method, harness.runner
        if head.harness is None:
            raise ContractError("the head has no harness to evolve; start the campaign with a harness seed")
        config = move.harness if move.harness is not None else HarnessMoveConfig()
        if config.method is not None and config.method != method.name:
            raise ContractError(f"the move is for method {config.method!r}, but the campaign runs {method.name!r}")
        state = self._method_state(head)
        rules.method_state_schema_matches(state, method.name, method.state_schema_version)
        context = EvolveContext(
            current=head.harness,
            policy=head.policy(),
            tasks=config.tasks or harness.tasks,
            records=self._recent_records,
            state=state,
            runner=runner,
            grader=harness.grader,
            episode_budget=harness.episode_budget if config.episode_budget is None else config.episode_budget,
        )
        with self._purpose("method"):
            result = method.propose(context)
        proposal = result.proposal
        if proposal is None:
            reason = f"{method.name} proposed nothing"
            cost = self._metering().close()
            # Recorded before the pending state moves, so a restart can see what the attempt spent.
            self._chain.record_skip(
                parent_id=head.release_id,
                method=method.name,
                reason=reason,
                state=result.state,
                episodes_spent=result.episodes_spent,
                label=config.label,
                cost=cost,
            )
            self._pending_method_state = result.state
            return StepOutcome(
                move=move, decision=None, evaluation=None, release=None, skipped_reason=reason, cost=cost
            )
        rules.spec_runnable(proposal.spec, runner.support)
        candidate = UpdateCandidate(
            kind="harness",
            parent_id=head.release_id,
            master=head.master,
            serving=head.serving,
            harness=proposal.spec,
            trained_numerics=head.trained_numerics,
            method_state=result.state,
            metrics={**_size_metrics(head.master), "method_episodes": float(result.episodes_spent)},
            move_label=config.label,
        )
        self._check_candidate(candidate)
        if harness.method_evaluator is None:
            with self._purpose("gate"):
                evaluation, decision = self._gate.judge(candidate, head, selector=method.selector)
            method_evaluation = evaluation
        else:
            method_gate = Gate(harness.method_evaluator, default_selector=method.selector)
            # The method's own validation run is its selection work, as upstream GEPA charges it to its budget,
            # so "gate" counts only the campaign gate's episodes, the same overhead for every move kind.
            with self._purpose("method"):
                method_evaluation, decision = method_gate.judge(candidate, head)
            evaluation = method_evaluation
            if decision.selected:
                with self._purpose("gate"):
                    evaluation, verdict = self._gate.judge(candidate, head)
                decision = _both(decision, verdict)
        self._recent_records = method_evaluation.records
        # The method learns from every verdict, so its new state is attached
        # whether the candidate is published or rejected.
        settled = replace(candidate, method_state=method.record(result.state, proposal, method_evaluation, decision))
        return self._publish_or_reject(move, settled, evaluation, decision)

    # ------------------------------------------------------------ shared

    def _settle(
        self, move: Move, candidate: UpdateCandidate, head: HoneRelease, selector: CandidateSelector | None
    ) -> StepOutcome:
        self._check_candidate(candidate)
        with self._purpose("gate"):
            evaluation, decision = self._gate.judge(candidate, head, selector=selector)
        return self._publish_or_reject(move, candidate, evaluation, decision)

    def _publish_or_reject(
        self, move: Move, candidate: UpdateCandidate, evaluation: EvaluationResult, decision: SelectionDecision
    ) -> StepOutcome:
        # Published and rejected candidates both carry the gate result, so the
        # record shows how close a rejection was, not only that it happened.
        cost = self._metering().close()
        candidate = replace(candidate, metrics={**candidate.metrics, **gate_metrics(evaluation), **cost.metrics()})
        if decision.selected:
            release = self._chain.commit(candidate, decision, cost=cost)
            self._pending_method_state = None
            return StepOutcome(move=move, decision=decision, evaluation=evaluation, release=release, cost=cost)
        self._chain.record_rejection(candidate, decision, cost=cost)
        self._pending_method_state = candidate.method_state
        return StepOutcome(move=move, decision=decision, evaluation=evaluation, release=None, cost=cost)

    def _metering(self) -> CostMeter:
        if self._meter is None:
            raise ContractError("a move's cost is metered only inside Campaign.advance")
        return self._meter

    @contextmanager
    def _purpose(self, purpose: EpisodePurpose) -> Iterator[None]:
        with self._metering().purpose(purpose):
            yield

    @staticmethod
    def _check_candidate(candidate: UpdateCandidate) -> None:
        rules.master_is_hf_safetensors(candidate)
        rules.serving_built_from_master(candidate)

    def _check_job(self, job: TrainJobSpec) -> None:
        rules.algorithm_requirements_met(job, self._trainer.support)
        if requirements_for(job).requires_rollout and self._rollout is None:
            raise ContractError(f"{job.algorithm} learns from rollouts, but the campaign has no rollout provider")

    def _method_state(self, head: HoneRelease) -> Mapping[str, object]:
        return self._pending_method_state if self._pending_method_state is not None else head.method_state

    def _batch_for(
        self, job: TrainJobSpec, head: HoneRelease, policy: PolicyRef, *, release_id: str | None
    ) -> TrainingBatch | None:
        requirements = requirements_for(job)
        if not requirements.requires_batch:
            return None
        if not requirements.requires_rollout:
            with self._purpose("demonstration"):
                return self._data.next_batch(requirements.data_contract())
        if self._rollout is None:
            raise ContractError(f"{job.algorithm} learns from rollouts, but the campaign has no rollout provider")
        with self._purpose("rollout"):
            batch = self._rollout.collect(
                self._harness.tasks,
                policy,
                head.harness if job.rollout == "harness" else None,
                release_id,
                self._harness.samples_per_task,
            )
        rules.batch_from_one_release(batch, head)
        return batch
