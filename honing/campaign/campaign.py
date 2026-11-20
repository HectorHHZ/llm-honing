"""A campaign: builds a candidate for each move, checks the rules, and settles it through the gate."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from honing.campaign import rules
from honing.core import (
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
    keyed_method_state,
    method_states,
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

    ``method`` runs a harness move that names no method. ``methods`` are the
    further methods a move can name. A method is named by its own ``name``, so
    no two of them may share one.
    """

    runner: EpisodeRunner
    grader: TaskGrader
    method: EvolveMethod
    tasks: tuple[Task, ...]
    episode_budget: int = 0
    samples_per_task: int = 1
    method_evaluator: Evaluator | None = None
    methods: tuple[EvolveMethod, ...] = ()

    def __post_init__(self) -> None:
        require(len(self.tasks) > 0, "a campaign needs at least one task")
        require(self.samples_per_task > 0, "samples_per_task must be > 0")
        names = [method.name for method in self.all_methods]
        require(len(set(names)) == len(names), f"a campaign names each harness method once, got {names}")

    @property
    def all_methods(self) -> tuple[EvolveMethod, ...]:
        """The default method first, then the others in the order given."""
        return (self.method, *self.methods)


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
    )


def _names(names: Iterable[str]) -> str:
    return ", ".join(repr(name) for name in names)


class _OwnStateSelector:
    """Shows a method's selector its own state on a candidate that carries every method's.

    A method's selector reads ``candidate.method_state`` as that method wrote
    it. In a campaign with several methods the candidate carries all of their
    states, so the selector is given a copy holding only its own.
    """

    def __init__(self, selector: CandidateSelector, own: Mapping[str, object]) -> None:
        self.name = selector.name
        self._selector = selector
        self._own = own

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        return self._selector.decide(replace(candidate, method_state=self._own), evaluation)


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

    ``trainer`` and ``harness.method`` run a move that names neither. A move
    can name any trainer in ``trainers`` and any method in ``harness.methods``
    instead, so one campaign can mix, for example, supervised training with a
    weight edit, and GEPA with Meta-Harness. A name the campaign does not have
    is refused before the move runs anything.

    With one method, a release's ``method_state`` is that method's state. With
    several it holds each method's own state under its name
    (:func:`~honing.core.keyed_method_state`), so a rollback restores them
    together; each method and its selector still see only their own.
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
        trainers: Mapping[str, Trainer] | None = None,
    ) -> None:
        self._chain = chain
        self._gate = gate
        self._trainer = trainer
        self._trainers = dict(trainers or {})
        require(all(self._trainers), "a registered trainer needs a name")
        self._data = data
        self._compression = compression
        self._harness = harness
        self._defaults = defaults
        self._rollout = rollout
        self._pending_method_state: Mapping[str, object] | None = None
        self._recent_records: tuple[EpisodeRecord, ...] = ()
        self._history: list[StepRecord] = []

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

    def start(self, source: ModelSource, work_dir: Path, harness_seed: HarnessSpec) -> HoneRelease:
        """Check the planes fit together, materialize the initial model, and publish it as release 0."""
        methods = self._harness.all_methods
        for method in methods:
            rules.method_supported_by_runner(method.requires, self._harness.runner.support)
        rules.spec_runnable(harness_seed, self._harness.runner.support)
        own_states = {method.name: method.initial_state(harness_seed) for method in methods}
        for method in methods:
            rules.method_state_schema_matches(own_states[method.name], method.name, method.state_schema_version)
        initial_state = keyed_method_state(own_states) if self._keyed else own_states[self._harness.method.name]
        snapshot = source.materialize(work_dir)
        for trainer in (self._trainer, *self._trainers.values()):
            rules.architecture_supported(snapshot.architecture, trainer.support)
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

        The scheduler's state is exported right after its choice, before the
        move's result is known, checked against its declared schema, and
        written into every record line from here on. The scheduler reads the
        result in the history at its next choice. The history is only in
        memory, so a restart resumes the scheduler from its last choice
        without the last move's result (see
        :class:`~honing.core.CoAdaptScheduler`).
        """
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

    def check_move(self, move: Move) -> None:
        """Refuse a move that names a trainer or a method this campaign does not have.

        :meth:`advance` refuses the same moves before it runs anything. A
        caller with a fixed schedule can check all of it before the first move.
        """
        if move.kind in ("train", "prune"):
            self._trainer_for(move)
        elif move.kind == "harness":
            self._method_for(move.harness if move.harness is not None else HarnessMoveConfig())

    def advance(self, move: Move) -> StepOutcome:
        """Build, check, and settle the candidate for one move, and add the move to the history."""
        parent_id = self._chain.head().release_id
        outcome = self._advance(move)
        self._history.append(_step_record(len(self._history), parent_id, outcome))
        return outcome

    def _advance(self, move: Move) -> StepOutcome:
        head = self._chain.head()
        if move.kind == "harness":
            return self._advance_harness(move, head)
        if move.kind == "train":
            trainer = self._trainer_for(move)
            job = move.train if move.train is not None else self._defaults.train
            candidate = self._train_candidate(job, head, trainer)
        elif move.kind == "prune":
            trainer = self._trainer_for(move)
            budget = move.budget if move.budget is not None else self._defaults.budget
            recovery = move.recovery if move.recovery is not None else self._defaults.recovery
            candidate = self._prune_candidate(budget, recovery, head, trainer)
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

    def _train_candidate(self, job: TrainJobSpec, head: HoneRelease, trainer: Trainer) -> UpdateCandidate:
        self._check_job(job, trainer)
        batch = self._batch_for(job, head, head.policy(), release_id=head.release_id)
        rules.batch_meets_contract(job, batch)
        result = trainer.train(job, head.master, batch)
        rules.architecture_supported(result.snapshot.architecture, trainer.support)
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
        self, budget: CompressionBudget, recovery: RecoverySpec | None, head: HoneRelease, trainer: Trainer
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
        rules.architecture_supported(plan.after, trainer.support)
        rules.recovery_teacher_fits_new_shape(step)
        if recovery_job is not None:
            self._check_job(recovery_job, trainer)

        snapshot = compression.editor.apply(head.master, plan)
        metrics = {"estimated_parameter_change": float(plan.estimated.parameters)}
        if recovery_job is not None:
            # The edited weights are not published, so their rollout samples name no release.
            batch = self._batch_for(recovery_job, head, _policy_of(snapshot), release_id=None)
            rules.batch_meets_contract(recovery_job, batch)
            result = trainer.train(recovery_job, snapshot, batch)
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
        harness, runner = self._harness, self._harness.runner
        config = move.harness if move.harness is not None else HarnessMoveConfig()
        method = self._method_for(config)
        if head.harness is None:
            raise ContractError("the head has no harness to evolve; start the campaign with a harness seed")
        carried = self._method_state(head)
        state = self._own_state(carried, method)
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
        result = method.propose(context)
        proposal = result.proposal
        if proposal is None:
            reason = f"{method.name} proposed nothing"
            skipped = self._with_own_state(carried, method, result.state)
            # Recorded before the pending state moves, so a restart can see what the attempt spent.
            self._chain.record_skip(
                parent_id=head.release_id,
                method=method.name,
                reason=reason,
                state=skipped,
                episodes_spent=result.episodes_spent,
                label=config.label,
            )
            self._pending_method_state = skipped
            return StepOutcome(move=move, decision=None, evaluation=None, release=None, skipped_reason=reason)
        rules.spec_runnable(proposal.spec, runner.support)
        candidate = UpdateCandidate(
            kind="harness",
            parent_id=head.release_id,
            master=head.master,
            serving=head.serving,
            harness=proposal.spec,
            trained_numerics=head.trained_numerics,
            method_state=self._with_own_state(carried, method, result.state),
            metrics={**_size_metrics(head.master), "method_episodes": float(result.episodes_spent)},
            move_label=config.label,
        )
        self._check_candidate(candidate)
        selector = _OwnStateSelector(method.selector, result.state) if self._keyed else method.selector
        if harness.method_evaluator is None:
            evaluation, decision = self._gate.judge(candidate, head, selector=selector)
            method_evaluation = evaluation
        else:
            method_gate = Gate(harness.method_evaluator, default_selector=selector)
            method_evaluation, decision = method_gate.judge(candidate, head)
            evaluation = method_evaluation
            if decision.selected:
                evaluation, verdict = self._gate.judge(candidate, head)
                decision = _both(decision, verdict)
        self._recent_records = method_evaluation.records
        # The method learns from every verdict, so its new state is attached
        # whether the candidate is published or rejected.
        recorded = method.record(result.state, proposal, method_evaluation, decision)
        settled = replace(candidate, method_state=self._with_own_state(carried, method, recorded))
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
        # Published and rejected candidates both carry the gate result, so the
        # record shows how close a rejection was, not only that it happened.
        candidate = replace(candidate, metrics={**candidate.metrics, **gate_metrics(evaluation)})
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

    def _check_job(self, job: TrainJobSpec, trainer: Trainer) -> None:
        rules.algorithm_requirements_met(job, trainer.support)
        if requirements_for(job).requires_rollout and self._rollout is None:
            raise ContractError(f"{job.algorithm} learns from rollouts, but the campaign has no rollout provider")

    def _method_state(self, head: HoneRelease) -> Mapping[str, object]:
        return self._pending_method_state if self._pending_method_state is not None else head.method_state

    # ------------------------------------------------------------ named trainers and methods

    @property
    def _keyed(self) -> bool:
        """Whether a release's method state holds several methods' states under their names."""
        return bool(self._harness.methods)

    def _trainer_for(self, move: Move) -> Trainer:
        if move.trainer is None:
            return self._trainer
        if move.trainer not in self._trainers:
            known = _names(sorted(self._trainers)) if self._trainers else "no named trainers"
            raise ContractError(f"the move is for trainer {move.trainer!r}, but the campaign has {known}")
        return self._trainers[move.trainer]

    def _method_for(self, config: HarnessMoveConfig) -> EvolveMethod:
        if config.method is None:
            return self._harness.method
        for method in self._harness.all_methods:
            if method.name == config.method:
                return method
        running = _names(method.name for method in self._harness.all_methods)
        raise ContractError(f"the move is for method {config.method!r}, but the campaign runs {running}")

    def _own_state(self, carried: Mapping[str, object], method: EvolveMethod) -> Mapping[str, object]:
        """``method``'s own state from the state a release carries."""
        if not self._keyed:
            return carried
        states = method_states(carried)
        if method.name not in states:
            raise ContractError(
                f"the head's method state holds {_names(sorted(states))}, and nothing for {method.name!r}"
            )
        return states[method.name]

    def _with_own_state(
        self, carried: Mapping[str, object], method: EvolveMethod, own: Mapping[str, object]
    ) -> Mapping[str, object]:
        """The state a release carries, with ``method``'s own replaced; the other methods' are kept."""
        if not self._keyed:
            return own
        return keyed_method_state({**method_states(carried), method.name: own})

    def _batch_for(
        self, job: TrainJobSpec, head: HoneRelease, policy: PolicyRef, *, release_id: str | None
    ) -> TrainingBatch | None:
        requirements = requirements_for(job)
        if not requirements.requires_batch:
            return None
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
