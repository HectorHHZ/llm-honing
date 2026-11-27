"""Run a Honing campaign through every plane from one JSON config.

The config names every component by import path (see
:mod:`honing.e2e.components`), so a real implementation replaces a fake by
changing one entry. The slots cover every plane:

- model: ``model_source``, and ``endpoints`` for runners that call the policy
- data and training: ``data``, ``rollout`` (optional), ``trainer``
- compression: ``scorer``, ``selector``, ``editor``, ``quantizer``, ``telemetry``
- harness: ``runner``, ``grader``, ``method``
- gate: ``evaluator`` (optional; the default plays the runner on ``tasks``)
- campaign: ``scheduler`` (optional; the default replays the moves of ``schedule``)

``components`` may also hold ``trainers`` (named entries) and ``methods`` (a
list). A train or prune step names one with ``"trainer"``, a harness step
with ``"method"`` (the method's own name); a step that names none runs the
``trainer`` or ``method`` slot. A ``direction_edit`` train step can state its
own ``"direction_edit": {"question": ..., "target": ..., "strength": ...}``,
all three together.

``tasks`` and ``harness_seed`` are factories too. ``tasks`` are the gate's;
an optional ``harness_tasks`` factory gives a harness move its own tasks (a
method's training split the gate never sees), and ``harness.episode_budget``
what a harness move may spend on its own episodes (default 0: a method that
runs episodes, as GEPA does, needs it). The ``schedule`` lists moves
(``train``, ``prune``, ``quantize``, ``harness``) and ``rollback`` steps; each
move is gated, and a rollback restores an earlier release and checks that its
weights and harness came back. The scheduler chooses every move until it
stops. A config without one gets a :class:`~honing.campaign.ScriptedScheduler`
over the moves of ``schedule``. A rollback is the pipeline's own step: it runs
once as many moves as precede it in ``schedule`` have run, whichever scheduler
chose them, and a run that ends before it fails. ``expect`` gives each step's
result (``published``, ``rolled back``, ``skipped``, or ``restored``); a run
whose results differ fails. Everything a run wrote goes under ``work_dir``:
``summary.json``, the release chain, and whatever the runner keeps. An
optional ``logging`` block logs the run to W&B (:mod:`honing.e2e.tracking`).
An optional ``budget`` block (:func:`~honing.campaign.cost.budget_from_config`)
stops the run once its spend reaches the limit; every step's cost and the
total are in ``summary.json`` either way. A ``usd`` budget also stops the run,
and fails it, after a move whose dollars it could not price.

``configs/e2e/all_planes.json`` runs fakes for every plane except the harness
runner, which is :class:`~honing.e2e.workspace_runner.WorkspaceEpisodeRunner`
with a stand-in agent process calling a stand-in OpenAI-compatible policy
server: the boundary OpenHands, Reef, or Claude Code sit behind.

Run: ``python scripts/e2e/honing_pipeline.py --config configs/e2e/all_planes.json``
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from honing.campaign import Campaign, CampaignDefaults, CompressionComponents, HarnessComponents, StepOutcome
from honing.campaign.cost import CampaignBudget, budget_from_config, cost_summary
from honing.core import (
    CandidateSelector,
    CoAdaptScheduler,
    CompressionBudget,
    ContractError,
    DataProvider,
    EpisodeRunner,
    Evaluator,
    EvolveMethod,
    HarnessMoveConfig,
    HarnessSpec,
    ImportanceScorer,
    ModelEditor,
    ModelSource,
    Move,
    MoveCost,
    NumericsPolicy,
    ParamMask,
    PolicyEndpoints,
    PruneSelector,
    Quantizer,
    QuantScheme,
    RecoverySpec,
    RolloutProvider,
    Task,
    TaskGrader,
    TeacherRef,
    TelemetryStore,
    Trainer,
    TrainJobSpec,
    UpdatePolicy,
)
from honing.e2e.components import (
    METHODS,
    SCHEDULER_SLOT,
    TRAINERS,
    PipelineContext,
    build_component,
    build_named_trainers_and_methods,
    direction_edit_spec,
    missing_requirements,
    resolve_references,
    scheduler_or_script,
)
from honing.e2e.tracking import (
    ChainRecords,
    NullTracker,
    RunTracker,
    WandbSettings,
    logging_settings,
    run_config,
    start_tracker,
    step_values,
)
from honing.release import AcceptAll, EpisodeEvaluator, Gate, NoRegressionSelector, ReleaseChain, WinLossSelector

# Built in this order; a factory can read the components built before it.
SLOTS: tuple[tuple[str, type, bool], ...] = (
    ("model_source", ModelSource, True),
    ("endpoints", PolicyEndpoints, False),
    ("trainer", Trainer, True),
    ("data", DataProvider, True),
    ("rollout", RolloutProvider, False),
    ("scorer", ImportanceScorer, True),
    ("selector", PruneSelector, True),
    ("editor", ModelEditor, True),
    ("quantizer", Quantizer, True),
    ("telemetry", TelemetryStore, True),
    ("runner", EpisodeRunner, True),
    ("grader", TaskGrader, True),
    ("method", EvolveMethod, True),
    ("evaluator", Evaluator, False),
    SCHEDULER_SLOT,
)
RESULTS = ("published", "rolled back", "skipped", "restored")
# As in Campaign.run: a scheduler that never stops still ends.
MAX_MOVES = 1000


def _selector(spec: Mapping[str, Any]) -> CandidateSelector:
    """``{"win_loss": {...}}``, ``{"no_regression": {"max_regression": 0.05}}``, or ``{"accept_all": {}}``."""
    ((name, args),) = spec.items()
    kinds = {"win_loss": WinLossSelector, "no_regression": NoRegressionSelector, "accept_all": AcceptAll}
    if name not in kinds:
        raise ContractError(f"unknown selector {name!r}; use one of {sorted(kinds)}")
    selector: CandidateSelector = kinds[name](**args)
    return selector


def _job(entry: Mapping[str, Any]) -> TrainJobSpec:
    """A train step's job; ``numerics`` is ``{"mode": ..., "scheme": {...}}`` for quantization-aware training."""
    teacher, numerics = entry.get("teacher"), entry.get("numerics")
    update = dict(entry.get("update_policy", {}))
    for key in ("lora_targets", "trainable_patterns"):
        if key in update:
            update[key] = tuple(update[key])
    if "masks" in update:
        update["masks"] = tuple(ParamMask(**mask) for mask in update["masks"])
    return TrainJobSpec(
        algorithm=entry.get("algorithm", "sft"),
        steps=int(entry.get("steps", 1)),
        rollout=entry.get("rollout", "none"),
        contract=entry.get("contract"),
        update_policy=UpdatePolicy(**update),
        teacher=TeacherRef(**teacher) if teacher else None,
        direction_edit=direction_edit_spec(entry.get("direction_edit")),
        numerics=(
            NumericsPolicy(
                mode=numerics["mode"], scheme=QuantScheme(**numerics["scheme"]) if numerics.get("scheme") else None
            )
            if numerics
            else NumericsPolicy()
        ),
    )


def _defaults(config: Mapping[str, Any]) -> CampaignDefaults:
    defaults = config.get("defaults", {})
    recovery = defaults.get("recovery")
    return CampaignDefaults(
        train=_job(defaults.get("train", {"algorithm": "sft", "steps": 1})),
        budget=CompressionBudget(**defaults.get("budget", {"keep_ratio": 0.75})),
        recovery=(
            RecoverySpec(
                algorithm=recovery["algorithm"],
                steps=int(recovery.get("steps", 1)),
                teacher=TeacherRef(**recovery["teacher"]) if recovery.get("teacher") else None,
            )
            if recovery
            else None
        ),
        scheme=QuantScheme(
            **defaults.get("scheme", {"weight_format": "int4", "granularity": "group", "group_size": 128})
        ),
    )


def _move(entry: Mapping[str, Any]) -> Move:
    kind = entry.get("kind")
    if kind == "train":
        return Move(kind="train", train=_job(entry), trainer=entry.get("trainer"))
    if kind == "prune":
        budget = CompressionBudget(keep_ratio=float(entry["keep_ratio"])) if "keep_ratio" in entry else None
        return Move(kind="prune", budget=budget, trainer=entry.get("trainer"))
    if kind == "quantize":
        return Move(kind="quantize", scheme=QuantScheme(**entry["scheme"]) if "scheme" in entry else None)
    if kind == "harness":
        return Move(kind="harness", harness=HarnessMoveConfig(label=entry.get("label"), method=entry.get("method")))
    raise ContractError(f"a pipeline schedules train, prune, quantize, harness, and rollback steps, not {kind!r}")


@dataclass(frozen=True)
class PipelineRun:
    """A started campaign, the steps to run, and where it writes."""

    campaign: Campaign
    schedule: tuple[Mapping[str, Any], ...]
    expect: tuple[str, ...]
    context: PipelineContext
    scheduler: CoAdaptScheduler
    tracking: WandbSettings | None = None


def build(config: Mapping[str, Any], root: Path, work_dir: Path | None = None) -> PipelineRun:
    """Check the environment, build every component, and start the campaign (release 0 is published)."""
    config = resolve_references(config)
    tracking = logging_settings(config.get("logging"))
    requires = dict(config.get("requires", {}))
    if tracking is not None:
        requires["env"] = [*requires.get("env", ()), *tracking.required_env]
    missing = missing_requirements(requires)
    if missing:
        raise ContractError("this config needs " + ", ".join(missing))
    work = (work_dir or root / config.get("work_dir", "runs/e2e-pipeline")).resolve()
    work.mkdir(parents=True, exist_ok=False)
    context = PipelineContext(root=root.resolve(), work_dir=work)
    budget = budget_from_config(config.get("budget"))
    context.shared["budget"] = budget
    components = config.get("components", {})
    unknown = set(components) - {slot for slot, _, _ in SLOTS} - {TRAINERS, METHODS}
    if unknown:
        raise ContractError(f"unknown components {sorted(unknown)}")
    for slot, protocol, required in SLOTS:
        if slot in components:
            context.components[slot] = build_component(components[slot], context, slot, protocol)
        elif required:
            raise ContractError(f"the config names no {slot} component")
    trainers, methods = build_named_trainers_and_methods(components, context)
    tasks = _tasks(config["tasks"], context, "tasks")
    harness_tasks = _tasks(config["harness_tasks"], context, "harness_tasks") if "harness_tasks" in config else tasks
    seed = build_component(config["harness_seed"], context, "harness_seed", HarnessSpec)
    episode_budget = config.get("harness", {}).get("episode_budget", 0)
    if not isinstance(episode_budget, int) or isinstance(episode_budget, bool) or episode_budget < 0:
        raise ContractError(f"harness.episode_budget must be an integer >= 0, got {episode_budget!r}")
    c = context.components
    selectors = config.get("selectors", {"default": {"win_loss": {}}})
    evaluator = c.get("evaluator") or EpisodeEvaluator(c["runner"], c["grader"], tasks)  # type: ignore[arg-type]
    gate = Gate(
        evaluator,  # type: ignore[arg-type]
        default_selector=_selector(selectors.get("default", {"win_loss": {}})),
        selectors={kind: _selector(spec) for kind, spec in selectors.items() if kind != "default"},
    )
    campaign = Campaign(
        chain=ReleaseChain(work / "releases"),
        gate=gate,
        trainer=c["trainer"],  # type: ignore[arg-type]
        data=c["data"],  # type: ignore[arg-type]
        compression=CompressionComponents(
            scorer=c["scorer"],  # type: ignore[arg-type]
            selector=c["selector"],  # type: ignore[arg-type]
            editor=c["editor"],  # type: ignore[arg-type]
            quantizer=c["quantizer"],  # type: ignore[arg-type]
            telemetry=c["telemetry"],  # type: ignore[arg-type]
        ),
        harness=HarnessComponents(
            runner=c["runner"],  # type: ignore[arg-type]
            grader=c["grader"],  # type: ignore[arg-type]
            method=c["method"],  # type: ignore[arg-type]
            tasks=harness_tasks,
            episode_budget=episode_budget,
            methods=methods,
        ),
        defaults=_defaults(config),
        rollout=c.get("rollout"),  # type: ignore[arg-type]
        trainers=trainers,
        budget=budget,
    )
    campaign.start(c["model_source"], work, harness_seed=seed)  # type: ignore[arg-type]
    schedule = tuple(config.get("schedule", ()))
    # Built here, so a bad step is refused before anything runs.
    moves = tuple(_move(entry) for entry in schedule if entry.get("kind") != "rollback")
    for move in moves:
        campaign.check_move(move)  # a step for a trainer or method the config does not build
    expect = tuple(config.get("expect", ()))
    if expect and (len(expect) != len(schedule) or set(expect) - set(RESULTS)):
        raise ContractError(f"expect needs one of {list(RESULTS)} per schedule step")
    return PipelineRun(campaign, schedule, expect, context, scheduler_or_script(c, moves), tracking)


def _tasks(spec: Mapping[str, Any], context: PipelineContext, slot: str) -> tuple[Task, ...]:
    tasks = build_component(spec, context, slot, tuple)
    if not tasks or not all(isinstance(task, Task) for task in tasks):  # type: ignore[attr-defined]
        raise ContractError(f"the {slot} factory must return a non-empty tuple of Task")
    return tasks  # type: ignore[return-value]


def _step(outcome: StepOutcome) -> dict[str, object]:
    evaluation, decision, release = outcome.evaluation, outcome.decision, outcome.release
    status = "skipped" if decision is None else "published" if release is not None else "rolled back"
    return {
        "move": outcome.move.kind,
        "result": status,
        "release": release.release_id if release is not None else None,
        "reason": outcome.skipped_reason if decision is None else decision.reason,
        "candidate_mean": statistics.fmean(evaluation.candidate_scores) if evaluation else None,
        "incumbent_mean": statistics.fmean(evaluation.incumbent_scores) if evaluation else None,
        "episodes": len(evaluation.records) if evaluation else 0,
        "cost": outcome.cost.to_record() if outcome.cost is not None else None,
    }


def _rollback(campaign: Campaign, entry: Mapping[str, Any]) -> dict[str, object]:
    target = campaign.chain.get(str(entry["to"]))
    head = campaign.rollback(target.release_id)
    restored = (
        head.release_id == target.release_id
        and head.master == target.master
        and head.serving == target.serving
        and head.harness == target.harness
    )
    return {
        "move": "rollback",
        "result": "restored" if restored else "not restored",
        "release": head.release_id,
        "reason": f"head restored to {target.release_id} with its weights and harness",
    }


def _stopped_at_limit(
    budget: CampaignBudget | None, report: Mapping[str, object] | None, costs: Sequence[MoveCost]
) -> bool:
    """Whether the budget stopped the run at its limit, right after the first move whose spend reached it."""
    if budget is None or report is None or report["stopped"] != "limit" or not costs:
        return False
    before = MoveCost()
    for cost in costs[:-1]:
        before = before.plus(cost)
    spent_before, spent_after = budget.spent(before), budget.spent(before.plus(costs[-1]))
    return spent_before is not None and spent_after is not None and spent_before < budget.limit <= spent_after


def run(pipeline: PipelineRun, tracker: RunTracker | None = None) -> dict[str, Any]:
    """Run the scheduler's moves and the schedule's rollbacks in order, and return the summary.

    ``failures`` names any step whose result was unexpected, and a run with
    ``expect`` fails when it ran another number of steps. A ``rollback`` entry
    the run never reached, because the scheduler stopped first, is a failure
    too. Neither counts when the budget stopped the run at its limit, after
    the first move whose spend reached it. A ``usd`` budget's stop on a move
    whose dollars are unknown is always a failure.
    """
    campaign = pipeline.campaign
    tracker = tracker or NullTracker()
    records = ChainRecords(campaign.chain.record_path)
    steps: list[dict[str, object]] = []
    costs: list[MoveCost] = []

    def log(step: dict[str, object]) -> None:
        steps.append(step)
        tracker.log_step(len(steps), step_values(step, records.new()))

    entries = list(pipeline.schedule)
    for _ in range(MAX_MOVES):
        # Campaign.step checks the budget too; checking here first keeps a rollback from running after the stop.
        if not campaign.admits_move():
            break
        while entries and entries[0].get("kind") == "rollback":
            log(_rollback(campaign, entries.pop(0)))
        if entries:
            entries.pop(0)  # the move entry this move takes the place of
        outcome = campaign.step(pipeline.scheduler)
        if outcome is None:
            break
        if outcome.cost is not None:
            costs.append(outcome.cost)
        log(_step(outcome))
    head = campaign.chain.head()
    failures = [
        f"step {index + 1} ({step['move']}): expected {wanted}, got {step['result']}"
        for index, (step, wanted) in enumerate(zip(steps, pipeline.expect, strict=False))
        if step["result"] != wanted
    ]
    budget = campaign.budget_report()
    if budget is not None and budget["stopped"] == "unknown_cost":
        failures.append(f"stopped: {budget['reason']}")
    # A budget that ends the run at its limit is how a budgeted run ends, so the steps it cut short are no failure.
    # Only a stop at the first move whose spend reached the limit counts: one later would mean a move ran unadmitted.
    if not _stopped_at_limit(campaign.budget, budget, costs):
        if pipeline.expect and len(steps) != len(pipeline.expect):
            failures.append(f"{len(steps)} steps ran, but expect lists {len(pipeline.expect)}")
        # A rollback the scheduler stopped before would otherwise pass unnoticed, with no step and no failure.
        first_left = len(pipeline.schedule) - len(entries)
        failures.extend(
            f"the run ended before schedule step {first_left + offset + 1} (rollback to {entry['to']}) ran"
            for offset, entry in enumerate(entries)
            if entry.get("kind") == "rollback"
        )
    releases = campaign.chain.releases()
    return {
        "head": head.release_id,
        "releases": [{"id": r.release_id, "kind": r.kind, "parameters": r.master.parameters} for r in releases],
        "steps": steps,
        "cost_total": cost_summary(costs),
        "budget": budget,
        "failures": failures,
        "work_dir": str(pipeline.context.work_dir),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Honing campaign through every plane from one config.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--root", type=Path, default=Path.cwd(), help="repository root; config paths are relative to it"
    )
    parser.add_argument("--work-dir", type=Path, help="output directory (must not exist); overrides the config")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    try:
        pipeline = build(config, args.root, args.work_dir)
    except ContractError as error:
        print(f"cannot start: {error}", file=sys.stderr)
        return 2
    work_dir = pipeline.context.work_dir
    try:
        tracker = start_tracker(
            pipeline.tracking,
            name=work_dir.name,
            work_dir=work_dir,
            root=args.root,
            config=run_config(config, args.root, "honing_pipeline"),
        )
    except Exception as error:  # wandb raises its own errors for a bad key, project or network
        print(f"cannot start logging: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    if tracker.url:
        print(f"wandb: {tracker.url}")
    failed = True
    try:
        summary = run(pipeline, tracker)
        (work_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        failed = bool(summary["failures"])
    finally:
        problems = tracker.finish(work_dir, config.get("requires", {}).get("env", ()), failed=failed)
    for index, step in enumerate(summary["steps"], start=1):
        scores = (
            f" (candidate {step['candidate_mean']:.3f} vs incumbent {step['incumbent_mean']:.3f})"
            if step.get("candidate_mean") is not None
            else ""
        )
        print(f"{index}. {step['move']}: {step['result']}{scores}: {step['reason']}")
    print(f"head {summary['head']}; releases {[r['kind'] for r in summary['releases']]}; in {summary['work_dir']}")
    for failure in summary["failures"]:
        print("unexpected:", failure, file=sys.stderr)
    for problem in problems:
        print("logging:", problem, file=sys.stderr)
    if tracker.url:
        print(f"wandb: {tracker.url}")
    return 1 if summary["failures"] or problems else 0


__all__ = ["SLOTS", "PipelineRun", "build", "main", "run"]
