"""Run a Honing campaign through every plane from one JSON config.

The config names every component by import path (see
:mod:`honing.e2e.components`), so a real implementation replaces a fake by
changing one entry. The slots cover every plane:

- model: ``model_source``, and ``endpoints`` for runners that call the policy
- data and training: ``data``, ``rollout`` (optional), ``trainer``
- compression: ``scorer``, ``selector``, ``editor``, ``quantizer``, ``telemetry``
- harness: ``runner``, ``grader``, ``method``
- gate: ``evaluator`` (optional; the default plays the runner on ``tasks``)

``tasks`` and ``harness_seed`` are factories too. The ``schedule`` lists moves
(``train``, ``prune``, ``quantize``, ``harness``) and ``rollback`` steps; each
move is gated, and a rollback restores an earlier release and checks that its
weights and harness came back. ``expect`` gives each step's result
(``published``, ``rolled back``, ``skipped``, or ``restored``); a run whose
results differ fails. Everything a run wrote goes under ``work_dir``:
``summary.json``, the release chain, and whatever the runner keeps.

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
from honing.core import (
    CandidateSelector,
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
    NumericsPolicy,
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
)
from honing.e2e.components import PipelineContext, build_component, missing_requirements
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
)
RESULTS = ("published", "rolled back", "skipped", "restored")


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
    return TrainJobSpec(
        algorithm=entry.get("algorithm", "sft"),
        steps=int(entry.get("steps", 1)),
        rollout=entry.get("rollout", "none"),
        contract=entry.get("contract"),
        teacher=TeacherRef(**teacher) if teacher else None,
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
        return Move(kind="train", train=_job(entry))
    if kind == "prune":
        budget = CompressionBudget(keep_ratio=float(entry["keep_ratio"])) if "keep_ratio" in entry else None
        return Move(kind="prune", budget=budget)
    if kind == "quantize":
        return Move(kind="quantize", scheme=QuantScheme(**entry["scheme"]) if "scheme" in entry else None)
    if kind == "harness":
        return Move(kind="harness", harness=HarnessMoveConfig(label=entry.get("label")))
    raise ContractError(f"a pipeline schedules train, prune, quantize, harness, and rollback steps, not {kind!r}")


@dataclass(frozen=True)
class PipelineRun:
    """A started campaign, the steps to run, and where it writes."""

    campaign: Campaign
    schedule: tuple[Mapping[str, Any], ...]
    expect: tuple[str, ...]
    context: PipelineContext


def build(config: Mapping[str, Any], root: Path, work_dir: Path | None = None) -> PipelineRun:
    """Check the environment, build every component, and start the campaign (release 0 is published)."""
    missing = missing_requirements(config.get("requires", {}))
    if missing:
        raise ContractError("this config needs " + ", ".join(missing))
    work = (work_dir or root / config.get("work_dir", "runs/e2e-pipeline")).resolve()
    work.mkdir(parents=True, exist_ok=False)
    context = PipelineContext(root=root.resolve(), work_dir=work)
    components = config.get("components", {})
    unknown = set(components) - {slot for slot, _, _ in SLOTS}
    if unknown:
        raise ContractError(f"unknown components {sorted(unknown)}")
    for slot, protocol, required in SLOTS:
        if slot in components:
            context.components[slot] = build_component(components[slot], context, slot, protocol)
        elif required:
            raise ContractError(f"the config names no {slot} component")
    tasks = build_component(config["tasks"], context, "tasks", tuple)
    seed = build_component(config["harness_seed"], context, "harness_seed", HarnessSpec)
    if not tasks or not all(isinstance(task, Task) for task in tasks):  # type: ignore[attr-defined]
        raise ContractError("the tasks factory must return a non-empty tuple of Task")
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
            tasks=tasks,  # type: ignore[arg-type]
        ),
        defaults=_defaults(config),
        rollout=c.get("rollout"),  # type: ignore[arg-type]
    )
    campaign.start(c["model_source"], work, harness_seed=seed)  # type: ignore[arg-type]
    schedule = tuple(config.get("schedule", ()))
    for entry in schedule:
        if entry.get("kind") != "rollback":
            _move(entry)  # refuse a bad step before anything runs
    expect = tuple(config.get("expect", ()))
    if expect and (len(expect) != len(schedule) or set(expect) - set(RESULTS)):
        raise ContractError(f"expect needs one of {list(RESULTS)} per schedule step")
    return PipelineRun(campaign, schedule, expect, context)


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
    }


def run(pipeline: PipelineRun) -> dict[str, Any]:
    """Run every step in order; returns the summary, with ``failures`` naming any step whose result was unexpected."""
    campaign = pipeline.campaign
    steps: list[dict[str, object]] = []
    for entry in pipeline.schedule:
        if entry.get("kind") == "rollback":
            target = campaign.chain.get(str(entry["to"]))
            head = campaign.rollback(target.release_id)
            restored = (
                head.release_id == target.release_id
                and head.master == target.master
                and head.serving == target.serving
                and head.harness == target.harness
            )
            steps.append(
                {
                    "move": "rollback",
                    "result": "restored" if restored else "not restored",
                    "release": head.release_id,
                    "reason": f"head restored to {target.release_id} with its weights and harness",
                }
            )
            continue
        steps.append(_step(campaign.advance(_move(entry))))
    head = campaign.chain.head()
    failures = [
        f"step {index + 1} ({step['move']}): expected {wanted}, got {step['result']}"
        for index, (step, wanted) in enumerate(zip(steps, pipeline.expect, strict=False))
        if step["result"] != wanted
    ]
    releases = campaign.chain.releases()
    return {
        "head": head.release_id,
        "releases": [{"id": r.release_id, "kind": r.kind, "parameters": r.master.parameters} for r in releases],
        "steps": steps,
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
    summary = run(pipeline)
    (pipeline.context.work_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
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
    return 1 if summary["failures"] else 0


__all__ = ["SLOTS", "PipelineRun", "build", "main", "run"]
