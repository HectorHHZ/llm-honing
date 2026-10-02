"""Run a short Honing campaign on RA2 from one JSON config.

The config names every component by import path, so a new trainer or agent is
tried by changing the config, not this code. Each entry is
``{"factory": "package.module:function", "args": {...}}``; the factory is
called as ``function(context, **args)`` and must return an object satisfying
the slot's protocol. Components are built in :data:`SLOTS` order, and a
factory can read the ones built before it from ``context.components``.

The campaign runs the config's ``schedule`` (for example one harness move and
one train move, each gated on the ``gate_tasks``) and writes the release
chain, every episode's run directory, and ``summary.json`` under ``work_dir``.

``configs/e2e/ra2_dry_run.json`` plays through the real runner and grader with
stand-in match and check processes and fake model, trainer, data, and method,
so it runs anywhere with no game, Node, or GPU.

Run: ``python scripts/e2e/ra2_campaign.py --config configs/e2e/ra2_dry_run.json``
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_args

from honing.backends.ra2.grader import RA2Grader
from honing.backends.ra2.harness import seed_harness
from honing.backends.ra2.runner import RA2EpisodeRunner
from honing.backends.ra2.scenarios import RA2Suite, Split, load_suite
from honing.campaign import (
    Campaign,
    CampaignDefaults,
    CompressionComponents,
    HarnessComponents,
    ScriptedScheduler,
    StepOutcome,
)
from honing.core import (
    CompressionBudget,
    ContractError,
    DataProvider,
    EpisodeRunner,
    Evaluator,
    EvolveMethod,
    HarnessMoveConfig,
    ImportanceMap,
    ModelSnapshot,
    ModelSource,
    Move,
    PolicyEndpoints,
    PrunePlan,
    QuantizePlan,
    QuantScheme,
    RolloutProvider,
    ServingWeights,
    SignalKind,
    TaskGrader,
    Trainer,
    TrainJobSpec,
    UtilizationTelemetry,
)
from honing.e2e.components import build_component, missing_requirements
from honing.release import EpisodeEvaluator, Gate, ReleaseChain, WinLossSelector

# Built in this order; ``rollout`` and ``evaluator`` are optional.
SLOTS: tuple[tuple[str, type, bool], ...] = (
    ("model_source", ModelSource, True),
    ("endpoints", PolicyEndpoints, True),
    ("runner", EpisodeRunner, True),
    ("grader", TaskGrader, True),
    ("trainer", Trainer, True),
    ("data", DataProvider, True),
    ("rollout", RolloutProvider, False),
    ("method", EvolveMethod, True),
    ("evaluator", Evaluator, False),
)


@dataclass
class E2EContext:
    """What a component factory can read: paths, the task suite, and the components built so far.

    ``shared`` lets factories that belong together, such as a fake model source
    and the fake trainer that must use the same fake storage, hand state to
    each other.
    """

    root: Path
    integration_dir: Path
    work_dir: Path
    suite: RA2Suite
    components: dict[str, object] = field(default_factory=dict)
    shared: dict[str, object] = field(default_factory=dict)

    def component(self, slot: str) -> Any:
        if slot not in self.components:
            raise ContractError(f"the {slot} component is needed here but was not built earlier")
        return self.components[slot]


class NoCompression:
    """Fills the Compression plane for a campaign that never prunes or quantizes; every call refuses."""

    signal: SignalKind = "weights"

    def _refuse(self) -> ContractError:
        return ContractError("this RA2 campaign has no compression plane; schedule only harness and train moves")

    def score(self, model: ModelSnapshot, telemetry: UtilizationTelemetry | None) -> ImportanceMap:
        raise self._refuse()

    def select(self, importance: ImportanceMap, model: ModelSnapshot, budget: CompressionBudget) -> PrunePlan:
        raise self._refuse()

    def apply(self, model: ModelSnapshot, plan: PrunePlan) -> ModelSnapshot:
        raise self._refuse()

    def quantize(self, model: ModelSnapshot, plan: QuantizePlan) -> ServingWeights:
        raise self._refuse()

    def observe(self, stats: UtilizationTelemetry) -> None:
        raise self._refuse()

    def window(self) -> UtilizationTelemetry | None:
        return None


# ---------------------------------------------------------------- factories for the RA2 components


def _expand(values: Sequence[str], context: E2EContext) -> list[str]:
    """Fill ``{python}``, ``{root}``, and ``{integration}`` in a command."""
    names = {"python": sys.executable, "root": str(context.root), "integration": str(context.integration_dir)}
    return [value.format(**names) for value in values]


def episode_runner(
    context: E2EContext, command: Sequence[str] | None = None, concurrency: int = 4
) -> RA2EpisodeRunner:
    """The RA2 runner, with the policy served by the ``endpoints`` component. ``command`` replaces Node."""
    return RA2EpisodeRunner(
        suite=context.suite,
        endpoints=context.component("endpoints"),
        integration_dir=context.integration_dir,
        runs_dir=context.work_dir / "runs",
        command=_expand(command, context) if command else None,
        concurrency=concurrency,
    )


def grader(
    context: E2EContext, audit_command: Sequence[str] | None = None, replay_command: Sequence[str] | None = None
) -> RA2Grader:
    """The RA2 grader. For real matches ``replay_command`` runs ``scripts/verify_replay.py`` with ``--game-dir``."""
    return RA2Grader(
        suite=context.suite,
        integration_dir=context.integration_dir,
        audit_command=_expand(audit_command, context) if audit_command else None,
        replay_command=_expand(replay_command, context) if replay_command else None,
    )


# ---------------------------------------------------------------- config


def _tasks(suite: RA2Suite, selection: Mapping[str, object]) -> tuple[Any, ...]:
    split = str(selection.get("split", "val"))
    if split not in get_args(Split):
        raise ContractError(f"unknown task split {split!r}")
    limit = selection.get("limit")
    return suite.tasks(split, limit=int(str(limit)) if limit is not None else None)  # type: ignore[arg-type]


def _move(entry: Mapping[str, object]) -> Move:
    kind = entry.get("kind")
    if kind == "harness":
        return Move(kind="harness", harness=HarnessMoveConfig(label=str(entry.get("label", "harness"))))
    if kind == "train":
        job = TrainJobSpec(
            algorithm=entry.get("algorithm", "sft"),  # type: ignore[arg-type]
            steps=int(str(entry.get("steps", 1))),
            contract=entry.get("contract", "typed_decisions"),  # type: ignore[arg-type]
        )
        return Move(kind="train", train=job)
    raise ContractError(f"an RA2 campaign schedules harness and train moves, not {kind!r}")


@dataclass(frozen=True)
class E2ERun:
    """A built campaign ready to run, and where it writes."""

    campaign: Campaign
    moves: tuple[Move, ...]
    context: E2EContext


def build(config: Mapping[str, Any], root: Path, work_dir: Path | None = None) -> E2ERun:
    """Check the environment, build every component, and start the campaign (release 0 is published)."""
    missing = missing_requirements(config.get("requires", {}))
    if missing:
        raise ContractError("this config needs " + ", ".join(missing))
    integration_dir = (root / config.get("integration_dir", "integrations/ra2")).resolve()
    work = (work_dir or root / config.get("work_dir", "runs/e2e")).resolve()
    work.mkdir(parents=True, exist_ok=False)
    suite = load_suite(integration_dir / config.get("suite", "tasks/suite.jsonl"))
    context = E2EContext(root=root.resolve(), integration_dir=integration_dir, work_dir=work, suite=suite)
    components = config.get("components", {})
    for slot, protocol, required in SLOTS:
        if slot in components:
            context.components[slot] = build_component(components[slot], context, slot, protocol)
        elif required:
            raise ContractError(f"the config names no {slot} component")
    unknown = set(components) - {slot for slot, _, _ in SLOTS}
    if unknown:
        raise ContractError(f"unknown components {sorted(unknown)}")
    c = context.components
    gate_tasks = _tasks(suite, config.get("gate_tasks", {"split": "val", "limit": 4}))
    evaluator = c.get("evaluator") or EpisodeEvaluator(c["runner"], c["grader"], gate_tasks)  # type: ignore[arg-type]
    none = NoCompression()
    campaign = Campaign(
        chain=ReleaseChain(work / "releases"),
        gate=Gate(evaluator, default_selector=WinLossSelector()),  # type: ignore[arg-type]
        trainer=c["trainer"],  # type: ignore[arg-type]
        data=c["data"],  # type: ignore[arg-type]
        compression=CompressionComponents(scorer=none, selector=none, editor=none, quantizer=none, telemetry=none),
        harness=HarnessComponents(
            runner=c["runner"],  # type: ignore[arg-type]
            grader=c["grader"],  # type: ignore[arg-type]
            method=c["method"],  # type: ignore[arg-type]
            tasks=_tasks(suite, config.get("harness_tasks", {"split": "val", "limit": 4})),
        ),
        defaults=CampaignDefaults(
            train=TrainJobSpec("sft", steps=1, contract="typed_decisions"),
            budget=CompressionBudget(keep_ratio=1.0),
            recovery=None,
            scheme=QuantScheme(weight_format="int4", granularity="group", group_size=128),
        ),
        rollout=c.get("rollout"),  # type: ignore[arg-type]
    )
    campaign.start(c["model_source"], work, harness_seed=seed_harness(integration_dir))  # type: ignore[arg-type]
    moves = tuple(_move(entry) for entry in config.get("schedule", ({"kind": "harness"}, {"kind": "train"})))
    return E2ERun(campaign, moves, context)


def summarize(outcomes: Sequence[StepOutcome], run: E2ERun) -> dict[str, object]:
    steps = []
    for outcome in outcomes:
        evaluation, decision = outcome.evaluation, outcome.decision
        status = "skipped" if decision is None else "published" if outcome.release is not None else "rolled back"
        steps.append(
            {
                "move": outcome.move.kind,
                "result": status,
                "release": outcome.release.release_id if outcome.release is not None else None,
                "reason": outcome.skipped_reason if decision is None else decision.reason,
                "candidate_mean": statistics.fmean(evaluation.candidate_scores) if evaluation else None,
                "incumbent_mean": statistics.fmean(evaluation.incumbent_scores) if evaluation else None,
                "episodes": len(evaluation.records) if evaluation else 0,
            }
        )
    work = run.context.work_dir
    return {
        "head": run.campaign.chain.head().release_id,
        "steps": steps,
        "releases": str(work / "releases"),
        "runs": str(work / "runs"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a short Honing campaign on RA2 from one config.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--root", type=Path, default=Path.cwd(), help="repository root; config paths are relative to it"
    )
    parser.add_argument("--work-dir", type=Path, help="output directory (must not exist); overrides the config")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    try:
        run = build(config, args.root, args.work_dir)
    except ContractError as error:
        print(f"cannot start: {error}", file=sys.stderr)
        return 2
    outcomes = run.campaign.run(ScriptedScheduler(run.moves))
    summary = summarize(outcomes, run)
    (run.context.work_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for step in summary["steps"]:  # type: ignore[attr-defined]
        scores = (
            f" (candidate {step['candidate_mean']:.2f} vs incumbent {step['incumbent_mean']:.2f}, "
            f"{step['episodes']} episodes)"
            if step["candidate_mean"] is not None
            else ""
        )
        print(f"{step['move']}: {step['result']}{scores}: {step['reason']}")
    print(f"head {summary['head']}; releases in {summary['releases']}; episodes in {summary['runs']}")
    return 0


__all__ = [
    "SLOTS",
    "E2EContext",
    "E2ERun",
    "NoCompression",
    "build",
    "episode_runner",
    "grader",
    "main",
    "missing_requirements",
    "summarize",
]
