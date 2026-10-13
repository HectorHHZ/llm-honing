"""Run a Honing campaign on tank duels from one JSON config.

The config names every component by import path, as RA2's does
(:mod:`honing.backends.ra2.e2e`): each entry is
``{"factory": "package.module:function", "args": {...}}``, called as
``function(context, **args)``. Components are built in :data:`SLOTS` order,
and a factory can read the ones built before it from ``context.components``.

The campaign runs the config's ``schedule`` of harness and train moves. Every
move is gated on ``gate_tasks`` (the ``val`` split); a harness method draws its
minibatches from ``harness_tasks`` and rollouts are played on them too (the
``train`` split). It writes the release chain, every round's run directory,
and ``summary.json`` under ``work_dir``.

``configs/e2e/tank_dry_run.json`` plays real rounds (the Node match process and
Honing's ``DecisionServer``) with a stand-in decision model, a fake trainer,
and GEPA reflecting through a scripted chat model, so it runs on a CPU with
Node and the ``gepa`` extra.

Run: ``python scripts/e2e/tank_campaign.py --config configs/e2e/tank_dry_run.json``
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

from honing.backends.tank.demonstrations import TankDemonstrationProvider
from honing.backends.tank.grader import TankGrader
from honing.backends.tank.harness import seed_harness
from honing.backends.tank.rollouts import TankRolloutProvider
from honing.backends.tank.runner import TankEpisodeRunner
from honing.backends.tank.scenarios import Split, TankSuite, load_suite
from honing.campaign import (
    Campaign,
    CampaignDefaults,
    CompressionComponents,
    HarnessComponents,
    ScriptedScheduler,
    StepOutcome,
)
from honing.core import (
    ChatModel,
    CompressionBudget,
    ContractError,
    DataProvider,
    EpisodeOutcome,
    EpisodeRunner,
    Evaluator,
    EvolveMethod,
    HarnessMoveConfig,
    HarnessSpec,
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
    Task,
    TaskGrader,
    Trainer,
    TrainJobSpec,
    UtilizationTelemetry,
)
from honing.e2e.components import build_component, missing_requirements
from honing.release import EpisodeEvaluator, Gate, ReleaseChain, WinLossSelector

DEFAULT_INTEGRATION = "integrations/tank-duel"
DEFAULT_SUITE = "tasks/suite.jsonl"

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
    """What a component factory can read: paths, the task suite and splits, and the components built so far."""

    root: Path
    integration_dir: Path
    work_dir: Path
    suite: TankSuite
    gate_tasks: tuple[Task, ...]
    harness_tasks: tuple[Task, ...]
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
        return ContractError("this tank campaign has no compression plane; schedule only harness and train moves")

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


class TankDeliveryProbe:
    """Satisfies :class:`~honing.e2e.conformance.DeliveryProbe` for tank rounds.

    A file counts as delivered when the round's manifest records it at its
    path with the content's SHA-256, and, for an operating note, when its text
    appears in a state the model was sent.
    """

    def delivered(self, outcome: EpisodeOutcome, spec: HarnessSpec) -> dict[str, bool]:
        import hashlib

        if outcome.workspace_location is None:
            return {path: False for path, _ in spec.files}
        run = Path(outcome.workspace_location)
        manifest = json.loads((run / "manifest.json").read_text())
        recorded = {entry["path"]: entry for entry in manifest["harness"]}
        log = (run / "decisions.jsonl").read_text(encoding="utf-8") if (run / "decisions.jsonl").exists() else ""
        states = [part for line in log.splitlines() for part in json.loads(line)["contextParts"]["essential"]]
        delivered = {}
        for path, file in spec.files:
            entry = recorded.get(path)
            if entry is None or entry["sha256"] != hashlib.sha256(file.content.encode()).hexdigest():
                delivered[path] = False
            elif entry["role"] == "note":
                delivered[path] = any(file.content.strip() in state for state in states)
            else:
                delivered[path] = True
        return delivered


# ---------------------------------------------------------------- factories for the tank components


def _expand(values: Sequence[str], root: Path, integration_dir: Path) -> list[str]:
    """Fill ``{python}``, ``{root}``, and ``{integration}`` in a command."""
    names = {"python": sys.executable, "root": str(root), "integration": str(integration_dir)}
    return [value.format(**names) for value in values]


def _place(context: Any) -> tuple[Path, Path, TankSuite]:
    """The integration directory, work directory, and suite, from this module's context or a pipeline's."""
    if isinstance(context, E2EContext):
        return context.integration_dir, context.work_dir, context.suite
    integration = (Path(context.root) / DEFAULT_INTEGRATION).resolve()
    suite = context.shared.get("tank_suite")
    if suite is None:
        suite = context.shared["tank_suite"] = load_suite(integration / DEFAULT_SUITE)
    return integration, Path(context.work_dir), suite


def episode_runner(context: Any, command: Sequence[str] | None = None, concurrency: int = 4) -> TankEpisodeRunner:
    """The tank runner, with the policy served by the ``endpoints`` component. ``command`` replaces Node."""
    integration, work, suite = _place(context)
    return TankEpisodeRunner(
        suite=suite,
        endpoints=context.component("endpoints"),
        integration_dir=integration,
        runs_dir=work / "runs",
        command=_expand(command, Path(context.root), integration) if command else None,
        concurrency=concurrency,
    )


def grader(context: Any) -> TankGrader:
    _, _, suite = _place(context)
    return TankGrader(suite=suite)


def rollouts(context: Any) -> TankRolloutProvider:
    """Sampled rounds through the ``runner`` and ``grader`` components."""
    return TankRolloutProvider(context.component("runner"), context.component("grader"))


def demonstrations(context: E2EContext, limit: int | None = None) -> TankDemonstrationProvider:
    """Planner-teacher rounds on the harness tasks (the train split), under the seed harness."""
    return TankDemonstrationProvider(
        context.component("runner"),
        context.harness_tasks[:limit],
        seed_harness(context.integration_dir),
    )


def gepa_method(context: E2EContext, chat: Mapping[str, Any], **config: Any) -> Any:
    """GEPA validating on the gate's tasks, reflecting through the chat model ``chat`` names."""
    from honing.harness.methods.gepa import GEPAConfig, GEPAMethod

    chat_model = build_component(chat, context, "chat", ChatModel)
    return GEPAMethod(chat_model, GEPAConfig(**config), [task.task_id for task in context.gate_tasks])  # type: ignore[arg-type]


def tasks(context: Any, split: str = "val", limit: int | None = 4) -> tuple[Task, ...]:
    """Tasks of one split, for a runner check config's ``tasks`` entry."""
    _, _, suite = _place(context)
    return _tasks(suite, {"split": split, "limit": limit})


def harness_seed(context: Any) -> HarnessSpec:
    """The seed harness, for a runner check config's ``harness_seed`` entry."""
    integration, _, _ = _place(context)
    return seed_harness(integration)


def delivery_probe(context: object) -> TankDeliveryProbe:
    return TankDeliveryProbe()


# ---------------------------------------------------------------- config


def _tasks(suite: TankSuite, selection: Mapping[str, object]) -> tuple[Task, ...]:
    split = str(selection.get("split", "val"))
    if split not in get_args(Split):
        raise ContractError(f"unknown task split {split!r}")
    limit = selection.get("limit")
    return suite.tasks(split, limit=int(str(limit)) if limit is not None else None)  # type: ignore[arg-type]


def _move(entry: Mapping[str, object]) -> Move:
    kind = entry.get("kind")
    if kind == "harness":
        budget = entry.get("episode_budget")
        return Move(
            kind="harness",
            harness=HarnessMoveConfig(
                label=str(entry.get("label", "harness")),
                episode_budget=int(str(budget)) if budget is not None else None,
            ),
        )
    if kind == "train":
        algorithm = str(entry.get("algorithm", "sft"))
        job = TrainJobSpec(
            algorithm=algorithm,  # type: ignore[arg-type]
            steps=int(str(entry.get("steps", 1))),
            contract=entry.get("contract", "typed_decisions"),  # type: ignore[arg-type]
            # On-policy algorithms play under the head's harness: the controller is part of the policy.
            rollout="harness" if algorithm == "grpo" else "none",
        )
        return Move(kind="train", train=job)
    raise ContractError(f"a tank campaign schedules harness and train moves, not {kind!r}")


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
    integration_dir = (root / config.get("integration_dir", DEFAULT_INTEGRATION)).resolve()
    work = (work_dir or root / config.get("work_dir", "runs/tank-e2e")).resolve()
    work.mkdir(parents=True, exist_ok=False)
    suite = load_suite(integration_dir / config.get("suite", DEFAULT_SUITE))
    context = E2EContext(
        root=root.resolve(),
        integration_dir=integration_dir,
        work_dir=work,
        suite=suite,
        gate_tasks=_tasks(suite, config.get("gate_tasks", {"split": "val", "limit": 4})),
        harness_tasks=_tasks(suite, config.get("harness_tasks", {"split": "train", "limit": 8})),
    )
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
    evaluator = c.get("evaluator") or EpisodeEvaluator(c["runner"], c["grader"], context.gate_tasks)  # type: ignore[arg-type]
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
            tasks=context.harness_tasks,
            episode_budget=int(config.get("episode_budget", 0)),
            samples_per_task=int(config.get("samples_per_task", 1)),
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
                "workspaces": [r.outcome.workspace_location for r in evaluation.records] if evaluation else [],
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
    parser = argparse.ArgumentParser(description="Run a Honing campaign on tank duels from one config.")
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
    "TankDeliveryProbe",
    "build",
    "delivery_probe",
    "demonstrations",
    "episode_runner",
    "gepa_method",
    "grader",
    "harness_seed",
    "main",
    "rollouts",
    "summarize",
    "tasks",
]
