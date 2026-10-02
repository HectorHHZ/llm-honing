"""Check the harness runner, or the trainer, a pipeline config names against Honing's contracts.

:func:`main` checks the runner (:func:`honing.e2e.conformance.check_runner`);
:func:`trainer_main` checks the trainer (:func:`honing.e2e.trainer_conformance.check_trainer`).

The runner, grader, policy endpoints, and model source are built from the
config's ``components`` exactly as the pipeline builds them, so the runner is
checked in the configuration a campaign will use. The config's ``conformance``
entry names the delivery probe (``{"probe": {"factory": ...}, "concurrent": 4}``);
without one, the delivery case fails and says so.

Run: ``python scripts/e2e/check_runner.py --config configs/e2e/all_planes.json`` or
``python scripts/e2e/check_trainer.py --config configs/e2e/all_planes.json``
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from honing.core import (
    ContractError,
    DataProvider,
    EpisodeRunner,
    HarnessSpec,
    ModelSource,
    PolicyEndpoints,
    RolloutProvider,
    TaskGrader,
    Trainer,
)
from honing.e2e.components import PipelineContext, build_component, missing_requirements
from honing.e2e.conformance import DeliveryProbe, RunnerCheck, check_runner
from honing.e2e.pipeline import _job
from honing.e2e.trainer_conformance import TrainerCheck, check_trainer

# What a runner check builds, in order: the policy's weights, how a runner reaches them, the runner, its grader.
SLOTS = (
    ("model_source", ModelSource, True),
    ("endpoints", PolicyEndpoints, False),
    ("runner", EpisodeRunner, True),
    ("grader", TaskGrader, True),
)


def run_checks(config: dict[str, Any], root: Path, work_dir: Path) -> list[RunnerCheck]:
    missing = missing_requirements(config.get("requires", {}))
    if missing:
        raise ContractError("this config needs " + ", ".join(missing))
    context = PipelineContext(root=root.resolve(), work_dir=work_dir.resolve())
    components = config.get("components", {})
    for slot, protocol, required in SLOTS:
        if slot in components:
            context.components[slot] = build_component(components[slot], context, slot, protocol)
        elif required:
            raise ContractError(f"the config names no {slot} component")
    tasks = build_component(config["tasks"], context, "tasks", tuple)
    seed = build_component(config["harness_seed"], context, "harness_seed", HarnessSpec)
    settings = config.get("conformance", {})
    probe = build_component(settings["probe"], context, "probe", DeliveryProbe) if "probe" in settings else None
    snapshot = context.components["model_source"].materialize(context.work_dir)  # type: ignore[attr-defined]
    return check_runner(
        context.components["runner"],  # type: ignore[arg-type]
        context.components["grader"],  # type: ignore[arg-type]
        seed=seed,  # type: ignore[arg-type]
        tasks=tasks,  # type: ignore[arg-type]
        snapshot=snapshot,
        probe=probe,  # type: ignore[arg-type]
        concurrent=int(settings.get("concurrent", 4)),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check a pipeline config's harness runner against Honing's contract.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--work-dir", type=Path, help="where episodes run (default: a temporary directory)")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    work = args.work_dir or Path(tempfile.mkdtemp(prefix="honing-runner-check-"))
    work.mkdir(parents=True, exist_ok=True)
    try:
        checks = run_checks(config, args.root, work)
    except ContractError as error:
        print(f"cannot start: {error}", file=sys.stderr)
        return 2
    for check in checks:
        print(f"{'pass' if check.passed else 'FAIL'}  {check.name:<10} {check.detail}")
    failed = [check.name for check in checks if not check.passed]
    print(f"runner check: {'ok' if not failed else 'failed: ' + ', '.join(failed)}; episodes in {work}")
    return 1 if failed else 0


# What a trainer check builds: the base weights, the trainer, and the data it trains on.
TRAINER_SLOTS = (
    ("model_source", ModelSource, True),
    ("trainer", Trainer, True),
    ("data", DataProvider, True),
    ("rollout", RolloutProvider, False),
)


def run_trainer_checks(config: dict[str, Any], root: Path, work_dir: Path) -> list[TrainerCheck]:
    """Build the trainer and its inputs as the pipeline does, and check the config's ``trainer_check.jobs``."""
    missing = missing_requirements(config.get("requires", {}))
    if missing:
        raise ContractError("this config needs " + ", ".join(missing))
    context = PipelineContext(root=root.resolve(), work_dir=work_dir.resolve())
    components = config.get("components", {})
    for slot, protocol, required in TRAINER_SLOTS:
        if slot in components:
            context.components[slot] = build_component(components[slot], context, slot, protocol)
        elif required:
            raise ContractError(f"the config names no {slot} component")
    tasks = build_component(config["tasks"], context, "tasks", tuple)
    seed = build_component(config["harness_seed"], context, "harness_seed", HarnessSpec)
    settings = config.get("trainer_check", {})
    jobs = [_job(entry) for entry in settings.get("jobs", [{"algorithm": "sft"}])]
    c = context.components
    return check_trainer(
        c["trainer"],  # type: ignore[arg-type]
        base=c["model_source"].materialize(context.work_dir),  # type: ignore[attr-defined]
        jobs=jobs,
        data=c["data"],  # type: ignore[arg-type]
        rollout=c.get("rollout"),  # type: ignore[arg-type]
        tasks=tasks,  # type: ignore[arg-type]
        harness=seed,  # type: ignore[arg-type]
        samples_per_task=int(settings.get("samples_per_task", 2)),
    )


def trainer_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check a pipeline config's trainer against Honing's contract.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--work-dir", type=Path, help="where the trainer writes (default: a temporary directory)")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    work = args.work_dir or Path(tempfile.mkdtemp(prefix="honing-trainer-check-"))
    work.mkdir(parents=True, exist_ok=True)
    try:
        checks = run_trainer_checks(config, args.root, work)
    except ContractError as error:
        print(f"cannot start: {error}", file=sys.stderr)
        return 2
    for check in checks:
        print(f"{'pass' if check.passed else 'FAIL'}  {check.name:<14} {check.detail}")
    failed = [check.name for check in checks if not check.passed]
    print(f"trainer check: {'ok' if not failed else 'failed: ' + ', '.join(failed)}; outputs in {work}")
    return 1 if failed else 0
