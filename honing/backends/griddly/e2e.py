"""Factories for the existing Honing component loader and runner conformance checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from honing.backends.griddly.demonstrations import GriddlyDemonstrationProvider
from honing.backends.griddly.grader import GriddlyGrader
from honing.backends.griddly.harness import seed_harness
from honing.backends.griddly.rollouts import GriddlyRolloutProvider
from honing.backends.griddly.runner import GriddlyEpisodeRunner
from honing.backends.griddly.scenarios import GriddlySuite, Split, load_suite
from honing.core import EpisodeOutcome, HarnessSpec, Task


def _integration(context: Any) -> Path:
    return (Path(context.root) / "integrations/griddly").resolve()


def _suite(context: Any) -> GriddlySuite:
    return load_suite(_integration(context) / "tasks/suite.jsonl")


def tasks(context: Any, split: Split = "val", limit: int | None = None) -> tuple[Task, ...]:
    return _suite(context).tasks(split, limit=limit)


def harness_seed(context: Any) -> HarnessSpec:
    return seed_harness(_integration(context))


def episode_runner(context: Any, python: str | None = None, concurrency: int = 1) -> GriddlyEpisodeRunner:
    integration = _integration(context)
    return GriddlyEpisodeRunner(
        suite=_suite(context),
        endpoints=context.component("endpoints"),
        integration_dir=integration,
        runs_dir=Path(context.work_dir) / "runs",
        concurrency=concurrency,
        command=[python, str(integration / "scripts/match.py")] if python is not None else None,
    )


def grader(context: Any, python: str | None = None) -> GriddlyGrader:
    integration = _integration(context)
    return GriddlyGrader(
        suite=_suite(context),
        integration_dir=integration,
        command=[python, str(integration / "scripts/verify_replay.py")] if python is not None else None,
    )


def demonstrations(context: Any, runs: list[str]) -> GriddlyDemonstrationProvider:
    return GriddlyDemonstrationProvider(
        [Path(path).resolve() for path in runs], suite=_suite(context), integration_dir=_integration(context)
    )


def rollouts(context: Any) -> GriddlyRolloutProvider:
    return GriddlyRolloutProvider(context.component("runner"), context.component("grader"))


class GriddlyDeliveryProbe:
    def delivered(self, outcome: EpisodeOutcome, spec: HarnessSpec) -> dict[str, bool]:
        run = Path(outcome.workspace_location or "")
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        recorded = {entry["path"]: entry for entry in manifest["harness"]}
        return {
            path: recorded.get(path, {}).get("sha256") == hashlib.sha256(file.content.encode("utf-8")).hexdigest()
            and (run / "harness" / path).read_text(encoding="utf-8") == file.content
            for path, file in spec.files
        }


def delivery_probe(context: Any) -> GriddlyDeliveryProbe:
    return GriddlyDeliveryProbe()
