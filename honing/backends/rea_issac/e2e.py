"""Component factories for Honing's existing JSON-configured evaluation tools.

The surrounding application supplies policy endpoints. These factories never
start a model, install game packages, or build a training campaign.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from honing.backends.rea_issac.grader import ReaIssacGrader
from honing.backends.rea_issac.harness import seed_harness
from honing.backends.rea_issac.runner import ReaIssacEpisodeRunner
from honing.backends.rea_issac.scenarios import ReaIssacSuite, Split, load_suite
from honing.core import HarnessSpec, Task


def _integration(context: Any) -> Path:
    return Path(context.root) / "integrations/rea_issac"


def _suite(context: Any) -> ReaIssacSuite:
    return load_suite(_integration(context) / "tasks/suite.jsonl")


def tasks(context: Any, split: Split = "val", limit: int | None = None) -> tuple[Task, ...]:
    return _suite(context).tasks(split, limit=limit)


def harness_seed(context: Any) -> HarnessSpec:
    return seed_harness(_integration(context))


def episode_runner(context: Any, python: str | None = None, concurrency: int = 1) -> ReaIssacEpisodeRunner:
    integration = _integration(context)
    command = [python, str(integration / "scripts/match.py")] if python is not None else None
    return ReaIssacEpisodeRunner(
        suite=_suite(context),
        endpoints=context.component("endpoints"),
        integration_dir=integration,
        runs_dir=Path(context.work_dir) / "runs",
        command=command,
        concurrency=concurrency,
    )


def grader(context: Any) -> ReaIssacGrader:
    return ReaIssacGrader(suite=_suite(context))
