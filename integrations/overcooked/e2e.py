"""Factories usable by the existing JSON component loader from a source checkout."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from integrations.overcooked.adapter import OvercookedEpisodeRunner, OvercookedGrader
from integrations.overcooked.harness import seed_harness
from integrations.overcooked.scenarios import load_suite


def tasks(context: Any, split: str = "val", limit: int | None = None) -> tuple:
    return load_suite().tasks(split, limit)


def harness_seed(context: Any) -> Any:
    return seed_harness()


def episode_runner(context: Any, python: str | None = None) -> OvercookedEpisodeRunner:
    return OvercookedEpisodeRunner(
        suite=load_suite(),
        endpoints=context.component("endpoints"),
        runs_dir=Path(context.work_dir) / "runs",
        python=python,
    )


def grader(context: Any, python: str | None = None) -> OvercookedGrader:
    return OvercookedGrader(suite=load_suite(), python=python)
