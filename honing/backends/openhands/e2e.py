"""Factories that put the OpenHands runner into a pipeline config (see ``honing.e2e.components``).

Paths in a factory's args may use ``{root}`` (the repository root) and environment variables.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from honing.backends.openhands.runner import (
    AGENT,
    DEFAULT_TOOLS,
    OpenHandsDeliveryProbe,
    OpenHandsEpisodeRunner,
    ToolPolicy,
    default_python,
)
from honing.core import HarnessFile, HarnessSpec
from honing.e2e.components import PipelineContext


def _path(value: str, context: PipelineContext) -> Path:
    return Path(os.path.expandvars(value.format(root=str(context.root))))


def runner(
    context: PipelineContext,
    python: str | None = None,
    worker: str = "{root}/integrations/openhands/worker.py",
    tools: Sequence[str] = DEFAULT_TOOLS,
    default_tools: Sequence[str] = DEFAULT_TOOLS,
    mcp_registry: Mapping[str, Mapping[str, object]] | None = None,
    mounts: Mapping[str, str] | None = None,
    concurrency: int = 1,
    timeout_seconds: float = 900.0,
    max_iterations: int = 30,
    max_budget_usd: float | None = None,
) -> OpenHandsEpisodeRunner:
    """The OpenHands runner; reads the ``endpoints`` component for the policy."""
    return OpenHandsEpisodeRunner(
        python=_path(python, context) if python else default_python(context.root),
        worker=_path(worker, context),
        workspaces=context.work_dir / "workspaces",
        endpoints=context.component("endpoints"),
        tools=ToolPolicy(
            allowed=frozenset(tools), default=tuple(default_tools), mcp_registry=dict(mcp_registry or {})
        ),
        mounts={name: _path(path, context) for name, path in (mounts or {}).items()},
        concurrency=concurrency,
        timeout_seconds=timeout_seconds,
        max_iterations=max_iterations,
        max_budget_usd=max_budget_usd,
    )


def probe(context: object) -> OpenHandsDeliveryProbe:
    """Factory for a config's ``conformance.probe`` entry."""
    return OpenHandsDeliveryProbe()


def seed_harness(context: object, instructions: str = "Solve the task.\n") -> HarnessSpec:
    """A minimal OpenHands harness: one instructions file."""
    return HarnessSpec(agent=AGENT, files=(("AGENTS.md", HarnessFile("instructions", instructions)),))
