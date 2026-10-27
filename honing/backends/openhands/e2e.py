"""Factories that put OpenHands into a pipeline config (see :mod:`honing.e2e.components`).

Paths in arguments may use ``{root}`` (the repository root), ``~``, and environment variables.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from honing.backends.openhands.harness import (
    AGENT,
    AGENT_SETTINGS,
    DEFAULT_TOOLS,
    MCP_SETTINGS,
    ToolPolicy,
    agent_settings,
    harness_from_directory,
    load_mcp_registry,
    mcp_settings,
    skill_files,
)
from honing.backends.openhands.runner import MountMode, OpenHandsDeliveryProbe, OpenHandsEpisodeRunner, default_python
from honing.core import HarnessFile, HarnessSpec
from honing.e2e.components import PipelineContext


def runner(
    context: PipelineContext,
    python: str | None = None,
    worker: str = "{root}/integrations/openhands/worker.py",
    tools: Sequence[str] = DEFAULT_TOOLS,
    default_tools: Sequence[str] = DEFAULT_TOOLS,
    tool_modules: Sequence[str] = (),
    mcp_registry: Mapping[str, Mapping[str, object]] | None = None,
    mcp_registry_file: str | None = None,
    mounts: Mapping[str, str] | None = None,
    mount_mode: MountMode = "copy",
    concurrency: int = 1,
    timeout_seconds: float = 900.0,
    max_iterations: int = 30,
    max_budget_usd: float | None = None,
) -> OpenHandsEpisodeRunner:
    """Builds the OpenHands runner from the ``endpoints`` component built before it.

    Args:
        context: The pipeline context.
        python: The OpenHands environment's interpreter; defaults to the README's.
        worker: The worker script.
        tools: Tools a harness may enable.
        default_tools: Tools the agent gets when the harness names none.
        tool_modules: Modules that register custom tools.
        mcp_registry: MCP server definitions by name.
        mcp_registry_file: A ``{"mcpServers": {...}}`` file whose servers join ``mcp_registry``.
        mounts: Directories put into every agent's working directory, by name.
        mount_mode: ``copy`` (each episode gets its own copy) or ``link``; see
            :class:`~honing.backends.openhands.runner.OpenHandsEpisodeRunner`.
        concurrency: How many agents run at once.
        timeout_seconds: The limit on one episode.
        max_iterations: The step limit.
        max_budget_usd: The cost limit on one episode.

    Returns:
        The runner.
    """
    registry: dict[str, Mapping[str, object]] = {}
    if mcp_registry_file:
        registry.update(load_mcp_registry(_path(mcp_registry_file, context)))
    registry.update(mcp_registry or {})
    return OpenHandsEpisodeRunner(
        python=_path(python, context) if python else default_python(context.root),
        worker=_path(worker, context),
        workspaces=context.work_dir / "workspaces",
        endpoints=context.component("endpoints"),
        tools=ToolPolicy(
            allowed=frozenset(tools),
            default=tuple(default_tools),
            mcp_registry=registry,
            tool_modules=tuple(tool_modules),
        ),
        mounts={name: _path(path, context) for name, path in (mounts or {}).items()},
        mount_mode=mount_mode,
        concurrency=concurrency,
        timeout_seconds=timeout_seconds,
        max_iterations=max_iterations,
        max_budget_usd=max_budget_usd,
    )


def probe(context: object) -> OpenHandsDeliveryProbe:
    """Builds the delivery probe for a config's ``conformance.probe`` entry.

    Args:
        context: The pipeline context (unused).

    Returns:
        The probe.
    """
    return OpenHandsDeliveryProbe()


def seed_harness(
    context: PipelineContext,
    directory: str | None = None,
    instructions: str | None = None,
    skills_dirs: Sequence[str] = (),
    tools: Sequence[str] | None = None,
    max_iterations: int | None = None,
    mcp_servers: Sequence[str] | None = None,
) -> HarnessSpec:
    """Builds an OpenHands harness from local files and arguments, each layered over the last.

    Args:
        context: The pipeline context.
        directory: A harness laid out on disk (see
            :func:`~honing.backends.openhands.harness.harness_from_directory`).
        instructions: Text for ``AGENTS.md``.
        skills_dirs: Local skills directories, such as ``~/.claude/skills``,
            added under ``skills/``.
        tools: Tools for ``openhands/agent.json``.
        max_iterations: A step limit for ``openhands/agent.json``.
        mcp_servers: Registry server names for ``openhands/mcp.json``.

    Returns:
        The harness.
    """
    spec = harness_from_directory(_path(directory, context)) if directory else HarnessSpec(agent=AGENT)
    if instructions is not None:
        spec = spec.with_file("AGENTS.md", HarnessFile("instructions", instructions))
    for skills in skills_dirs:
        for path, file in skill_files(_path(skills, context)):
            spec = spec.with_file(path, file)
    if tools is not None or max_iterations is not None:
        spec = spec.with_file(AGENT_SETTINGS, agent_settings(tools, max_iterations))
    if mcp_servers is not None:
        spec = spec.with_file(MCP_SETTINGS, mcp_settings(mcp_servers))
    return spec


def _path(value: str, context: PipelineContext) -> Path:
    return Path(os.path.expandvars(value.format(root=str(context.root)))).expanduser()
