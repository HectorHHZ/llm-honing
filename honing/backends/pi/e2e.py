"""Factories that put pi into a pipeline config (see :mod:`honing.e2e.components`).

Paths in arguments may use ``{root}`` (the repository root), ``~``, and environment variables.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

from honing.backends.pi.harness import AGENT, DEFAULT_TOOLS, SETTINGS, ToolPolicy, settings_file
from honing.backends.pi.runner import MountMode, PiDeliveryProbe, PiEpisodeRunner
from honing.core import ContractError, HarnessFile, HarnessSpec
from honing.e2e.components import PipelineContext


def runner(
    context: PipelineContext,
    node: str = "node",
    worker: str = "{root}/integrations/pi/worker.mjs",
    tools: Sequence[str] = DEFAULT_TOOLS,
    default_tools: Sequence[str] = DEFAULT_TOOLS,
    mounts: Mapping[str, str] | None = None,
    mount_mode: MountMode = "copy",
    inherit_environment: Sequence[str] = (),
    model_settings: Mapping[str, object] | None = None,
    concurrency: int = 1,
    timeout_seconds: float = 900.0,
) -> PiEpisodeRunner:
    """Builds the pi runner from the ``endpoints`` component built before it.

    Args:
        context: The pipeline context.
        node: The Node.js executable, a path or a name looked up on ``PATH``.
        worker: The worker script.
        tools: Tools a harness may enable.
        default_tools: Tools the agent gets when the harness names none.
        mounts: Directories put into every agent's working directory, by name.
        mount_mode: ``copy`` (each episode gets its own copy) or ``link``; see
            :class:`~honing.backends.pi.runner.PiEpisodeRunner`.
        inherit_environment: Names of further variables the worker takes from
            Honing's environment, such as ``HTTPS_PROXY`` or ``SSL_CERT_FILE``.
        model_settings: Fields for the model's ``models.json`` entry, such as ``contextWindow``.
        concurrency: How many agents run at once.
        timeout_seconds: The limit on one episode.

    Returns:
        The runner.

    Raises:
        ContractError: ``node`` is not found.
    """
    executable = shutil.which(_path(node, context) if "/" in node else node)
    if executable is None:
        raise ContractError(f"no Node.js executable {node!r}; pi needs Node.js 22.19 or newer")
    return PiEpisodeRunner(
        node=Path(executable),
        worker=Path(_path(worker, context)),
        workspaces=context.work_dir / "workspaces",
        endpoints=context.component("endpoints"),
        tools=ToolPolicy(allowed=frozenset(tools), default=tuple(default_tools)),
        mounts={name: Path(_path(path, context)) for name, path in (mounts or {}).items()},
        mount_mode=mount_mode,
        inherit_environment=tuple(inherit_environment),
        model_settings=model_settings,
        concurrency=concurrency,
        timeout_seconds=timeout_seconds,
    )


def probe(context: object) -> PiDeliveryProbe:
    """Builds the delivery probe for a config's ``conformance.probe`` entry.

    Args:
        context: The pipeline context (unused).

    Returns:
        The probe.
    """
    return PiDeliveryProbe()


def seed_harness(
    context: PipelineContext,
    instructions: str | None = None,
    system_prompt: str | None = None,
    tools: Sequence[str] | None = None,
    thinking_level: str | None = None,
) -> HarnessSpec:
    """Builds a pi harness from arguments.

    Args:
        context: The pipeline context.
        instructions: Text for ``AGENTS.md``, appended to pi's system prompt.
        system_prompt: Text for ``SYSTEM.md``, which replaces pi's system prompt.
        tools: Tools for ``pi/settings.json``.
        thinking_level: The thinking level for ``pi/settings.json``.

    Returns:
        The harness.
    """
    spec = HarnessSpec(agent=AGENT)
    if instructions is not None:
        spec = spec.with_file("AGENTS.md", HarnessFile("instructions", instructions))
    if system_prompt is not None:
        spec = spec.with_file("SYSTEM.md", HarnessFile("instructions", system_prompt))
    if tools is not None or thinking_level is not None:
        spec = spec.with_file(SETTINGS, settings_file(tools, thinking_level))
    return spec


def _path(value: str, context: PipelineContext) -> str:
    return str(Path(os.path.expandvars(value.format(root=str(context.root)))).expanduser())
