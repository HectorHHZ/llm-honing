"""Factories that put Meta-Harness into a pipeline config (see :mod:`honing.e2e.components`).

Meta-Harness searches on the config's ``harness_tasks`` and its proposer reads
their trajectories, so the gate's ``tasks`` must be held out from them; its
selector refuses a gate on any search task. ``harness.episode_budget`` must
cover two runs of the search tasks and a smoke test (see
:class:`~honing.harness.methods.meta_harness.MetaHarnessMethod`).

``configs/e2e/meta_harness_dry_run.json`` runs it with a scripted proposer.

Paths in arguments may use ``{root}`` (the repository root), ``{run}`` (the
run directory's name), ``{tmp}`` (the system's temporary directory), and ``~``.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from honing.core import ProposerAgent
from honing.e2e.components import PipelineContext, build_component
from honing.harness.methods.meta_harness import MetaHarnessMethod
from honing.harness.proposers import DEFAULT_TOOLS, ClaudeCodeProposer


def method(context: PipelineContext, proposer: Mapping[str, Any], work_dir: str | None = None) -> MetaHarnessMethod:
    """Meta-Harness with the proposer agent ``proposer`` names.

    Args:
        context: The pipeline context.
        proposer: A component entry whose factory returns a ``ProposerAgent``.
        work_dir: The method's directory, where the archive is written for the
            proposer; defaults to ``meta-harness`` under the run's directory.
            A real proposer should work outside the repository, for example
            in ``{tmp}/meta-harness-{run}``.

    Returns:
        The method.
    """
    agent = build_component(proposer, context, "proposer", ProposerAgent)
    directory = _path(work_dir, context) if work_dir else context.work_dir / "meta-harness"
    return MetaHarnessMethod(agent, directory)  # type: ignore[arg-type]


def claude_code(
    context: PipelineContext,
    model: str,
    effort: str | None = None,
    timeout_seconds: float | None = None,
    settings: str | Mapping[str, Any] | None = None,
    base_url: str | None = None,
    inherit_environment: Sequence[str] = (),
    tools: Sequence[str] = DEFAULT_TOOLS,
) -> ClaudeCodeProposer:
    """Claude Code as the proposer, logging its sessions under ``proposer-logs`` in the run's directory.

    Args:
        context: The pipeline context.
        model: The model the CLI runs.
        effort: The CLI's ``--effort``.
        timeout_seconds: The limit on one session.
        settings: Claude Code settings for ``--settings``, such as permission
            rules denying paths outside the archive and the Bash sandbox: a
            file, or the settings themselves, written to
            ``proposer-settings.json`` in the run's directory, where every
            string may use the placeholders.
        base_url: An Anthropic-compatible API root to call instead of Anthropic's.
        inherit_environment: Further variables the CLI takes from the environment.
        tools: The tools the agent may use.

    Returns:
        The proposer.
    """
    if isinstance(settings, Mapping):
        written = context.work_dir / "proposer-settings.json"
        written.write_text(json.dumps(_expand(settings, context), indent=2) + "\n", encoding="utf-8")
        settings_file: Path | None = written
    else:
        settings_file = _path(settings, context) if settings else None
    return ClaudeCodeProposer(
        model,
        context.work_dir / "proposer-logs",
        tools=tools,
        effort=effort,
        timeout_seconds=timeout_seconds,
        settings=settings_file,
        base_url=base_url,
        inherit_environment=inherit_environment,
    )


def _placeholders(context: PipelineContext) -> dict[str, str]:
    return {"root": str(context.root), "run": context.work_dir.name, "tmp": tempfile.gettempdir()}


def _path(value: str, context: PipelineContext) -> Path:
    return Path(value.format(**_placeholders(context))).expanduser()


def _expand(value: Any, context: PipelineContext) -> Any:
    if isinstance(value, str):
        return value.format(**_placeholders(context))
    if isinstance(value, Mapping):
        return {key: _expand(item, context) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item, context) for item in value]
    return value
