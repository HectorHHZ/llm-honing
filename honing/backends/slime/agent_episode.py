"""The files between :mod:`~honing.backends.slime.agent_runner` and :mod:`~honing.backends.slime.agent_worker`.

The runner writes ``episode.json`` (:func:`~honing.backends.slime.agent_runner.episode_record`)
and the worker reads it (:func:`read_episode`); the worker's module docstring lists what
it writes back. This module holds the names both sides use, and imports only the
standard library and the sandbox's limits, so importing the runner does not import the
worker: the worker runs as ``python -m honing.backends.slime.agent_worker``, which would
otherwise execute a module the package had already imported.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from honing.backends.slime.sandbox import EXEC_GRACE_SECONDS, FILE_SECONDS, ContainerLimits

SCHEMA_VERSION = 1
EPISODE_FILE = "episode.json"
RESULT_FILE = "result.json"
STATE_FILE = "state.json"
TRANSCRIPT_FILE = "transcript.jsonl"
LOG_FILE = "worker.log"
WORKSPACE_DIR = "workspace"
HARNESS_DIR = "harness"
WORKSPACE_OUT_DIR = "workspace_out"

# The agent's working directory in the sandbox. Slime's run_agent writes the CLI's output
# to {workdir}/.harness/trajectory.jsonl (harness/common.py:109-117 at the pin).
WORKDIR = "/workspace"
SLIME_META_DIR = ".harness"

STATUS_COMPLETED = "completed"
STATUS_TIME_BUDGET_EXCEEDED = "time_budget_exceeded"
STATUS_INFRA_ERROR = "infra_error"
STATUSES = (STATUS_COMPLETED, STATUS_TIME_BUDGET_EXCEEDED, STATUS_INFRA_ERROR)
EXIT_INFRA = 3
# slime.agent.sandbox.EXIT_TIME_BUDGET_EXCEEDED: exec_and_wait's code when the budget ran out.
EXIT_TIME_BUDGET_EXCEEDED = -1

# The worker's limits on the steps outside the setup and the agent's budget, in seconds. The
# runner's deadline for the worker adds them up (``AgentEpisodeSettings.worker_timeout_seconds``),
# each exec with the time the sandbox allows past its own limit.
# Slime's steps between the setup and the agent's clock, which starts only when exec_and_wait
# polls against the budget: the per-exec timeouts that ClaudeCodeHarness.run passes, by the
# slime function that passes them (paths relative to ``slime/agent/``), and exec_and_wait's
# write_file of the launcher. tests/contracts/test_slime_agent_api.py checks them against slime.
SLIME_PRE_LAUNCH_EXEC_SECONDS: Mapping[str, tuple[int, ...]] = {
    "sandbox.py:ensure_agent_user": (60,),
    "harness/claude_code.py:ClaudeCodeHarness.write_config": (60,),
    "harness/common.py:run_agent": (30,),
    "sandbox.py:exec_and_wait": (30, 30),
}
SLIME_PRE_LAUNCH_FILE_WRITES = 1
PRE_LAUNCH_SECONDS = (
    sum(seconds + EXEC_GRACE_SECONDS for limits in SLIME_PRE_LAUNCH_EXEC_SECONDS.values() for seconds in limits)
    + SLIME_PRE_LAUNCH_FILE_WRITES * FILE_SECONDS
)
# Stopping the agent before the copy-out: a check that ps and pkill are there, one listing of
# its processes, one pkill, then polls until none is left.
STOP_KILL_SECONDS = 60
STOP_POLLS = 25
STOP_POLL_SECONDS = 10
STOP_POLL_INTERVAL_SECONDS = 0.2
STOP_AGENT_SECONDS = (
    2 * (STOP_POLL_SECONDS + EXEC_GRACE_SECONDS)
    + (STOP_KILL_SECONDS + EXEC_GRACE_SECONDS)
    + STOP_POLLS * (STOP_POLL_SECONDS + EXEC_GRACE_SECONDS + STOP_POLL_INTERVAL_SECONDS)
)
# Copying out: the transcript's read_file, then the tar of the working directory.
COPY_SECONDS = 300
COPY_OUT_SECONDS = FILE_SECONDS + COPY_SECONDS + EXEC_GRACE_SECONDS


def sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class EpisodeFileError(ValueError):
    """``episode.json`` is missing a field or holds a value of the wrong type."""


@dataclass(frozen=True)
class PlacedFile:
    """One file in the agent's working directory, and the harness paths whose content it carries."""

    target: str
    sha256: str
    paths: tuple[str, ...]


@dataclass(frozen=True)
class Episode:
    """``episode.json``, checked; :func:`~honing.backends.slime.agent_runner.episode_record` writes it."""

    task_id: str
    prompt: str
    sglang_url: str
    served_model: str
    tokenizer_location: str
    image: str
    time_budget_seconds: int
    setup_timeout_seconds: int
    copy_out_max_bytes: int
    limits: ContainerLimits
    network: str
    host_alias: str
    lifetime_seconds: int
    name_prefix: str
    placement: tuple[PlacedFile, ...]
    install_cli: bool = True
    cli_extra_args: str = ""
    adapter_bind_host: str | None = None
    tool_parser: str | None = None
    reasoning_parser: str | None = None
    max_turns_per_sid: int | None = None
    max_context_tokens: int = 0
    fork_threshold_tokens: int | None = None
    max_new_tokens: int | None = None
    sampling_defaults: Mapping[str, object] = field(default_factory=dict)


def _field(record: Mapping[str, Any], key: str, kind: type | tuple[type, ...], *, optional: bool = False) -> Any:
    if key not in record:
        if optional:
            return None
        raise EpisodeFileError(f"episode.json has no {key!r}")
    value = record[key]
    if value is None and optional:
        return None
    kinds = kind if isinstance(kind, tuple) else (kind,)
    # bool is an int, and an int field must not take true or false.
    if not isinstance(value, kinds) or (isinstance(value, bool) and bool not in kinds):
        names = " or ".join(item.__name__ for item in kinds)
        raise EpisodeFileError(f"episode.json {key!r} must be {names}, not {type(value).__name__}")
    return value


def _section(record: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = _field(record, key, dict)
    assert isinstance(value, dict)
    return value


def read_episode(path: Path) -> Episode:
    """Read and check ``episode.json``; raise :class:`EpisodeFileError` for anything malformed."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as error:  # not only JSONDecodeError: a huge integer raises a plain ValueError
        raise EpisodeFileError(f"episode.json is not JSON: {error}") from None
    if not isinstance(record, dict):
        raise EpisodeFileError("episode.json must hold a JSON object")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise EpisodeFileError(
            f"episode.json schema_version must be {SCHEMA_VERSION}, not {record.get('schema_version')!r}"
        )
    task, policy, adapter = _section(record, "task"), _section(record, "policy"), _section(record, "adapter")
    agent, sandbox, cli = _section(record, "agent"), _section(record, "sandbox"), _section(record, "cli")
    placement = _field(record, "placement", list)
    placed: list[PlacedFile] = []
    for entry in placement:
        if not isinstance(entry, dict):
            raise EpisodeFileError("each placement entry must be a JSON object")
        paths = _field(entry, "paths", list)
        if not paths or not all(isinstance(item, str) for item in paths):
            raise EpisodeFileError("a placement entry's paths must be a non-empty list of strings")
        placed.append(PlacedFile(_field(entry, "target", str), _field(entry, "sha256", str), tuple(paths)))
    sampling = _field(adapter, "sampling_defaults", dict, optional=True) or {}
    try:
        limits = ContainerLimits(
            cpus=float(_field(sandbox, "cpus", (int, float))),
            memory_bytes=_field(sandbox, "memory_bytes", int),
            pids=_field(sandbox, "pids", int),
        )
    except ValueError as error:
        raise EpisodeFileError(str(error)) from None
    episode = Episode(
        task_id=_field(task, "task_id", str),
        prompt=_field(task, "prompt", str),
        sglang_url=_field(policy, "sglang_url", str),
        served_model=_field(policy, "served_model", str),
        tokenizer_location=_field(policy, "tokenizer_location", str),
        image=_field(sandbox, "image", str),
        time_budget_seconds=_field(agent, "time_budget_seconds", int),
        setup_timeout_seconds=_field(agent, "setup_timeout_seconds", int),
        copy_out_max_bytes=_field(agent, "copy_out_max_bytes", int),
        limits=limits,
        network=_field(sandbox, "network", str),
        host_alias=_field(sandbox, "host_alias", str),
        lifetime_seconds=_field(sandbox, "lifetime_seconds", int),
        name_prefix=_field(sandbox, "name_prefix", str),
        placement=tuple(placed),
        install_cli=_field(cli, "install", bool),
        cli_extra_args=_field(cli, "extra_args", str, optional=True) or "",
        adapter_bind_host=_field(adapter, "bind_host", str, optional=True),
        tool_parser=_field(adapter, "tool_parser", str, optional=True),
        reasoning_parser=_field(adapter, "reasoning_parser", str, optional=True),
        max_turns_per_sid=_field(adapter, "max_turns_per_sid", int, optional=True),
        max_context_tokens=_field(adapter, "max_context_tokens", int, optional=True) or 0,
        fork_threshold_tokens=_field(adapter, "fork_threshold_tokens", int, optional=True),
        max_new_tokens=_field(adapter, "max_new_tokens", int, optional=True),
        sampling_defaults=dict(sampling),
    )
    for name in ("time_budget_seconds", "setup_timeout_seconds", "copy_out_max_bytes", "lifetime_seconds"):
        if getattr(episode, name) <= 0:
            raise EpisodeFileError(f"episode.json {name} must be positive")
    if episode.max_new_tokens is not None and episode.max_new_tokens <= 0:
        raise EpisodeFileError("episode.json max_new_tokens must be positive or null")
    return episode
