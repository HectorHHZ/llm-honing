"""Run OpenHands agents as Honing episodes.

OpenHands needs Python 3.12 and its own litellm pin, so it never runs in
Honing's interpreter. :class:`OpenHandsEpisodeRunner` prepares an episode
directory and starts ``integrations/openhands/worker.py`` with the OpenHands
environment's Python; the two sides share only files. Per episode:

    <workspaces>/<task>--<spec>--<uid>/
        episode.json      written here: task, policy endpoint, limits, allowlists (never the API key)
        harness/<path>    the HarnessSpec's files at their spec paths
        work/             the agent's working directory; each mount is linked in by name
        result.json, transcript.jsonl, delivery.json, openhands/, agent.log   written by the worker

The harness controls which tools and MCP servers the agent gets, through two
settings files (see ``integrations/openhands/README.md``), but only among what
the operator allows: :meth:`OpenHandsEpisodeRunner.check_spec` refuses a harness
that names a tool off the allowlist or an MCP server not in the registry, before
any episode starts. An evolution method can therefore change the agent's
toolset without being able to make it run an arbitrary command.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import signal
import sys
import uuid
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from honing.core import (
    CampaignRuleError,
    ContractError,
    EpisodeOutcome,
    HarnessSpec,
    PolicyEndpoints,
    PolicyRef,
    RunnerSupport,
    Task,
    require,
)

AGENT = "openhands"
AGENT_SETTINGS = "openhands/agent.json"
MCP_SETTINGS = "openhands/mcp.json"
TIMED_OUT = -9
# Read-only exploration and editing tools bundled with openhands-tools; terminal also runs commands.
DEFAULT_TOOLS = ("terminal", "file_editor")


@dataclass(frozen=True)
class ToolPolicy:
    """What the operator lets a harness give the agent.

    ``allowed`` is every tool a harness may enable in ``openhands/agent.json``,
    and ``default`` what the agent gets when the harness names none.
    ``mcp_registry`` maps a server name to its OpenHands MCP server
    definition; a harness enables servers by name in ``openhands/mcp.json``
    and never supplies a command itself.
    """

    allowed: frozenset[str] = frozenset(DEFAULT_TOOLS)
    default: tuple[str, ...] = DEFAULT_TOOLS
    mcp_registry: Mapping[str, Mapping[str, object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require(set(self.default) <= self.allowed, f"default tools {list(self.default)} must be allowed")

    def problems(self, spec: HarnessSpec) -> list[str]:
        """Every way ``spec``'s settings ask for something this policy does not allow."""
        problems = []
        for path, check in ((AGENT_SETTINGS, self._agent), (MCP_SETTINGS, self._mcp)):
            file = spec.get(path)
            if file is None:
                continue
            if file.kind != "settings":
                problems.append(f"{path} must be a settings file, not {file.kind}")
                continue
            try:
                settings = json.loads(file.content)
            except json.JSONDecodeError as error:
                problems.append(f"{path} is not JSON: {error}")
                continue
            if not isinstance(settings, dict):
                problems.append(f"{path} must be a JSON object")
                continue
            problems += check(path, settings)
        return problems

    def _agent(self, path: str, settings: Mapping[str, object]) -> list[str]:
        problems = []
        unknown = set(settings) - {"tools", "max_iterations"}
        if unknown:
            problems.append(f"{path} has unknown keys {sorted(unknown)}")
        tools = settings.get("tools", list(self.default))
        if not isinstance(tools, list) or not all(isinstance(tool, str) for tool in tools):
            problems.append(f"{path}: tools must be a list of tool names")
        else:
            refused = sorted(set(tools) - self.allowed)
            if refused:
                problems.append(f"{path} enables tools {refused}; the operator allows {sorted(self.allowed)}")
        iterations = settings.get("max_iterations")
        if iterations is not None and not (isinstance(iterations, int) and iterations > 0):
            problems.append(f"{path}: max_iterations must be a positive integer")
        return problems

    def _mcp(self, path: str, settings: Mapping[str, object]) -> list[str]:
        unknown = set(settings) - {"enabled"}
        enabled = settings.get("enabled", [])
        if unknown:
            return [f"{path} has unknown keys {sorted(unknown)}; a harness enables registry servers by name only"]
        if not isinstance(enabled, list) or not all(isinstance(name, str) for name in enabled):
            return [f"{path}: enabled must be a list of server names"]
        refused = sorted(set(enabled) - set(self.mcp_registry))
        if refused:
            return [f"{path} enables MCP servers {refused}; the registry has {sorted(self.mcp_registry)}"]
        return []


class OpenHandsEpisodeRunner:
    """Satisfies :class:`~honing.core.EpisodeRunner` by running one OpenHands conversation per episode.

    ``python`` is the OpenHands environment's interpreter and ``worker`` the
    worker script. ``mounts`` maps a name to a directory linked into every
    agent's working directory (a task's document corpus, for example); the link
    is not write-protected, so mount data the agent may not change from a copy.
    ``concurrency`` caps how many agents run at once; an agent still running
    after ``timeout_seconds`` is killed with its child processes and reported
    with exit code -9. ``max_iterations`` and ``max_budget_usd`` bound one
    conversation; a harness may lower the iteration limit, never raise it.

    The policy's API key is read from the environment variable its endpoints
    name (``HostedEndpoints``) and passed to the worker through the environment,
    never written into the episode directory.
    """

    support = RunnerSupport(
        agents=frozenset({AGENT}),
        file_kinds=frozenset({"instructions", "skill", "settings"}),
        programmable_loop=False,
        local_policy_only=False,
    )

    def __init__(
        self,
        *,
        python: Path,
        worker: Path,
        workspaces: Path,
        endpoints: PolicyEndpoints,
        tools: ToolPolicy | None = None,
        mounts: Mapping[str, Path] | None = None,
        environment: Mapping[str, str] | None = None,
        concurrency: int = 1,
        timeout_seconds: float = 900.0,
        max_iterations: int = 30,
        max_budget_usd: float | None = None,
        llm_timeout_seconds: int = 120,
    ) -> None:
        if concurrency <= 0 or timeout_seconds <= 0 or max_iterations <= 0:
            raise ContractError("concurrency, timeout_seconds, and max_iterations must be positive")
        if max_budget_usd is not None and max_budget_usd <= 0:
            raise ContractError("max_budget_usd must be positive when set")
        self._python = python
        self._worker = worker
        self._workspaces = workspaces
        self._endpoints = endpoints
        self._tools = tools or ToolPolicy()
        self._mounts = {name: Path(path).resolve() for name, path in (mounts or {}).items()}
        for name, path in self._mounts.items():
            require("/" not in name and name not in {"", ".", ".."}, f"a mount name is one path segment, got {name!r}")
            require(path.is_dir(), f"mount {name!r} points at {path}, which is not a directory")
        self._environment = dict(environment or {})
        self._concurrency = concurrency
        self._timeout = timeout_seconds
        self._limits = {
            "max_iterations": max_iterations,
            "max_budget_usd": max_budget_usd,
            "llm_timeout_seconds": llm_timeout_seconds,
        }
        # One semaphore per event loop: a campaign runs each gate in its own asyncio.run.
        self._slots: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
            weakref.WeakKeyDictionary()
        )

    def check_spec(self, spec: HarnessSpec) -> None:
        """Refuse, before any episode, a harness this runner cannot run or whose toolset the operator forbids."""
        if spec.agent != AGENT:
            raise ContractError(f"the OpenHands runner runs agent {AGENT!r}, not {spec.agent!r}")
        problems = self._tools.problems(spec)
        if problems:
            raise CampaignRuleError("harness_tools_allowed", "; ".join(problems))

    def _policy(self, policy: PolicyRef) -> dict[str, object]:
        endpoint = self._endpoints.resolve(policy)
        if endpoint.runtime != "sglang":
            raise ContractError(f"OpenHands needs an OpenAI-compatible endpoint, not a {endpoint.runtime} one")
        entry: dict[str, object] = {
            "base_url": endpoint.base_url,
            "model": endpoint.served_model,
            "checksum": policy.checkpoint.checksum,
            "api_key_env": None,
        }
        hosted = getattr(self._endpoints, "hosted", None)
        if callable(hosted):
            model = hosted(policy)
            entry["api_key_env"] = model.api_key_env
            if model.input_usd_per_million is not None:
                entry["input_cost_per_token"] = model.input_usd_per_million / 1e6
            if model.output_usd_per_million is not None:
                entry["output_cost_per_token"] = model.output_usd_per_million / 1e6
        return entry

    def _prepare(self, spec: HarnessSpec, task: Task, policy: PolicyRef) -> Path:
        root = self._workspaces / f"{task.task_id}--{spec.spec_id[:12]}--{uuid.uuid4().hex[:8]}"
        harness, work = root / "harness", root / "work"
        work.mkdir(parents=True)
        for path, file in spec.files:
            target = harness / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(file.content, encoding="utf-8")
        for name, mount in self._mounts.items():
            (work / name).symlink_to(mount, target_is_directory=True)
        episode = {
            "task": {"task_id": task.task_id, "prompt": task.prompt},
            "spec_id": spec.spec_id,
            "policy": self._policy(policy),
            "harness": [{"path": path, "kind": file.kind} for path, file in spec.files],
            "limits": self._limits,
            "tools": {"allowed": sorted(self._tools.allowed), "default": list(self._tools.default)},
            "mcp_registry": {name: dict(server) for name, server in self._tools.mcp_registry.items()},
        }
        (root / "episode.json").write_text(json.dumps(episode, indent=2) + "\n", encoding="utf-8")
        return root

    async def run(self, spec: HarnessSpec, task: Task, policy: PolicyRef) -> EpisodeOutcome:
        self.check_spec(spec)
        slots = self._slots.setdefault(asyncio.get_running_loop(), asyncio.Semaphore(self._concurrency))
        async with slots:
            # Prepared inside the slot, so an episode directory's age is how long its agent has run.
            root = self._prepare(spec, task, policy)
            exit_code = await self._start(root)
        result_path = root / "result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        return EpisodeOutcome(
            task_id=task.task_id,
            spec_id=spec.spec_id,
            policy_checksum=policy.checkpoint.checksum,
            exit_code=exit_code,
            final_output=str(result.get("final_output", "")),
            transcript_location=str(root / "transcript.jsonl"),
            workspace_location=str(root),
        )

    async def _start(self, root: Path) -> int:
        with (root / "agent.log").open("wb") as log:
            process = await asyncio.create_subprocess_exec(
                str(self._python),
                str(self._worker),
                str(root),
                cwd=root / "work",
                env={**os.environ, **self._environment, "OPENHANDS_SUPPRESS_BANNER": "1"},
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
                # Its own process group, so a timeout also stops the shells the agent's tools started.
                start_new_session=True,
            )
            try:
                return await asyncio.wait_for(process.wait(), self._timeout)
            except asyncio.TimeoutError:  # the builtin TimeoutError only from Python 3.11
                _kill_group(process.pid)
                await process.wait()
                return TIMED_OUT
            except asyncio.CancelledError:
                # A gate whose other episode failed cancels this one; the agent must not outlive it.
                _kill_group(process.pid)
                await asyncio.shield(process.wait())
                raise


def _kill_group(pid: int) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGKILL)


class OpenHandsDeliveryProbe:
    """Satisfies :class:`~honing.e2e.conformance.DeliveryProbe` from the worker's own report.

    A file counts as delivered when the worker's ``delivery.json`` names a
    route for it (system message, skill, agent settings, MCP, or a workspace
    file) and records the hash of the content the spec holds, so a file that
    was written but never loaded into the agent does not count.
    """

    def delivered(self, outcome: EpisodeOutcome, spec: HarnessSpec) -> dict[str, bool]:
        report_path = None if outcome.workspace_location is None else Path(outcome.workspace_location)
        files: Mapping[str, Mapping[str, object]] = {}
        if report_path is not None and (report_path / "delivery.json").exists():
            files = json.loads((report_path / "delivery.json").read_text()).get("files", {})
        result = {}
        for path, file in spec.files:
            entry = files.get(path, {})
            result[path] = bool(entry.get("route")) and entry.get("sha256") == _sha256(file.content)
        return result


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def default_python(root: Path) -> Path:
    """The OpenHands environment's interpreter, as ``integrations/openhands/README.md`` sets it up."""
    scripts = "Scripts" if sys.platform == "win32" else "bin"
    return root / "integrations" / "openhands" / ".venv" / scripts / "python"


__all__ = [
    "AGENT",
    "AGENT_SETTINGS",
    "DEFAULT_TOOLS",
    "MCP_SETTINGS",
    "OpenHandsDeliveryProbe",
    "OpenHandsEpisodeRunner",
    "ToolPolicy",
    "default_python",
]
