"""Run one OpenHands conversation per Honing episode, in OpenHands' own environment.

OpenHands needs Python 3.12 and its own litellm pin, so this module never
imports it: it writes an episode directory and starts
``integrations/openhands/worker.py`` with the OpenHands environment's Python.
The two share only that directory.
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
from pathlib import Path

from honing.backends.llm_gateway import GatewayEndpoints
from honing.backends.openhands.harness import AGENT, AgentPlan, ToolPolicy, plan_agent
from honing.core import (
    ContractError,
    EpisodeOutcome,
    HarnessSpec,
    PolicyEndpoints,
    PolicyRef,
    RunnerSupport,
    Task,
    require,
)

TIMED_OUT = -9


class OpenHandsEpisodeRunner:
    """Satisfies :class:`~honing.core.EpisodeRunner` with OpenHands agents.

    Each episode gets ``<workspaces>/<task>--<spec>--<id>/`` holding
    ``episode.json`` (task, policy endpoint, the resolved :class:`AgentPlan`,
    limits), ``harness/`` (the spec's files), and ``work/`` (the agent's working
    directory). The worker adds ``result.json``, ``transcript.jsonl`` (every
    OpenHands event), ``delivery.json``, and ``openhands/`` (OpenHands' own store).

    Args:
        python: The OpenHands environment's interpreter.
        worker: The worker script.
        workspaces: Where episode directories are created.
        endpoints: Resolves a policy to its OpenAI-compatible endpoint. With
            :class:`~honing.backends.llm_gateway.GatewayEndpoints`, the worker also
            gets the model's API key (through the environment, never on disk)
            and prices.
        tools: What a harness may give the agent.
        mounts: Directories linked into every working directory, by name. The
            links are not write-protected.
        environment: Extra environment variables for the worker.
        concurrency: How many agents run at once.
        timeout_seconds: An agent still running after this is killed with its
            child processes, and the outcome's exit code is -9.
        max_iterations: The step limit; a harness may lower it.
        max_budget_usd: Stops a conversation once it has cost this much.
        llm_timeout_seconds: The limit on one model call.

    Raises:
        ContractError: A limit is not positive, or a mount is not a directory.
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
        require(
            concurrency > 0 and timeout_seconds > 0 and max_iterations > 0 and llm_timeout_seconds > 0,
            "concurrency, timeout_seconds, max_iterations, and llm_timeout_seconds must be positive",
        )
        require(max_budget_usd is None or max_budget_usd > 0, "max_budget_usd must be positive when set")
        self._python = python
        self._worker = worker
        self._workspaces = workspaces
        self._endpoints = endpoints
        self._tools = tools or ToolPolicy()
        self._mounts = {name: Path(path).expanduser().resolve() for name, path in (mounts or {}).items()}
        for name, path in self._mounts.items():
            require("/" not in name and name not in {"", ".", ".."}, f"a mount name is one path segment, got {name!r}")
            require(path.is_dir(), f"mount {name!r} points at {path}, which is not a directory")
        self._environment = dict(environment or {})
        self._concurrency = concurrency
        self._timeout = timeout_seconds
        self._max_iterations = max_iterations
        self._limits = {"max_budget_usd": max_budget_usd, "llm_timeout_seconds": llm_timeout_seconds}
        # One semaphore per event loop: a campaign runs each gate in its own asyncio.run.
        self._slots: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
            weakref.WeakKeyDictionary()
        )

    def check_spec(self, spec: HarnessSpec) -> AgentPlan:
        """Resolves a harness against the runner's tool policy, before any episode runs.

        Args:
            spec: The harness.

        Returns:
            The plan the worker would build.

        Raises:
            ContractError: The harness is for another agent.
            CampaignRuleError: The harness asks for something the policy forbids,
                or is malformed (see :func:`~honing.backends.openhands.harness.plan_agent`).
        """
        return plan_agent(spec, self._tools, self._max_iterations)

    async def run(self, spec: HarnessSpec, task: Task, policy: PolicyRef) -> EpisodeOutcome:
        """Runs one OpenHands conversation on ``task`` under ``spec`` against ``policy``.

        Args:
            spec: The harness.
            task: The task; its prompt is the conversation's first message.
            policy: The weights to run against.

        Returns:
            The outcome, with ``final_output`` the agent's last message and
            ``workspace_location`` the episode directory.

        Raises:
            ContractError: The harness is for another agent, the endpoint does
                not speak OpenAI-compatible chat, or the API key is not set.
            CampaignRuleError: As :meth:`check_spec`.
        """
        plan = self.check_spec(spec)
        access = self._access(policy)
        slots = self._slots.setdefault(asyncio.get_running_loop(), asyncio.Semaphore(self._concurrency))
        async with slots:
            root = self._prepare(spec, task, plan, access)
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

    def _access(self, policy: PolicyRef) -> dict[str, object]:
        endpoint = self._endpoints.resolve(policy)
        if endpoint.runtime != "sglang":
            raise ContractError(f"OpenHands needs an OpenAI-compatible endpoint, not a {endpoint.runtime} one")
        access: dict[str, object] = {"base_url": endpoint.base_url, "model": endpoint.served_model}
        if isinstance(self._endpoints, GatewayEndpoints):
            model = self._endpoints.model(policy)
            key = model.api_key_env
            if key is not None and not (self._environment.get(key) or os.environ.get(key)):
                raise ContractError(f"{model.model} needs its API key in the environment variable {key}")
            access["api_key_env"] = key
            if model.input_usd_per_million is not None:
                access["input_cost_per_token"] = model.input_usd_per_million / 1e6
            if model.output_usd_per_million is not None:
                access["output_cost_per_token"] = model.output_usd_per_million / 1e6
        return access

    def _prepare(self, spec: HarnessSpec, task: Task, plan: AgentPlan, access: Mapping[str, object]) -> Path:
        root = self._workspaces / f"{task.task_id}--{spec.spec_id[:12]}--{uuid.uuid4().hex[:8]}"
        work = root / "work"
        work.mkdir(parents=True)
        for path, file in spec.files:
            target = root / "harness" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(file.content, encoding="utf-8")
        for name, mount in self._mounts.items():
            (work / name).symlink_to(mount, target_is_directory=True)
        episode = {
            "task": {"task_id": task.task_id, "prompt": task.prompt},
            "spec_id": spec.spec_id,
            "policy": dict(access),
            "agent": plan.to_json(),
            "limits": self._limits,
        }
        (root / "episode.json").write_text(json.dumps(episode, indent=2) + "\n", encoding="utf-8")
        return root

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
                # Its own process group, so stopping it also stops the shells the agent's tools started.
                start_new_session=True,
            )
            try:
                return await asyncio.wait_for(process.wait(), self._timeout)
            except asyncio.TimeoutError:  # the builtin TimeoutError only from Python 3.11
                _kill_group(process.pid)
                await process.wait()
                return TIMED_OUT
            except asyncio.CancelledError:
                _kill_group(process.pid)
                await asyncio.shield(process.wait())
                raise


class OpenHandsDeliveryProbe:
    """Satisfies :class:`~honing.e2e.conformance.DeliveryProbe` from the worker's report.

    A file counts as delivered only when the worker reports loading it with the
    content the spec holds, so a file written to disk but never loaded does not count.
    """

    def delivered(self, outcome: EpisodeOutcome, spec: HarnessSpec) -> dict[str, bool]:
        """Tells which of the spec's files reached the agent.

        Args:
            outcome: The episode's outcome.
            spec: The harness the episode ran.

        Returns:
            Whether each of the spec's paths was delivered.
        """
        report = None if outcome.workspace_location is None else Path(outcome.workspace_location) / "delivery.json"
        loaded: Mapping[str, Mapping[str, object]] = {}
        if report is not None and report.exists():
            loaded = json.loads(report.read_text()).get("files", {})
        return {
            path: loaded.get(path, {}).get("sha256") == hashlib.sha256(file.content.encode()).hexdigest()
            for path, file in spec.files
        }


def default_python(root: Path) -> Path:
    """Returns the OpenHands environment's interpreter, as ``integrations/openhands/README.md`` creates it.

    Args:
        root: The repository root.
    """
    return root / "integrations" / "openhands" / ".venv" / ("Scripts" if sys.platform == "win32" else "bin") / "python"


def _kill_group(pid: int) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGKILL)
