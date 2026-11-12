"""Run one pi session per Honing episode, in pi's own Node.js process.

pi is a Node.js program, so this module never imports it: it writes an episode
directory and starts ``integrations/pi/worker.mjs``, which runs pi through its
SDK. The two share only that directory.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import shutil
import signal
import uuid
import weakref
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Literal

from honing.backends.llm_gateway import GatewayEndpoints
from honing.backends.pi.harness import AGENT, AgentPlan, ToolPolicy, plan_agent
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

TIMED_OUT = -9
# How long a worker asked to stop may take to abort its session, which stops the agent's shell commands.
STOP_GRACE_SECONDS = 10.0
REFUSED = 2

MountMode = Literal["copy", "link"]

# What the worker inherits from Honing's environment. Everything else, such as HF_TOKEN or cloud credentials,
# stays out of reach of the agent's shell. PI_* is not inherited: the runner sets pi's own variables.
_INHERITED = frozenset({"PATH", "HOME", "LANG", "TMPDIR", "TERM"})
_INHERITED_PREFIXES = ("LC_",)
# Endpoints that speak OpenAI-compatible chat: an SGLang server, or any other provider, router, or server.
_CHAT_RUNTIMES = frozenset({"sglang", "openai_chat"})
# The one provider an episode's models.json defines.
PROVIDER = "honing"


class PiEpisodeRunner:
    """Satisfies :class:`~honing.core.EpisodeRunner` with pi agents.

    Each episode gets ``<workspaces>/<task>--<spec>--<id>/`` holding
    ``episode.json`` (task, model, the resolved :class:`AgentPlan`),
    ``agent/models.json`` (pi's agent directory, with one provider for the
    policy's endpoint), ``harness/`` (the spec's files), and ``work/`` (the
    agent's working directory). The worker adds ``result.json``,
    ``transcript.jsonl`` (every pi session event), and ``delivery.json``.

    Args:
        node: The Node.js executable, 22.19 or newer.
        worker: The worker script.
        workspaces: Where episode directories are created.
        endpoints: Resolves a policy to its OpenAI-compatible endpoint. With
            :class:`~honing.backends.llm_gateway.GatewayEndpoints`, the worker also
            gets the model's API key (through the environment, never on disk)
            and prices.
        tools: What a harness may give the agent.
        mounts: Directories put into every working directory, by name.
        mount_mode: ``copy``, the default, gives each episode its own copy of
            every mount, so an agent that edits one cannot change what another
            episode reads. ``link`` symlinks them instead, for a directory too
            large to copy; the links are shared and not write-protected.
        environment: Extra environment variables for the worker. The worker
            sees only these, ``PATH``, ``HOME``, ``LANG``, ``TMPDIR``, ``TERM``,
            ``LC_*``, pi's own settings, the policy's API key, and the variables
            ``inherit_environment`` names.
        inherit_environment: Names of further variables the worker takes from
            Honing's environment, such as ``HTTPS_PROXY`` or ``SSL_CERT_FILE``.
            Only the names are configured; the values stay in the environment.
        model_settings: Fields merged into the model's ``models.json`` entry,
            such as ``contextWindow`` and ``maxTokens`` for a local model with a
            short context.
        concurrency: How many agents run at once.
        timeout_seconds: An agent still running after this is stopped, and the
            outcome's exit code is -9. The worker gets SIGTERM and aborts its
            session, which kills the agent's running shell commands; after
            ``STOP_GRACE_SECONDS`` its process group gets SIGKILL. pi has no
            step limit of its own, so this is the only one.

    Raises:
        ContractError: A limit is not positive, a mount is not a directory,
            ``mount_mode`` is neither ``copy`` nor ``link``, or an inherited name
            is not a variable name.
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
        node: Path,
        worker: Path,
        workspaces: Path,
        endpoints: PolicyEndpoints,
        tools: ToolPolicy | None = None,
        mounts: Mapping[str, Path] | None = None,
        mount_mode: MountMode = "copy",
        environment: Mapping[str, str] | None = None,
        inherit_environment: Iterable[str] = (),
        model_settings: Mapping[str, object] | None = None,
        concurrency: int = 1,
        timeout_seconds: float = 900.0,
    ) -> None:
        require(concurrency > 0 and timeout_seconds > 0, "concurrency and timeout_seconds must be positive")
        self._node = node
        # Resolved here: the worker starts in the episode's working directory.
        self._worker = worker.resolve()
        self._workspaces = workspaces
        self._endpoints = endpoints
        self._tools = tools or ToolPolicy()
        self._mounts = {name: Path(path).expanduser().resolve() for name, path in (mounts or {}).items()}
        for name, path in self._mounts.items():
            require("/" not in name and name not in {"", ".", ".."}, f"a mount name is one path segment, got {name!r}")
            require(path.is_dir(), f"mount {name!r} points at {path}, which is not a directory")
        require(mount_mode in ("copy", "link"), f"mount_mode must be 'copy' or 'link', got {mount_mode!r}")
        self._mount_mode = mount_mode
        self._environment = dict(environment or {})
        self._inherit = frozenset(inherit_environment)
        for name in self._inherit:
            require(bool(name) and "=" not in name, f"inherit_environment names variables, got {name!r}")
        self._model_settings = dict(model_settings or {})
        self._concurrency = concurrency
        self._timeout = timeout_seconds
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
                or is malformed (see :func:`~honing.backends.pi.harness.plan_agent`).
        """
        return plan_agent(spec, self._tools, reasoning=self._model_settings.get("reasoning") is True)

    async def run(self, spec: HarnessSpec, task: Task, policy: PolicyRef) -> EpisodeOutcome:
        """Runs one pi session on ``task`` under ``spec`` against ``policy``.

        Args:
            spec: The harness.
            task: The task; its prompt is the session's first message.
            policy: The weights to run against.

        Returns:
            The outcome, with ``final_output`` the agent's last message and
            ``workspace_location`` the episode directory. An episode that did not
            finish (it timed out, pi aborted it, or the model's provider kept
            failing) has a nonzero exit code, and ``final_output`` starts with
            the reason, so the grader scores it as a failed episode and its
            feedback says why. A harness that :meth:`check_spec` refuses starts
            no agent: the exit code is 2 and ``final_output`` gives the reason.

        Raises:
            ContractError: The harness is for another agent, the endpoint does
                not speak OpenAI-compatible chat, or the API key is not set.
        """
        try:
            plan = self.check_spec(spec)
        except CampaignRuleError as refusal:
            return self._refused(spec, task, policy, refusal)
        access = self._access(policy)
        slots = self._slots.setdefault(asyncio.get_running_loop(), asyncio.Semaphore(self._concurrency))
        async with slots:
            # Off the event loop: copying a mount can take a while, and other episodes are running.
            root = await asyncio.to_thread(self._prepare, spec, task, plan, access)
            exit_code = await self._start(root, self._worker_environment(root, access.get("api_key_env")))
        result_path = root / "result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        output = str(result.get("final_output", ""))
        if exit_code != 0:
            output = "\n\n".join(part for part in (self._failure(exit_code, result), output) if part)
        return EpisodeOutcome(
            task_id=task.task_id,
            spec_id=spec.spec_id,
            policy_checksum=policy.checkpoint.checksum,
            exit_code=exit_code,
            final_output=output,
            transcript_location=str(root / "transcript.jsonl"),
            workspace_location=str(root),
        )

    def _failure(self, exit_code: int, result: Mapping[str, object]) -> str:
        if exit_code == TIMED_OUT:
            return f"The agent did not finish within {self._timeout:g} seconds and was stopped."
        status, error = result.get("status", "failed"), result.get("error")
        return f"The agent stopped before finishing ({status}): {error}" if error else f"The agent stopped ({status})."

    def _refused(self, spec: HarnessSpec, task: Task, policy: PolicyRef, refusal: CampaignRuleError) -> EpisodeOutcome:
        root = self._workspaces / f"{task.task_id}--{spec.spec_id[:12]}--{uuid.uuid4().hex[:8]}"
        root.mkdir(parents=True)
        reason = f"The harness was refused before the agent started: {refusal}"
        result = {"final_output": reason, "status": "refused", "error": str(refusal)}
        (root / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return EpisodeOutcome(
            task_id=task.task_id,
            spec_id=spec.spec_id,
            policy_checksum=policy.checkpoint.checksum,
            exit_code=REFUSED,
            final_output=reason,
            workspace_location=str(root),
        )

    def _worker_environment(self, root: Path, api_key_env: object) -> dict[str, str]:
        inherited = {
            name: value
            for name, value in os.environ.items()
            if name in _INHERITED
            or name.startswith(_INHERITED_PREFIXES)
            or name == api_key_env
            or name in self._inherit
        }
        pi = {
            "PI_CODING_AGENT_DIR": str(root / "agent"),
            "PI_OFFLINE": "1",
            "PI_SKIP_VERSION_CHECK": "1",
            "PI_TELEMETRY": "0",
        }
        return {**inherited, **self._environment, **pi}

    def _access(self, policy: PolicyRef) -> dict[str, object]:
        endpoint = self._endpoints.resolve(policy)
        if endpoint.runtime not in _CHAT_RUNTIMES:
            raise ContractError(f"pi needs an OpenAI-compatible endpoint, not a {endpoint.runtime} one")
        access: dict[str, object] = {"base_url": endpoint.base_url, "model": endpoint.served_model}
        if isinstance(self._endpoints, GatewayEndpoints):
            model = self._endpoints.model(policy)
            key = model.api_key_env
            if key is not None and not (self._environment.get(key) or os.environ.get(key)):
                raise ContractError(f"{model.model} needs its API key in the environment variable {key}")
            access["api_key_env"] = key
            if model.input_usd_per_million is not None and model.output_usd_per_million is not None:
                # pi prices cached input separately; at the full input price, a cost is never understated.
                price = model.input_usd_per_million
                access["cost"] = {
                    "input": price,
                    "output": model.output_usd_per_million,
                    "cacheRead": price,
                    "cacheWrite": price,
                }
        return access

    def _models(self, access: Mapping[str, object]) -> dict[str, object]:
        model: dict[str, object] = {"id": access["model"], **self._model_settings}
        if "cost" in access:
            model["cost"] = access["cost"]
        # "$NAME" makes pi read the key from the environment, so models.json never holds it.
        key = access.get("api_key_env")
        provider = {
            "baseUrl": access["base_url"],
            "api": "openai-completions",
            "apiKey": f"${key}" if key else "unused",
            "models": [model],
        }
        return {"providers": {PROVIDER: provider}}

    def _prepare(self, spec: HarnessSpec, task: Task, plan: AgentPlan, access: Mapping[str, object]) -> Path:
        root = self._workspaces / f"{task.task_id}--{spec.spec_id[:12]}--{uuid.uuid4().hex[:8]}"
        work = root / "work"
        work.mkdir(parents=True)
        (root / "agent").mkdir()
        for path, file in spec.files:
            target = root / "harness" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(file.content, encoding="utf-8")
        for name, mount in self._mounts.items():
            if self._mount_mode == "copy":
                shutil.copytree(mount, work / name, ignore_dangling_symlinks=True)
            else:
                (work / name).symlink_to(mount, target_is_directory=True)
        models = json.dumps(self._models(access), indent=2) + "\n"
        (root / "agent" / "models.json").write_text(models, encoding="utf-8")
        episode = {
            "task": {"task_id": task.task_id, "prompt": task.prompt},
            "spec_id": spec.spec_id,
            "model": {"provider": PROVIDER, "id": access["model"], "api_key_env": access.get("api_key_env")},
            "agent": plan.to_json(),
        }
        (root / "episode.json").write_text(json.dumps(episode, indent=2) + "\n", encoding="utf-8")
        return root

    async def _start(self, root: Path, environment: Mapping[str, str]) -> int:
        with (root / "agent.log").open("wb") as log:
            process = await asyncio.create_subprocess_exec(
                str(self._node),
                str(self._worker),
                str(root),
                cwd=root / "work",
                env=dict(environment),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
                # Its own process group, so a worker that does not stop in time is killed with what it started.
                start_new_session=True,
            )
            try:
                return await asyncio.wait_for(process.wait(), self._timeout)
            except asyncio.TimeoutError:  # the builtin TimeoutError only from Python 3.11
                await _stop(process)
                return TIMED_OUT
            except asyncio.CancelledError:
                await asyncio.shield(_stop(process))
                raise


class PiDeliveryProbe:
    """Satisfies :class:`~honing.e2e.conformance.DeliveryProbe` from the worker's report.

    The worker reports a file only after checking that it reached pi: an
    instruction file's text is in pi's system prompt, a skill is listed there,
    and the settings' tools are the session's. A file counts as delivered only
    when the reported hash matches the spec's content.
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


async def _stop(process: asyncio.subprocess.Process) -> None:
    """Asks the worker to stop, then kills its process group if it has not within the grace period.

    pi starts each shell command in a process group of its own, out of reach of
    a signal to the worker's group; only the worker aborting its session stops them.
    """
    _signal_group(process.pid, signal.SIGTERM)
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(process.wait(), STOP_GRACE_SECONDS)
    _signal_group(process.pid, signal.SIGKILL)
    await process.wait()


def _signal_group(pid: int, number: signal.Signals) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, number)
