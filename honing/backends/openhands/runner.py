"""Run one OpenHands conversation per Honing episode, in OpenHands' own environment.

OpenHands needs Python 3.12 and its own litellm pin, so this module never
imports it: it writes an episode directory and starts
``integrations/openhands/worker.py`` with the OpenHands environment's Python,
on the host or, with a :class:`Container`, in a Docker container that mounts
only that directory. The two share only that directory.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import shutil
import signal
import sys
import uuid
import weakref
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from honing.backends.llm_gateway import GatewayEndpoints
from honing.backends.openhands.harness import AGENT, AgentPlan, ToolPolicy, plan_agent
from honing.core import (
    CampaignRuleError,
    ContractError,
    EpisodeOutcome,
    HarnessSpec,
    HoningError,
    PolicyEndpoints,
    PolicyRef,
    RunnerSupport,
    Task,
    require,
)

TIMED_OUT = -9
REFUSED = 2
# The worker's exit code when it could not run the episode at all (integrations/openhands/worker.py).
WORKER_FAILED = 1
# Where the worker script is mounted in a container.
CONTAINER_WORKER = "/opt/honing/worker.py"

MountMode = Literal["copy", "link"]

# What the worker inherits from Honing's environment. Everything else, such as HF_TOKEN or cloud credentials,
# stays out of reach of the agent's terminal.
_INHERITED = frozenset({"PATH", "HOME", "LANG", "TMPDIR", "TERM"})
_INHERITED_PREFIXES = ("LC_", "OPENHANDS_")
# Endpoints that speak OpenAI-compatible chat: an SGLang server, or any other provider, router, or server.
_CHAT_RUNTIMES = frozenset({"sglang", "openai_chat"})
# ``docker run``'s exit codes when the container itself failed: the daemon or ``run`` (125), or the command could not
# be invoked (126) or found (127) in the image.
_DOCKER_FAILURES = frozenset({125, 126, 127})
# What the docker CLI needs to reach the daemon. None of it enters the container.
_DOCKER_CLIENT = frozenset({"PATH", "HOME", "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG", "DOCKER_CERT_PATH"})
# Worker variables that describe the host, so a container does not get them.
_HOST_ONLY = frozenset({"PATH", "HOME", "TMPDIR"})
# The wait before the n-th retry of a container that did not start is n times this.
_START_RETRY_SECONDS = 5.0


class EpisodeInfrastructureError(HoningError):
    """The episode could not run: its container did not start, or the worker crashed.

    Raised rather than scored 0, as a grader command that cannot start is: a
    failure of the machine, the image, or the model's API would otherwise be
    charged to the harness.
    """


@dataclass(frozen=True)
class Container:
    """Runs each episode's worker in a Docker container instead of on the host.

    Only the episode directory (at its host path) and the worker script
    (read-only) are mounted, so the agent's terminal cannot reach anything else
    on the host, such as a task suite's hidden grader files. The container runs
    as the host user, with ``HOME`` in the episode directory, and gets the
    worker's variables except the host's ``PATH``, ``HOME``, and ``TMPDIR``.
    A policy served on the host's loopback is not reachable from the default
    network as ``127.0.0.1``: address it as ``host.docker.internal`` on Docker
    Desktop, or pass ``--network host`` on Linux.

    Attributes:
        image: An image with the OpenHands environment
            (``integrations/openhands/Dockerfile``) and what the tasks need.
        python: The OpenHands environment's interpreter in the image.
        run_args: Further ``docker run`` arguments.
        start_attempts: How many times an episode tries to start its
            container; Docker Desktop occasionally refuses a mount under load.
    """

    image: str
    python: str = "python"
    run_args: tuple[str, ...] = ()
    start_attempts: int = 3

    def __post_init__(self) -> None:
        require(bool(self.image), "a container needs an image")
        require(self.start_attempts > 0, f"start_attempts must be positive, got {self.start_attempts}")


class OpenHandsEpisodeRunner:
    """Satisfies :class:`~honing.core.EpisodeRunner` with OpenHands agents.

    Each episode gets ``<workspaces>/<task>--<spec>--<id>/`` holding
    ``episode.json`` (task, policy endpoint, the resolved :class:`AgentPlan`,
    limits), ``harness/`` (the spec's files), and ``work/`` (the agent's working
    directory). The worker adds ``result.json``, ``transcript.jsonl`` (every
    OpenHands event), ``delivery.json``, and ``openhands/`` (OpenHands' own store).

    An episode whose worker crashes (exit code 1), or whose container does not
    start, raises :class:`EpisodeInfrastructureError` instead of returning an
    outcome to be graded. A harness OpenHands cannot load, such as a skill
    whose frontmatter does not parse, is the harness's failure, not the
    machine's: the worker refuses it with exit code 2, and it is graded.

    Args:
        python: The OpenHands environment's interpreter.
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
            large to copy; the links are shared and not write-protected, and a
            container cannot follow them.
        seed_workspace: A task's starting files, a directory copied into
            ``work/`` before the mounts, or ``None``; for a task suite, the
            manifest's ``workspace`` (:mod:`honing.harness.tasks`). It may
            hold no links.
        container: Runs the worker in a Docker container; see :class:`Container`.
        environment: Extra environment variables for the worker. The worker
            sees only these, ``PATH``, ``HOME``, ``LANG``, ``TMPDIR``, ``TERM``,
            ``LC_*``, ``OPENHANDS_*``, the policy's API key, and the variables
            ``inherit_environment`` names.
        inherit_environment: Names of further variables the worker takes from
            Honing's environment, such as ``HTTPS_PROXY``, ``NO_PROXY``, or
            ``SSL_CERT_FILE`` on a network that needs a proxy or its own CA
            bundle. Only the names are configured; the values stay in the
            environment.
        concurrency: How many agents run at once.
        timeout_seconds: An agent still running after this is killed with its
            child processes, and the outcome's exit code is -9.
        max_iterations: The step limit; a harness may lower it.
        max_budget_usd: Stops a conversation once it has cost this much.
        llm_timeout_seconds: The limit on one model call.

    Raises:
        ContractError: A limit is not positive, a mount is not a directory,
            ``mount_mode`` is neither ``copy`` nor ``link`` (or is ``link`` with
            a container), or an inherited name is not a variable name.
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
        mount_mode: MountMode = "copy",
        seed_workspace: Callable[[Task], Path | None] | None = None,
        container: Container | None = None,
        environment: Mapping[str, str] | None = None,
        inherit_environment: Iterable[str] = (),
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
        require(mount_mode in ("copy", "link"), f"mount_mode must be 'copy' or 'link', got {mount_mode!r}")
        require(container is None or mount_mode == "copy", "a container cannot follow linked mounts; copy them")
        self._mount_mode = mount_mode
        self._seed_workspace = seed_workspace
        self._container = container
        self._environment = dict(environment or {})
        self._inherit = frozenset(inherit_environment)
        for name in self._inherit:
            require(bool(name) and "=" not in name, f"inherit_environment names variables, got {name!r}")
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
            ``workspace_location`` the episode directory. A harness that
            :meth:`check_spec` refuses starts no agent: the outcome's exit code
            is 2 and ``final_output`` gives the reason, so the gate scores and
            rejects it like any failed episode.

        Raises:
            ContractError: The harness is for another agent, the endpoint does
                not speak OpenAI-compatible chat, the API key is not set, or the
                task's starting files hold a link.
            EpisodeInfrastructureError: The worker crashed, or the container
                did not start.
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
            exit_code = await self._start(root, self._worker_environment(access.get("api_key_env")))
        result_path = root / "result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        if exit_code == WORKER_FAILED:
            reason = result.get("error") or f"status {result.get('status', 'unknown')}"
            raise EpisodeInfrastructureError(
                f"the worker could not run task {task.task_id!r} ({reason}); see {root / 'agent.log'}"
            )
        return EpisodeOutcome(
            task_id=task.task_id,
            spec_id=spec.spec_id,
            policy_checksum=policy.checkpoint.checksum,
            exit_code=exit_code,
            final_output=str(result.get("final_output", "")),
            transcript_location=str(root / "transcript.jsonl"),
            workspace_location=str(root),
        )

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

    def _worker_environment(self, api_key_env: object) -> dict[str, str]:
        inherited = {
            name: value
            for name, value in os.environ.items()
            if name in _INHERITED
            or name.startswith(_INHERITED_PREFIXES)
            or name == api_key_env
            or name in self._inherit
        }
        return {**inherited, **self._environment, "OPENHANDS_SUPPRESS_BANNER": "1"}

    def _access(self, policy: PolicyRef) -> dict[str, object]:
        endpoint = self._endpoints.resolve(policy)
        if endpoint.runtime not in _CHAT_RUNTIMES:
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
        seed = self._seed_workspace(task) if self._seed_workspace is not None else None
        if seed is not None:
            _refuse_links(task, seed)
            shutil.copytree(seed, work, dirs_exist_ok=True)
        if self._container is not None:
            (root / "home").mkdir()
        for path, file in spec.files:
            target = root / "harness" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(file.content, encoding="utf-8")
        for name, mount in self._mounts.items():
            if self._mount_mode == "copy":
                shutil.copytree(mount, work / name, ignore_dangling_symlinks=True)
            else:
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

    async def _start(self, root: Path, environment: Mapping[str, str]) -> int:
        container = self._container
        if container is None:
            command = [str(self._python), str(self._worker), str(root)]
            return await self._run_process(command, root / "work", environment, root / "agent.log", None)
        client = {name: value for name, value in os.environ.items() if name in _DOCKER_CLIENT}
        exit_code = 0
        for attempt in range(1, container.start_attempts + 1):
            name = f"honing-{uuid.uuid4().hex[:12]}"
            command = _docker_command(container, root.resolve(), self._worker.resolve(), name, environment)
            exit_code = await self._run_process(command, root, {**client, **environment}, root / "agent.log", name)
            if exit_code not in _DOCKER_FAILURES:
                return exit_code
            if attempt < container.start_attempts:
                await asyncio.sleep(_START_RETRY_SECONDS * attempt)
        raise EpisodeInfrastructureError(
            f"the container for {root.name} did not start in {container.start_attempts} attempts"
            f" (docker run exited {exit_code}); see {root / 'agent.log'}"
        )

    async def _run_process(
        self, command: Sequence[str], cwd: Path, environment: Mapping[str, str], log_path: Path, container: str | None
    ) -> int:
        # Appended to, so a container that is retried keeps the log of each attempt.
        with log_path.open("ab") as log:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=cwd,
                env=dict(environment),
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
                await _kill_container(container, environment)
                return TIMED_OUT
            except asyncio.CancelledError:
                _kill_group(process.pid)
                await asyncio.shield(process.wait())
                await asyncio.shield(_kill_container(container, environment))
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
    # macOS refuses (EPERM) to signal a group whose only member is a zombie, as when the worker has just exited.
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGKILL)


async def _kill_container(name: str | None, environment: Mapping[str, str]) -> None:
    """Stop a container whose ``docker run`` was killed: killing the client does not stop the container."""
    if name is None:
        return
    with contextlib.suppress(OSError):
        process = await asyncio.create_subprocess_exec(
            "docker",
            "kill",
            name,
            env=dict(environment),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await process.wait()


def _docker_command(
    container: Container, root: Path, worker: Path, name: str, environment: Mapping[str, str]
) -> list[str]:
    """``docker run`` for one episode. Variables are passed by name, so their values never reach the command line."""
    command = [
        "docker",
        "run",
        "--rm",
        "--init",
        "--name",
        name,
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--volume",
        f"{root}:{root}",
        "--volume",
        f"{worker}:{CONTAINER_WORKER}:ro",
        "--workdir",
        str(root / "work"),
        "--env",
        f"HOME={root / 'home'}",
    ]
    for variable in sorted(set(environment) - _HOST_ONLY):
        command.extend(["--env", variable])
    return [*command, *container.run_args, container.image, container.python, CONTAINER_WORKER, str(root)]


def _refuse_links(task: Task, seed: Path) -> None:
    """A link in a task's starting files would be followed out of the workspace, perhaps to hidden grader files."""
    for path in (seed, *seed.rglob("*")):
        if path.is_symlink():
            raise ContractError(f"task {task.task_id!r} starts from {seed}, which holds the link {path}")
