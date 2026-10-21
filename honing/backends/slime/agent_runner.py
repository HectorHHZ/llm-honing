"""Claude Code episodes on slime's agent layer, each run as a worker in Slime's environment.

:class:`SlimeEpisodeRunner` satisfies :class:`~honing.core.EpisodeRunner` for
``claude_code``. Per episode it resolves the policy to an SGLang server, copies the
task's workspace from the suite, places the harness files where Claude Code reads
them (:func:`place_harness`), and runs :mod:`~honing.backends.slime.agent_worker` in
Slime's environment (:class:`~honing.backends.slime.SlimeEnvironment`), because
slime's adapter and harness import torch and Honing does not. The worker serves
slime's ``AnthropicAdapter``, runs slime's ``ClaudeCodeHarness`` in a
:class:`~honing.backends.slime.LocalContainerSandbox`, and copies the workspace and
the transcript back. The outcome's ``workspace_location`` is that copy, for a
:class:`~honing.core.TaskGrader` to grade.

What the agent does is the episode's outcome, whatever it is: a nonzero exit, or
``-1`` when it ran out of its time budget, with the workspace it left; a prompt that
outgrew the model's context ends the agent's call as if it hit its length limit. A
workspace the agent left that cannot come back as it is (larger than the cap, or with
paths that differ only in case or are not valid UTF-8) is the outcome's ``failure``,
which graders score 0.
What stops the episode from saying anything about the harness raises
:class:`AgentWorkerError`: the
policy server failing a call, Docker failing, the CLI not installing, or the worker
crashing or running past its own deadline.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import re
import shutil
import signal
import uuid
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from honing.backends.slime import agent_episode as files
from honing.backends.slime.sandbox import ContainerLimits
from honing.backends.slime.trainer import SlimeEnvironment
from honing.core import (
    ContractError,
    EpisodeOutcome,
    HarnessSpec,
    HoningError,
    PolicyEndpoint,
    PolicyEndpoints,
    PolicyRef,
    RunnerSupport,
    Task,
)
from honing.harness.tasks import TaskSuite

AGENT = "claude_code"
SUPPORT = RunnerSupport(
    agents=frozenset({AGENT}),
    file_kinds=frozenset({"instructions", "skill", "command", "settings"}),
    programmable_loop=False,
    local_policy_only=True,
)

# What the worker inherits from this process, by name, besides what the runner sets.
# Docker's variables reach the daemon (colima's socket, a config without a credential
# helper), the tarball paths are where slime's harness installs the CLI from, and the
# Hugging Face ones say where tokenizers are cached and whether to go online.
PASSTHROUGH = (
    "PATH",
    "HOME",
    "TMPDIR",
    "LANG",
    "DOCKER_HOST",
    "DOCKER_CONFIG",
    "DOCKER_CONTEXT",
    "SLIME_AGENT_NODE_TARBALL",
    "SLIME_AGENT_CC_TARBALL",
    "HF_HOME",
    "HF_HUB_OFFLINE",
)
WORKER_MODULE = "honing.backends.slime.agent_worker"
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_REMOVE_SECONDS = 60.0
_INSPECT_SECONDS = 60.0


class AgentWorkerError(HoningError):
    """An episode's worker could not run it, for a reason that is not the agent's. The message names its log."""


@dataclass(frozen=True)
class AgentEpisodeSettings:
    """How each episode runs. Every field reaches the worker through ``episode.json``.

    ``time_budget_seconds`` is the agent's; slime stops waiting for it then. Setting up
    the sandbox (creating and starting the container, copying the workspace in,
    installing the CLI) has its own limit. The image must already be on the daemon:
    the runner checks before its first episode, and no episode pulls it. The
    whole worker gets both, plus three fixed sums of per-call limits, each exec with
    the time the sandbox allows past its own limit: slime's steps between the setup
    and the agent's clock (``agent_episode.PRE_LAUNCH_SECONDS``: creating the agent
    user, writing the CLI's config, launching it), and the worker's own limits on
    stopping the agent and copying out (``STOP_AGENT_SECONDS`` and
    ``COPY_OUT_SECONDS``). ``worker_margin_seconds`` covers the rest: loading the
    tokenizer, which has no limit, and steps whose limits are not summed here (the
    policy server's preflight, slime's last poll and output tail once the budget runs
    out, removing the sandbox). Past all that the worker is killed. The sandbox must
    live at least that long. ``copy_out_max_bytes`` caps the
    workspace copied back, as a tar; a larger one is the outcome's ``failure``.

    The adapter fields go to slime's ``AnthropicAdapter``: ``tool_parser`` and
    ``reasoning_parser`` are SGLang parser names (they need SGLang importable in the
    worker), ``max_turns_per_sid`` caps the agent's model calls, and
    ``max_new_tokens``, when set, replaces slime's default of 4096 per call.
    ``cli_extra_args`` is appended to the CLI's command line (slime's
    ``SLIME_AGENT_CC_EXTRA_ARGS``). With ``install_cli`` off, the image must already
    have ``/usr/local/bin/claude``. ``adapter_bind_host`` defaults to what the
    sandbox's host alias reaches: ``127.0.0.1`` on a Mac, the bridge gateway on Linux.
    ``max_context_tokens`` stops the agent's call before the prompt overflows the
    model; 0 takes the server's longest input from its ``/server_info``.

    ``network`` must be Docker's default ``bridge``: the sandbox reaches the adapter
    through ``host-gateway``, which Docker resolves to that network's gateway, so on
    another network the CLI would call an address nothing listens on.
    """

    image: str
    time_budget_seconds: int = 1800
    setup_timeout_seconds: int = 600
    worker_margin_seconds: int = 300
    copy_out_max_bytes: int = 64 * 2**20
    limits: ContainerLimits = field(default_factory=ContainerLimits)
    network: str = "bridge"
    host_alias: str = "host.docker.internal"
    sandbox_lifetime_seconds: int = 4 * 3600
    adapter_bind_host: str | None = None
    tool_parser: str | None = None
    reasoning_parser: str | None = None
    max_turns_per_sid: int | None = None
    max_context_tokens: int = 0
    fork_threshold_tokens: int | None = None
    max_new_tokens: int | None = None
    sampling_defaults: Mapping[str, object] = field(default_factory=dict)
    install_cli: bool = True
    cli_extra_args: str = ""

    def __post_init__(self) -> None:
        if not self.image:
            raise ContractError("AgentEpisodeSettings.image must name an image")
        for name in (
            "time_budget_seconds",
            "setup_timeout_seconds",
            "worker_margin_seconds",
            "copy_out_max_bytes",
            "sandbox_lifetime_seconds",
        ):
            if getattr(self, name) <= 0:
                raise ContractError(f"AgentEpisodeSettings.{name} must be positive")
        if self.max_new_tokens is not None and self.max_new_tokens <= 0:
            raise ContractError("AgentEpisodeSettings.max_new_tokens must be positive or None")
        if self.network != "bridge":
            raise ContractError(
                f"Claude Code on slime runs on only Docker's default 'bridge' network, not {self.network!r}:"
                " the sandbox reaches the adapter through host-gateway, which is that network's gateway"
            )
        if self.sandbox_lifetime_seconds < self.worker_timeout_seconds:
            raise ContractError(
                f"the sandbox must outlive the worker's deadline, {self.worker_timeout_seconds}s,"
                f" not {self.sandbox_lifetime_seconds}s"
            )

    @property
    def worker_timeout_seconds(self) -> int:
        fixed = files.PRE_LAUNCH_SECONDS + files.STOP_AGENT_SECONDS + files.COPY_OUT_SECONDS
        return math.ceil(self.setup_timeout_seconds + self.time_budget_seconds + fixed + self.worker_margin_seconds)


@dataclass(frozen=True)
class Placement:
    """One file in the agent's working directory, and the harness paths it carries."""

    target: str
    content: str
    paths: tuple[str, ...]


def _name(path: str, kind: str) -> str:
    posix = PurePosixPath(path)
    name = posix.parent.name if posix.name == "SKILL.md" and posix.parent.name else posix.stem
    if not _NAME.fullmatch(name):
        raise ContractError(f"the {kind} at {path!r} gives the name {name!r}, which Claude Code cannot load")
    return name


def place_harness(spec: HarnessSpec) -> tuple[Placement, ...]:
    """Where each harness file goes in the agent's working directory, for Claude Code.

    - ``instructions`` → ``CLAUDE.md``. Several are joined in path order, separated by
      a blank line; one is written as it is.
    - ``skill`` → ``.claude/skills/<name>/SKILL.md``, the name being the directory of a
      path ending in ``SKILL.md``, else the file's stem.
    - ``command`` → ``.claude/commands/<name>.md``, the name being the file's stem.
    - ``settings`` → ``.claude/settings.json``, the project's settings; at most one,
      and it must be a JSON object. Slime writes the user's own settings, under
      ``/home/agent``, and this does not touch them.

    Other kinds, another agent, two files with one target, or settings that are not a
    JSON object raise :class:`~honing.core.ContractError`.
    """
    if spec.agent != AGENT:
        raise ContractError(f"this runner runs {AGENT!r}, not {spec.agent!r}")
    unsupported = sorted(spec.kinds - SUPPORT.file_kinds)
    if unsupported:
        raise ContractError(f"Claude Code on slime takes {sorted(SUPPORT.file_kinds)}, not {unsupported}")
    files = sorted(spec.files, key=lambda item: item[0])
    placements: list[Placement] = []
    instructions = [(path, file) for path, file in files if file.kind == "instructions"]
    if len(instructions) == 1:
        placements.append(Placement("CLAUDE.md", instructions[0][1].content, (instructions[0][0],)))
    elif instructions:
        joined = "\n\n".join(file.content.rstrip("\n") for _, file in instructions) + "\n"
        placements.append(Placement("CLAUDE.md", joined, tuple(path for path, _ in instructions)))
    settings = [(path, file) for path, file in files if file.kind == "settings"]
    if len(settings) > 1:
        raise ContractError(f"Claude Code reads one project settings file, and the spec has {len(settings)}")
    for path, file in settings:
        try:
            parsed = json.loads(file.content)
        except ValueError:
            parsed = None
        if not isinstance(parsed, dict):
            raise ContractError(f"the settings file {path!r} must hold a JSON object")
        placements.append(Placement(".claude/settings.json", file.content, (path,)))
    for path, file in files:
        if file.kind == "skill":
            placements.append(Placement(f".claude/skills/{_name(path, 'skill')}/SKILL.md", file.content, (path,)))
        elif file.kind == "command":
            placements.append(Placement(f".claude/commands/{_name(path, 'command')}.md", file.content, (path,)))
    targets = [placement.target for placement in placements]
    duplicated = sorted({target for target in targets if targets.count(target) > 1})
    if duplicated:
        raise ContractError(f"several harness files go to {duplicated}")
    return tuple(placements)


def sglang_root(endpoint: PolicyEndpoint) -> str:
    """SGLang's native root for slime's adapter, which posts token ids to ``/generate``.

    An endpoint's ``base_url`` may be the OpenAI-compatible root (``.../v1``) that
    other runners talk to; that suffix is dropped. The worker checks the result
    answers SGLang's ``/get_model_info`` before the episode starts.
    """
    if endpoint.runtime != "sglang":
        raise ContractError(f"Claude Code on slime needs an SGLang server, not a {endpoint.runtime!r} endpoint")
    url = endpoint.base_url.rstrip("/")
    return url.removesuffix("/v1")


def _check_workspace(workspace: Path, placements: Sequence[Placement]) -> None:
    taken = [files.SLIME_META_DIR, *(placement.target for placement in placements)]
    for target in taken:
        parts = PurePosixPath(target).parts
        for depth in range(1, len(parts) + 1):
            prefix = workspace.joinpath(*parts[:depth])
            if os.path.lexists(prefix) and (depth == len(parts) or not prefix.is_dir() or prefix.is_symlink()):
                raise ContractError(
                    f"the task's workspace already has {'/'.join(parts[:depth])!r}, where {target} goes"
                )


def episode_record(
    task: Task,
    spec: HarnessSpec,
    policy: PolicyRef,
    endpoint: PolicyEndpoint,
    settings: AgentEpisodeSettings,
    placements: Sequence[Placement],
) -> dict[str, object]:
    """``episode.json`` for one episode, as :func:`~honing.backends.slime.agent_episode.read_episode` reads it."""
    return {
        "schema_version": files.SCHEMA_VERSION,
        "task": {"task_id": task.task_id, "prompt": task.prompt},
        "spec_id": spec.spec_id,
        "policy_checksum": policy.checkpoint.checksum,
        "policy": {
            "sglang_url": sglang_root(endpoint),
            "served_model": endpoint.served_model,
            "tokenizer_location": endpoint.tokenizer_location,
        },
        "adapter": {
            "bind_host": settings.adapter_bind_host,
            "tool_parser": settings.tool_parser,
            "reasoning_parser": settings.reasoning_parser,
            "max_turns_per_sid": settings.max_turns_per_sid,
            "max_context_tokens": settings.max_context_tokens,
            "fork_threshold_tokens": settings.fork_threshold_tokens,
            "max_new_tokens": settings.max_new_tokens,
            "sampling_defaults": dict(settings.sampling_defaults),
        },
        "agent": {
            "time_budget_seconds": settings.time_budget_seconds,
            "setup_timeout_seconds": settings.setup_timeout_seconds,
            "copy_out_max_bytes": settings.copy_out_max_bytes,
        },
        "sandbox": {
            "image": settings.image,
            "cpus": settings.limits.cpus,
            "memory_bytes": settings.limits.memory_bytes,
            "pids": settings.limits.pids,
            "network": settings.network,
            "host_alias": settings.host_alias,
            "lifetime_seconds": settings.sandbox_lifetime_seconds,
            "name_prefix": f"honing-episode-{uuid.uuid4().hex[:12]}",
        },
        "cli": {"install": settings.install_cli, "extra_args": settings.cli_extra_args},
        "placement": [
            {
                "target": placement.target,
                "sha256": files.sha256_text(placement.content),
                "paths": list(placement.paths),
            }
            for placement in placements
        ],
        "capture_samples": False,
    }


def _read_result(path: Path) -> dict[str, object] | None:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return result if isinstance(result, dict) else None


class SlimeEpisodeRunner:
    """Satisfies :class:`~honing.core.EpisodeRunner` with Claude Code, run by slime's agent layer in a container.

    ``suite`` gives each task's workspace by ``task_id``; ``endpoints`` resolves the
    policy to an SGLang server; ``environment`` says how to start a Python with Slime
    on its path. ``episodes`` holds one directory per episode, with the files the
    worker read and wrote and its ``worker.log``. ``concurrency`` caps how many run at
    once. The worker sees only the variables in ``passthrough`` that are set here, and
    ``environment.env``. The runner's own ``docker`` calls get the same variables, so
    they find the ``docker`` the worker does and reach the same daemon.
    """

    support = SUPPORT

    def __init__(
        self,
        *,
        suite: TaskSuite,
        endpoints: PolicyEndpoints,
        environment: SlimeEnvironment,
        settings: AgentEpisodeSettings,
        episodes: Path,
        concurrency: int = 4,
        passthrough: Sequence[str] = PASSTHROUGH,
        environ: Mapping[str, str] | None = None,
        worker_module: str = WORKER_MODULE,
        terminate_grace_seconds: float = 30.0,
    ) -> None:
        if concurrency <= 0 or terminate_grace_seconds <= 0:
            raise ContractError("concurrency and terminate_grace_seconds must be positive")
        self._suite = suite
        self._endpoints = endpoints
        self._environment = environment
        self._settings = settings
        # Absolute, because the worker runs in its episode's directory and is handed the path.
        self._episodes = episodes.resolve()
        self._concurrency = concurrency
        self._passthrough = tuple(passthrough)
        self._environ = os.environ if environ is None else environ
        self._worker_module = worker_module
        self._grace = terminate_grace_seconds
        self._image_checked = False
        self._slots: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
            weakref.WeakKeyDictionary()
        )

    def _prepare(self, spec: HarnessSpec, task: Task, policy: PolicyRef) -> Path:
        placements = place_harness(spec)
        endpoint = self._endpoints.resolve(policy)
        source = self._suite.entry(task.task_id).workspace
        _check_workspace(source, placements)
        record = episode_record(task, spec, policy, endpoint, self._settings, placements)
        directory = self._episodes / f"{task.task_id}--{spec.spec_id[:12]}--{uuid.uuid4().hex[:8]}"
        directory.mkdir(parents=True)
        shutil.copytree(source, directory / files.WORKSPACE_DIR, symlinks=True)
        for placement in placements:
            target = directory / files.HARNESS_DIR / placement.target
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(placement.content, encoding="utf-8")
        (directory / files.EPISODE_FILE).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        return directory

    def _worker_env(self) -> dict[str, str]:
        return {name: self._environ[name] for name in self._passthrough if name in self._environ}

    async def _check_image(self) -> None:
        """Raise unless the episodes' image is on the daemon the worker uses; the worker never pulls one.

        Checked once per runner, before its first episode. An image removed after that
        makes that episode's sandbox fail to start, which raises too.
        """
        if self._image_checked:
            return
        image = self._settings.image
        try:
            inspect = await asyncio.create_subprocess_exec(
                "docker",
                "image",
                "inspect",
                image,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
                env=self._worker_env(),
            )
        except OSError as error:
            raise AgentWorkerError(f"cannot run docker to check the image {image}: {error}") from None
        try:
            _, err = await asyncio.wait_for(inspect.communicate(), _INSPECT_SECONDS)
        except asyncio.TimeoutError:  # the builtin TimeoutError only from Python 3.11
            with contextlib.suppress(ProcessLookupError):
                inspect.kill()
            await inspect.wait()
            raise AgentWorkerError(f"docker image inspect {image} did not finish in {_INSPECT_SECONDS:.0f}s") from None
        if inspect.returncode != 0:
            detail = err.decode(errors="replace").strip()[-500:]
            raise AgentWorkerError(
                f"the image {image} is not on the Docker daemon; build or pull {image} first: {detail}"
            )
        self._image_checked = True

    async def run(self, spec: HarnessSpec, task: Task, policy: PolicyRef) -> EpisodeOutcome:
        await self._check_image()
        directory = self._prepare(spec, task, policy)
        command = self._environment.command([self._environment.python, "-m", self._worker_module, str(directory)], {})
        # One semaphore per event loop: a campaign runs each gate in its own asyncio.run.
        slots = self._slots.setdefault(asyncio.get_running_loop(), asyncio.Semaphore(self._concurrency))
        async with slots:
            code = await self._run_worker(command, directory)
        return self._outcome(spec, task, policy, directory, code)

    async def _run_worker(self, command: list[str], directory: Path) -> int:
        log_path = directory / files.LOG_FILE
        with log_path.open("wb") as log:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=directory,
                env=self._worker_env(),
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = self._settings.worker_timeout_seconds
            try:
                return await asyncio.wait_for(process.wait(), deadline)
            except asyncio.TimeoutError:  # the builtin TimeoutError only from Python 3.11
                await self._stop(process, directory)
                raise AgentWorkerError(f"the worker ran past {deadline}s and was killed; see {log_path}") from None
            except BaseException:
                stopping = asyncio.ensure_future(self._stop(process, directory))
                # Finish stopping even if cancelled again meanwhile; the first cancellation is re-raised below.
                while not stopping.done():
                    with contextlib.suppress(asyncio.CancelledError):
                        await asyncio.shield(stopping)
                raise

    async def _stop(self, process: asyncio.subprocess.Process, directory: Path) -> None:
        """SIGTERM the worker, which removes its sandbox; SIGKILL it after the grace; then remove the sandbox."""
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            await asyncio.wait_for(process.wait(), self._grace)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
        # Whatever the worker started in its session (docker clients, the adapter's threads) goes too.
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)
        state = _read_result(directory / files.STATE_FILE)
        sandbox_id = state.get("sandbox_id") if state else None
        if isinstance(sandbox_id, str) and sandbox_id:
            remover = await asyncio.create_subprocess_exec(
                "docker",
                "rm",
                "--force",
                sandbox_id,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                env=self._worker_env(),
            )
            try:
                await asyncio.wait_for(remover.wait(), _REMOVE_SECONDS)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    remover.kill()
                await remover.wait()

    def _outcome(self, spec: HarnessSpec, task: Task, policy: PolicyRef, directory: Path, code: int) -> EpisodeOutcome:
        log = directory / files.LOG_FILE
        result = _read_result(directory / files.RESULT_FILE)
        if result is None:
            raise AgentWorkerError(f"the worker exited {code} without a readable {files.RESULT_FILE}; see {log}")
        status = result.get("status")
        if status == files.STATUS_INFRA_ERROR:
            raise AgentWorkerError(f"the episode could not run: {result.get('error')}; see {log}")
        exit_code = result.get("exit_code")
        workspace = result.get("workspace")
        # A failure is the agent's, and leaves no workspace; anything else must name the copied one.
        failure = result.get("failure")
        if (
            code != 0
            or status not in files.STATUSES
            or not isinstance(exit_code, int)
            or isinstance(exit_code, bool)
            or not (failure is None or (isinstance(failure, str) and failure))
            or workspace != (None if failure else files.WORKSPACE_OUT_DIR)
        ):
            raise AgentWorkerError(f"the worker exited {code} with a malformed {files.RESULT_FILE}; see {log}")
        output = result.get("final_output")
        transcript = directory / files.TRANSCRIPT_FILE
        return EpisodeOutcome(
            task_id=task.task_id,
            spec_id=spec.spec_id,
            policy_checksum=policy.checkpoint.checksum,
            exit_code=exit_code,
            final_output=output if isinstance(output, str) else "",
            transcript_location=str(transcript) if result.get("transcript") and transcript.is_file() else None,
            workspace_location=None if failure else str(directory / files.WORKSPACE_OUT_DIR),
            failure=failure,
        )


class SlimeAgentDeliveryProbe:
    """Satisfies :class:`~honing.e2e.conformance.DeliveryProbe`: what the worker found in place before the CLI started.

    The worker checks each placed file's content in the sandbox just before it starts
    Claude Code and records it per harness path in ``result.json``.
    """

    def delivered(self, outcome: EpisodeOutcome, spec: HarnessSpec) -> dict[str, bool]:
        result = None
        if outcome.workspace_location is not None:
            result = _read_result(Path(outcome.workspace_location).parent / files.RESULT_FILE)
        delivery = result.get("delivery") if result else None
        found = delivery if isinstance(delivery, dict) else {}
        return {path: found.get(path) is True for path, _ in spec.files}
