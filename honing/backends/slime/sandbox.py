"""A Docker container on this machine as slime's agent ``Sandbox``.

Slime runs its coding agents in ``slime.agent.sandbox.E2BSandbox``, a remote service.
:class:`LocalContainerSandbox` satisfies the same ``Sandbox`` protocol at the pinned
commit (``SLIME_REVISION``) with a local container, so the helpers built on it
(``exec_and_wait``, ``ensure_agent_user``, ``install_npm_cli``) and slime's
``ClaudeCodeHarness`` run unchanged. The agent runs with its permission checks off
(``--permission-mode bypassPermissions``), so the container is what isolates it.

The protocol is structural, so this module imports neither slime nor the rest of
Honing, only the standard library. It runs wherever Honing does, and in Slime's
image with the Honing source on ``PYTHONPATH``, like :mod:`~honing.backends.slime.worker`.

The image needs bash, coreutils (``timeout``), ``useradd`` and git for
``ensure_agent_user``, and tar for ``install_node22``; ``sandbox.Dockerfile`` next to
this module is the smallest one. The agent reaches the host as ``host.docker.internal``
(``host-gateway``: the bridge gateway on Linux, the Mac's loopback under colima and
Docker Desktop), so an adapter on the host is at ``http://host.docker.internal:<port>``.
On Linux it must listen on the bridge gateway's address, not only on 127.0.0.1.

What the container does not limit: outbound network (the default ``bridge`` network
reaches the internet and the host's network) and disk use.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import tempfile
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import IO, TypeVar

_LOGGER = logging.getLogger(__name__)
_T = TypeVar("_T")

ExecResult = tuple[int, str, str]

# How long ``timeout`` waits after its TERM before it sends KILL.
_KILL_AFTER_SECONDS = 5
# How much longer than a command's own limit the host waits for ``docker exec`` itself.
_CLIENT_GRACE_SECONDS = 15.0
_CREATE_SECONDS = 600.0
_START_SECONDS = 60.0
_REMOVE_SECONDS = 30.0
_FILE_SECONDS = 600.0
_REAP_SECONDS = 5.0
# coreutils ``timeout`` exits 124 when it stopped the command, 137 when it had to KILL it.
_TIMED_OUT_CODES = frozenset({124, 137})
# Docker's own failures, as opposed to the command's: the daemon refused the exec (no
# such container or user), the CLI could not reach the daemon, or the runtime could not
# start the program.
_DOCKER_FAILURE = re.compile(
    r"^(Error response from daemon:|Cannot connect to the Docker daemon|failed to connect to the docker API"
    r"|OCI runtime exec failed)"
)
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_WRITE_SCRIPT = 'mkdir -p -- "$(dirname -- "$1")" && cat > "$1"'


class SandboxError(RuntimeError):
    """Docker could not do what the sandbox asked: create, exec, write, or remove.

    A ``RuntimeError``, as slime's own sandbox raises, because this module does not
    import Honing's errors.
    """


class SandboxTimeoutError(SandboxError, TimeoutError):
    """A command ran past its time limit and was killed, with everything it started in its group."""


@dataclass(frozen=True)
class ContainerLimits:
    """Resources one sandbox may use. Swap is off: the memory limit is the whole limit."""

    cpus: float = 2.0
    memory_bytes: int = 4 * 2**30
    pids: int = 1024

    def __post_init__(self) -> None:
        # Docker refuses less than 6 MiB of memory.
        if self.cpus <= 0 or self.memory_bytes < 6 * 2**20 or self.pids <= 0:
            raise ValueError(f"limits must be positive, memory at least 6 MiB: {self}")


async def _run(
    argv: Sequence[str], *, stdin: bytes | Path | None = None, timeout_seconds: float
) -> tuple[int, bytes, bytes] | None:
    """Run a docker CLI command on the host. ``None`` means it ran past ``timeout_seconds`` and was killed.

    The client is killed and reaped on every way out, cancellation included, so no
    transport is left to a closed loop. Killing the client does not stop what it
    started in a container: the caller's in-container ``timeout`` and the container's
    removal do.
    """
    with contextlib.ExitStack() as files:
        source: int | IO[bytes] = asyncio.subprocess.DEVNULL
        if isinstance(stdin, Path):
            source = files.enter_context(stdin.open("rb"))
        elif stdin is not None:
            source = asyncio.subprocess.PIPE
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=source,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        data = stdin if isinstance(stdin, bytes) else None
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(data), timeout_seconds)
        except asyncio.TimeoutError:
            await _kill(process)
            return None
        except BaseException:
            await _kill(process)
            raise
    assert process.returncode is not None
    return process.returncode, stdout, stderr


async def _through_cancellation(task: asyncio.Future[_T]) -> bool:
    """Wait for ``task`` to finish even if the caller is cancelled meanwhile; return whether it was.

    ``asyncio.shield`` keeps the task running; each cancellation of the caller is held
    and reported, for the caller to re-raise once the task is done. A cancellation can
    land in the same step as the task finishes, so a finished task is not taken to have
    been cancelled: only ``task.cancelled()`` says that.
    """
    held = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:  # noqa: PERF203 (catching each cancellation is the loop's purpose)
            if task.cancelled():
                break
            held = True
        except Exception:  # the task's own failure, which the caller reads from the task
            break
    return held


async def _kill(process: asyncio.subprocess.Process) -> None:
    with contextlib.suppress(ProcessLookupError):
        process.kill()
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(process.wait(), _REAP_SECONDS)


def _decode(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _env_file(env: dict[str, str]) -> str:
    """Write ``env`` to a file only this user can read, for ``docker exec --env-file``.

    A file rather than ``-e NAME=VALUE`` because the arguments show in the host's
    process list, and the values include the agent's session token. Docker reads the
    file a line at a time, so a value cannot hold a newline.
    """
    for name, value in env.items():
        if not _ENV_NAME.fullmatch(name):
            raise ValueError(f"not an environment variable name: {name!r}")
        if "\n" in value or "\r" in value or "\0" in value:
            raise ValueError(f"the value of {name} holds a line break or NUL, which docker's env file cannot carry")
    descriptor, path = tempfile.mkstemp(prefix="honing-sandbox-env-")
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.writelines(f"{name}={value}\n" for name, value in env.items())
    return path


class LocalContainerSandbox:
    """slime's ``Sandbox`` as a Docker container that is removed on every way out.

    ``async with`` creates the container and removes it afterwards: on success, on an
    exception, and on cancellation, a second cancellation during removal included.
    The container's main process is ``sleep lifetime_seconds`` and it runs with
    ``--rm``, so one whose owner was killed outright removes itself when that ends;
    all carry the label ``honing.sandbox=1``.

    ``exec`` runs ``cmd`` under ``bash -c`` as ``user``, in ``/`` with that user's
    ``HOME``, inside coreutils ``timeout``: the limit is enforced in the container,
    because killing the host's ``docker exec`` client leaves the command running.
    A command past its limit is killed with its process group and raises
    :class:`SandboxTimeoutError`, as slime's E2B sandbox raises. A command that
    detaches with ``setsid`` (``exec_and_wait``) leaves the group and runs on until
    the container goes. ``idempotent`` is accepted and ignored: nothing is retried.
    """

    def __init__(
        self,
        image: str,
        *,
        limits: ContainerLimits | None = None,
        network: str = "bridge",
        host_alias: str = "host.docker.internal",
        host_address: str = "host-gateway",
        lifetime_seconds: int = 3600,
        docker: str = "docker",
        name_prefix: str = "honing-sandbox",
    ) -> None:
        if lifetime_seconds <= 0:
            raise ValueError(f"lifetime_seconds must be positive, not {lifetime_seconds}")
        self.image = image
        self.limits = limits if limits is not None else ContainerLimits()
        self.network = network
        self.host_alias = host_alias
        self.host_address = host_address
        self.lifetime_seconds = lifetime_seconds
        self.docker = docker
        # Named before it exists, so it can be removed by name whatever point creation reached.
        self.sandbox_id = f"{name_prefix}-{uuid.uuid4().hex}"
        self._entered = False

    def create_argv(self) -> list[str]:
        """The ``docker create`` command for the container; the image must already be present."""
        limits = self.limits
        return [
            self.docker,
            "create",
            "--pull",
            "never",
            "--rm",
            "--init",
            "--name",
            self.sandbox_id,
            "--label",
            "honing.sandbox=1",
            "--cpus",
            str(limits.cpus),
            "--memory",
            str(limits.memory_bytes),
            "--memory-swap",
            str(limits.memory_bytes),
            "--pids-limit",
            str(limits.pids),
            "--network",
            self.network,
            "--add-host",
            f"{self.host_alias}:{self.host_address}",
            "--security-opt",
            "no-new-privileges",
            self.image,
            "sleep",
            str(self.lifetime_seconds),
        ]

    async def __aenter__(self) -> LocalContainerSandbox:
        if self._entered:
            raise SandboxError(f"sandbox {self.sandbox_id} was already entered")
        self._entered = True
        try:
            await self._pull()
            await self._create()
            self._check("start", await _run([self.docker, "start", self.sandbox_id], timeout_seconds=_START_SECONDS))
        except BaseException:
            # ``async with`` does not call __aexit__ when __aenter__ fails, and the container
            # may exist by now. Creation always finishes before this runs, so a container
            # created at all is found by name.
            await self._remove(raise_on_failure=False)
            raise
        return self

    async def _pull(self) -> None:
        """Pull the image if it is missing. Cancelling a pull is safe: no container exists yet."""
        present = await _run([self.docker, "image", "inspect", self.image], timeout_seconds=_START_SECONDS)
        if present is None or present[0] != 0:
            self._check(
                "pull", await _run([self.docker, "pull", "--quiet", self.image], timeout_seconds=_CREATE_SECONDS)
            )

    async def _create(self) -> None:
        """``docker create`` the container, finishing even if cancelled meanwhile.

        A ``docker run`` (or a ``docker create``) whose client is killed can still create
        the container after the removal by name has run, and a container that is created
        but never started is removed by neither ``--rm`` nor its ``sleep``. So creation,
        which takes seconds once the image is present, is waited out, and a cancellation
        is re-raised after it, when removal by name will find the container.
        """
        creation = asyncio.ensure_future(_run(self.create_argv(), timeout_seconds=_CREATE_SECONDS))
        if await _through_cancellation(creation):
            raise asyncio.CancelledError
        self._check("create", creation.result())

    def _check(self, step: str, result: tuple[int, bytes, bytes] | None) -> None:
        if result is None:
            raise SandboxError(f"docker {step} did not finish in time for {self.sandbox_id} ({self.image})")
        if result[0] != 0:
            raise SandboxError(
                f"docker {step} failed (exit={result[0]}) for {self.image}: {_decode(result[2])[-1000:]}"
            )

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        # A failed removal raises only when nothing else is on its way out, so it never hides the body's error.
        await self._remove(raise_on_failure=exc_type is None)

    async def _remove(self, *, raise_on_failure: bool) -> None:
        """``docker rm -f`` this container, and wait for it even through a cancellation.

        The removal runs as its own task behind ``asyncio.shield``: a cancellation that
        arrives meanwhile is held until the removal has finished, then re-raised.
        """
        removal = asyncio.ensure_future(
            _run([self.docker, "rm", "--force", self.sandbox_id], timeout_seconds=_REMOVE_SECONDS)
        )
        cancelled = await _through_cancellation(removal)
        failure: str | None = None
        if removal.cancelled():
            failure = "the removal itself was cancelled"
        elif (error := removal.exception()) is not None:
            failure = f"{type(error).__name__}: {error}"
        elif (result := removal.result()) is None:
            failure = f"docker rm did not finish in {_REMOVE_SECONDS:.0f}s"
        elif result[0] != 0:
            failure = f"docker rm failed (exit={result[0]}): {_decode(result[2])[-500:]}"
        if failure is not None:
            message = f"could not remove sandbox {self.sandbox_id}: {failure}"
            if raise_on_failure and not cancelled:
                raise SandboxError(message)
            _LOGGER.warning("%s", message)
        if cancelled:
            raise asyncio.CancelledError

    async def _exec(
        self,
        argv: Sequence[str],
        *,
        user: str,
        env: dict[str, str] | None,
        stdin: bytes | Path | None,
        timeout_seconds: float,
    ) -> tuple[int, str, str] | None:
        """``docker exec`` ``argv`` as ``user``. ``None`` means the client ran past ``timeout_seconds``."""
        env_path = _env_file(env) if env else None
        try:
            command = [self.docker, "exec", "--user", user]
            if stdin is not None:
                command.append("--interactive")
            if env_path is not None:
                command += ["--env-file", env_path]
            result = await _run([*command, self.sandbox_id, *argv], stdin=stdin, timeout_seconds=timeout_seconds)
        finally:
            if env_path is not None:
                os.unlink(env_path)
        if result is None:
            return None
        code, stdout, stderr = result
        out, err = _decode(stdout), _decode(stderr)
        if code in (1, 126, 127) and (_DOCKER_FAILURE.match(err) or _DOCKER_FAILURE.match(out)):
            raise SandboxError(f"docker exec failed in {self.sandbox_id} (exit={code}): {(err or out)[-1000:]}")
        return code, out, err

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
        idempotent: bool = True,
    ) -> ExecResult:
        # coreutils reads a zero limit as no limit at all.
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, not {timeout}")
        argv = ["timeout", "--kill-after", str(_KILL_AFTER_SECONDS), str(timeout), "bash", "-c", cmd]
        client_seconds = timeout + _KILL_AFTER_SECONDS + _CLIENT_GRACE_SECONDS
        started = time.monotonic()
        result = await self._exec(argv, user=user, env=env, stdin=None, timeout_seconds=client_seconds)
        # A command can exit 124 by itself; one that did so before its limit was not stopped.
        if result is None or (result[0] in _TIMED_OUT_CODES and time.monotonic() - started >= timeout):
            raise SandboxTimeoutError(f"exec ran past its {timeout}s limit in {self.sandbox_id}: {cmd[:120]}")
        code, out, err = result
        if check and code != 0:
            raise RuntimeError(f"docker exec failed (exit={code}): {cmd[:120]}\n{err[:400]}")
        return code, out, err

    async def write_file(self, sandbox_path: str, content: str | bytes | Path, *, user: str = "root") -> None:
        """Write ``content`` to ``sandbox_path`` as ``user``, creating its directory.

        A host ``Path`` is streamed from disk, so a large tarball is never held in
        memory. Nothing is mounted, so this works where the container's host cannot
        see this machine's files (colima's VM, for one).
        """
        data = content.encode("utf-8") if isinstance(content, str) else content
        result = await self._exec(
            ["sh", "-c", _WRITE_SCRIPT, "sh", sandbox_path],
            user=user,
            env=None,
            stdin=data,
            timeout_seconds=_FILE_SECONDS,
        )
        if result is None:
            raise SandboxTimeoutError(f"writing {sandbox_path} did not finish in {_FILE_SECONDS:.0f}s")
        if result[0] != 0:
            raise SandboxError(f"cannot write {sandbox_path} as {user} (exit={result[0]}): {result[2][-500:]}")

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        """The file's text, or ``""`` if it cannot be read: what slime's E2B sandbox returns."""
        try:
            result = await self._exec(
                ["cat", "--", sandbox_path], user=user, env=None, stdin=None, timeout_seconds=_FILE_SECONDS
            )
        except SandboxError:
            return ""
        if result is None or result[0] != 0:
            return ""
        return result[1]
