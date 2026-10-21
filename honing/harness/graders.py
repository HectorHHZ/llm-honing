"""Grade an episode by running the task's grader command over the files the agent left behind.

:class:`CommandGrader` looks the task up in its :class:`~honing.harness.tasks.TaskSuite`,
opens a sandbox over the episode's ``workspace_location``, copies the suite's
``grader_files`` into a directory of their own outside the workspace, and runs the task's
grader argv there (in the workspace when there are none). The command finds the agent's
files through ``HONING_WORKSPACE``. The command's last non-empty stdout line must be one JSON object,
``{"score": <number in [0, 1]>, "feedback": <string>}``.

The episode runner's sandbox is gone by the time a grader runs (``run_graded`` grades
after ``run`` returns), so the grader opens its own over the workspace the runner
recorded. Two kinds of failure are told apart:

- the agent's work failed: the workspace holds a file that cannot be copied safely
  (a FIFO, socket, or device) or that the grader may not read, or the grader exits
  nonzero, prints something that is not a grade,
  reports a score outside ``[0, 1]``, times out, or cannot be started. That is a score
  of 0 with the reason as feedback, so one bad episode never aborts a gate. So is an
  outcome whose runner set ``failure``: the agent left nothing to grade, and no
  command runs.
- the infrastructure failed: the outcome names no workspace and no failure, or the
  sandbox cannot be opened. That raises, because scoring it 0 would charge a broken
  runner to the harness.
"""

from __future__ import annotations

import asyncio
import errno
import json
import math
import os
import re
import shutil
import signal
import stat
import tempfile
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Protocol, runtime_checkable

from honing.core import ContractError, EpisodeOutcome, Grade, HoningError, Task, require
from honing.harness.tasks import ManifestTask, TaskSuite

FINAL_OUTPUT_ENV = "HONING_FINAL_OUTPUT_FILE"
METADATA_ENV = "HONING_TASK_METADATA"
TASK_ID_ENV = "HONING_TASK_ID"
WORKSPACE_ENV = "HONING_WORKSPACE"

# The exit code a shell reports for a command it cannot find.
_NOT_RUNNABLE = 127
# How long to keep reading output after the command's process group is gone. Only
# a process that left the group can still hold the pipes open that long.
_DRAIN_SECONDS = 5.0
_EXIT_POLL_SECONDS = 0.02
_STDERR_IN_FEEDBACK_CHARS = 500


class UngradableWorkspaceError(HoningError):
    """The agent left its workspace in a state the grader cannot work in; it scores 0, it does not abort.

    A sandbox raises it from opening, and the message becomes the feedback.
    """


@dataclass(frozen=True)
class CommandResult:
    """What one command produced. ``stdout`` and ``stderr`` are the tails a sandbox kept."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


@runtime_checkable
class CommandSandbox(Protocol):
    """A working copy of an episode's workspace that a grader command runs in.

    The part of slime's ``Sandbox`` a grader needs, declared here because only
    ``honing/backends`` may import slime. The sandbox supplies ``PATH``, ``HOME``,
    the locale, and ``HONING_WORKSPACE`` (the workspace as the command sees it); the
    caller's ``env`` is added to those and nothing else is inherited, so the host's
    secrets stay out of the grader.
    """

    async def add_grader_files(self, source: Path) -> str:
        """Copy the host directory ``source`` into a new directory and return its path as the command sees it.

        The directory is outside the workspace and not at any fixed path relative to
        it, so the agent's code, which a grader runs from the workspace, cannot open
        the hidden tests by a relative path.
        """
        ...

    async def write_file(self, name: str, content: bytes) -> str:
        """Write a file outside the workspace and return its path as the command sees it."""
        ...

    async def exec(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        timeout_seconds: float,
        max_output_bytes: int,
        cwd: str | None = None,
    ) -> CommandResult:
        """Run ``argv`` in ``cwd``, or in the workspace. A command that cannot start exits 127; a timeout kills it."""
        ...


SandboxOpener = Callable[[Path], AbstractAsyncContextManager[CommandSandbox]]


def _kill_group(pid: int) -> None:
    """Kill the process group the command leads, which holds everything it started."""
    with suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGKILL)


async def _exit_code(process: asyncio.subprocess.Process) -> int:
    """The command's exit code, as soon as it has exited.

    ``Process.wait`` will not do: from Python 3.11 it returns only once every pipe
    has closed as well (``base_subprocess._call_connection_lost``), and a background
    process the command started can hold them open. ``returncode`` is set as soon
    as the command itself is reaped, on every version.
    """
    while (code := process.returncode) is None:
        await asyncio.sleep(_EXIT_POLL_SECONDS)
    return code


async def _tail(stream: asyncio.StreamReader, limit: int, kept: bytearray) -> None:
    """Read ``stream`` to its end, keeping its last ``limit`` bytes in ``kept`` as it goes.

    ``kept`` is the caller's, so what was read survives the read being cancelled.
    """
    while chunk := await stream.read(65536):
        kept += chunk
        if len(kept) > limit:
            del kept[:-limit]


class LocalWorkspace:
    """A grading sandbox on this machine: a temporary copy of the workspace and a subprocess.

    It does not prevent reward hacking: do not optimize against a grader run in it
    (GEPA, RL) until #16 lands. It keeps grading from changing the agent's files,
    but neither the command nor the agent's code it runs is isolated from the
    machine. Both run as the current user with this machine's ``PATH`` and can read
    any file that user can, the suite's hidden cases included (through a parent
    process's working directory, for one). It is for graders checked in with the
    suite. Open one with :func:`local_workspace`.
    """

    def __init__(self, root: Path) -> None:
        self.workspace = root / "workspace"
        self._scratch = root / "scratch"
        self._home = root / "home"
        self.grader_directories: list[Path] = []

    async def add_grader_files(self, source: Path) -> str:
        # A directory of its own, not one beside the workspace: agent code run from the
        # workspace could open a sibling as ``../<name>``.
        directory = Path(tempfile.mkdtemp(prefix="honing-grader-"))
        self.grader_directories.append(directory)
        await asyncio.to_thread(shutil.copytree, source, directory / "files", symlinks=True)
        return str(directory / "files")

    async def write_file(self, name: str, content: bytes) -> str:
        path = self._scratch / name
        await asyncio.to_thread(path.write_bytes, content)
        return str(path)

    async def exec(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        timeout_seconds: float,
        max_output_bytes: int,
        cwd: str | None = None,
    ) -> CommandResult:
        environment = {
            "PATH": os.environ.get("PATH", os.defpath),
            "HOME": str(self._home),
            "TMPDIR": str(self._scratch),
            "LANG": "C.UTF-8",
            WORKSPACE_ENV: str(self.workspace),
            **env,
        }
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=self.workspace if cwd is None else cwd,
                env=environment,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as error:
            return CommandResult(exit_code=_NOT_RUNNABLE, stdout="", stderr=f"cannot run {argv[0]!r}: {error}")
        if process.stdout is None or process.stderr is None:
            raise ContractError("a grader command's stdout and stderr must be pipes")
        stdout, stderr = bytearray(), bytearray()
        readers = [
            asyncio.ensure_future(_tail(process.stdout, max_output_bytes, stdout)),
            asyncio.ensure_future(_tail(process.stderr, max_output_bytes, stderr)),
        ]
        waited = timed_out = False
        try:
            # Only the command itself is waited for: a background process it started
            # can hold the pipes open long after the grade is printed.
            try:
                await asyncio.wait_for(_exit_code(process), timeout_seconds)
            except asyncio.TimeoutError:
                timed_out = True
            waited = True
        finally:
            # Every way out, a cancelled grade included, stops everything the command
            # started: it leads its own process group.
            _kill_group(process.pid)
            if not waited:
                # Cancelled, or failed: reap the command and close its transport on this
                # loop before the exception goes on, or both are left to a closed loop.
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(process.wait(), _DRAIN_SECONDS)
                for reader in readers:
                    reader.cancel()
        exit_code = await _exit_code(process)
        _, still_reading = await asyncio.wait(readers, timeout=_DRAIN_SECONDS)
        for reader in still_reading:
            reader.cancel()
        return CommandResult(
            exit_code=exit_code,
            stdout=bytes(stdout).decode("utf-8", errors="replace"),
            stderr=bytes(stderr).decode("utf-8", errors="replace"),
            timed_out=timed_out,
        )


def _copy_regular_file(workspace: Path, source: str, destination: str) -> object:
    """``copytree``'s file copy, refusing anything that is not a regular file before opening it.

    Opening a FIFO blocks until something writes to it, and a device can be read
    forever, so either would hang the copy outside any timeout.
    """
    mode = os.lstat(source).st_mode
    if not stat.S_ISREG(mode):
        kind = "FIFO" if stat.S_ISFIFO(mode) else "socket" if stat.S_ISSOCK(mode) else "device or special file"
        name = Path(source).relative_to(workspace).as_posix()
        raise UngradableWorkspaceError(f"the workspace holds a {kind}, {name!r}, which the grader does not copy")
    return shutil.copy2(source, destination)


_ERRNO = re.compile(r"\[Errno (\d+)\]")


def _unreadable(error: shutil.Error, workspace: Path) -> UngradableWorkspaceError | None:
    """The refusal for a copy that failed only because the agent left files the grader may not read.

    ``copytree`` reports each failure as ``(source, destination, str(error))``,
    so the errno is read from the message, and the source is checked to really be
    unreadable: a destination that could not be written, a full disk say, is not
    the agent's doing and returns ``None``.
    """
    failures = error.args[0] if error.args else None
    if not isinstance(failures, list) or not failures:
        return None
    names = []
    for source, _, reason in failures:
        found = _ERRNO.match(str(reason))
        path = Path(source)
        if found is None or int(found.group(1)) not in (errno.EACCES, errno.EPERM):
            return None
        if not path.is_relative_to(workspace) or os.access(path, os.R_OK | (os.X_OK if path.is_dir() else 0)):
            return None
        names.append(path.relative_to(workspace).as_posix())
    # Sorted, because ``copytree`` reports in directory order, which differs between
    # filesystems, and the same workspace must get the same feedback everywhere.
    listed = ", ".join(map(repr, sorted(names)))
    return UngradableWorkspaceError(f"the workspace holds files the grader may not read: {listed}")


def _remove(root: Path) -> None:
    """Delete ``root`` even where the agent made a directory read-only."""
    # ``os.walk`` lists a directory before descending into it, so each one is
    # made writable and listable from its parent, before the walk reaches it.
    for directory, subdirectories, _ in os.walk(root):
        for name in subdirectories:
            path = Path(directory) / name
            if not path.is_symlink():
                path.chmod(path.stat().st_mode | stat.S_IRWXU)
    shutil.rmtree(root, ignore_errors=True)


@asynccontextmanager
async def local_workspace(workspace: Path) -> AsyncGenerator[LocalWorkspace, None]:
    """Copy ``workspace`` into a temporary directory, so grading never changes the agent's files.

    Links are copied as links rather than followed, so a link the agent left
    pointing at ``/`` does not copy the machine. A FIFO, socket, or device file
    is refused with :class:`UngradableWorkspaceError` without being opened, and so
    is a file or directory the grader may not read, the workspace root included;
    any other copy failure raises.
    """
    root = Path(tempfile.mkdtemp(prefix="honing-grade-"))
    sandbox = LocalWorkspace(root)
    try:
        copy = partial(_copy_regular_file, workspace)
        try:
            await asyncio.to_thread(shutil.copytree, workspace, root / "workspace", symlinks=True, copy_function=copy)
        except shutil.Error as error:
            refusal = _unreadable(error, workspace)
            if refusal is None:
                raise
            raise refusal from None
        except PermissionError:
            # ``copytree`` lists the root before its own error collection starts, so an
            # unreadable root raises directly. Only that is the agent's doing.
            if os.access(workspace, os.R_OK | os.X_OK):
                raise
            refusal = UngradableWorkspaceError(f"the grader may not read the workspace itself, {str(workspace)!r}")
            raise refusal from None
        (root / "scratch").mkdir()
        (root / "home").mkdir()
        yield sandbox
    finally:
        for directory in sandbox.grader_directories:
            await asyncio.to_thread(_remove, directory)
        await asyncio.to_thread(_remove, root)


def _with_stderr(reason: str, stderr: str) -> str:
    tail = stderr.strip()[-_STDERR_IN_FEEDBACK_CHARS:]
    return f"{reason}; stderr ends: {tail}" if tail else reason


def grade_from_result(result: CommandResult, timeout_seconds: float) -> Grade:
    """Turn a grader command's result into a grade. Anything that is not a valid grade scores 0."""
    if result.timed_out:
        return Grade(score=0.0, feedback=f"the grader timed out after {timeout_seconds:g} s")
    if result.exit_code != 0:
        return Grade(score=0.0, feedback=_with_stderr(f"the grader exited {result.exit_code}", result.stderr))
    # Only "\n" ends a line: ``str.splitlines`` also splits on characters such as
    # U+2028 that JSON allows unescaped inside a string.
    lines = [line.removesuffix("\r") for line in result.stdout.split("\n") if line.strip()]
    if not lines:
        return Grade(score=0.0, feedback=_with_stderr("the grader printed nothing on stdout", result.stderr))
    last = lines[-1].strip()
    try:
        reply = json.loads(last)
    except ValueError:  # JSONDecodeError, or an integer longer than Python will convert
        return Grade(score=0.0, feedback=f"the grader's last stdout line is not JSON: {last[:200]!r}")
    if not isinstance(reply, dict):
        return Grade(score=0.0, feedback=f"the grader's last stdout line is not a JSON object: {last[:200]!r}")
    score, feedback = reply.get("score"), reply.get("feedback")
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        return Grade(score=0.0, feedback=f"the grader's score must be a number, got {score!r}")
    try:
        value = float(score)
    except OverflowError:
        # An integer too large for a float, which ``math.isfinite`` would raise on too.
        value = math.inf
    if not (math.isfinite(value) and 0.0 <= value <= 1.0):
        return Grade(score=0.0, feedback=f"the grader's score must be in [0, 1], got {_short(score)}")
    if not isinstance(feedback, str):
        return Grade(score=0.0, feedback=f"the grader's feedback must be a string, got {feedback!r}")
    return Grade(score=value, feedback=feedback)


def _short(value: object) -> str:
    text = repr(value)
    return text if len(text) <= 40 else f"{text[:37]}..."


def _truncate(feedback: str | None, limit: int) -> str | None:
    if feedback is None or len(feedback) <= limit:
        return feedback
    return feedback[:limit] + " [truncated]"


class CommandGrader:
    """Grades an episode by running its task's grader command over a copy of the agent's workspace.

    The command runs in its own copy of the task's ``grader_files``, outside the
    workspace, or in the workspace when the task has none. It sees three variables
    besides what the sandbox supplies:
    ``HONING_TASK_ID``; ``HONING_FINAL_OUTPUT_FILE``, a file holding the agent's
    final text, for tasks graded on the reply rather than on files; and
    ``HONING_TASK_METADATA``, a file holding the manifest line's ``metadata`` as
    JSON. Feedback is cut to ``max_feedback_chars``, since a reflective method
    puts it in a prompt.

    A grader that imports the agent's code in its own process can be forged: that
    code can print a grade, exit before the real one is printed, or replace what
    computes it. The example suite (``examples/tasks/python-bugfix``) shows the
    safe shape: a parent that holds the expected values and never imports the
    agent's code, and a child per case that runs it from the workspace.

    A grader that runs the agent's code should first load its hidden data and
    remove it from disk: that code runs as the same user, so it can search the
    disk for anything left there, this grader's own directory included. What
    stays in the grader's memory is still in reach (``/proc/<pid>/mem``, or a
    debugger where the system allows one); only a sandbox that isolates the
    agent's code (#16) closes that.
    """

    def __init__(
        self,
        suite: TaskSuite,
        *,
        open_sandbox: SandboxOpener = local_workspace,
        timeout_seconds: float = 600.0,
        max_output_bytes: int = 64 * 1024,
        max_feedback_chars: int = 4096,
    ) -> None:
        require(timeout_seconds > 0, f"timeout_seconds must be > 0, got {timeout_seconds}")
        require(max_output_bytes > 0, f"max_output_bytes must be > 0, got {max_output_bytes}")
        require(max_feedback_chars > 0, f"max_feedback_chars must be > 0, got {max_feedback_chars}")
        self._suite = suite
        self._open_sandbox = open_sandbox
        self._timeout_seconds = timeout_seconds
        self._max_output_bytes = max_output_bytes
        self._max_feedback_chars = max_feedback_chars

    async def grade(self, task: Task, outcome: EpisodeOutcome) -> Grade:
        entry = self._suite.entry(task.task_id)
        if outcome.failure is not None:
            return Grade(score=0.0, feedback=_truncate(outcome.failure, self._max_feedback_chars))
        if outcome.workspace_location is None:
            raise ContractError(f"the outcome for task {task.task_id!r} names no workspace to grade")
        try:
            result = await self._run(entry, outcome, Path(outcome.workspace_location))
        except UngradableWorkspaceError as refused:
            return Grade(score=0.0, feedback=_truncate(str(refused), self._max_feedback_chars))
        grade = grade_from_result(result, self._timeout_seconds)
        return Grade(score=grade.score, feedback=_truncate(grade.feedback, self._max_feedback_chars))

    async def _run(self, entry: ManifestTask, outcome: EpisodeOutcome, workspace: Path) -> CommandResult:
        async with self._open_sandbox(workspace) as sandbox:
            cwd = None if entry.grader_files is None else await sandbox.add_grader_files(entry.grader_files)
            env = {
                TASK_ID_ENV: entry.task_id,
                FINAL_OUTPUT_ENV: await sandbox.write_file("final_output.txt", outcome.final_output.encode("utf-8")),
                METADATA_ENV: await sandbox.write_file(
                    "metadata.json", json.dumps(entry.metadata, sort_keys=True).encode("utf-8")
                ),
            }
            return await sandbox.exec(
                entry.grader,
                env=env,
                timeout_seconds=self._timeout_seconds,
                max_output_bytes=self._max_output_bytes,
                cwd=cwd,
            )
