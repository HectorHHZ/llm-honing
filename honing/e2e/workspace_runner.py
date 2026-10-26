"""A reference harness runner at the boundary an external agent framework sits behind.

OpenHands, Reef, Claude Code, and similar frameworks all take the same shape
when Honing runs them: write the harness files where the agent reads them, start
the agent on one task with the policy's endpoint, wait, and collect what it
wrote. :class:`WorkspaceEpisodeRunner` is that shape with nothing else: a fresh
workspace per episode, the harness files at their spec paths, ``task.json``,
and an agent command run as a subprocess with the endpoint in its environment.
It is what ``configs/e2e/all_planes.json`` runs, and a starting point for a
real framework's runner.

The agent writes two files into its workspace: ``result.json`` with
``final_output``, and ``transcript.jsonl``. The runner judges nothing; a
:class:`~honing.core.TaskGrader` does.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import uuid
import weakref
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from honing.core import ContractError, EpisodeOutcome, HarnessSpec, PolicyEndpoints, PolicyRef, RunnerSupport, Task

TIMED_OUT = -9
# What every agent process gets from the runner's own environment; anything
# else (an API key, for one) must be named in ``forward_environment``.
BASE_ENVIRONMENT = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT")
# Files the runner or the agent writes; a task's starting files may not use these names.
RUNNER_FILES = frozenset({"task.json", "agent.log", "result.json", "transcript.jsonl"})


class WorkspaceEpisodeRunner:
    """Satisfies :class:`~honing.core.EpisodeRunner` by running an agent command in a fresh workspace.

    The agent process gets ``HONING_POLICY_URL``, ``HONING_POLICY_MODEL`` and
    ``HONING_POLICY_CHECKSUM`` (the endpoint the policy resolves to) and
    ``HONING_WORKSPACE``. ``support`` is what the agent framework can run;
    ``concurrency`` caps how many agents run at once; an agent still running
    after ``timeout_seconds`` is killed and reported with exit code -9, and so
    is one whose episode is cancelled.

    The agent's environment is ``BASE_ENVIRONMENT`` and the variables named in
    ``forward_environment`` (where set), then ``environment``, then the
    ``HONING_`` variables: not the runner's whole environment, which may hold
    keys the agent has no use for. ``seed_workspace`` gives a task's starting
    files (a directory copied in before the harness files), or ``None``. A
    symbolic link in it is copied as a link, and must be relative and resolve
    inside that directory.
    """

    def __init__(
        self,
        *,
        command: Sequence[str],
        workspaces: Path,
        endpoints: PolicyEndpoints,
        support: RunnerSupport,
        environment: Mapping[str, str] | None = None,
        forward_environment: Sequence[str] = (),
        seed_workspace: Callable[[Task], Path | None] | None = None,
        concurrency: int = 4,
        timeout_seconds: float = 600.0,
    ) -> None:
        if concurrency <= 0 or timeout_seconds <= 0:
            raise ContractError("concurrency and timeout_seconds must be positive")
        self.support = support
        self._command = list(command)
        self._workspaces = workspaces
        self._endpoints = endpoints
        self._environment = dict(environment or {})
        self._forward = (*BASE_ENVIRONMENT, *forward_environment)
        self._seed_workspace = seed_workspace
        self._concurrency = concurrency
        self._timeout = timeout_seconds
        self._slots: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
            weakref.WeakKeyDictionary()
        )

    def _prepare(self, spec: HarnessSpec, task: Task) -> Path:
        if spec.agent not in self.support.agents:
            raise ContractError(f"this runner runs agents {sorted(self.support.agents)}, not {spec.agent!r}")
        workspace = self._workspaces / f"{task.task_id}--{spec.spec_id[:12]}--{uuid.uuid4().hex[:8]}"
        source = self._seed_workspace(task) if self._seed_workspace is not None else None
        if source is None:
            workspace.mkdir(parents=True)
        else:
            workspace.parent.mkdir(parents=True, exist_ok=True)
            _refuse_links_that_leave(task, source)
            # A link stays a link: following it would copy in whatever it points at, hidden cases included.
            shutil.copytree(source, workspace, symlinks=True)
            clashes = sorted(
                name
                for name in (*RUNNER_FILES, *(path for path, _ in spec.files))
                if os.path.lexists(workspace / name)
            )
            if clashes:
                raise ContractError(
                    f"task {task.task_id!r}'s starting files use names the runner or the harness writes: {clashes}"
                )
        for path, file in spec.files:
            target = workspace / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(file.content, encoding="utf-8")
        (workspace / "task.json").write_text(json.dumps({"task_id": task.task_id, "prompt": task.prompt}))
        return workspace

    async def run(self, spec: HarnessSpec, task: Task, policy: PolicyRef) -> EpisodeOutcome:
        endpoint = self._endpoints.resolve(policy)
        workspace = self._prepare(spec, task)
        env = {
            **{name: os.environ[name] for name in self._forward if name in os.environ},
            **self._environment,
            "HONING_POLICY_URL": endpoint.base_url,
            "HONING_POLICY_MODEL": endpoint.served_model,
            "HONING_POLICY_CHECKSUM": policy.checkpoint.checksum,
            "HONING_WORKSPACE": str(workspace),
        }
        # One semaphore per event loop: a campaign runs each gate in its own asyncio.run.
        slots = self._slots.setdefault(asyncio.get_running_loop(), asyncio.Semaphore(self._concurrency))
        async with slots:
            with (workspace / "agent.log").open("wb") as log:
                process = await asyncio.create_subprocess_exec(
                    *self._command, cwd=workspace, env=env, stdout=log, stderr=asyncio.subprocess.STDOUT
                )
                try:
                    exit_code = await asyncio.wait_for(process.wait(), self._timeout)
                except asyncio.TimeoutError:  # the builtin TimeoutError only from Python 3.11
                    exit_code = TIMED_OUT
                finally:
                    # On a timeout, and on cancellation too: when one episode of a
                    # gather fails, asyncio.run cancels the rest, and an agent left
                    # running would keep calling the policy after the run is over.
                    if process.returncode is None:
                        process.kill()
                        await process.wait()
        result = workspace / "result.json"
        final_output = str(json.loads(result.read_text()).get("final_output", "")) if result.exists() else ""
        return EpisodeOutcome(
            task_id=task.task_id,
            spec_id=spec.spec_id,
            policy_checksum=policy.checkpoint.checksum,
            exit_code=exit_code,
            final_output=final_output,
            transcript_location=str(workspace / "transcript.jsonl"),
            workspace_location=str(workspace),
        )


class WorkspaceDeliveryProbe:
    """Satisfies :class:`~honing.e2e.conformance.DeliveryProbe` for runners that leave the harness files in place.

    A file counts as delivered when the episode's workspace holds it at its
    spec path with the same content. A framework that renders the harness
    elsewhere (into an agent's configuration, for example) needs its own probe.
    """

    def delivered(self, outcome: EpisodeOutcome, spec: HarnessSpec) -> dict[str, bool]:
        if outcome.workspace_location is None:
            return {path: False for path, _ in spec.files}
        workspace = Path(outcome.workspace_location)
        return {
            path: (workspace / path).is_file() and (workspace / path).read_text(encoding="utf-8") == file.content
            for path, file in spec.files
        }


def workspace_probe(context: object) -> WorkspaceDeliveryProbe:
    """Factory for a config's ``conformance.probe`` entry."""
    return WorkspaceDeliveryProbe()


def _refuse_links_that_leave(task: Task, source: Path) -> None:
    """Refuse a link the copy would keep that points outside the task's starting files.

    The copied link would resolve from the agent's workspace, so an absolute one, or one
    that climbs out, would let the agent read or write files outside its workspace.
    """
    root = source.resolve()
    for directory, dirnames, filenames in os.walk(source):
        for name in (*dirnames, *filenames):
            path = Path(directory) / name
            if not path.is_symlink():
                continue
            if os.path.isabs(os.readlink(path)) or not path.resolve().is_relative_to(root):
                raise ContractError(
                    f"task {task.task_id!r}'s starting files hold a link that is absolute or leaves them: "
                    f"{path.relative_to(source).as_posix()} -> {os.readlink(path)}"
                )
