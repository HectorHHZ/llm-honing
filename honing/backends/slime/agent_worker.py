"""One Claude Code episode in Slime's environment, run as ``python -m honing.backends.slime.agent_worker EPISODE_DIR``.

Slime's agent adapter and harness import torch (``slime.agent.trajectory`` through
``slime.utils.types``, and ``slime.agent.harness`` through ``slime.utils.misc``), and
Honing's process never does. So :class:`~honing.backends.slime.agent_runner.SlimeEpisodeRunner`
runs each episode as this worker in Slime's environment, and only files cross.

In, written by the runner into ``EPISODE_DIR``:

``episode.json``
    The task, the policy (SGLang's native root URL, the served model's name, the
    tokenizer's location), the adapter, sandbox and CLI settings, the time budget, and
    ``placement``: where each harness file goes in the agent's working directory.
``workspace/``
    The task's starting files.
``harness/``
    The harness files, already at their places (``CLAUDE.md``, ``.claude/...``).

Out, written here:

``state.json``
    The sandbox's container name, written before the container exists, so the runner
    can remove it if it has to kill this worker.
``result.json``
    ``status`` (``completed``, ``time_budget_exceeded`` or ``infra_error``), the CLI's
    ``exit_code``, whether the agent was killed at its deadline, how many of the
    agent's processes were still running before the copy-out, ``final_output``,
    which harness files were in place, turn and token counts, how many calls
    overflowed the model's context, the context limit in force, the CLI version,
    SGLang's model info, timings, ``failure`` and ``error``. Claude Code's
    own ``total_cost_usd`` is not recorded: it prices the served model at Anthropic's
    rates, so it means nothing here.
``transcript.jsonl``
    The CLI's ``stream-json`` output. Slime sends the CLI's stderr into the same file,
    so not every line is JSON.
``workspace_out/``
    The working directory after the episode, without slime's ``.harness/`` and without
    the placed harness files. It is missing when the agent left a workspace that
    cannot come back as it is; ``failure`` in ``result.json`` then says why, and
    ``workspace`` is null.

The worker exits 0 when the episode ran, whatever the agent did, and 3 when it could
not run it (``status: infra_error``): the policy server did not answer, Docker failed,
the CLI did not install. Anything else is a crash, and ``result.json`` may be missing.

Slime is imported lazily, so this module also imports where Slime is absent, for
the runner's constants and for tests of the parts that need no Slime.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import importlib
import io
import json
import logging
import os
import re
import shlex
import signal
import subprocess
import sys
import tarfile
import time
import unicodedata
import urllib.request
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

from honing.backends.slime.agent_episode import (
    COPY_SECONDS,
    EPISODE_FILE,
    EXIT_INFRA,
    EXIT_TIME_BUDGET_EXCEEDED,
    HARNESS_DIR,
    LOG_FILE,
    RESULT_FILE,
    SCHEMA_VERSION,
    SLIME_META_DIR,
    STATE_FILE,
    STATUS_COMPLETED,
    STATUS_INFRA_ERROR,
    STATUS_TIME_BUDGET_EXCEEDED,
    STOP_KILL_SECONDS,
    STOP_POLL_INTERVAL_SECONDS,
    STOP_POLL_SECONDS,
    STOP_POLLS,
    TRANSCRIPT_FILE,
    WORKDIR,
    WORKSPACE_DIR,
    WORKSPACE_OUT_DIR,
    Episode,
    EpisodeFileError,
    PlacedFile,
    read_episode,
)
from honing.backends.slime.sandbox import LocalContainerSandbox

_LOGGER = logging.getLogger("honing.backends.slime.agent_worker")

SANDBOX_TRANSCRIPT = f"{WORKDIR}/{SLIME_META_DIR}/trajectory.jsonl"
# ClaudeCodeHarness launches /usr/local/bin/claude (harness/claude_code.py:41,58).
CLI_PATH = "/usr/local/bin/claude"


_PREFLIGHT_SECONDS = 30.0
_SHUTDOWN_WAIT_SECONDS = 5.0

# SGLang's answers when a request does not fit the model's context, at v0.5.15.post1 (the base of
# Slime's image): the tokenizer manager's two checks (tokenizer_manager.py:958-997) and the
# scheduler's (managers/utils.py:177-192). /generate returns each as a 400 with the message in
# the body, and slime raises it as "sglang upstream 400: <body>" (adapters/common.py:485-495).
_CONTEXT_OVERFLOW = re.compile(
    r"sglang upstream 400: .*(?:is longer than the model's context length"
    r"|exceeds the model's maximum context length|exceeds the maximum allowed length)",
    re.DOTALL,
)

# What this worker calls in slime at the pin, checked against slime's source by
# tests/contracts/test_slime_agent_api.py. Parameters after ``self``; ``*name`` is
# keyword-only. Paths are relative to ``slime/agent/``.
SLIME_SIGNATURES: Mapping[str, tuple[str, ...]] = {
    "harness/common.py:BaseHarness.run": (
        "sb",
        "*workdir",
        "*session_id",
        "*adapter_url",
        "*time_budget_sec",
        "*prompt",
    ),
    "harness/claude_code.py:ClaudeCodeHarness.install_cli": ("sb",),
    "adapters/common.py:BaseAdapter.__init__": (
        "*tokenizer",
        "*sglang_url",
        "*tool_parser",
        "*reasoning_parser",
        "*max_turns_per_sid",
        "*fork_threshold_tokens",
        "*debug_callback",
    ),
    "adapters/common.py:BaseAdapter.open_session": ("sid", "*sampling_defaults", "*max_context_tokens"),
    "adapters/common.py:BaseAdapter.drop_session": ("sid", "*wait_timeout"),
    "adapters/common.py:call_sglang_generate": ("prompt_ids", "session", "body", "*adapter", "*session_id"),
    "aiohttp_threaded.py:run_app_in_thread": (
        "app",
        "*host",
        "*port",
        "*thread_name",
        "*start_timeout_sec",
        "*runner_kwargs",
    ),
}
# Module-level and class-level constants this worker relies on, by the same paths.
SLIME_CONSTANTS: Mapping[str, object] = {
    "sandbox.py:EXIT_TIME_BUDGET_EXCEEDED": EXIT_TIME_BUDGET_EXCEEDED,
    "harness/claude_code.py:ClaudeCodeHarness.node_tarball_env": "SLIME_AGENT_NODE_TARBALL",
    "harness/claude_code.py:ClaudeCodeHarness.cli_tarball_env": "SLIME_AGENT_CC_TARBALL",
    "harness/claude_code.py:ClaudeCodeHarness.extra_args_env": "SLIME_AGENT_CC_EXTRA_ARGS",
}
CLI_TARBALL_ENVS = ("SLIME_AGENT_NODE_TARBALL", "SLIME_AGENT_CC_TARBALL")
CLI_EXTRA_ARGS_ENV = "SLIME_AGENT_CC_EXTRA_ARGS"


def final_output(transcript: str) -> str:
    """The ``result`` of the last ``{"type": "result"}`` event, or ``""``.

    Slime sends the CLI's stderr into the same file as its ``stream-json`` stdout
    (``exec_and_wait`` redirects both), so lines that are not JSON objects are skipped.
    """
    output = ""
    for line in transcript.split("\n"):
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("type") == "result":
            result = event.get("result")
            output = result if isinstance(result, str) else ""
    return output


def pack_workspace(episode_dir: Path, target: Path) -> Path:
    """Tar the task's workspace with the placed harness files over it, for one ``write_file``."""
    with tarfile.open(target, "w") as archive:
        for name in (WORKSPACE_DIR, HARNESS_DIR):
            source = episode_dir / name
            if source.is_dir():
                for child in sorted(source.iterdir()):
                    archive.add(child, arcname=child.name)
    return target


def copy_out_command(placement: tuple[PlacedFile, ...], max_bytes: int) -> str:
    """Tar the working directory to stdout as base64, at most ``max_bytes + 1`` bytes of tar.

    ``read_file`` returns text only, so binary files cross as base64 through ``exec``.
    One byte past the limit is kept, so the caller can tell a full workspace from one
    that was cut off. ``pipefail`` makes a failed tar fail the command, which would
    otherwise exit with ``base64``'s code; ``head`` closing the pipe past the limit can
    also kill tar with SIGPIPE, so the caller checks the size before the exit code.
    """
    excludes = [f"./{SLIME_META_DIR}", *(f"./{placed.target}" for placed in placement)]
    flags = " ".join(f"--exclude={shlex.quote(item)}" for item in excludes)
    return f"set -o pipefail; tar cf - -C {WORKDIR} {flags} . | head -c {max_bytes + 1} | base64 -w0"


class AgentWorkspaceError(Exception):
    """The agent left a workspace that cannot come back as it is; the message is the outcome's ``failure``."""


def decode_workspace(encoded: str, max_bytes: int) -> bytes:
    """The copied-out tar; :class:`AgentWorkspaceError` when the agent left more than ``max_bytes`` of it."""
    data = base64.b64decode(encoded, validate=True)
    if len(data) > max_bytes:
        raise AgentWorkspaceError(
            f"the agent left a workspace larger than {max_bytes} bytes as a tar; it was not copied back"
        )
    return data


def case_collisions(names: Iterable[str]) -> list[tuple[str, str]]:
    """Pairs of paths, and of their parent directories, that differ only in case or Unicode normalization.

    The tar is made on Linux, where ``Foo.py`` and ``foo.py`` are two files; a Mac's
    disk is case-insensitive and normalization-insensitive, so one would overwrite the
    other there, or a file would land where a directory is. Refused on every host, so
    one workspace gets one outcome wherever it is graded.
    """
    seen: dict[str, str] = {}
    pairs: set[tuple[str, str]] = set()
    for name in names:
        parts = PurePosixPath(name).parts
        for depth in range(1, len(parts) + 1):
            path = "/".join(parts[:depth])
            # Unicode's canonical caseless match: NFD(casefold(NFD(path))). casefold, unlike
            # lower, folds every case distinction, ß and ss included.
            key = unicodedata.normalize("NFD", unicodedata.normalize("NFD", path).casefold())
            other = seen.setdefault(key, path)
            if other != path:
                first, second = sorted((other, path))
                pairs.add((first, second))
    return sorted(pairs)


_SURROGATE = re.compile("[\ud800-\udfff]")


def non_utf8_names(names: Iterable[str]) -> list[bytes]:
    """The paths whose bytes are not valid UTF-8, which a Mac's disk refuses (EILSEQ).

    Linux keeps any bytes in a name, and ``tarfile`` reads them back with surrogate
    escapes. Refused on every host, like :func:`case_collisions`.
    """
    # A str fails to encode as UTF-8 exactly when it holds a surrogate, which is how the escapes arrive.
    return sorted(name.encode("utf-8", "surrogateescape") for name in names if _SURROGATE.search(name))


def unpack_workspace(data: bytes, target: Path) -> list[str]:
    """Extract the copied-out tar into ``target``; return the members skipped as unsafe.

    The tar is the agent's: a link out of the directory, an absolute path, or a device
    file is skipped (``tarfile``'s ``data`` filter) rather than failing the episode,
    because the agent made it. Paths that are not valid UTF-8, or that one
    case-insensitive disk would take for one, raise :class:`AgentWorkspaceError` before
    anything is written.
    """
    skipped: list[str] = []

    def keep(member: tarfile.TarInfo, path: str) -> tarfile.TarInfo | None:
        try:
            return tarfile.data_filter(member, path)
        except tarfile.FilterError:
            skipped.append(member.name)
            return None

    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        members = archive.getmembers()
        invalid = non_utf8_names(member.name for member in members)
        if invalid:
            listed = ", ".join(repr(name) for name in invalid[:5])
            more = f" and {len(invalid) - 5} more" if len(invalid) > 5 else ""
            raise AgentWorkspaceError(
                f"the agent left paths that are not valid UTF-8, which a Mac's disk cannot hold: {listed}{more}"
            )
        collisions = case_collisions(member.name for member in members)
        if collisions:
            listed = ", ".join(repr(pair) for pair in collisions[:5])
            more = f" and {len(collisions) - 5} more" if len(collisions) > 5 else ""
            raise AgentWorkspaceError(
                "the agent left paths that differ only in case or Unicode normalization, which a case-insensitive"
                f" disk cannot keep apart: {listed}{more}"
            )
        target.mkdir(parents=True, exist_ok=True)
        archive.extractall(target, members=members, filter=keep)
    return skipped


def prune_placed_directories(root: Path, placement: tuple[PlacedFile, ...]) -> None:
    """Remove the directories that held only placed harness files (``.claude/skills/x/``), now left empty.

    The copy-out leaves the placed files out but keeps their directories; one the
    agent also wrote into is not empty, and stays.
    """
    for placed in placement:
        parent = (root / placed.target).parent
        while parent != root and parent.is_dir() and not parent.is_symlink() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@dataclass
class _Counts:
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    upstream_errors: int = 0
    context_overflows: int = 0


def is_context_overflow(error: BaseException) -> bool:
    """Whether slime's error is SGLang refusing a request that does not fit the model's context."""
    return isinstance(error, RuntimeError) and _CONTEXT_OVERFLOW.match(str(error)) is not None


def context_limit(configured: int, server_info: object) -> int:
    """The prompt length at which slime stops a call before sending it: the setting, else the server's.

    SGLang's ``/server_info`` reports ``max_req_input_len``, the longest input its
    scheduler accepts (one less than the context, or the KV pool if that is smaller).
    Slime returns an empty reply cut off at the length limit once a prompt reaches the
    limit, and lowers ``max_new_tokens`` so prompt and reply fit under it
    (``adapters/common.py:457-468``), so SGLang's own checks never fire. 0 is no limit.
    """
    if configured > 0:
        return configured
    value = server_info.get("max_req_input_len") if isinstance(server_info, dict) else None
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return 0


def _count_turns(common: Any, counts: _Counts) -> None:
    """Wrap slime's ``call_sglang_generate`` to count turns, tokens, and failed calls to the policy server.

    The adapter answers a failed call with a 500, which the CLI may retry or give up
    on; either way the episode then says nothing about the harness, so the worker
    reports it as an infrastructure fault. A prompt that outgrew the model's context
    is the agent's doing, not the server's: it gets the empty reply cut off at the
    length limit that slime itself returns when it stops such a call first, and is
    counted apart. ``_run_turn`` looks the function up by its module-level name,
    which slime keeps module-level so tests can replace it (``adapters/common.py:442-452``
    at the pin).
    """
    original = common.call_sglang_generate

    async def counted(prompt_ids: Any, session: Any, body: Any, *, adapter: Any, session_id: Any = None) -> Any:
        try:
            turn = await original(prompt_ids, session, body, adapter=adapter, session_id=session_id)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if is_context_overflow(error):
                counts.context_overflows += 1
                return common.TurnRecord(prompt_ids=list(prompt_ids), output_ids=[], finish_reason="length")
            counts.upstream_errors += 1
            raise
        counts.turns += 1
        counts.input_tokens += len(turn.prompt_ids)
        counts.output_tokens += len(turn.output_ids)
        return turn

    common.call_sglang_generate = counted


def _slime() -> SimpleNamespace:
    return SimpleNamespace(
        adapters=importlib.import_module("slime.agent.adapters"),
        common=importlib.import_module("slime.agent.adapters.common"),
        harness=importlib.import_module("slime.agent.harness"),
        threaded=importlib.import_module("slime.agent.aiohttp_threaded"),
        transformers=importlib.import_module("transformers"),
    )


def _get_json(url: str) -> object:
    with urllib.request.urlopen(url, timeout=_PREFLIGHT_SECONDS) as response:
        return json.loads(response.read())


def _bind_host(episode: Episode) -> str:
    """Where the adapter listens: the address the sandbox's host alias reaches.

    Under colima and Docker Desktop ``host-gateway`` is the Mac's loopback, so
    ``127.0.0.1`` serves the sandbox without opening the adapter to the network. On
    Linux it is the bridge network's gateway (``sandbox.py:16-19``).
    """
    if episode.adapter_bind_host:
        return episode.adapter_bind_host
    if sys.platform == "darwin":
        return "127.0.0.1"
    inspected = subprocess.run(
        ["docker", "network", "inspect", episode.network, "--format", "{{(index .IPAM.Config 0).Gateway}}"],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return inspected.stdout.strip()


class _InfraError(Exception):
    """The episode could not run for a reason that is not the agent's."""


async def _delivery(sandbox: LocalContainerSandbox, placement: tuple[PlacedFile, ...]) -> dict[str, bool]:
    """Which harness paths are in the working directory with their content, just before the CLI starts."""
    delivered: dict[str, bool] = {}
    for placed in placement:
        code, out, _ = await sandbox.exec(
            f"sha256sum -- {shlex.quote(f'{WORKDIR}/{placed.target}')}", timeout=COPY_SECONDS
        )
        ok = code == 0 and out.split(" ", 1)[0] == placed.sha256
        delivered.update(dict.fromkeys(placed.paths, ok))
    return delivered


def _check_cli_tarballs(episode: Episode) -> None:
    """Fail before any sandbox exists when the CLI is to be installed from tarballs that are not there.

    Slime's ``install_cli`` reads them from the environment with ``os.environ[...]``,
    and a missing file would only surface once the sandbox is up.
    """
    if not episode.install_cli:
        return
    problems = []
    for name in CLI_TARBALL_ENVS:
        value = os.environ.get(name, "")
        if not value:
            problems.append(f"{name} is not set")
        elif not Path(value).is_file():
            problems.append(f"{name}={value} is not a file")
    if problems:
        raise _InfraError(f"cli.install is set, but {'; '.join(problems)}")


_STOP_TOOLS = ("ps", "pkill")


async def _agent_processes(sandbox: LocalContainerSandbox) -> str:
    # A zombie (stat Z) is dead already, waiting for the container's init to reap it. The
    # pipeline exits with awk's code, so a nonzero one is the listing itself failing.
    code, out, err = await sandbox.exec("ps -u agent -o pid=,stat= | awk '$2 !~ /^Z/'", timeout=STOP_POLL_SECONDS)
    if code != 0:
        raise _InfraError(f"cannot list the agent's processes (exit {code}): {err.strip()[-200:]}")
    return out.strip()


async def _stop_agent(sandbox: LocalContainerSandbox) -> int:
    """Kill every process of the ``agent`` user, and check none is left (a killed one may take a moment to go).

    Returns how many were running when it started. ``ps`` and ``pkill`` must be in
    the image (procps); without them the listing would come back empty and the agent
    would pass for stopped while it still writes, so their absence raises.
    """
    tools = " ".join(_STOP_TOOLS)
    code, out, err = await sandbox.exec(
        f'for tool in {tools}; do command -v "$tool" > /dev/null || echo "$tool"; done', timeout=STOP_POLL_SECONDS
    )
    missing = out.split()
    if missing:
        raise _InfraError(f"the sandbox image has no {' or '.join(missing)}, which stopping the agent needs")
    if code != 0:
        raise _InfraError(
            f"cannot check for {' and '.join(_STOP_TOOLS)} in the sandbox (exit {code}): {err.strip()[-200:]}"
        )
    running = await _agent_processes(sandbox)
    code, _, err = await sandbox.exec("pkill -KILL -u agent", timeout=STOP_KILL_SECONDS)
    if code not in (0, 1):  # 1: no process matched
        raise _InfraError(f"pkill failed (exit {code}): {err.strip()[-200:]}")
    for _ in range(STOP_POLLS):
        out = await _agent_processes(sandbox)
        if not out:
            return len(running.splitlines())
        await asyncio.sleep(STOP_POLL_INTERVAL_SECONDS)
    raise _InfraError(f"the agent's processes outlived pkill: {out.strip()[:200]}")


async def _run_episode(episode_dir: Path, episode: Episode, result: dict[str, Any]) -> None:
    """Run the episode, filling ``result`` as it goes; raise :class:`_InfraError` when it cannot run."""
    started = time.monotonic()
    _check_cli_tarballs(episode)
    slime = _slime()
    try:
        tokenizer = slime.transformers.AutoTokenizer.from_pretrained(
            episode.tokenizer_location, trust_remote_code=False
        )
    except Exception as error:
        raise _InfraError(f"cannot load the tokenizer from {episode.tokenizer_location}: {error}") from error
    try:
        result["model_info"] = _get_json(f"{episode.sglang_url}/get_model_info")
    except Exception as error:
        raise _InfraError(
            f"the policy server at {episode.sglang_url} did not answer /get_model_info: {error}"
        ) from error

    server_info = None
    if episode.max_context_tokens <= 0:
        try:
            server_info = _get_json(f"{episode.sglang_url}/server_info")
        except Exception as error:  # a server that does not report it: SGLang's own 400 is then caught per call
            _LOGGER.warning("the policy server did not answer /server_info, so no context limit is set: %s", error)
    result["max_context_tokens"] = context_limit(episode.max_context_tokens, server_info)

    counts = _Counts()
    _count_turns(slime.common, counts)
    sampling = dict(episode.sampling_defaults)
    if episode.max_new_tokens is not None:
        sampling["max_new_tokens"] = episode.max_new_tokens
    adapter = slime.adapters.AnthropicAdapter(
        tokenizer=tokenizer,
        sglang_url=episode.sglang_url,
        tool_parser=episode.tool_parser,
        reasoning_parser=episode.reasoning_parser,
        max_turns_per_sid=episode.max_turns_per_sid,
        fork_threshold_tokens=episode.fork_threshold_tokens,
    )
    handle = slime.threaded.run_app_in_thread(
        adapter.app,
        host=_bind_host(episode),
        port=0,
        thread_name="honing-anthropic-adapter",
        # As slime's coding-agent example serves it (examples/coding_agent_rl/generate.py:161-170): a client
        # that goes away cancels its turn, and only slow or failed requests are logged.
        runner_kwargs={"handler_cancellation": True, "access_log_class": slime.threaded.FilteredAccessLogger},
    )
    session = f"honing-{uuid.uuid4().hex}"
    timings: dict[str, float] = result["timings"]
    try:
        adapter.open_session(session, sampling_defaults=sampling, max_context_tokens=result["max_context_tokens"])
        sandbox = LocalContainerSandbox(
            episode.image,
            limits=episode.limits,
            network=episode.network,
            host_alias=episode.host_alias,
            lifetime_seconds=episode.lifetime_seconds,
            name_prefix=episode.name_prefix,
            # The runner checked the image is there; pulling one is not part of an episode.
            pull_missing=False,
        )
        result["sandbox_id"] = sandbox.sandbox_id
        _write_json(episode_dir / STATE_FILE, {"sandbox_id": sandbox.sandbox_id, "adapter_port": handle.port})
        harness = slime.harness.ClaudeCodeHarness()
        async with contextlib.AsyncExitStack() as stack:

            async def set_up() -> None:
                # Creating and starting the container count against the setup timeout too. Entered
                # through the stack, the sandbox is removed on the way out wherever the timeout lands.
                await stack.enter_async_context(sandbox)
                await _set_up(episode_dir, episode, sandbox, harness, result)

            try:
                await asyncio.wait_for(set_up(), episode.setup_timeout_seconds)
            except asyncio.TimeoutError:  # the builtin TimeoutError only from Python 3.11
                raise _InfraError(f"setting up the sandbox ran past {episode.setup_timeout_seconds}s") from None
            timings["setup_seconds"] = round(time.monotonic() - started, 3)
            if episode.cli_extra_args:
                os.environ[CLI_EXTRA_ARGS_ENV] = episode.cli_extra_args
            else:
                os.environ.pop(CLI_EXTRA_ARGS_ENV, None)
            agent_started = time.monotonic()
            exit_code = await harness.run(
                sandbox,
                workdir=WORKDIR,
                session_id=session,
                adapter_url=f"http://{episode.host_alias}:{handle.port}",
                time_budget_sec=episode.time_budget_seconds,
                prompt=episode.prompt,
            )
            timings["agent_seconds"] = round(time.monotonic() - agent_started, 3)
            result["exit_code"] = int(exit_code)
            # Stop the agent before the workspace is copied, so the copy is what it left and no
            # process of its is still writing while tar reads (tar would fail, and that would
            # read as the infrastructure's fault). exec_and_wait stops waiting at the deadline
            # but leaves the CLI running, and a CLI that exited may have left processes behind.
            result["agent_processes_stopped"] = await _stop_agent(sandbox)
            result["agent_stopped_at_deadline"] = exit_code == EXIT_TIME_BUDGET_EXCEEDED
            copy_started = time.monotonic()
            await _copy_out(episode_dir, episode, sandbox, result)
            timings["copy_out_seconds"] = round(time.monotonic() - copy_started, 3)
    finally:
        # Drain the session's turns before the server goes, so none is left pending on a stopped loop.
        await adapter.drop_session(session, wait_timeout=_SHUTDOWN_WAIT_SECONDS)
        handle.stop()
        result.update(
            turns=counts.turns,
            input_tokens=counts.input_tokens,
            output_tokens=counts.output_tokens,
            upstream_errors=counts.upstream_errors,
            context_overflows=counts.context_overflows,
        )
    if counts.upstream_errors:
        raise _InfraError(f"the policy server failed {counts.upstream_errors} of the agent's calls; see {LOG_FILE}")
    result["status"] = (
        STATUS_TIME_BUDGET_EXCEEDED if result["exit_code"] == EXIT_TIME_BUDGET_EXCEEDED else STATUS_COMPLETED
    )


async def _set_up(
    episode_dir: Path, episode: Episode, sandbox: LocalContainerSandbox, harness: Any, result: dict[str, Any]
) -> None:
    archive = pack_workspace(episode_dir, episode_dir / ".copy-in.tar")
    try:
        await sandbox.write_file("/tmp/honing-copy-in.tar", archive)
    finally:
        archive.unlink(missing_ok=True)
    await sandbox.exec(
        f"mkdir -p {WORKDIR} && tar xf /tmp/honing-copy-in.tar -C {WORKDIR} --no-same-owner"
        " && rm -f /tmp/honing-copy-in.tar",
        check=True,
        timeout=COPY_SECONDS,
    )
    if episode.install_cli:
        await harness.install_cli(sandbox)
    code, out, err = await sandbox.exec(f"{CLI_PATH} --version", user="root", timeout=60)
    if code != 0:
        raise _InfraError(f"{CLI_PATH} does not run in {episode.image} (exit {code}): {(err or out)[-500:]}")
    result["cli_version"] = out.strip()
    result["delivery"] = await _delivery(sandbox, episode.placement)


async def _copy_out(
    episode_dir: Path, episode: Episode, sandbox: LocalContainerSandbox, result: dict[str, Any]
) -> None:
    transcript = await sandbox.read_file(SANDBOX_TRANSCRIPT)
    if transcript:
        (episode_dir / TRANSCRIPT_FILE).write_text(transcript, encoding="utf-8")
        result["transcript"] = TRANSCRIPT_FILE
    result["final_output"] = final_output(transcript)
    # The tar runs as root (exec's default), which may read every file the agent left, so
    # a tar that fails is not the agent's doing.
    code, out, err = await sandbox.exec(
        copy_out_command(episode.placement, episode.copy_out_max_bytes), timeout=COPY_SECONDS
    )
    try:
        # Size first: past the cap, head closing the pipe may have killed tar, and that exit is expected.
        data = decode_workspace(out, episode.copy_out_max_bytes)
        if code != 0:
            raise _InfraError(f"cannot copy the workspace out (exit {code}): {err[-500:]}")
        skipped = unpack_workspace(data, episode_dir / WORKSPACE_OUT_DIR)
    except AgentWorkspaceError as refusal:
        result["failure"] = str(refusal)
        return
    except ValueError as error:  # not base64
        raise _InfraError(f"cannot copy the workspace out: {error}") from error
    except (tarfile.TarError, OSError) as error:
        raise _InfraError(f"cannot unpack the workspace: {error}") from error
    prune_placed_directories(episode_dir / WORKSPACE_OUT_DIR, episode.placement)
    result["workspace"] = WORKSPACE_OUT_DIR
    result["skipped_members"] = skipped


def _empty_result() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS_INFRA_ERROR,
        "exit_code": None,
        "final_output": "",
        "transcript": None,
        "workspace": None,
        "samples": None,
        "delivery": {},
        "turns": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "upstream_errors": 0,
        "context_overflows": 0,
        "max_context_tokens": 0,
        "sandbox_id": None,
        "cli_version": None,
        "model_info": None,
        "skipped_members": [],
        "agent_stopped_at_deadline": False,
        "agent_processes_stopped": 0,
        "timings": {},
        "failure": None,
        "error": None,
    }


async def _main(episode_dir: Path, episode: Episode) -> dict[str, Any]:
    result = _empty_result()
    task = asyncio.current_task()
    assert task is not None
    loop = asyncio.get_running_loop()
    # The runner sends SIGTERM when it gives up on the worker: cancel, so the sandbox is removed on the way out.
    loop.add_signal_handler(signal.SIGTERM, task.cancel)
    try:
        await _run_episode(episode_dir, episode, result)
    except asyncio.CancelledError:
        result.update(status=STATUS_INFRA_ERROR, error="the worker was terminated")
    except _InfraError as error:
        result.update(status=STATUS_INFRA_ERROR, error=str(error))
        _LOGGER.error("episode could not run: %s", error)
    except Exception as error:  # the episode's own code raised: an infrastructure fault, reported with its trace
        result.update(status=STATUS_INFRA_ERROR, error=f"{type(error).__name__}: {error}")
        _LOGGER.exception("episode failed")
    finally:
        loop.remove_signal_handler(signal.SIGTERM)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="honing.backends.slime.agent_worker")
    parser.add_argument("episode_dir", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    episode_dir: Path = args.episode_dir
    try:
        episode = read_episode(episode_dir / EPISODE_FILE)
    except (OSError, EpisodeFileError) as error:
        result = _empty_result()
        result["error"] = f"cannot read {EPISODE_FILE}: {error}"
        _write_json(episode_dir / RESULT_FILE, result)
        return EXIT_INFRA
    result = asyncio.run(_main(episode_dir, episode))
    _write_json(episode_dir / RESULT_FILE, result)
    return EXIT_INFRA if result["status"] == STATUS_INFRA_ERROR else 0


if __name__ == "__main__":
    sys.exit(main())
