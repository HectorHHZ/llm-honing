"""Policy endpoints that start an SGLang engine for each checkpoint a runner asks for, including ones a campaign
has just trained.

:meth:`SGLangPolicyEndpoints.resolve` maps a policy to a running server by its
checkpoint checksum. The first time a checksum is asked for, the checkpoint
at ``policy.checkpoint.location`` is checked against that checksum (Honing
computes it from the files, :func:`~honing.backends.hf_checkpoint.checkpoint_checksum`),
then started on a free GPU from ``gpus``. When every GPU is serving, the
server used least recently is stopped to make room. Later calls for the same
checksum reuse the running server. :meth:`close` stops them all.

A running server is checked on every call: one whose process has exited is
down, and otherwise its health route is probed unless it passed less than
``health_ttl_seconds`` ago, with ``health_timeout_seconds`` for an answer. A
probe that fails is repeated once at once, and the server is down only if both
fail, so a hung server is found within about twice ``health_timeout_seconds``.
For ``"jev_decision"`` this checks the decision server's process and its
``/v1/health`` route only, not the SGLang engine inside it. A server that is
down is stopped and started again once on the same GPU; if that
start fails, or the server goes down again after its restart, the call raises
:class:`~honing.core.ContractError` naming the checkpoint, and so does every
later call for it. The one restart belongs to a started server, not to the
checkpoint: a server stopped to make room and started again later gets a new
one.

Every start, probe, restart and stop runs on one long-lived thread the map
owns, and :meth:`~SGLangPolicyEndpoints.resolve` waits for its answer; calls
for the same checkpoint share one answer, so one start. A server dies with the
thread that started it (:func:`~honing.backends.sglang.engine.die_with_parent`),
so it outlives the event loop of the runner that asked for it, and lives until
:meth:`~SGLangPolicyEndpoints.close` or the end of the process. A call that finds its server
running, and passed within the TTL, returns at once without that thread, so a
start or probe for one checkpoint never holds up the others' running servers.
Runners call it off their event loop, through
:func:`~honing.harness.endpoints.resolve_endpoint`.

The ``runtime`` the runners speak decides what is started:

- ``"sglang"``: the engine itself with its tokenizer, answering
  OpenAI-compatible chat and completions (a workspace runner's agent, or a
  :class:`~honing.harness.models.LiteLLMChatModel` with ``api_base`` set to it).
- ``"jev_decision"``: Honing's decision server for the checkpoint
  (``honing.backends.jev.server --family lettered``), answering ``/v1/decide``
  for the RA2 and tank runners, with its engine started inside it.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future, InvalidStateError
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import honing
from honing.backends.hf_checkpoint import checkpoint_checksum
from honing.backends.sglang.checkpoint import context_length
from honing.backends.sglang.engine import SGLangEngine, SGLangError, SGLangSettings, die_with_parent
from honing.core import ContractError, EndpointRuntime, PolicyEndpoint, PolicyRef
from honing.harness.endpoints import HEALTH_TIMEOUT_SECONDS, HEALTH_TTL_SECONDS, answers_twice, server_answers

REPO = Path(honing.__file__).resolve().parents[1]


@dataclass(frozen=True)
class ServingSettings:
    """Where and how checkpoints are served.

    ``python`` is the SGLang environment's interpreter. Each entry of ``gpus``
    is one server's device, or several devices separated by commas for a
    checkpoint that needs tensor parallelism (``"0,1,2,3"`` serves it on four).
    A decision server runs
    from ``decision_python`` (default ``python``), which needs transformers for
    the tokenizer. ``verify`` checks each checkpoint against the policy's
    checksum before serving it. A running server's health is trusted for
    ``health_ttl_seconds`` after it passes, and two probes in a row that get no
    200 within ``health_timeout_seconds`` each count it as down.
    """

    python: Path
    gpus: tuple[str, ...]
    runtime: EndpointRuntime = "sglang"
    decision_python: Path | None = None
    max_length: int = 8192
    prefix_cache: bool = True
    verify: bool = True
    log_dir: Path | None = None
    start_timeout: float = 900.0
    health_ttl_seconds: float = HEALTH_TTL_SECONDS
    health_timeout_seconds: float = HEALTH_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if not self.gpus:
            raise ValueError("ServingSettings needs at least one GPU")
        if self.health_ttl_seconds < 0 or self.health_timeout_seconds <= 0:
            raise ValueError("ServingSettings needs health_ttl_seconds >= 0 and health_timeout_seconds > 0")


@dataclass(frozen=True)
class StartedServer:
    """What starting a server returns: the URL handed to runners, the route that answers 200 while it is up, and
    whether its process has exited."""

    url: str
    health_url: str
    exited: Callable[[], bool]


@dataclass
class _Served:
    endpoint: PolicyEndpoint
    gpu: str
    stack: ExitStack
    checkpoint: Path
    health_url: str
    exited: Callable[[], bool]
    checked_at: float  # when the server last answered its health route
    restarts: int = 0


class SGLangPolicyEndpoints:
    """A :class:`~honing.core.PolicyEndpoints` that serves each policy's checkpoint on demand. Use it as a context
    manager, or call :meth:`close`."""

    def __init__(
        self,
        settings: ServingSettings,
        start: Callable[[Path, str, str, ExitStack], StartedServer] | None = None,
        probe: Callable[[str, float], bool] = server_answers,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.settings = settings
        self._start = start or self._start_server
        self._probe = probe
        self._clock = clock
        # Guards the fields below. Never held across a start, a stop or a probe, so a call whose server is running
        # and fresh returns while the owner thread works on another checkpoint.
        self._lock = threading.Lock()
        self._served: OrderedDict[str, _Served] = OrderedDict()
        # A checkpoint whose server failed to start, or went down past its one restart, fails again at once, with
        # the first error: a campaign's runner resolves once per episode, and every retry would load the model
        # again before failing.
        self._failed: dict[str, str] = {}
        # One job per checksum at a time: a call that finds one pending waits for its answer, so one start.
        self._pending: dict[str, Future[PolicyEndpoint]] = {}
        self._working: str | None = None  # the checksum the owner thread is serving now
        self._closed = False
        self._jobs: queue.SimpleQueue[tuple[PolicyRef, Future[PolicyEndpoint]] | None] = queue.SimpleQueue()
        # Every server is started from this thread, so it lives as long as the map (die_with_parent). A daemon, so
        # a process that exits, on a Ctrl-C too, does not wait for a start in progress; on Linux its server then dies
        # with the process.
        self._owner = threading.Thread(target=self._serve, name="sglang-endpoints", daemon=True)
        self._owner.start()

    def __enter__(self) -> SGLangPolicyEndpoints:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def serving(self) -> dict[str, str]:
        """Checksum to GPU of every running server, least recently used first."""
        with self._lock:
            return {key: served.gpu for key, served in self._served.items()}

    def resolve(self, policy: PolicyRef) -> PolicyEndpoint:
        """The endpoint serving ``policy``, starting, probing or restarting its server as needed.

        A server whose process is running and that passed its health check within the TTL is returned at once, on
        the calling thread. Anything else waits for the owner thread, which does one job at a time: a call for
        another checkpoint that needs a start or a probe waits for the job ahead of it.
        """
        key = policy.checkpoint.checksum
        with self._lock:
            if self._closed:
                raise ContractError(f"the SGLang endpoints are closed; {policy.checkpoint.location} is not served")
            served = self._served.get(key)
            if served is not None and not served.exited() and self._within_ttl(served):
                self._served.move_to_end(key)
                return served.endpoint
            if key in self._failed:
                raise self._failed_earlier(policy, key)
            future = self._pending.get(key)
            if future is None:
                future = self._pending[key] = Future()
                self._jobs.put((policy, future))
        return future.result()

    def _within_ttl(self, served: _Served) -> bool:
        return self._clock() - served.checked_at < self.settings.health_ttl_seconds

    def _failed_earlier(self, policy: PolicyRef, key: str) -> ContractError:
        return ContractError(
            f"the server for {policy.checkpoint.location} ({key[:12]}) failed earlier: {self._failed[key]}"
        )

    def _serve(self) -> None:
        """The owner thread: answer each job in turn until :meth:`close`."""
        while (job := self._jobs.get()) is not None:
            policy, future = job
            key = policy.checkpoint.checksum
            with self._lock:
                closed = self._closed
                self._working = None if closed else key
            answer: PolicyEndpoint | BaseException
            try:
                if closed:
                    raise ContractError(f"the SGLang endpoints are closed; {policy.checkpoint.location} is not served")
                answer = self._serve_one(policy)
            except BaseException as error:  # handed to the waiting callers; the thread goes on
                answer = error
            with self._lock:
                self._pending.pop(key, None)
                self._working = None
            _settle(future, answer)

    def _serve_one(self, policy: PolicyRef) -> PolicyEndpoint:
        key = policy.checkpoint.checksum
        with self._lock:
            served = self._served.get(key)
            if served is not None:
                self._served.move_to_end(key)
            elif key in self._failed:
                raise self._failed_earlier(policy, key)
        if served is not None:
            if self._up(served):
                return served.endpoint
            return self._restart(key, served)
        checkpoint = Path(policy.checkpoint.location)
        if policy.checkpoint.weight_format != "hf_safetensors" or not (checkpoint / "config.json").exists():
            raise ContractError(f"{checkpoint} is not a Hugging Face checkpoint an SGLang engine can load")
        if self.settings.verify and checkpoint_checksum(checkpoint) != key:
            raise ContractError(f"{checkpoint} does not hold the weights with checksum {key[:12]}")
        oldest = None
        with self._lock:
            self._refuse_if_closed(checkpoint)
            used = {served.gpu for served in self._served.values()}
            free = [gpu for gpu in self.settings.gpus if gpu not in used]
            if not free:  # stop the server used least recently
                _, oldest = self._served.popitem(last=False)
                free = [oldest.gpu]
        if oldest is not None:
            oldest.stack.close()
        stack = ExitStack()
        try:
            started = self._start(checkpoint, key, free[0], stack)
        except Exception as error:
            stack.close()
            with self._lock:
                self._failed[key] = f"{type(error).__name__}: {error}"
            raise
        except BaseException:
            stack.close()
            raise
        return self._register(key, checkpoint, free[0], stack, started, restarts=0)

    def _refuse_if_closed(self, checkpoint: Path) -> None:
        """With the lock held, before a start: after :meth:`close`, nothing new is started."""
        if self._closed:
            raise ContractError(f"the SGLang endpoints were closed; {checkpoint} was not started")

    def _register(
        self, key: str, checkpoint: Path, gpu: str, stack: ExitStack, started: StartedServer, restarts: int
    ) -> PolicyEndpoint:
        endpoint = PolicyEndpoint(
            base_url=started.url,
            tokenizer_location=str(checkpoint),
            served_model=key[:16],
            runtime=self.settings.runtime,
        )
        with self._lock:
            closed = self._closed
            if not closed:
                # The start answered its health route moments ago (the engine's launch loop, the decision server's
                # ready line), so the first check is due one TTL from now.
                self._served[key] = _Served(
                    endpoint, gpu, stack, checkpoint, started.health_url, started.exited, self._clock(), restarts
                )
        if closed:  # close() ran during the start and did not wait for it, so the server is stopped here
            stack.close()
            raise ContractError(f"the SGLang endpoints were closed while {checkpoint} started; it was stopped")
        return endpoint

    def _up(self, served: _Served) -> bool:
        if served.exited():  # down, no need to ask
            return False
        if self._within_ttl(served):
            return True
        if not answers_twice(self._probe, served.health_url, self.settings.health_timeout_seconds):
            return False
        with self._lock:
            served.checked_at = self._clock()
        return True

    def _restart(self, key: str, served: _Served) -> PolicyEndpoint:
        """Stop a server that is down and start it again on the same GPU, once; past that, fail the checkpoint."""
        with self._lock:
            self._served.pop(key, None)
        served.stack.close()  # stop the old server before starting its replacement on the same GPU
        named = f"{served.checkpoint} ({key[:12]})"
        with self._lock:
            if served.restarts >= 1:
                self._failed[key] = "it went down again after its one restart"
                raise ContractError(f"the server for {named} {self._failed[key]}")
            self._refuse_if_closed(served.checkpoint)
        stack = ExitStack()
        try:
            started = self._start(served.checkpoint, key, served.gpu, stack)
        except Exception as error:
            stack.close()
            with self._lock:
                self._failed[key] = f"it went down and did not restart: {type(error).__name__}: {error}"
            raise ContractError(f"the server for {named} {self._failed[key]}") from error
        except BaseException:
            stack.close()
            raise
        return self._register(key, served.checkpoint, served.gpu, stack, started, restarts=served.restarts + 1)

    def close(self) -> None:
        """Stop every running server and refuse later calls; any thread may call it, the owner thread too.

        A start in progress is not waited for: the owner thread stops its server when the start returns, and the
        end of the process stops it before that. Calls waiting for other checkpoints fail at once. When nothing is
        starting, the owner thread has exited by the time this returns.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            working = self._working
            servers = list(self._served.values())
            self._served.clear()
            waiting = [future for key, future in self._pending.items() if key != working]
        self._jobs.put(None)
        for future in waiting:
            _settle(future, ContractError("the SGLang endpoints were closed before this checkpoint was served"))
        for served in servers:
            served.stack.close()
        if working is None and threading.current_thread() is not self._owner:
            self._owner.join()

    def _log(self, key: str, what: str) -> Path | None:
        if self.settings.log_dir is None:
            return None
        self.settings.log_dir.mkdir(parents=True, exist_ok=True)
        return self.settings.log_dir / f"{what}-{key[:16]}.log"

    def _start_server(self, checkpoint: Path, key: str, gpu: str, stack: ExitStack) -> StartedServer:
        """Start the server for one checkpoint on one GPU; register its shutdown on ``stack``; return its URLs."""
        settings = self.settings
        if settings.runtime == "sglang":
            engine_settings = SGLangSettings(
                python=settings.python,
                gpu=gpu,
                chat=True,
                served_model=key[:16],
                context_length=context_length(checkpoint),
                prefix_cache=settings.prefix_cache,
                start_timeout=settings.start_timeout,
                log_path=self._log(key, "engine"),
            )
            engine = stack.enter_context(SGLangEngine.launch(checkpoint, engine_settings))
            return StartedServer(engine.openai_url, f"{engine.url}/health", engine.exited)
        command = [
            str(settings.decision_python or settings.python), "-m", "honing.backends.jev.server",
            "--family", "lettered", "--checkpoint", str(checkpoint), "--checksum", key,
            "--engine", "sglang", "--sglang-python", str(settings.python), "--gpu", gpu,
            "--max-len", str(settings.max_length), *(() if settings.prefix_cache else ("--no-prefix-cache",)),
        ]  # fmt: skip
        log = self._log(key, "decision")
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")]))}
        if log is not None:
            env["SGLANG_LOG"] = str(self._log(key, "engine"))
        errors = log.open("ab") if log is not None else subprocess.DEVNULL
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=errors, env=env, text=True, preexec_fn=die_with_parent
        )
        stack.callback(_stop_server, process)
        line = process.stdout.readline() if process.stdout else ""
        try:
            ready: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError:
            raise SGLangError(f"the decision server for {key[:12]} did not start; see {log}") from None
        url: str = ready["url"]
        return StartedServer(url, f"{url}/v1/health", lambda: process.poll() is not None)


def _settle(future: Future[PolicyEndpoint], answer: PolicyEndpoint | BaseException) -> None:
    """Hand ``answer`` to the callers waiting on ``future``, unless :meth:`SGLangPolicyEndpoints.close` failed it
    first."""
    with contextlib.suppress(InvalidStateError):
        if isinstance(answer, BaseException):
            future.set_exception(answer)
        else:
            future.set_result(answer)


def _stop_server(process: subprocess.Popen[str]) -> None:
    """SIGTERM the decision server, which stops its engine as it exits."""
    process.terminate()
    try:
        process.wait(timeout=120)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


if __name__ == "__main__":  # python -m honing.backends.sglang.endpoints <sglang-python> <gpu> <checkpoint> [runtime]
    from honing.backends.hf_checkpoint import hf_snapshot

    snapshot = hf_snapshot(Path(sys.argv[3]))
    runtime = sys.argv[4] if len(sys.argv) > 4 else "sglang"
    with SGLangPolicyEndpoints(ServingSettings(python=Path(sys.argv[1]), gpus=(sys.argv[2],), runtime=runtime)) as e:  # type: ignore[arg-type]
        print(e.resolve(PolicyRef(snapshot.checkpoint, snapshot.architecture, "master")))
