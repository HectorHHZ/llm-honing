"""Policy endpoints that start an SGLang engine for each checkpoint a runner asks for, including ones a campaign
has just trained.

:meth:`SGLangPolicyEndpoints.resolve` maps a policy to a running server by its
checkpoint checksum. The first time a checksum is asked for, the checkpoint
at ``policy.checkpoint.location`` is checked against that checksum (Honing
computes it from the files, :func:`~honing.backends.sglang.checkpoint.checkpoint_checksum`),
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

The ``runtime`` the runners speak decides what is started:

- ``"sglang"``: the engine itself with its tokenizer, answering
  OpenAI-compatible chat and completions (a workspace runner's agent, or a
  :class:`~honing.harness.models.LiteLLMChatModel` with ``api_base`` set to it).
- ``"jev_decision"``: Honing's decision server for the checkpoint
  (``honing.backends.jev.server --family lettered``), answering ``/v1/decide``
  for the RA2 and tank runners, with its engine started inside it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import honing
from honing.backends.sglang.checkpoint import checkpoint_checksum, context_length
from honing.backends.sglang.engine import SGLangEngine, SGLangError, SGLangSettings, die_with_parent
from honing.core import ContractError, EndpointRuntime, PolicyEndpoint, PolicyRef
from honing.harness.endpoints import HEALTH_TIMEOUT_SECONDS, HEALTH_TTL_SECONDS, server_answers

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
        self._served: OrderedDict[str, _Served] = OrderedDict()
        self._lock = threading.Lock()
        # A checkpoint whose server failed to start, or went down past its one restart, fails again at once, with
        # the first error: a campaign's runner resolves once per episode, and every retry would load the model
        # again before failing.
        self._failed: dict[str, str] = {}

    def __enter__(self) -> SGLangPolicyEndpoints:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def serving(self) -> dict[str, str]:
        """Checksum to GPU of every running server, least recently used first."""
        return {key: served.gpu for key, served in self._served.items()}

    def resolve(self, policy: PolicyRef) -> PolicyEndpoint:
        """The endpoint serving ``policy``, starting, probing or restarting its server as needed.

        The health probe and a restart run under the lock on purpose, as a cold start does: runners resolve on one
        event-loop thread, and holding it stops two threads from restarting the same server. A call for any other
        checkpoint waits meanwhile.
        """
        key = policy.checkpoint.checksum
        with self._lock:
            if key in self._served:
                served = self._served[key]
                self._served.move_to_end(key)
                if self._up(served):
                    return served.endpoint
                return self._restart(key, served)
            if key in self._failed:
                raise ContractError(
                    f"the server for {policy.checkpoint.location} ({key[:12]}) failed earlier: {self._failed[key]}"
                )
            checkpoint = Path(policy.checkpoint.location)
            if policy.checkpoint.weight_format != "hf_safetensors" or not (checkpoint / "config.json").exists():
                raise ContractError(f"{checkpoint} is not a Hugging Face checkpoint an SGLang engine can load")
            if self.settings.verify and checkpoint_checksum(checkpoint) != key:
                raise ContractError(f"{checkpoint} does not hold the weights with checksum {key[:12]}")
            used = {served.gpu for served in self._served.values()}
            free = [gpu for gpu in self.settings.gpus if gpu not in used]
            if not free:  # stop the server used least recently
                _, oldest = self._served.popitem(last=False)
                oldest.stack.close()
                free = [oldest.gpu]
            stack = ExitStack()
            try:
                started = self._start(checkpoint, key, free[0], stack)
            except Exception as error:
                stack.close()
                self._failed[key] = f"{type(error).__name__}: {error}"
                raise
            except BaseException:
                stack.close()
                raise
            return self._register(key, checkpoint, free[0], stack, started, restarts=0)

    def _register(
        self, key: str, checkpoint: Path, gpu: str, stack: ExitStack, started: StartedServer, restarts: int
    ) -> PolicyEndpoint:
        endpoint = PolicyEndpoint(
            base_url=started.url,
            tokenizer_location=str(checkpoint),
            served_model=key[:16],
            runtime=self.settings.runtime,
        )
        # The start answered its health route moments ago (the engine's launch loop, the decision server's ready
        # line), so the first check is due one TTL from now.
        self._served[key] = _Served(
            endpoint, gpu, stack, checkpoint, started.health_url, started.exited, self._clock(), restarts
        )
        return endpoint

    def _up(self, served: _Served) -> bool:
        if served.exited():  # down, no need to ask
            return False
        if self._clock() - served.checked_at < self.settings.health_ttl_seconds:
            return True
        # One immediate retry: a busy engine can leave a single probe waiting past the timeout, and a restart stops
        # a server other episodes are using.
        timeout = self.settings.health_timeout_seconds
        if not (self._probe(served.health_url, timeout) or self._probe(served.health_url, timeout)):
            return False
        served.checked_at = self._clock()
        return True

    def _restart(self, key: str, served: _Served) -> PolicyEndpoint:
        """Stop a server that is down and start it again on the same GPU, once; past that, fail the checkpoint."""
        del self._served[key]
        served.stack.close()  # stop the old server before starting its replacement on the same GPU
        named = f"{served.checkpoint} ({key[:12]})"
        if served.restarts >= 1:
            self._failed[key] = "it went down again after its one restart"
            raise ContractError(f"the server for {named} {self._failed[key]}")
        stack = ExitStack()
        try:
            started = self._start(served.checkpoint, key, served.gpu, stack)
        except Exception as error:
            stack.close()
            self._failed[key] = f"it went down and did not restart: {type(error).__name__}: {error}"
            raise ContractError(f"the server for {named} {self._failed[key]}") from error
        except BaseException:
            stack.close()
            raise
        return self._register(key, served.checkpoint, served.gpu, stack, started, restarts=served.restarts + 1)

    def close(self) -> None:
        with self._lock:
            while self._served:
                _, served = self._served.popitem(last=False)
                served.stack.close()

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


def _stop_server(process: subprocess.Popen[str]) -> None:
    """SIGTERM the decision server, which stops its engine as it exits."""
    process.terminate()
    try:
        process.wait(timeout=120)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


if __name__ == "__main__":  # python -m honing.backends.sglang.endpoints <sglang-python> <gpu> <checkpoint> [runtime]
    from honing.backends.sglang.checkpoint import hf_snapshot

    snapshot = hf_snapshot(Path(sys.argv[3]))
    runtime = sys.argv[4] if len(sys.argv) > 4 else "sglang"
    with SGLangPolicyEndpoints(ServingSettings(python=Path(sys.argv[1]), gpus=(sys.argv[2],), runtime=runtime)) as e:  # type: ignore[arg-type]
        print(e.resolve(PolicyRef(snapshot.checkpoint, snapshot.architecture, "master")))
