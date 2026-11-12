"""Endpoint resolution runs off the runners' event loop (#154).

Runners resolve through :func:`~honing.harness.endpoints.resolve_endpoint`, on a thread, and
:class:`~honing.backends.sglang.endpoints.SGLangPolicyEndpoints` starts, probes and restarts on one thread it
owns. Starts and probes here are fakes that block on events or sleep; one test starts real child processes,
on Linux only.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import pytest

from honing.backends.sglang.endpoints import ServingSettings, SGLangPolicyEndpoints, StartedServer
from honing.backends.sglang.engine import die_with_parent
from honing.core import ContractError, EpisodeRunner, HarnessSpec, PolicyEndpoint, PolicyRef, RunnerSupport, Task
from honing.harness.endpoints import StaticEndpoint, StaticPolicyEndpoints, resolve_endpoint
from tests.fakes.model import FakeModelSource, FakeWorld
from tests.test_sglang_serving import FakeClock, policy, write_checkpoint

REPO = Path(__file__).resolve().parents[1]
WAIT_SECONDS = 10.0  # a bound on every wait, so a broken test fails instead of hanging


class Gate:
    """A fake start or probe that blocks: ``entered`` is set when it is called, and it returns once ``opened``."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.opened = threading.Event()

    def wait(self) -> None:
        self.entered.set()
        assert self.opened.wait(WAIT_SECONDS), "the gate was never opened"


class Starts:
    """Starts and stops fake servers; ``block`` names the start numbers that wait on ``gate`` first."""

    def __init__(self, gate: Gate | None = None, block: frozenset[int] = frozenset()) -> None:
        self.gate = gate or Gate()
        self.block = block
        self.started: list[str] = []
        self.stopped: list[int] = []
        self.exited: list[bool] = []
        self.threads: list[threading.Thread] = []

    def __call__(self, checkpoint: Path, key: str, gpu: str, stack: ExitStack) -> StartedServer:
        n = len(self.started)
        self.started.append(checkpoint.name)
        self.exited.append(False)
        self.threads.append(threading.current_thread())
        if n in self.block:
            self.gate.wait()
        stack.callback(self.stopped.append, n)
        return StartedServer(
            f"http://{checkpoint.name}.{n}/v1", f"http://{checkpoint.name}.{n}/health", lambda: self.exited[n]
        )


def serving(
    starts: Starts,
    probe: Callable[[str, float], bool] = lambda url, timeout: True,
    gpus: tuple[str, ...] = ("0", "1"),
) -> tuple[SGLangPolicyEndpoints, FakeClock]:
    clock = FakeClock()
    settings = ServingSettings(python=Path("python"), gpus=gpus, health_ttl_seconds=30.0)
    return SGLangPolicyEndpoints(settings, start=starts, probe=probe, clock=clock), clock


def eventually(condition: Callable[[], bool]) -> bool:
    deadline = time.monotonic() + WAIT_SECONDS
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


async def entered(gate: Gate) -> None:
    """Wait on the loop until ``gate``'s start or probe has been called."""
    deadline = time.monotonic() + WAIT_SECONDS
    while not gate.entered.is_set():
        assert time.monotonic() < deadline, "the start or probe was never called"
        await asyncio.sleep(0.01)


def in_thread(call: Callable[[], Any]) -> tuple[threading.Thread, list[Any]]:
    """Run ``call`` on a new thread; its result, or the exception it raised, lands in the returned list."""
    out: list[Any] = []

    def run() -> None:
        try:
            out.append(call())
        except BaseException as error:  # the test inspects it
            out.append(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, out


# The loop keeps running (acceptance 1).


@pytest.mark.parametrize("blocked", ["start", "probe", "restart", "static probe"])
def test_episodes_on_the_loop_keep_running_while_a_resolve_waits(tmp_path: Path, blocked: str) -> None:
    ticks = [0]
    during: list[int] = []

    def hold() -> None:  # in place of a start or probe that takes a while
        before = ticks[0]
        time.sleep(0.3)
        during.append(ticks[0] - before)

    p = policy(write_checkpoint(tmp_path / "m"))
    endpoints: Any
    if blocked == "static probe":

        def static_probe(url: str, timeout: float) -> bool:
            hold()
            return True

        entry = StaticEndpoint("http://m/v1", "m", "tok", health_url="http://m/health")
        endpoints = StaticPolicyEndpoints({p.checkpoint.checksum: entry}, probe=static_probe)
    else:

        def start(checkpoint: Path, key: str, gpu: str, stack: ExitStack) -> StartedServer:
            if blocked == "start" or (blocked == "restart" and exited):
                hold()
            return StartedServer("http://m/v1", "http://m/health", lambda: bool(exited))

        def probe(url: str, timeout: float) -> bool:
            if blocked == "probe":
                hold()
            return True

        exited: list[bool] = []
        endpoints, clock = serving(Starts(), probe)
        endpoints._start = start  # the start above, which reads ``exited``
        if blocked != "start":
            endpoints.resolve(p)  # running, and fresh
            clock.now += 30.0  # the probe is due
            if blocked == "restart":
                exited.append(True)

    async def main() -> PolicyEndpoint:
        async def tick() -> None:
            while True:
                ticks[0] += 1
                await asyncio.sleep(0.005)

        ticker = asyncio.create_task(tick())
        try:
            return await resolve_endpoint(endpoints, p)
        finally:
            ticker.cancel()

    assert asyncio.run(main()).base_url == "http://m/v1"
    assert len(during) == 1 and during[0] >= 10, f"the loop ticked {during} times while the {blocked} ran"


# One start for concurrent resolves (acceptance 4).


def test_two_threads_resolving_one_checkpoint_at_once_start_one_server(tmp_path: Path) -> None:
    starts = Starts(block=frozenset({0}))
    endpoints, _ = serving(starts)
    p = policy(write_checkpoint(tmp_path / "m"))
    first, first_out = in_thread(lambda: endpoints.resolve(p))
    assert starts.gate.entered.wait(WAIT_SECONDS)
    second, second_out = in_thread(lambda: endpoints.resolve(p))
    time.sleep(0.1)  # the second call is waiting on the first one's start now
    starts.gate.opened.set()
    first.join(WAIT_SECONDS)
    second.join(WAIT_SECONDS)
    assert first_out == second_out and isinstance(first_out[0], PolicyEndpoint)
    assert starts.started == ["m"]
    endpoints.close()


def test_resolves_gathered_on_the_loop_start_one_server(tmp_path: Path) -> None:
    starts = Starts(block=frozenset({0}))
    endpoints, _ = serving(starts)
    p = policy(write_checkpoint(tmp_path / "m"))

    async def main() -> list[PolicyEndpoint]:
        async def open_gate() -> None:
            await entered(starts.gate)
            await asyncio.sleep(0.05)
            starts.gate.opened.set()

        _, *answers = await asyncio.gather(open_gate(), *(resolve_endpoint(endpoints, p) for _ in range(20)))
        return answers

    assert len(set(asyncio.run(main()))) == 1 and starts.started == ["m"]
    endpoints.close()


class CountingEndpoints:
    """A map whose resolve for checksum ``slow`` blocks on ``gate``; records each call and the threads in it."""

    def __init__(self, slow: str) -> None:
        self.slow = slow
        self.gate = Gate()
        self.calls: list[str] = []
        self.threads: set[int] = set()

    def resolve(self, policy: PolicyRef) -> PolicyEndpoint:
        self.calls.append(policy.checkpoint.checksum)
        self.threads.add(threading.get_ident())
        if policy.checkpoint.checksum == self.slow:
            self.gate.wait()
        return PolicyEndpoint(f"http://{policy.checkpoint.checksum}/v1", "tok", "m")


def policy_named(checksum: str, tmp_path: Path) -> PolicyRef:
    snapshot = FakeModelSource(FakeWorld()).materialize(tmp_path)
    checkpoint = type(snapshot.checkpoint)(snapshot.checkpoint.location, snapshot.checkpoint.weight_format, checksum)
    return PolicyRef(checkpoint, snapshot.architecture, "master")


def test_episodes_resolving_one_policy_share_one_call_and_a_warm_one_is_not_held_up(tmp_path: Path) -> None:
    endpoints = CountingEndpoints(slow="cold")
    cold, warm = policy_named("cold", tmp_path), policy_named("warm", tmp_path)

    async def main() -> None:
        waiting = [asyncio.ensure_future(resolve_endpoint(endpoints, cold)) for _ in range(50)]
        # Every episode of the cold policy is waiting; the warm one still answers at once.
        assert (await asyncio.wait_for(resolve_endpoint(endpoints, warm), 2.0)).base_url == "http://warm/v1"
        endpoints.gate.opened.set()
        assert {endpoint.base_url for endpoint in await asyncio.gather(*waiting)} == {"http://cold/v1"}

    asyncio.run(main())
    assert sorted(endpoints.calls) == ["cold", "warm"]
    assert threading.get_ident() not in endpoints.threads


def test_two_checkpoints_on_one_health_url_probe_it_once_at_a_time(tmp_path: Path) -> None:
    gate = Gate()
    calls: list[str] = []

    def probe(url: str, timeout: float) -> bool:
        calls.append(url)
        gate.wait()
        return True

    entry = StaticEndpoint("http://m/v1", "m", "tok", health_url="http://m/health")
    endpoints = StaticPolicyEndpoints({"a": entry, "b": entry}, probe=probe)
    a, a_out = in_thread(lambda: endpoints.resolve(policy_named("a", tmp_path / "a")))
    assert gate.entered.wait(WAIT_SECONDS)
    b, b_out = in_thread(lambda: endpoints.resolve(policy_named("b", tmp_path / "b")))
    time.sleep(0.1)  # b waits for a's probe
    gate.opened.set()
    a.join(WAIT_SECONDS)
    b.join(WAIT_SECONDS)
    assert a_out == b_out == [entry.endpoint()] and calls == ["http://m/health"]


# The fast path: running and fresh only.


@pytest.mark.parametrize("state", ["fresh", "exited", "past the ttl"])
def test_only_a_running_server_checked_within_the_ttl_is_returned_without_the_owner_thread(
    tmp_path: Path, state: str
) -> None:
    probes: list[str] = []

    def probe(url: str, timeout: float) -> bool:
        probes.append(url)
        return True

    starts = Starts(block=frozenset({1}))  # start 0 is b's; start 1, a's, holds the owner thread
    endpoints, clock = serving(starts, probe)
    b = policy(write_checkpoint(tmp_path / "b"))
    a = policy(write_checkpoint(tmp_path / "a", b"\1"))
    endpoints.resolve(b)
    if state == "exited":
        starts.exited[0] = True
    elif state == "past the ttl":
        clock.now += 30.0
    busy, _ = in_thread(lambda: endpoints.resolve(a))
    assert starts.gate.entered.wait(WAIT_SECONDS)
    asking, out = in_thread(lambda: endpoints.resolve(b))
    asking.join(0.3)
    if state == "fresh":
        assert out and out[0].base_url == "http://b.0/v1" and probes == []
    else:
        assert not out, f"a resolve of a server {state} returned without the owner thread"
    starts.gate.opened.set()
    asking.join(WAIT_SECONDS)
    busy.join(WAIT_SECONDS)
    expected = {
        "fresh": ("http://b.0/v1", []),
        "exited": ("http://b.2/v1", []),
        "past the ttl": ("http://b.0/v1", ["http://b.0/health"]),
    }
    assert (out[0].base_url, probes) == expected[state]
    endpoints.close()


# Every runner resolves off the loop (acceptance 3).


class ReachedResolveError(Exception):
    pass


class RecordingEndpoints:
    """Records the thread each resolve runs on, then stops the episode."""

    def __init__(self) -> None:
        self.threads: list[int] = []

    def resolve(self, policy: PolicyRef) -> PolicyEndpoint:
        self.threads.append(threading.get_ident())
        raise ReachedResolveError


def _craftax(tmp_path: Path, endpoints: Any) -> tuple[EpisodeRunner, HarnessSpec, Task]:
    from honing.backends.craftax.harness import seed_harness
    from honing.backends.craftax.runner import CraftaxEpisodeRunner
    from honing.backends.craftax.scenarios import load_suite

    integration = REPO / "integrations/craftax"
    suite = load_suite(integration / "tasks/suite.jsonl")
    runner = CraftaxEpisodeRunner(
        suite=suite, endpoints=endpoints, integration_dir=integration, runs_dir=tmp_path, command=["false"]
    )
    return runner, seed_harness(integration), suite.tasks("val")[0]


def _griddly(tmp_path: Path, endpoints: Any) -> tuple[EpisodeRunner, HarnessSpec, Task]:
    from honing.backends.griddly.harness import seed_harness
    from honing.backends.griddly.runner import GriddlyEpisodeRunner
    from honing.backends.griddly.scenarios import load_suite

    integration = REPO / "integrations/griddly"
    suite = load_suite(integration / "tasks/suite.jsonl")
    runner = GriddlyEpisodeRunner(
        suite=suite, endpoints=endpoints, integration_dir=integration, runs_dir=tmp_path, command=["false"]
    )
    return runner, seed_harness(integration), suite.tasks("val")[0]


def _rea_issac(tmp_path: Path, endpoints: Any) -> tuple[EpisodeRunner, HarnessSpec, Task]:
    from honing.backends.rea_issac.harness import seed_harness
    from honing.backends.rea_issac.runner import ReaIssacEpisodeRunner
    from honing.backends.rea_issac.scenarios import load_suite

    integration = REPO / "integrations/rea_issac"
    suite = load_suite(integration / "tasks/suite.jsonl")
    runner = ReaIssacEpisodeRunner(
        suite=suite, endpoints=endpoints, integration_dir=integration, runs_dir=tmp_path, command=["false"]
    )
    return runner, seed_harness(integration), suite.tasks("val")[0]


def _ra2(tmp_path: Path, endpoints: Any) -> tuple[EpisodeRunner, HarnessSpec, Task]:
    from honing.backends.ra2.scenarios import load_suite
    from tests.test_ra2_backend import SPEC, SUITE_PATH, _runner

    return _runner(tmp_path, endpoints), SPEC, load_suite(SUITE_PATH).tasks("val")[0]


def _tank(tmp_path: Path, endpoints: Any) -> tuple[EpisodeRunner, HarnessSpec, Task]:
    from honing.backends.tank.scenarios import load_suite
    from tests.test_tank_backend import SEED, SUITE_PATH, _runner

    return _runner(tmp_path, endpoints=endpoints), SEED, load_suite(SUITE_PATH).tasks("val")[0]


def _openhands(tmp_path: Path, endpoints: Any) -> tuple[EpisodeRunner, HarnessSpec, Task]:
    from tests.test_openhands_runner import SEED, TASK, _runner

    return _runner(tmp_path, endpoints=endpoints), SEED, TASK


def _workspace(tmp_path: Path, endpoints: Any) -> tuple[EpisodeRunner, HarnessSpec, Task]:
    from honing.e2e.workspace_runner import WorkspaceEpisodeRunner
    from tests.fakes.builder import TASKS
    from tests.fakes.harness import SEED_HARNESS

    support = RunnerSupport(
        agents=frozenset({"claude_code"}), file_kinds=frozenset({"instructions", "skill"}), programmable_loop=False
    )
    runner = WorkspaceEpisodeRunner(command=["false"], workspaces=tmp_path, endpoints=endpoints, support=support)
    return runner, SEED_HARNESS, TASKS[0]


RUNNERS = {
    "craftax": _craftax,
    "griddly": _griddly,
    "openhands": _openhands,
    "ra2": _ra2,
    "rea_issac": _rea_issac,
    "tank": _tank,
    "workspace": _workspace,
}


@pytest.mark.parametrize("name", sorted(RUNNERS))
def test_every_runner_resolves_off_its_event_loop(tmp_path: Path, name: str) -> None:
    endpoints = RecordingEndpoints()
    runner, spec, task = RUNNERS[name](tmp_path, endpoints)
    snapshot = FakeModelSource(FakeWorld()).materialize(tmp_path / "model")
    loop_threads: list[int] = []

    async def main() -> None:
        loop_threads.append(threading.get_ident())
        await runner.run(spec, task, PolicyRef(snapshot.checkpoint, snapshot.architecture, "master"))

    with pytest.raises(ReachedResolveError):
        asyncio.run(main())
    assert len(endpoints.threads) == 1 and endpoints.threads != loop_threads


# Cancellation and close.


def test_a_cancelled_resolve_leaves_its_server_with_the_map_and_other_callers_get_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    errors: list[object] = []
    monkeypatch.setattr(threading, "excepthook", errors.append)
    starts = Starts(block=frozenset({0}))
    endpoints, _ = serving(starts)
    p = policy(write_checkpoint(tmp_path / "m"))
    answers: list[PolicyEndpoint] = []

    async def first_gate() -> None:
        cancelled = asyncio.ensure_future(resolve_endpoint(endpoints, p))
        kept = asyncio.ensure_future(resolve_endpoint(endpoints, p))
        await entered(starts.gate)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        starts.gate.opened.set()
        answers.append(await kept)  # one caller's cancellation is not the others'

    asyncio.run(first_gate())
    assert answers[0].base_url == "http://m.0/v1"

    starts.gate = Gate()
    starts.block = frozenset({1})
    starts.exited[0] = True  # the next resolve restarts the server, and that start blocks

    async def second_gate() -> None:  # cancelled while the start runs; asyncio.run ends with it still running
        resolving = asyncio.ensure_future(resolve_endpoint(endpoints, p))
        await entered(starts.gate)
        resolving.cancel()

    began = time.monotonic()
    asyncio.run(second_gate())
    assert time.monotonic() - began < 2.0 and not starts.gate.opened.is_set()
    starts.gate.opened.set()
    assert eventually(lambda: endpoints.serving == {p.checkpoint.checksum: "0"})
    assert eventually(lambda: not any(t.name == "honing-resolve" for t in threading.enumerate()))
    endpoints.close()
    assert starts.stopped == [0, 1] and errors == []


def test_close_does_not_wait_for_a_start_and_stops_its_server_when_the_start_returns(tmp_path: Path) -> None:
    starts = Starts(block=frozenset({0}))
    endpoints, _ = serving(starts)
    a = policy(write_checkpoint(tmp_path / "a"))
    b = policy(write_checkpoint(tmp_path / "b", b"\1"))
    starting, starting_out = in_thread(lambda: endpoints.resolve(a))
    assert starts.gate.entered.wait(WAIT_SECONDS)
    queued, queued_out = in_thread(lambda: endpoints.resolve(b))  # waits behind a's start
    time.sleep(0.1)
    closing, _ = in_thread(endpoints.close)
    closing.join(2.0)
    assert not closing.is_alive(), "close() waited for the start in progress"
    queued.join(2.0)
    assert isinstance(queued_out[0], ContractError), "a call queued behind the start was not failed at close"
    starts.gate.opened.set()
    starting.join(WAIT_SECONDS)
    assert isinstance(starting_out[0], ContractError) and "closed while" in str(starting_out[0])
    assert starts.stopped == [0] and starts.started == ["a"] and endpoints.serving == {}
    assert eventually(lambda: not endpoints._owner.is_alive())
    with pytest.raises(ContractError, match="closed"):
        endpoints.resolve(a)


def test_close_from_the_owner_thread_stops_the_servers_and_returns(tmp_path: Path) -> None:
    starts = Starts()
    endpoints, _ = serving(starts)
    endpoints.resolve(policy(write_checkpoint(tmp_path / "a")))
    closed = threading.Event()

    def start_that_closes(checkpoint: Path, key: str, gpu: str, stack: ExitStack) -> StartedServer:
        endpoints.close()  # on the owner thread
        closed.set()
        return starts(checkpoint, key, gpu, stack)

    endpoints._start = start_that_closes
    with pytest.raises(ContractError, match="closed while"):
        endpoints.resolve(policy(write_checkpoint(tmp_path / "b", b"\1")))
    assert closed.is_set() and starts.stopped == [0, 1]
    assert eventually(lambda: not endpoints._owner.is_alive())


# Process exit: a Ctrl-C does not wait for a start in progress.

CTRL_C = """
import asyncio, os, signal, sys, threading, time
from pathlib import Path
from honing.backends.sglang.endpoints import ServingSettings, SGLangPolicyEndpoints, StartedServer
from honing.harness.endpoints import resolve_endpoint
from tests.test_sglang_serving import policy, write_checkpoint

def start(checkpoint, key, gpu, stack):
    print("starting", flush=True)
    time.sleep(60)  # a cold start
    return StartedServer("http://m/v1", "http://m/health", lambda: False)

async def main(endpoints, p):
    asyncio.get_running_loop().call_later(0.5, os.kill, os.getpid(), signal.SIGINT)
    await resolve_endpoint(endpoints, p)

p = policy(write_checkpoint(Path(sys.argv[1]) / "m"))
with SGLangPolicyEndpoints(ServingSettings(python=Path("python"), gpus=("0",)), start=start) as endpoints:
    asyncio.run(main(endpoints, p))
"""


def test_a_ctrl_c_during_a_cold_start_exits_without_waiting_for_it(tmp_path: Path) -> None:
    began = time.monotonic()
    done = subprocess.run(
        [sys.executable, "-c", CTRL_C, str(tmp_path)], cwd=REPO, capture_output=True, text=True, timeout=50
    )
    assert "starting" in done.stdout and "KeyboardInterrupt" in done.stderr, done.stderr
    assert time.monotonic() - began < 15.0, "the process waited for the start"


# A server outlives the event loop that resolved it (acceptance 2). die_with_parent works on Linux only.

linux = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="die_with_parent is a no-op off Linux")


def _alive(pid: int) -> bool:
    """Whether ``pid`` runs; a zombie waiting to be reaped counts as gone."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ")[1][0] != "Z"
    except (FileNotFoundError, IndexError):
        return False


@pytest.fixture
def children() -> Iterator[list[subprocess.Popen[bytes]]]:
    started: list[subprocess.Popen[bytes]] = []
    yield started
    for child in started:  # whatever a failed test left
        child.kill()
        child.wait()


def _real_start(
    children: list[subprocess.Popen[bytes]], stopped: list[int]
) -> Callable[[Path, str, str, ExitStack], StartedServer]:
    def start(checkpoint: Path, key: str, gpu: str, stack: ExitStack) -> StartedServer:
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"], preexec_fn=die_with_parent)
        children.append(child)

        def stop() -> None:
            stopped.append(child.pid)
            child.terminate()
            child.wait()

        stack.callback(stop)
        return StartedServer("http://m/v1", "http://m/health", lambda: child.poll() is not None)

    return start


@linux
def test_a_server_outlives_the_loop_and_the_thread_that_resolved_it_and_stops_on_close(
    tmp_path: Path, children: list[subprocess.Popen[bytes]]
) -> None:
    settings = ServingSettings(python=Path("python"), gpus=("0", "1"))
    stopped: list[int] = []
    endpoints = SGLangPolicyEndpoints(settings, start=_real_start(children, stopped))
    asyncio.run(resolve_endpoint(endpoints, policy(write_checkpoint(tmp_path / "a"))))
    caller, _ = in_thread(lambda: endpoints.resolve(policy(write_checkpoint(tmp_path / "b", b"\1"))))
    caller.join(WAIT_SECONDS)
    assert not caller.is_alive() and len(children) == 2
    time.sleep(1.0)  # a SIGTERM from a parent thread's exit would have arrived
    assert [child.poll() for child in children] == [None, None]
    closing, _ = in_thread(endpoints.close)  # from yet another thread
    closing.join(WAIT_SECONDS)
    # Stopped by close() itself, not by the owner thread's exit (die_with_parent would kill them then too).
    assert sorted(stopped) == sorted(child.pid for child in children)
    assert all(child.poll() is not None for child in children)
    assert not endpoints._owner.is_alive()


EXIT_WITHOUT_CLOSE = """
import asyncio, sys, subprocess
from pathlib import Path
from honing.backends.sglang.endpoints import ServingSettings, SGLangPolicyEndpoints, StartedServer
from honing.backends.sglang.engine import die_with_parent
from honing.harness.endpoints import resolve_endpoint
from tests.test_sglang_serving import policy, write_checkpoint

def start(checkpoint, key, gpu, stack):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"], preexec_fn=die_with_parent)
    print(child.pid, flush=True)
    return StartedServer("http://m/v1", "http://m/health", lambda: child.poll() is not None)

endpoints = SGLangPolicyEndpoints(ServingSettings(python=Path("python"), gpus=("0",)), start=start)
asyncio.run(resolve_endpoint(endpoints, policy(write_checkpoint(Path(sys.argv[1]) / "m"))))
"""  # exits without close()


@linux
def test_a_server_dies_with_the_process_that_started_it(tmp_path: Path) -> None:
    done = subprocess.run(
        [sys.executable, "-c", EXIT_WITHOUT_CLOSE, str(tmp_path)], cwd=REPO, capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0, done.stderr
    pid = int(done.stdout.split()[0])
    try:
        assert eventually(lambda: not _alive(pid)), "the server outlived its process"
    finally:
        if _alive(pid):
            os.kill(pid, signal.SIGKILL)
