"""LocalContainerSandbox: slime's agent helpers against a real container, and its edge cases against a fake CLI.

The Docker tests skip without a daemon and the slime ones without a checkout of
``slime.agent`` on ``PYTHONPATH`` (``scripts/fetch-slime-agent.sh``); the
``test-sandbox`` CI job sets ``HONING_REQUIRE_DOCKER=1`` and ``HONING_REQUIRE_SLIME=1``
so neither can skip there. Tests that need exact timing (a second cancellation
during removal, a client that hangs) run against ``tests/fakes/docker_cli.py``
instead, and need neither.
"""

from __future__ import annotations

import asyncio
import hashlib
import http.server
import importlib
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from types import ModuleType
from typing import ClassVar

import pytest

from honing.backends.slime import ContainerLimits, LocalContainerSandbox, SandboxError, SandboxTimeoutError
from honing.backends.slime import sandbox as sandbox_module
from tests.conftest import REPO_ROOT, require

DOCKERFILE = REPO_ROOT / "honing" / "backends" / "slime" / "sandbox.Dockerfile"
SMALL = ContainerLimits(cpus=1.0, memory_bytes=512 * 2**20, pids=256)

Factory = Callable[..., LocalContainerSandbox]


def docker(*argv: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *argv], capture_output=True, text=True, check=check)


def exists(name: str) -> bool:
    return docker("ps", "--all", "--filter", f"name=^{name}$", "--format", "{{.Names}}").stdout.strip() == name


@pytest.fixture(scope="session")
def image() -> str:
    if shutil.which("docker") is None or docker("info", check=False).returncode != 0:
        require("docker", "no docker daemon is reachable")
    tag = f"honing-sandbox-test:{hashlib.sha256(DOCKERFILE.read_bytes()).hexdigest()[:12]}"
    if docker("image", "inspect", tag, check=False).returncode != 0:
        docker("build", "--quiet", "--tag", tag, "--file", str(DOCKERFILE), str(DOCKERFILE.parent))
    return tag


@pytest.fixture(scope="module")
def slime_sandbox() -> ModuleType:
    try:
        import slime.agent.sandbox as module  # type: ignore[import-not-found]
    except ImportError:
        require("slime", "slime.agent.sandbox is not importable; run scripts/fetch-slime-agent.sh and set PYTHONPATH")
    return module


@pytest.fixture
def make_sandbox(image: str) -> Iterator[Factory]:
    """Sandboxes on the test image with small limits; any a test leaves behind are removed and fail it."""
    made: list[LocalContainerSandbox] = []

    def make(**overrides: object) -> LocalContainerSandbox:
        settings: dict[str, object] = {"limits": SMALL, "lifetime_seconds": 600, **overrides}
        sandbox = LocalContainerSandbox(image, **settings)  # type: ignore[arg-type]
        made.append(sandbox)
        return sandbox

    yield make
    left = [sandbox.sandbox_id for sandbox in made if exists(sandbox.sandbox_id)]
    for name in left:
        docker("rm", "--force", name, check=False)
    assert left == [], f"containers left behind: {left}"


async def _wait_until(predicate: Callable[[], bool], seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.1)
    return predicate()


# --- slime's helpers in a real container ------------------------------------------------------------------------


@pytest.mark.integration
def test_exec_and_wait_returns_the_exit_code_and_the_output(make_sandbox: Factory, slime_sandbox: ModuleType) -> None:
    async def scenario() -> tuple[tuple[int, str], tuple[int, str]]:
        async with make_sandbox() as sandbox:
            # The spawn exec returns at once; the command finishes seconds later, detached. Not
            # `exit 3`: the launcher runs cmd inline, so an exit skips its marker (sandbox.py:112).
            failed = await slime_sandbox.exec_and_wait(
                sandbox, cmd="sleep 2; echo out; bash -c 'exit 3'", time_budget_sec=60, tag="t1", want_output=True
            )
            passed = await slime_sandbox.exec_and_wait(sandbox, cmd="true", time_budget_sec=60, tag="t2")
            return failed, passed

    assert asyncio.run(scenario()) == ((3, "out\n"), (0, ""))


@pytest.mark.integration
def test_exec_and_wait_past_its_budget_returns_the_marker(make_sandbox: Factory, slime_sandbox: ModuleType) -> None:
    async def scenario() -> int:
        async with make_sandbox() as sandbox:
            code, _ = await slime_sandbox.exec_and_wait(sandbox, cmd="sleep 60", time_budget_sec=6, tag="slow")
            return int(code)

    assert asyncio.run(scenario()) == slime_sandbox.EXIT_TIME_BUDGET_EXCEEDED


@pytest.mark.integration
def test_ensure_agent_user_makes_an_agent_that_owns_the_workdir(
    make_sandbox: Factory, slime_sandbox: ModuleType
) -> None:
    async def scenario() -> list[str]:
        async with make_sandbox() as sandbox:
            await sandbox.exec("mkdir -p /workspace", check=True)
            await slime_sandbox.ensure_agent_user(sandbox, "/workspace")
            await slime_sandbox.ensure_agent_user(sandbox, "/workspace")  # a second run does not fail
            return [
                (await sandbox.exec("id -un; echo $HOME", user="agent"))[1],
                (await sandbox.exec("stat -c %U /workspace"))[1],
                # `git config --add` appends, so each run adds another "*".
                (await sandbox.exec("git config --system --get-all safe.directory | sort -u"))[1],
            ]

    assert asyncio.run(scenario()) == ["agent\n/home/agent\n", "agent\n", "*\n"]


def _slime_harness(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import ``slime.agent.harness`` without torch.

    It imports torch through ``slime.utils.misc`` for ``SingletonMeta`` alone, so this
    stands in a module holding only that class (verbatim, ``misc.py:50-61`` at the pin).
    The harness modules are dropped from ``sys.modules`` again, so no other test sees
    them imported against the stand-in.
    """
    misc = ModuleType("slime.utils.misc")

    class SingletonMeta(type):
        _instances: ClassVar[dict[type, object]] = {}

        def __call__(cls, *args: object, **kwargs: object) -> object:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
            return cls._instances[cls]

    misc.SingletonMeta = SingletonMeta  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "slime.utils", ModuleType("slime.utils"))
    monkeypatch.setitem(sys.modules, "slime.utils.misc", misc)
    try:
        return importlib.import_module("slime.agent.harness")
    finally:
        for name in [name for name in sys.modules if name.startswith("slime.agent.harness")]:
            del sys.modules[name]


@pytest.mark.integration
def test_install_npm_cli_installs_through_the_sandbox(
    make_sandbox: Factory, slime_sandbox: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The Node and CLI tarballs are fakes with the real layout: a top directory whose bin/
    # holds node, npm and npx, and an npm package under package/.
    common = _slime_harness(monkeypatch).common
    npm = (
        "#!/bin/bash\n"
        'if [ "$1" = "--version" ]; then echo 10.0.0-fake; exit 0; fi\n'
        'mkdir -p /usr/local/lib/fake-cli && tar xzf "${@: -1}" -C /usr/local/lib/fake-cli'
        " && ln -sf /usr/local/lib/fake-cli/package/cli.sh /usr/local/bin/claude\n"
    )
    node_tarball = _tarball(
        tmp_path / "node22.tar",
        {"node-v22/bin/node": "#!/bin/bash\necho v22.0.0-fake\n", "node-v22/bin/npm": npm, "node-v22/bin/npx": npm},
    )
    cli_tarball = _tarball(tmp_path / "cli.tgz", {"package/cli.sh": "#!/bin/bash\necho fake-claude 0.0.1\n"})

    async def scenario() -> tuple[int, str, str]:
        async with make_sandbox() as sandbox:
            await common.install_npm_cli(
                sandbox,
                node_runtime=node_tarball,
                npm_package=cli_tarball,
                check_cmd="ls -la /usr/local/bin/claude && /usr/local/bin/claude --version",
            )
            return await sandbox.exec("claude --version && node --version")

    code, out, _ = asyncio.run(scenario())
    assert (code, out) == (0, "fake-claude 0.0.1\nv22.0.0-fake\n")


def _tarball(path: Path, files: dict[str, str]) -> Path:
    with tarfile.open(path, "w:gz" if path.suffix == ".tgz" else "w") as archive:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o755
            archive.addfile(info, io.BytesIO(data))
    return path


@pytest.mark.integration
@pytest.mark.slow
def test_the_real_claude_code_cli_installs_when_its_tarballs_are_given(
    make_sandbox: Factory, slime_sandbox: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Node 22 for the container's architecture and the @anthropic-ai/claude-code npm tarball,
    # named the way slime's ClaudeCodeHarness reads them. npm fetches the CLI's platform
    # package from the registry, so the container needs the network.
    if not os.environ.get("SLIME_AGENT_NODE_TARBALL") or not os.environ.get("SLIME_AGENT_CC_TARBALL"):
        pytest.skip("set SLIME_AGENT_NODE_TARBALL and SLIME_AGENT_CC_TARBALL to install the real CLI")
    harness = _slime_harness(monkeypatch).ClaudeCodeHarness()

    async def scenario() -> tuple[int, str, str]:
        async with make_sandbox(limits=ContainerLimits(cpus=1.0, memory_bytes=2**30, pids=512)) as sandbox:
            await harness.install_cli(sandbox)
            return await sandbox.exec("/usr/local/bin/claude --version && node --version")

    code, out, err = asyncio.run(scenario())
    assert code == 0, err
    assert "Claude Code" in out


# --- files ------------------------------------------------------------------------------------------------------


@pytest.mark.integration
def test_write_file_writes_text_bytes_and_a_host_file(make_sandbox: Factory, tmp_path: Path) -> None:
    raw = bytes(range(256)) * 3  # not UTF-8
    host = tmp_path / "large.bin"
    host.write_bytes(os.urandom(8 * 2**20))

    async def scenario() -> list[str]:
        async with make_sandbox() as sandbox:
            await sandbox.exec("useradd -m agent", check=True)
            await sandbox.write_file("/tmp/new/dir/text.txt", "héllo\nwörld")
            await sandbox.write_file("/tmp/raw.bin", raw)
            await sandbox.write_file("/tmp/large.bin", host)
            await sandbox.write_file("/home/agent/mine.txt", "x", user="agent")
            sums = (await sandbox.exec("sha256sum /tmp/new/dir/text.txt /tmp/raw.bin /tmp/large.bin"))[1]
            owner = (await sandbox.exec("stat -c %U /home/agent/mine.txt"))[1]
            return [line.split()[0] for line in sums.splitlines()] + [owner]

    expected = [hashlib.sha256(data).hexdigest() for data in ("héllo\nwörld".encode(), raw, host.read_bytes())]
    assert asyncio.run(scenario()) == [*expected, "agent\n"]


@pytest.mark.integration
def test_write_file_fails_where_the_user_may_not_write(make_sandbox: Factory) -> None:
    async def scenario() -> None:
        async with make_sandbox() as sandbox:
            await sandbox.exec("useradd -m agent", check=True)
            await sandbox.write_file("/root/denied.txt", "x", user="agent")

    with pytest.raises(SandboxError, match=r"cannot write /root/denied\.txt as agent"):
        asyncio.run(scenario())


@pytest.mark.integration
def test_read_file_returns_the_text_or_empty_like_slimes_sandbox(make_sandbox: Factory) -> None:
    async def scenario() -> tuple[str, str, str]:
        async with make_sandbox() as sandbox:
            await sandbox.write_file("/tmp/a.txt", "one\ntwo\n")
            return (
                await sandbox.read_file("/tmp/a.txt"),
                await sandbox.read_file("/tmp/missing.txt"),
                await sandbox.read_file("/tmp/a.txt", user="nosuchuser"),
            )

    assert asyncio.run(scenario()) == ("one\ntwo\n", "", "")


# --- exec -------------------------------------------------------------------------------------------------------


@pytest.mark.integration
def test_exec_keeps_the_streams_apart_and_passes_the_env_intact(make_sandbox: Factory) -> None:
    env = {"PLAIN": "a=b c", "QUOTES": 'it\'s "q"', "EMPTY": "", "SPACES": "  both  "}

    async def scenario() -> tuple[tuple[int, str, str], tuple[int, str, str]]:
        async with make_sandbox() as sandbox:
            streams = await sandbox.exec("echo out; echo err >&2; exit 7")
            printed = await sandbox.exec('printf \'[%s]\' "$PLAIN" "$QUOTES" "${EMPTY-unset}" "$SPACES"', env=env)
            return streams, printed

    streams, printed = asyncio.run(scenario())
    assert streams == (7, "out\n", "err\n")
    assert printed == (0, '[a=b c][it\'s "q"][][  both  ]', "")


@pytest.mark.integration
def test_exec_with_check_raises_on_a_nonzero_exit(make_sandbox: Factory) -> None:
    async def scenario() -> None:
        async with make_sandbox() as sandbox:
            await sandbox.exec("echo why >&2; exit 2", check=True)

    with pytest.raises(RuntimeError, match=r"exit=2") as raised:
        asyncio.run(scenario())
    # The command failed, not docker: a plain RuntimeError, as slime's E2B sandbox raises.
    assert not isinstance(raised.value, SandboxError)
    assert "why" in str(raised.value)


@pytest.mark.integration
def test_exec_past_its_limit_raises_and_kills_its_process_group(make_sandbox: Factory) -> None:
    async def scenario() -> tuple[float, str]:
        async with make_sandbox() as sandbox:
            started = time.monotonic()
            with pytest.raises(SandboxTimeoutError):
                await sandbox.exec("sleep 61 & sleep 62", timeout=2)
            elapsed = time.monotonic() - started
            await asyncio.sleep(1)
            # [s]leep keeps pgrep's own command line from matching itself.
            return elapsed, (await sandbox.exec("pgrep -af '[s]leep 6[12]' || true"))[1]

    elapsed, survivors = asyncio.run(scenario())
    assert elapsed < 15
    assert survivors == ""


@pytest.mark.integration
def test_exec_as_a_missing_user_is_dockers_failure(make_sandbox: Factory) -> None:
    async def scenario() -> None:
        async with make_sandbox() as sandbox:
            await sandbox.exec("true", user="nosuchuser")

    with pytest.raises(SandboxError, match="unable to find user nosuchuser"):
        asyncio.run(scenario())


# --- the container ----------------------------------------------------------------------------------------------


@pytest.mark.integration
def test_the_container_has_the_limits_and_the_host_alias(make_sandbox: Factory) -> None:
    async def scenario() -> tuple[dict[str, object], str]:
        async with make_sandbox() as sandbox:
            inspected = json.loads(docker("inspect", sandbox.sandbox_id).stdout)[0]
            hosts = (await sandbox.exec("cat /etc/hosts"))[1]
            return inspected, hosts

    inspected, hosts = asyncio.run(scenario())
    host_config = inspected["HostConfig"]
    assert host_config["NanoCpus"] == 10**9
    assert host_config["Memory"] == host_config["MemorySwap"] == 512 * 2**20
    assert host_config["PidsLimit"] == 256
    assert host_config["NetworkMode"] == "bridge"
    assert host_config["ExtraHosts"] == ["host.docker.internal:host-gateway"]
    assert host_config["SecurityOpt"] == ["no-new-privileges"]
    assert host_config["Init"] is True
    assert host_config["AutoRemove"] is True
    assert inspected["Config"]["Labels"]["honing.sandbox"] == "1"
    assert "host.docker.internal" in hosts


@pytest.mark.integration
def test_the_container_reaches_a_server_on_the_host(make_sandbox: Factory) -> None:
    # The adapter's case: a server on the host, reached through the sandbox's host alias.
    # Under colima and Docker Desktop host-gateway reaches the host's loopback; on Linux it
    # is the bridge gateway, so the server listens there. The alias is one only --add-host
    # can resolve: colima resolves host.docker.internal by itself.
    if sys.platform == "darwin":
        bind = "127.0.0.1"
    else:
        bind = docker("network", "inspect", "bridge", "--format", "{{(index .IPAM.Config 0).Gateway}}").stdout.strip()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"adapter-ok")

        def log_message(self, *args: object) -> None:
            pass

    server = http.server.ThreadingHTTPServer((bind, 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    request = (
        f"exec 3<>/dev/tcp/honing-test-host/{server.server_address[1]}"
        " && printf 'GET / HTTP/1.0\\r\\n\\r\\n' >&3 && cat <&3 | tail -n 1"
    )

    async def scenario() -> tuple[int, str, str]:
        async with make_sandbox(host_alias="honing-test-host") as sandbox:
            return await sandbox.exec(request, timeout=20)

    try:
        code, out, err = asyncio.run(scenario())
    finally:
        server.shutdown()
    assert (code, out) == (0, "adapter-ok"), err


@pytest.mark.integration
def test_a_failed_create_raises_and_leaves_nothing(image: str) -> None:
    sandbox = LocalContainerSandbox("Not A Valid Reference", limits=SMALL)

    async def scenario() -> None:
        async with sandbox:
            pass

    with pytest.raises(SandboxError, match="docker pull failed"):
        asyncio.run(scenario())
    assert not exists(sandbox.sandbox_id)


@pytest.mark.integration
def test_the_container_is_removed_on_exit_and_on_an_error(make_sandbox: Factory) -> None:
    clean, failing = make_sandbox(), make_sandbox()

    async def scenario() -> list[bool]:
        seen = []
        async with clean:
            seen.append(exists(clean.sandbox_id))
        with pytest.raises(ValueError, match="body"):
            async with failing:
                seen.append(exists(failing.sandbox_id))
                raise ValueError("body")
        return seen

    assert asyncio.run(scenario()) == [True, True]
    assert not exists(clean.sandbox_id)
    assert not exists(failing.sandbox_id)


@pytest.mark.integration
def test_the_container_is_removed_when_a_running_exec_is_cancelled(make_sandbox: Factory) -> None:
    sandbox = make_sandbox()

    async def scenario() -> None:
        async def body() -> None:
            async with sandbox:
                await sandbox.exec("sleep 60", timeout=120)

        task = asyncio.ensure_future(body())
        assert await _wait_until(lambda: exists(sandbox.sandbox_id), 60)
        await asyncio.sleep(1)  # the exec is under way
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert not exists(sandbox.sandbox_id)


@pytest.mark.integration
def test_the_container_is_removed_when_cancelled_during_creation(make_sandbox: Factory) -> None:
    sandbox = make_sandbox()

    async def scenario() -> None:
        async def body() -> None:
            async with sandbox:
                await asyncio.sleep(3600)

        task = asyncio.ensure_future(body())
        # Cancel as soon as docker has the container, whether or not `docker run` has returned.
        assert await _wait_until(lambda: exists(sandbox.sandbox_id), 60)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert not exists(sandbox.sandbox_id)


@pytest.mark.integration
def test_the_container_is_removed_through_repeated_cancellation(make_sandbox: Factory) -> None:
    sandbox = make_sandbox()

    async def scenario() -> None:
        entered = asyncio.Event()

        async def body() -> None:
            async with sandbox:
                entered.set()
                await asyncio.sleep(3600)

        task = asyncio.ensure_future(body())
        await entered.wait()
        for _ in range(4):  # the first starts the removal, the rest land while it runs
            task.cancel()
            await asyncio.sleep(0.02)
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert not exists(sandbox.sandbox_id)


@pytest.mark.integration
@pytest.mark.slow
def test_an_abandoned_container_removes_itself_after_its_lifetime(make_sandbox: Factory) -> None:
    sandbox = make_sandbox(lifetime_seconds=3)

    async def scenario() -> bool:
        await sandbox.__aenter__()  # and never __aexit__, as when the owner is killed outright
        return await _wait_until(lambda: not exists(sandbox.sandbox_id), 30)

    assert asyncio.run(scenario())


# --- against a fake CLI: exact timing, and what docker is passed ------------------------------------------------


@pytest.fixture
def fake_docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[dict[str, object]], Path]:
    """Point a sandbox's ``docker=`` at ``tests/fakes/docker_cli.py``; returns a setter for its behaviour."""
    log = tmp_path / "docker.log"
    log.touch()
    cli = tmp_path / "docker"
    cli.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{REPO_ROOT / "tests/fakes/docker_cli.py"}" "$@"\n')
    cli.chmod(0o755)
    monkeypatch.setenv("FAKE_DOCKER_LOG", str(log))

    def behave(behaviour: dict[str, object]) -> Path:
        monkeypatch.setenv("FAKE_DOCKER", json.dumps(behaviour))
        return cli

    return behave


def _log(cli: Path) -> list[dict[str, object]]:
    lines = Path(os.environ["FAKE_DOCKER_LOG"]).read_text().splitlines()
    return [json.loads(line) for line in lines]


def _calls(cli: Path, command: str) -> list[dict[str, object]]:
    return [entry for entry in _log(cli) if entry.get("argv", [""])[0] == command]  # type: ignore[index]


def _alive(pid: object) -> bool:
    assert isinstance(pid, int)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.integration  # starts subprocesses
def test_the_env_goes_through_a_private_file_and_not_the_command_line(
    fake_docker: Callable[[dict[str, object]], Path],
) -> None:
    cli = fake_docker({})
    sandbox = LocalContainerSandbox("image", docker=str(cli))

    async def scenario() -> None:
        async with sandbox:
            await sandbox.exec("true", env={"TOKEN": "s3cret value"})

    asyncio.run(scenario())
    execs = _calls(cli, "exec")
    assert len(execs) == 1
    assert execs[0]["env_file"] == "TOKEN=s3cret value\n"
    assert execs[0]["env_mode"] == 0o600
    argv = execs[0]["argv"]
    assert isinstance(argv, list)
    assert not any("s3cret" in part for part in argv)
    assert not Path(argv[argv.index("--env-file") + 1]).exists()
    assert argv[argv.index(sandbox.sandbox_id) + 1 :] == ["timeout", "--kill-after", "5", "120", "bash", "-c", "true"]


@pytest.mark.integration
def test_removal_finishes_through_repeated_cancellation(fake_docker: Callable[[dict[str, object]], Path]) -> None:
    cli = fake_docker({"rm": {"seconds": 1.5}})
    sandbox = LocalContainerSandbox("image", docker=str(cli))

    async def scenario() -> None:
        entered = asyncio.Event()

        async def body() -> None:
            async with sandbox:
                entered.set()
                await asyncio.sleep(3600)

        task = asyncio.ensure_future(body())
        await entered.wait()
        task.cancel()
        await asyncio.sleep(0.5)  # the removal is under way
        task.cancel()
        await asyncio.sleep(0.3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert {"done": "rm"} in _log(cli)


def _cancel_after(monkeypatch: pytest.MonkeyPatch, subcommand: str) -> list[asyncio.Future[object]]:
    """Cancel the owner in the same loop step as ``docker <subcommand>`` returns; append the owner to the result.

    The real step finishes first, so the cancellation lands on a call that is already
    done: the race a timer can only hit by luck.
    """
    owner: list[asyncio.Future[object]] = []
    real = sandbox_module._run

    async def run(argv: Sequence[str], **kwargs: object) -> object:
        result = await real(argv, **kwargs)  # type: ignore[arg-type]
        if argv[1] == subcommand:
            owner[0].cancel()
        return result

    monkeypatch.setattr(sandbox_module, "_run", run)
    return owner


@pytest.mark.integration
@pytest.mark.parametrize("body_raises", [False, True], ids=["clean", "raising"])
def test_a_cancellation_that_lands_as_removal_finishes_is_kept(
    fake_docker: Callable[[dict[str, object]], Path],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    body_raises: bool,
) -> None:
    # A removal that finished is a removal that succeeded: no SandboxError on a clean exit,
    # no warning, and the cancellation still reaches the owner.
    cli = fake_docker({})
    owner = _cancel_after(monkeypatch, "rm")
    sandbox = LocalContainerSandbox("image", docker=str(cli))

    async def body() -> None:
        async with sandbox:
            if body_raises:
                raise ValueError("body")

    async def scenario() -> None:
        owner.append(asyncio.ensure_future(body()))
        with pytest.raises(asyncio.CancelledError):
            await owner[0]

    with caplog.at_level(logging.WARNING, logger=sandbox_module.__name__):
        asyncio.run(scenario())
    assert {"done": "rm"} in _log(cli)
    assert [record.getMessage() for record in caplog.records] == []


@pytest.mark.integration
def test_a_cancel_during_create_waits_for_it_then_removes(fake_docker: Callable[[dict[str, object]], Path]) -> None:
    # A killed `docker create` client can leave the daemon to create the container after the
    # removal ran; one that is never started is never removed. So creation is waited out.
    cli = fake_docker({"create": {"seconds": 1.5}})
    sandbox = LocalContainerSandbox("image", docker=str(cli))

    async def scenario() -> None:
        task = asyncio.ensure_future(sandbox.__aenter__())
        await asyncio.sleep(0.5)  # creation is under way
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    log = _log(cli)
    removal = next(index for index, entry in enumerate(log) if entry.get("argv", [""])[0] == "rm")  # type: ignore[index]
    assert log.index({"done": "create"}) < removal
    assert {"done": "rm"} in log
    assert _calls(cli, "start") == []


@pytest.mark.integration
def test_without_pull_missing_a_missing_image_raises_and_nothing_is_pulled(
    fake_docker: Callable[[dict[str, object]], Path],
) -> None:
    cli = fake_docker({"image": {"exit": 1, "stderr": "No such image: image"}})
    sandbox = LocalContainerSandbox("image", docker=str(cli), pull_missing=False)

    async def scenario() -> None:
        async with sandbox:
            pass

    with pytest.raises(SandboxError, match="the image image is not present; build or pull it first"):
        asyncio.run(scenario())
    assert _calls(cli, "pull") == []
    assert _calls(cli, "create") == []


@pytest.mark.integration
def test_a_cancel_during_start_removes_by_name(fake_docker: Callable[[dict[str, object]], Path]) -> None:
    cli = fake_docker({"start": {"seconds": 30}})
    sandbox = LocalContainerSandbox("image", docker=str(cli))

    async def scenario() -> None:
        task = asyncio.ensure_future(sandbox.__aenter__())
        await _wait_until(lambda: bool(_calls(cli, "start")), 10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert [entry["argv"] for entry in _calls(cli, "rm")] == [["rm", "--force", sandbox.sandbox_id]]
    assert {"done": "rm"} in _log(cli)
    [start] = _calls(cli, "start")
    assert not _alive(start["pid"])  # the client was killed, not left running


@pytest.mark.integration
def test_a_cancel_just_after_create_leaves_no_container(
    make_sandbox: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The container exists, created but not started, which neither --rm nor its sleep removes.
    sandbox = make_sandbox()
    owner = _cancel_after(monkeypatch, "create")
    seen: list[str] = []

    async def scenario() -> None:
        owner.append(asyncio.ensure_future(sandbox.__aenter__()))
        with pytest.raises(asyncio.CancelledError):
            await owner[0]

    original_remove = LocalContainerSandbox._remove

    async def remove(self: LocalContainerSandbox, *, raise_on_failure: bool) -> None:
        # What the removal finds: proof the cancel landed between create and start.
        seen.append(docker("inspect", "--format", "{{.State.Status}}", self.sandbox_id, check=False).stdout.strip())
        await original_remove(self, raise_on_failure=raise_on_failure)

    monkeypatch.setattr(LocalContainerSandbox, "_remove", remove)
    asyncio.run(scenario())
    assert seen == ["created"]
    assert not exists(sandbox.sandbox_id)


@pytest.mark.integration
def test_a_failed_removal_raises_only_when_nothing_else_is_raised(
    fake_docker: Callable[[dict[str, object]], Path],
) -> None:
    cli = fake_docker({"rm": {"exit": 1, "stderr": "Error response from daemon: busy"}})

    async def clean() -> None:
        async with LocalContainerSandbox("image", docker=str(cli)):
            pass

    async def failing() -> None:
        async with LocalContainerSandbox("image", docker=str(cli)):
            raise ValueError("body")

    with pytest.raises(SandboxError, match="could not remove"):
        asyncio.run(clean())
    with pytest.raises(ValueError, match="body"):
        asyncio.run(failing())


@pytest.mark.integration
def test_a_hung_client_is_killed_after_the_limit(
    fake_docker: Callable[[dict[str, object]], Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The in-container timeout normally ends the command; this is the host's backstop for
    # a client that does not return (a stream held open, a stuck daemon).
    monkeypatch.setattr(sandbox_module, "_CLIENT_GRACE_SECONDS", 0.5)
    monkeypatch.setattr(sandbox_module, "_KILL_AFTER_SECONDS", 0)
    cli = fake_docker({"exec": {"seconds": 30}})

    async def scenario() -> float:
        async with LocalContainerSandbox("image", docker=str(cli)) as sandbox:
            started = time.monotonic()
            with pytest.raises(SandboxTimeoutError):
                await sandbox.exec("sleep 30", timeout=1)
            return time.monotonic() - started

    assert asyncio.run(scenario()) < 10
    [hung] = _calls(cli, "exec")
    assert not _alive(hung["pid"])


@pytest.mark.integration
@pytest.mark.parametrize(
    ("behaviour", "expected"),
    [
        ({"exit": 1, "stderr": "plain failure\n"}, (1, "", "plain failure\n")),
        ({"exit": 124}, (124, "", "")),  # exited 124 by itself, well inside its limit
    ],
)
def test_the_commands_own_exit_codes_come_back(
    fake_docker: Callable[[dict[str, object]], Path], behaviour: dict[str, object], expected: tuple[int, str, str]
) -> None:
    cli = fake_docker({"exec": behaviour})

    async def scenario() -> tuple[int, str, str]:
        async with LocalContainerSandbox("image", docker=str(cli)) as sandbox:
            return await sandbox.exec("anything", timeout=100)

    assert asyncio.run(scenario()) == expected


@pytest.mark.integration
def test_a_daemon_error_is_the_sandboxs_failure(fake_docker: Callable[[dict[str, object]], Path]) -> None:
    cli = fake_docker({"exec": {"exit": 1, "stderr": "Error response from daemon: No such container: x\n"}})

    async def scenario() -> None:
        async with LocalContainerSandbox("image", docker=str(cli)) as sandbox:
            await sandbox.exec("true")

    with pytest.raises(SandboxError, match="No such container"):
        asyncio.run(scenario())


@pytest.mark.integration
@pytest.mark.parametrize(
    "behaviour",
    [
        # Docker 28.0.4 (CI): exit 126, the bare message on stdout.
        {"exit": 126, "stdout": "unable to find user nosuchuser: no matching entries in passwd file\n"},
        # Docker 29.5.2: exit 1, the daemon's prefix on stderr.
        {
            "exit": 1,
            "stderr": "Error response from daemon: unable to find user nosuchuser:"
            " no matching entries in passwd file\n",
        },
    ],
    ids=["docker-28", "docker-29"],
)
def test_a_missing_exec_user_is_the_sandboxs_failure_in_either_wording(
    fake_docker: Callable[[dict[str, object]], Path], behaviour: dict[str, object]
) -> None:
    cli = fake_docker({"exec": behaviour})

    async def scenario() -> None:
        async with LocalContainerSandbox("image", docker=str(cli)) as sandbox:
            await sandbox.exec("true", user="nosuchuser")

    with pytest.raises(SandboxError, match="unable to find user nosuchuser"):
        asyncio.run(scenario())


# --- arguments --------------------------------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "env", [{"BAD NAME": "x"}, {"1X": "x"}, {"X": "line\nbreak"}, {"X": "carriage\rreturn"}, {"X": "nul\0"}]
)
def test_an_env_docker_cannot_carry_is_refused(env: dict[str, str]) -> None:
    async def scenario() -> None:
        await LocalContainerSandbox("image").exec("true", env=env)

    with pytest.raises(ValueError, match=r"environment variable name|line break"):
        asyncio.run(scenario())


@pytest.mark.unit
@pytest.mark.parametrize("timeout", [0, -1])
def test_a_limit_that_is_not_positive_is_refused(timeout: int) -> None:
    # coreutils' timeout reads 0 as no limit at all.
    with pytest.raises(ValueError, match="timeout must be positive"):
        asyncio.run(LocalContainerSandbox("image").exec("true", timeout=timeout))


@pytest.mark.unit
@pytest.mark.parametrize("limits", [{"cpus": 0}, {"memory_bytes": 2**20}, {"pids": 0}], ids=["cpus", "memory", "pids"])
def test_limits_must_be_positive(limits: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="limits must be positive"):
        ContainerLimits(**limits)  # type: ignore[arg-type]


@pytest.mark.unit
def test_the_lifetime_must_be_positive() -> None:
    with pytest.raises(ValueError, match="lifetime_seconds"):
        LocalContainerSandbox("image", lifetime_seconds=0)


@pytest.mark.unit
def test_each_sandbox_has_its_own_name_before_it_exists() -> None:
    first, second = LocalContainerSandbox("image"), LocalContainerSandbox("image")
    assert first.sandbox_id != second.sandbox_id
    assert first.sandbox_id.startswith("honing-sandbox-")
    assert first.create_argv()[first.create_argv().index("--name") + 1] == first.sandbox_id
