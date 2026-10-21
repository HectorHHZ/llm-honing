"""The slime agent worker end to end: slime's real adapter and harness, a real container, a fake model and CLI.

Each test runs :class:`~honing.backends.slime.SlimeEpisodeRunner` with the real worker
in the slime agent environment (``scripts/setup-slime-agent-env.sh``, found through
``HONING_SLIME_AGENT_ENV``), on Docker. The policy is ``tests/fakes/sglang_generate.py``,
the tokenizer ``tests/fakes/byte_tokenizer``, and the CLI ``tests/fakes/claude_code/claude``,
installed by slime's own ``install_cli`` from fake Node and CLI tarballs. The real
Claude Code CLI against this path is the next pull request's test.

They skip without Docker or the environment; ``HONING_REQUIRE_DOCKER=1`` and
``HONING_REQUIRE_SLIME_AGENT=1`` turn the skips into failures.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import shutil
import socket
import subprocess
import tarfile
import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path

import pytest

from honing.backends.slime import (
    AgentEpisodeSettings,
    AgentWorkerError,
    ContainerLimits,
    SlimeAgentDeliveryProbe,
    SlimeEnvironment,
    SlimeEpisodeRunner,
)
from honing.backends.slime import agent_episode as files
from honing.backends.slime import agent_worker as worker
from honing.core import EpisodeOutcome, HarnessFile, HarnessSpec, PolicyEndpoint, PolicyRef, Task
from honing.harness.graders import CommandGrader
from honing.harness.tasks import load_task_suite
from tests.conftest import REPO_ROOT, require
from tests.fakes.harness import StaticPolicyEndpoints
from tests.fakes.model import FakeModelSource, FakeWorld
from tests.fakes.sglang_generate import FakeSGLang, agent_script, write_call

SANDBOX_DOCKERFILE = REPO_ROOT / "honing" / "backends" / "slime" / "sandbox.Dockerfile"
CLI_DIR = REPO_ROOT / "tests" / "fakes" / "claude_code"
TOKENIZER = REPO_ROOT / "tests" / "fakes" / "byte_tokenizer"
SUITE = load_task_suite(REPO_ROOT / "examples" / "tasks" / "python-bugfix" / "manifest.jsonl")
FIX = (REPO_ROOT / "examples" / "tasks" / "python-bugfix" / "clamp" / "reference" / "clamp.py").read_text()
SNAPSHOT = FakeModelSource(FakeWorld()).materialize(Path("unused"))
POLICY = PolicyRef(SNAPSHOT.checkpoint, SNAPSHOT.architecture, "master")
TASK = Task("clamp", SUITE.entry("clamp").prompt)
HARNESS = HarnessSpec(
    "claude_code",
    (
        ("CLAUDE.md", HarnessFile("instructions", "MARKER-INSTRUCTIONS: fix the bug and say so.\n")),
        ("skills/review/SKILL.md", HarnessFile("skill", "---\nname: review\n---\nReview.\n")),
        ("skills/lint.md", HarnessFile("skill", "---\nname: lint\n---\nLint.\n")),
        ("commands/fix.md", HarnessFile("command", "Fix it.\n")),
        ("settings.json", HarnessFile("settings", '{"markerSetting": true}\n')),
    ),
)
SOLVES = agent_script(write_call("/workspace/clamp.py", FIX), "Fixed clamp.")


def docker(*argv: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *argv], capture_output=True, text=True, check=check)


@pytest.fixture(scope="module")
def agent_env() -> Path:
    location = os.environ.get("HONING_SLIME_AGENT_ENV")
    if not location or not (Path(location) / "env" / "bin" / "python").is_file():
        require(
            "slime_agent",
            "HONING_SLIME_AGENT_ENV is not a slime agent environment; run scripts/setup-slime-agent-env.sh",
        )
    return Path(str(location))


@pytest.fixture(scope="module")
def image() -> str:
    """The sandbox image with python3, which the fake CLI runs on."""
    if shutil.which("docker") is None or docker("info", check=False).returncode != 0:
        require("docker", "no docker daemon is reachable")
    base = f"honing-sandbox-test:{hashlib.sha256(SANDBOX_DOCKERFILE.read_bytes()).hexdigest()[:12]}"
    if docker("image", "inspect", base, check=False).returncode != 0:
        docker("build", "--quiet", "--tag", base, "--file", str(SANDBOX_DOCKERFILE), str(SANDBOX_DOCKERFILE.parent))
    digest = hashlib.sha256(SANDBOX_DOCKERFILE.read_bytes() + (CLI_DIR / "Dockerfile").read_bytes()).hexdigest()
    tag = f"honing-agent-test:{digest[:12]}"
    if docker("image", "inspect", tag, check=False).returncode != 0:
        docker("build", "--quiet", "--build-arg", f"BASE={base}", "--tag", tag, str(CLI_DIR))
    return tag


def _tarball(path: Path, entries: Mapping[str, str]) -> Path:
    with tarfile.open(path, "w:gz" if path.suffix == ".tgz" else "w") as archive:
        for name, text in entries.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o755
            archive.addfile(info, io.BytesIO(data))
    return path


@pytest.fixture(scope="module")
def cli_tarballs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """Node and CLI tarballs in the layout slime's install_npm_cli unpacks; npm links the fake CLI."""
    directory = tmp_path_factory.mktemp("cli")
    npm = (
        "#!/bin/bash\n"
        'if [ "$1" = "--version" ]; then echo 10.0.0-fake; exit 0; fi\n'
        'mkdir -p /usr/local/lib/fake-cli && tar xzf "${@: -1}" -C /usr/local/lib/fake-cli'
        " && ln -sf /usr/local/lib/fake-cli/package/claude /usr/local/bin/claude\n"
    )
    node = _tarball(
        directory / "node22.tar",
        {"node-v22/bin/node": "#!/bin/bash\necho v22.0.0-fake\n", "node-v22/bin/npm": npm, "node-v22/bin/npx": npm},
    )
    cli = _tarball(directory / "cli.tgz", {"package/claude": (CLI_DIR / "claude").read_text()})
    return {"SLIME_AGENT_NODE_TARBALL": str(node), "SLIME_AGENT_CC_TARBALL": str(cli)}


@pytest.fixture
def no_containers_left() -> Iterator[None]:
    yield
    left = docker("ps", "--all", "--filter", "name=honing-episode-", "--format", "{{.Names}}").stdout.split()
    for name in left:
        docker("rm", "--force", name, check=False)
    assert left == [], f"containers left behind: {left}"


Make = Callable[..., SlimeEpisodeRunner]


@pytest.fixture
def make_runner(
    agent_env: Path, image: str, cli_tarballs: dict[str, str], tmp_path: Path, no_containers_left: None
) -> Make:
    def make(
        base_url: str, *, environ: Mapping[str, str | None] | None = None, **overrides: object
    ) -> SlimeEpisodeRunner:
        settings: dict[str, object] = {
            "image": image,
            "limits": ContainerLimits(cpus=1.0, memory_bytes=512 * 2**20, pids=256),
            "time_budget_seconds": 120,
            "setup_timeout_seconds": 300,
            "worker_margin_seconds": 120,
            **overrides,
        }
        endpoint = PolicyEndpoint(base_url=base_url, tokenizer_location=str(TOKENIZER), served_model="fake")
        return SlimeEpisodeRunner(
            suite=SUITE,
            endpoints=StaticPolicyEndpoints({POLICY.checkpoint.checksum: endpoint}),
            environment=SlimeEnvironment(
                honing_root=REPO_ROOT,
                python=str(agent_env / "env" / "bin" / "python"),
                python_path=(str(agent_env / "slime"),),
            ),
            settings=AgentEpisodeSettings(**settings),  # type: ignore[arg-type]
            episodes=tmp_path / "episodes",
            environ={
                name: value
                for name, value in {**os.environ, **cli_tarballs, "HF_HUB_OFFLINE": "1", **(environ or {})}.items()
                if value is not None
            },
        )

    return make


def run(runner: SlimeEpisodeRunner) -> EpisodeOutcome:
    return asyncio.run(runner.run(HARNESS, TASK, POLICY))


def result_of(outcome: EpisodeOutcome) -> dict[str, object]:
    assert outcome.workspace_location is not None
    return json.loads((Path(outcome.workspace_location).parent / files.RESULT_FILE).read_text())


def log_of(tmp_path: Path) -> str:
    (directory,) = (tmp_path / "episodes").iterdir()
    return (directory / files.LOG_FILE).read_text()


@pytest.mark.integration
@pytest.mark.slow
def test_an_episode_runs_through_slime_and_its_workspace_grades(make_runner: Make, tmp_path: Path) -> None:
    with FakeSGLang(SOLVES) as model:
        runner = make_runner(f"{model.url}/v1", max_new_tokens=1234, cli_extra_args="--fake-flag 'two words'")
        outcome = run(runner)
    assert (outcome.exit_code, outcome.final_output) == (0, "Fixed clamp.")
    out = Path(str(outcome.workspace_location))
    assert (out / "clamp.py").read_text() == FIX.strip()
    # Neither slime's directory nor the placed harness files come back with the workspace.
    assert sorted(path.name for path in out.iterdir()) == sorted(
        path.name for path in SUITE.entry("clamp").workspace.iterdir()
    )
    assert asyncio.run(CommandGrader(SUITE).grade(TASK, outcome)).score == 1.0

    # What the model saw: the instructions, the skills, the command, the settings, on every call.
    assert len(model.requests) == 2
    for request in model.requests:
        assert "MARKER-INSTRUCTIONS" in request.prompt
        assert "[skills] lint review" in request.prompt
        assert "[commands] fix.md" in request.prompt
        assert "[settings] markerSetting" in request.prompt
        assert request.sampling_params["max_new_tokens"] == 1234
    assert SlimeAgentDeliveryProbe().delivered(outcome, HARNESS) == {path: True for path, _ in HARNESS.files}

    result = result_of(outcome)
    assert (result["status"], result["agent_stopped_at_deadline"]) == ("completed", False)
    assert result["agent_processes_stopped"] == 0
    assert (result["turns"], result["upstream_errors"]) == (2, 0)
    # Token ids are bytes here, so the counts are the prompts' and the replies' byte lengths.
    assert result["input_tokens"] == sum(len(request.prompt.encode()) for request in model.requests)
    assert result["output_tokens"] == len(write_call("/workspace/clamp.py", FIX).encode()) + len(b"Fixed clamp.")
    assert result["cli_version"] == "0.0.1-fake (Claude Code)"
    assert result["model_info"] == {"model_path": "fake-byte-model", "tokenizer_path": "fake-byte-model"}
    assert "cost" not in json.dumps(result)

    transcript = Path(str(outcome.transcript_location)).read_text().splitlines()
    assert any(line.startswith("[fake-claude:unrecognized_model]") for line in transcript)
    init = json.loads(transcript[0])
    assert init["argv"][-2:] == ["--fake-flag", "two words"]
    log = log_of(tmp_path)
    assert "Task was destroyed" not in log and "Traceback" not in log, log
    # Successful requests are not logged, and the worker module runs once (no runpy double import).
    assert "aiohttp.access" not in log and "RuntimeWarning" not in log, log


@pytest.mark.integration
@pytest.mark.slow
def test_an_agent_past_its_time_budget_is_stopped_and_graded_on_what_it_left(
    make_runner: Make, tmp_path: Path
) -> None:
    with FakeSGLang(SOLVES, delay_seconds=40) as model:
        outcome = run(make_runner(model.url, time_budget_seconds=8))
    assert outcome.exit_code == -1
    result = result_of(outcome)
    assert (result["status"], result["agent_stopped_at_deadline"]) == ("time_budget_exceeded", True)
    assert asyncio.run(CommandGrader(SUITE).grade(TASK, outcome)).score < 1.0
    log = log_of(tmp_path)
    assert "Task was destroyed" not in log and "Traceback" not in log, log


@pytest.mark.integration
@pytest.mark.slow
def test_the_clis_own_exit_code_is_the_outcome(make_runner: Make) -> None:
    with FakeSGLang(agent_script(write_call("/workspace/clamp.py", FIX), "EXIT 3")) as model:
        outcome = run(make_runner(model.url))
    assert (outcome.exit_code, outcome.final_output) == (3, "EXIT 3")


@pytest.mark.integration
@pytest.mark.slow
def test_a_policy_server_that_fails_a_call_makes_the_episode_raise(make_runner: Make) -> None:
    with (
        FakeSGLang(SOLVES, status=500) as model,
        pytest.raises(AgentWorkerError, match="the policy server failed 1 of"),
    ):
        run(make_runner(model.url))


@pytest.mark.integration
def test_an_unreachable_policy_server_raises_before_any_sandbox(make_runner: Make) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with pytest.raises(AgentWorkerError, match="did not answer /get_model_info"):
        run(make_runner(f"http://127.0.0.1:{port}"))


@pytest.mark.integration
@pytest.mark.slow
def test_an_image_without_the_cli_raises_when_the_install_is_off(make_runner: Make) -> None:
    with FakeSGLang(SOLVES) as model, pytest.raises(AgentWorkerError, match="/usr/local/bin/claude does not run"):
        run(make_runner(model.url, install_cli=False))


@pytest.mark.integration
@pytest.mark.slow
def test_a_cancelled_episode_stops_the_worker_which_removes_its_sandbox(make_runner: Make, tmp_path: Path) -> None:
    # The runner sends SIGTERM; the worker cancels its episode, so the sandbox goes on the way out and the
    # worker still writes result.json. The runner's own `docker rm` afterwards is only a backstop.
    with FakeSGLang(SOLVES, delay_seconds=60) as model:
        runner = make_runner(model.url)

        async def scenario() -> None:
            task = asyncio.ensure_future(runner.run(HARNESS, TASK, POLICY))
            for _ in range(600):
                if model.requests:
                    break
                await asyncio.sleep(0.1)
            assert model.requests, "the agent never called the model"
            task.cancel()
            await task

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(scenario())
    (directory,) = (tmp_path / "episodes").iterdir()
    result = json.loads((directory / files.RESULT_FILE).read_text())
    assert (result["status"], result["error"]) == ("infra_error", "the worker was terminated")
    assert "could not remove sandbox" not in (directory / files.LOG_FILE).read_text()


def docker_wrapper(directory: Path, cases: str) -> dict[str, str]:
    """A ``docker`` first on the worker's ``PATH`` that runs the bash ``cases`` on ``$1``, then the real docker."""
    real = shutil.which("docker")
    assert real is not None
    bin_dir = directory / "bin"
    bin_dir.mkdir()
    wrapper = bin_dir / "docker"
    wrapper.write_text(f'#!/bin/bash\ncase "$1" in\n{cases}\nesac\nexec "{real}" "$@"\n')
    wrapper.chmod(0o755)
    return {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}


@pytest.mark.integration
def test_the_worker_never_pulls_a_missing_image(make_runner: Make, tmp_path: Path) -> None:
    # The runner's check finds the image; it is gone by the time the worker's sandbox looks.
    pulls, seen = tmp_path / "pulls", tmp_path / "inspected"
    cases = (
        f'  pull) echo "$*" >> "{pulls}"; exit 1 ;;\n'
        f'  image) if [ -e "{seen}" ]; then echo "No such image" >&2; exit 1; fi; touch "{seen}" ;;'
    )
    with FakeSGLang(SOLVES) as model, pytest.raises(AgentWorkerError, match="is not present; build or pull it first"):
        run(make_runner(model.url, environ=docker_wrapper(tmp_path, cases)))
    assert not pulls.exists()


@pytest.mark.integration
@pytest.mark.slow
def test_creating_the_sandbox_counts_against_the_setup_timeout(make_runner: Make, tmp_path: Path) -> None:
    # A slow `docker create` stands in for a slow daemon. Without the CLI the rest of the setup fails within
    # seconds ("does not run"), so only a timeout that covers creating the sandbox reports running past it.
    slow = docker_wrapper(tmp_path, "  create) sleep 30 ;;")
    with FakeSGLang(SOLVES) as model, pytest.raises(AgentWorkerError, match="setting up the sandbox ran past 10s"):
        run(make_runner(model.url, setup_timeout_seconds=10, install_cli=False, environ=slow))
    assert model.requests == []


@pytest.mark.integration
@pytest.mark.parametrize(
    ("environ", "message"),
    [
        (
            {"SLIME_AGENT_NODE_TARBALL": None, "SLIME_AGENT_CC_TARBALL": None},
            "SLIME_AGENT_NODE_TARBALL is not set; SLIME_AGENT_CC_TARBALL is not set",
        ),
        ({"SLIME_AGENT_CC_TARBALL": "/no/such/cli.tgz"}, "SLIME_AGENT_CC_TARBALL=/no/such/cli.tgz is not a file"),
    ],
)
def test_missing_cli_tarballs_raise_before_any_sandbox(
    make_runner: Make, tmp_path: Path, environ: Mapping[str, str | None], message: str
) -> None:
    # Slime's install_cli reads the tarballs with os.environ[...] once the sandbox is up; the worker
    # checks them first, so the episode fails at once with the names, and no container is made.
    started = time.monotonic()
    with FakeSGLang(SOLVES) as model, pytest.raises(AgentWorkerError, match=f"cli.install is set, but {message}"):
        run(make_runner(model.url, environ=environ))
    assert time.monotonic() - started < 30
    (directory,) = (tmp_path / "episodes").iterdir()
    assert not (directory / files.STATE_FILE).exists()
    assert model.requests == []


@pytest.mark.integration
@pytest.mark.slow
def test_an_agent_over_its_turn_cap_is_an_outcome_not_an_infrastructure_fault(make_runner: Make) -> None:
    # The adapter answers the call past max_turns_per_sid with a 429, which the fake CLI, like the
    # real one once it stops retrying, reports as an error and exits nonzero. That is the agent
    # running out of turns: the episode is graded, and nothing counts it as the policy server failing.
    with FakeSGLang(SOLVES) as model:
        outcome = run(make_runner(model.url, max_turns_per_sid=1))
    assert (outcome.exit_code, outcome.final_output) == (1, "API 429")
    assert len(model.requests) == 1
    result = result_of(outcome)
    assert (result["status"], result["turns"], result["upstream_errors"]) == ("completed", 1, 0)
    assert asyncio.run(CommandGrader(SUITE).grade(TASK, outcome)).score == 1.0


@pytest.mark.integration
@pytest.mark.slow
def test_processes_the_agent_left_running_are_stopped_before_the_copy_out(make_runner: Make, tmp_path: Path) -> None:
    # The CLI exits 0 but leaves a process that keeps rewriting a file. Were it still running, tar would
    # see the file change while reading it and fail, which the worker must report as its own fault.
    with FakeSGLang(agent_script(write_call("/workspace/clamp.py", FIX), "BACKGROUND")) as model:
        outcome = run(make_runner(model.url))
    assert (outcome.exit_code, outcome.final_output, outcome.failure) == (0, "BACKGROUND", None)
    assert (Path(str(outcome.workspace_location)) / "background.bin").is_file()
    result = result_of(outcome)
    assert (result["status"], result["agent_stopped_at_deadline"]) == ("completed", False)
    # The leftover writer: its shell, and the head it may be running at that moment.
    assert result["agent_processes_stopped"] >= 1
    assert asyncio.run(CommandGrader(SUITE).grade(TASK, outcome)).score == 1.0


@pytest.mark.integration
@pytest.mark.slow
def test_a_prompt_past_the_models_context_is_an_outcome_not_an_infrastructure_fault(make_runner: Make) -> None:
    # The server does not report its longest input, so slime sends the prompt and SGLang answers with its 400.
    # The worker hands the CLI an empty reply cut off at the length limit, as slime does when it stops first.
    with FakeSGLang(SOLVES, context_len=64, server_info=False) as model:
        outcome = run(make_runner(model.url))
    assert len(model.requests) == 1
    result = result_of(outcome)
    assert (result["status"], result["upstream_errors"], result["context_overflows"]) == ("completed", 0, 1)
    assert (result["turns"], result["max_context_tokens"]) == (0, 0)
    assert (outcome.exit_code, outcome.failure) == (0, None)
    assert asyncio.run(CommandGrader(SUITE).grade(TASK, outcome)).score < 1.0


@pytest.mark.integration
@pytest.mark.slow
def test_the_servers_longest_input_stops_the_agent_before_sglang_refuses(make_runner: Make) -> None:
    with FakeSGLang(SOLVES, context_len=64) as model:
        outcome = run(make_runner(model.url))
    assert model.requests == []
    result = result_of(outcome)
    assert (result["status"], result["context_overflows"], result["max_context_tokens"]) == ("completed", 0, 63)
    assert outcome.exit_code == 0


@pytest.mark.integration
@pytest.mark.slow
def test_the_context_limit_also_caps_the_reply_so_the_request_fits(make_runner: Make) -> None:
    # A limit well past the prompts: slime lowers max_new_tokens to what is left, so SGLang never refuses.
    with FakeSGLang(SOLVES, context_len=20_000) as model:
        outcome = run(make_runner(model.url, max_new_tokens=19_000))
    assert asyncio.run(CommandGrader(SUITE).grade(TASK, outcome)).score == 1.0
    for request in model.requests:
        assert len(request.prompt.encode()) + request.sampling_params["max_new_tokens"] <= 19_999
    assert result_of(outcome)["context_overflows"] == 0


@pytest.mark.integration
@pytest.mark.slow
def test_a_workspace_over_the_cap_grades_zero_with_the_reason(make_runner: Make, tmp_path: Path) -> None:
    with FakeSGLang(SOLVES) as model:
        outcome = run(make_runner(model.url, copy_out_max_bytes=2048))
    assert (outcome.exit_code, outcome.workspace_location) == (0, None)
    assert outcome.failure == "the agent left a workspace larger than 2048 bytes as a tar; it was not copied back"
    grade = asyncio.run(CommandGrader(SUITE).grade(TASK, outcome))
    assert (grade.score, grade.feedback) == (0.0, outcome.failure)
    (directory,) = (tmp_path / "episodes").iterdir()
    result = json.loads((directory / files.RESULT_FILE).read_text())
    assert (result["status"], result["workspace"]) == ("completed", None)


@pytest.mark.integration
@pytest.mark.slow
def test_files_that_differ_only_in_case_grade_zero_with_the_reason(make_runner: Make) -> None:
    # The task already has clamp.py; Clamp.py next to it would overwrite it on a case-insensitive disk.
    with FakeSGLang(agent_script(write_call("/workspace/Clamp.py", FIX), "Fixed.")) as model:
        outcome = run(make_runner(model.url))
    assert outcome.workspace_location is None
    assert "('Clamp.py', 'clamp.py')" in str(outcome.failure)
    assert asyncio.run(CommandGrader(SUITE).grade(TASK, outcome)).score == 0.0


@pytest.mark.integration
def test_a_tar_that_fails_fails_the_copy_out_command(image: str) -> None:
    # No /workspace: tar exits 2. With pipefail the pipeline says so; without it, base64's 0 would hide it.
    # The copy-out runs as root, as the sandbox's exec does by default.
    command = worker.copy_out_command((), 1024)
    failed = docker("run", "--rm", "--user", "root", "--entrypoint", "bash", image, "-c", command, check=False)
    assert failed.returncode == 2, failed
    ok = docker(
        "run",
        "--rm",
        "--user",
        "root",
        "--entrypoint",
        "bash",
        image,
        "-c",
        f"mkdir /workspace; {command}",
        check=False,
    )
    assert ok.returncode == 0, ok
