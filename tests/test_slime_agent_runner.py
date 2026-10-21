"""SlimeEpisodeRunner with a fake worker: harness placement, the episode files, the worker's environment and deadline.

No Slime and no Docker: the worker is ``tests/fakes/slime_agent_worker.py``, which
reads ``episode.json`` with the real worker's reader, and Docker is
``tests/fakes/docker_cli.py``. ``tests/test_slime_agent_worker.py`` runs the real
worker; the worker's helpers that need no Slime are tested at the end of this file.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import sys
import tarfile
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest

from honing.backends.slime import (
    AgentEpisodeSettings,
    AgentWorkerError,
    Placement,
    SlimeAgentDeliveryProbe,
    SlimeEnvironment,
    SlimeEpisodeRunner,
)
from honing.backends.slime import agent_episode as files
from honing.backends.slime import agent_runner as runner_module
from honing.backends.slime import agent_worker as worker
from honing.backends.slime import place_harness
from honing.core import ContractError, HarnessFile, HarnessSpec, PolicyEndpoint, PolicyRef, Task
from honing.e2e.conformance import check_runner
from honing.harness.graders import CommandGrader
from honing.harness.tasks import TaskSuite, load_task_suite
from tests.conftest import REPO_ROOT
from tests.fakes.harness import StaticPolicyEndpoints
from tests.fakes.model import FakeModelSource, FakeWorld

EXAMPLE_SUITE = REPO_ROOT / "examples" / "tasks" / "python-bugfix" / "manifest.jsonl"
FAKE_DOCKER = REPO_ROOT / "tests" / "fakes" / "docker_cli.py"
SNAPSHOT = FakeModelSource(FakeWorld()).materialize(Path("unused"))
POLICY = PolicyRef(SNAPSHOT.checkpoint, SNAPSHOT.architecture, "master")
ENDPOINT = PolicyEndpoint(base_url="http://sglang.test:30000/v1/", tokenizer_location="/models/tok", served_model="m")
# Fixed so that a later test cannot pass because an earlier one changed os.environ.
ENVIRON = {
    "PATH": os.environ["PATH"],
    "HOME": "/home/tester",
    "DOCKER_HOST": "unix:///tmp/fake-docker.sock",
    "SECRET_TOKEN": "must-not-reach-the-worker",
}
# The fakes' own variables, passed through to the worker and the runner's docker as well.
FAKE_NAMES = ("FAKE_AGENT_WORKER", "FAKE_DOCKER_LOG", "FAKE_DOCKER")


def spec(*files: tuple[str, str, str]) -> HarnessSpec:
    return HarnessSpec("claude_code", tuple((path, HarnessFile(kind, content)) for path, kind, content in files))  # type: ignore[arg-type]


EVERY_KIND = spec(
    ("instructions/main.md", "instructions", "Be careful.\n"),
    ("skills/review/SKILL.md", "skill", "---\nname: review\n---\nReview.\n"),
    ("skills/lint.md", "skill", "---\nname: lint\n---\nLint.\n"),
    ("commands/fix.md", "command", "Fix it.\n"),
    ("settings.json", "settings", '{"model": "x"}\n'),
)


def suite(tmp_path: Path, files: Mapping[str, str] | None = None) -> TaskSuite:
    workspace = tmp_path / "suite" / "t1" / "workspace"
    workspace.mkdir(parents=True)
    for name, text in (files or {"main.py": "print('hi')\n"}).items():
        (workspace / name).parent.mkdir(parents=True, exist_ok=True)
        (workspace / name).write_text(text)
    line = {"task_id": "t1", "prompt": "fix main.py", "workspace": "t1/workspace", "grader": ["true"], "split": "val"}
    manifest = tmp_path / "suite" / "manifest.jsonl"
    manifest.write_text(json.dumps(line) + "\n")
    return load_task_suite(manifest)


def make_runner(
    tmp_path: Path,
    *,
    knobs: Mapping[str, object] | None = None,
    task_suite: TaskSuite | None = None,
    settings: AgentEpisodeSettings | None = None,
    grace: float = 5.0,
    environ: Mapping[str, str] = ENVIRON,
    episodes: Path | None = None,
    docker: Mapping[str, object] | None = None,
) -> SlimeEpisodeRunner:
    """A runner whose worker is the fake one and whose ``docker``, first on the worker's ``PATH``, is the fake CLI.

    ``docker`` is the fake CLI's behaviour (``$FAKE_DOCKER``); it logs to ``tmp_path / "docker.log"``.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    cli = bin_dir / "docker"
    cli.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE_DOCKER}" "$@"\n')
    cli.chmod(0o755)
    fake_docker = {
        "PATH": f"{bin_dir}{os.pathsep}{environ['PATH']}",
        "FAKE_DOCKER_LOG": str(tmp_path / "docker.log"),
        "FAKE_DOCKER": json.dumps(docker or {}),
    }
    return SlimeEpisodeRunner(
        suite=task_suite or suite(tmp_path),
        endpoints=StaticPolicyEndpoints({POLICY.checkpoint.checksum: ENDPOINT}),
        environment=SlimeEnvironment(honing_root=REPO_ROOT, python=sys.executable, python_path=()),
        settings=settings or AgentEpisodeSettings(image="honing-agent:test"),
        episodes=tmp_path / "episodes" if episodes is None else episodes,
        passthrough=(*runner_module.PASSTHROUGH, *FAKE_NAMES),
        environ={**environ, **fake_docker, "FAKE_AGENT_WORKER": json.dumps(knobs or {})},
        worker_module="tests.fakes.slime_agent_worker",
        terminate_grace_seconds=grace,
    )


def run(runner: SlimeEpisodeRunner, harness: HarnessSpec = EVERY_KIND, task_id: str = "t1"):  # type: ignore[no-untyped-def]
    return asyncio.run(runner.run(harness, Task(task_id, "fix main.py"), POLICY))


def episode_dir(tmp_path: Path) -> Path:
    (directory,) = (tmp_path / "episodes").iterdir()
    return directory


@pytest.fixture
def docker_log(tmp_path: Path) -> Path:
    return tmp_path / "docker.log"


def removed(log: Path) -> list[list[str]]:
    if not log.exists():
        return []
    entries = [json.loads(line) for line in log.read_text().splitlines()]
    return [entry["argv"] for entry in entries if "argv" in entry and entry["argv"][0] == "rm"]


# --- placement ---------------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_every_supported_kind_goes_where_claude_code_reads_it() -> None:
    assert place_harness(EVERY_KIND) == (
        Placement("CLAUDE.md", "Be careful.\n", ("instructions/main.md",)),
        Placement(".claude/settings.json", '{"model": "x"}\n', ("settings.json",)),
        # Skills and commands in path order.
        Placement(".claude/commands/fix.md", "Fix it.\n", ("commands/fix.md",)),
        Placement(".claude/skills/lint/SKILL.md", "---\nname: lint\n---\nLint.\n", ("skills/lint.md",)),
        Placement(".claude/skills/review/SKILL.md", "---\nname: review\n---\nReview.\n", ("skills/review/SKILL.md",)),
    )


@pytest.mark.unit
def test_several_instruction_files_become_one_claude_md_in_path_order() -> None:
    placed = place_harness(spec(("b.md", "instructions", "second\n\n"), ("a.md", "instructions", "first")))
    assert placed == (Placement("CLAUDE.md", "first\n\nsecond\n", ("a.md", "b.md")),)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("harness", "message"),
    [
        (spec(("hooks/pre.sh", "hook", "echo\n")), r"not \['hook'\]"),
        (HarnessSpec("codex"), "not 'codex'"),
        (spec(("a.json", "settings", "{}"), ("b.json", "settings", "{}")), "one project settings file"),
        (spec(("s.json", "settings", "[1, 2]")), "must hold a JSON object"),
        (spec(("s.json", "settings", "not json")), "must hold a JSON object"),
        (spec(("skills/.hidden.md", "skill", "x")), "cannot load"),
        (spec(("skills/a/SKILL.md", "skill", "x"), ("more/a.md", "skill", "y")), r"several harness files go to"),
    ],
)
def test_a_harness_claude_code_cannot_take_is_refused(harness: HarnessSpec, message: str) -> None:
    with pytest.raises(ContractError, match=message):
        place_harness(harness)


@pytest.mark.unit
def test_the_native_sglang_root_drops_the_openai_suffix() -> None:
    assert runner_module.sglang_root(ENDPOINT) == "http://sglang.test:30000"
    plain = PolicyEndpoint(base_url="http://h:1", tokenizer_location="t", served_model="m")
    assert runner_module.sglang_root(plain) == "http://h:1"
    decisions = PolicyEndpoint(base_url="http://h:1", tokenizer_location="t", served_model="m", runtime="jev_decision")
    with pytest.raises(ContractError, match="needs an SGLang server"):
        runner_module.sglang_root(decisions)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("files", "where"),
    [
        ({"CLAUDE.md": "the task's own\n"}, "CLAUDE.md"),
        ({".harness/x": "slime's directory\n"}, ".harness"),
        ({".claude": "a file where a directory goes\n"}, ".claude"),
    ],
)
def test_a_workspace_that_already_has_a_placed_path_is_refused(
    tmp_path: Path, files: Mapping[str, str], where: str
) -> None:
    runner = make_runner(tmp_path, task_suite=suite(tmp_path, files))
    with pytest.raises(ContractError, match=f"already has '{where}'"):
        run(runner)
    assert not (tmp_path / "episodes").exists()


@pytest.mark.unit
def test_a_workspace_with_other_files_under_dot_claude_is_accepted(tmp_path: Path) -> None:
    runner = make_runner(tmp_path, task_suite=suite(tmp_path, {".claude/agents/x.md": "x\n", "main.py": "x\n"}))
    assert run(runner).exit_code == 0


# --- one episode -------------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_an_episode_writes_the_files_the_worker_reads_and_returns_its_outcome(tmp_path: Path) -> None:
    outcome = run(make_runner(tmp_path))
    directory = episode_dir(tmp_path)
    record = json.loads((directory / files.EPISODE_FILE).read_text())
    assert record["task"] == {"task_id": "t1", "prompt": "fix main.py"}
    assert record["spec_id"] == EVERY_KIND.spec_id
    assert record["policy_checksum"] == POLICY.checkpoint.checksum
    assert record["policy"] == {
        "sglang_url": "http://sglang.test:30000",
        "served_model": "m",
        "tokenizer_location": "/models/tok",
    }
    assert record["cli"] == {"install": True, "extra_args": ""}
    assert record["adapter"]["max_new_tokens"] is None
    assert record["sandbox"]["image"] == "honing-agent:test"
    assert [(entry["target"], entry["paths"]) for entry in record["placement"]] == [
        (placement.target, list(placement.paths)) for placement in place_harness(EVERY_KIND)
    ]
    # The task's files and the placed harness files, kept apart for the worker to overlay.
    assert (directory / "workspace" / "main.py").read_text() == "print('hi')\n"
    assert (directory / "harness" / ".claude" / "commands" / "fix.md").read_text() == "Fix it.\n"
    assert (directory / "harness" / "CLAUDE.md").read_text() == "Be careful.\n"
    assert outcome.task_id == "t1"
    assert outcome.spec_id == EVERY_KIND.spec_id
    assert outcome.policy_checksum == POLICY.checkpoint.checksum
    assert (outcome.exit_code, outcome.final_output) == (0, "done")
    assert outcome.workspace_location == str(directory / "workspace_out")
    assert (directory / "workspace_out" / "agent.txt").is_file()
    assert outcome.transcript_location == str(directory / "transcript.jsonl")
    assert SlimeAgentDeliveryProbe().delivered(outcome, EVERY_KIND) == {path: True for path, _ in EVERY_KIND.files}


@pytest.mark.unit
def test_a_relative_episodes_directory_is_taken_from_where_the_runner_was_made(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The worker runs in its episode's directory, so a path relative to the runner's would not resolve there.
    monkeypatch.chdir(tmp_path)
    outcome = run(make_runner(tmp_path, episodes=Path("episodes")))
    directory = episode_dir(tmp_path)
    assert outcome.workspace_location == str(directory / "workspace_out")
    assert (directory / "workspace_out" / "agent.txt").is_file()


def _docker_calls(log: Path, command: str) -> list[list[str]]:
    entries = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return [entry["argv"] for entry in entries if "argv" in entry and entry["argv"][0] == command]


@pytest.mark.integration  # starts subprocesses
def test_a_missing_image_raises_before_any_episode(tmp_path: Path, docker_log: Path) -> None:
    missing = {"image": {"exit": 1, "stderr": "Error response from daemon: No such image: honing-agent:test"}}
    runner = make_runner(tmp_path, docker=missing)
    with pytest.raises(AgentWorkerError, match=r"build or pull honing-agent:test first.*No such image"):
        run(runner)
    assert not (tmp_path / "episodes").exists()
    assert _docker_calls(docker_log, "image") == [["image", "inspect", "honing-agent:test"]]


@pytest.mark.integration
def test_the_image_is_checked_once_per_runner(tmp_path: Path, docker_log: Path) -> None:
    runner = make_runner(tmp_path)
    run(runner)
    run(runner)
    assert len(list((tmp_path / "episodes").iterdir())) == 2
    assert _docker_calls(docker_log, "image") == [["image", "inspect", "honing-agent:test"]]


@pytest.mark.unit
def test_the_optional_cli_and_token_settings_reach_the_worker(tmp_path: Path) -> None:
    settings = AgentEpisodeSettings(
        image="i", max_new_tokens=8192, cli_extra_args="--disallowedTools WebSearch", install_cli=False
    )
    run(make_runner(tmp_path, settings=settings))
    episode = files.read_episode(episode_dir(tmp_path) / files.EPISODE_FILE)
    assert (episode.max_new_tokens, episode.cli_extra_args, episode.install_cli) == (
        8192,
        "--disallowedTools WebSearch",
        False,
    )


@pytest.mark.unit
def test_the_worker_sees_only_the_allowed_environment(tmp_path: Path) -> None:
    run(make_runner(tmp_path))
    seen = json.loads((episode_dir(tmp_path) / "fake_env.json").read_text())
    env = seen["env"]
    assert "SECRET_TOKEN" not in env
    assert env["DOCKER_HOST"] == ENVIRON["DOCKER_HOST"]
    assert env["HOME"] == ENVIRON["HOME"]
    assert env["PYTHONPATH"] == str(REPO_ROOT)
    # Python itself may add LC_CTYPE (PEP 538) and macOS __CF_USER_TEXT_ENCODING.
    extra = set(env) - {*runner_module.PASSTHROUGH, *FAKE_NAMES, "PYTHONPATH", "PYTHONUNBUFFERED"}
    assert extra <= {"LC_CTYPE", "__CF_USER_TEXT_ENCODING"}
    assert seen["argv"][1:] == [str(episode_dir(tmp_path))]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("knobs", "expected"),
    [
        ({"status": "time_budget_exceeded", "exit_code": -1}, -1),
        ({"exit_code": 3, "final_output": "EXIT 3"}, 3),
    ],
)
def test_what_the_agent_did_is_the_outcome(tmp_path: Path, knobs: Mapping[str, object], expected: int) -> None:
    outcome = run(make_runner(tmp_path, knobs=knobs))
    assert outcome.exit_code == expected
    assert outcome.workspace_location is not None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("knobs", "message"),
    [
        ({"status": "infra_error"}, "could not run: fake infrastructure fault"),
        ({"result": "missing", "worker_exit": 1}, "exited 1 without a readable result.json"),
        ({"result": "malformed"}, "exited 0 without a readable result.json"),
        ({"worker_exit": 2}, "exited 2 with a malformed result.json"),
        ({"exit_code": True}, "exited 0 with a malformed result.json"),
        ({"status": "finished"}, "malformed result.json"),
        ({"workspace": None}, "malformed result.json"),
        ({"failure": "too big"}, "malformed result.json"),
        ({"failure": "", "workspace": None}, "malformed result.json"),
        ({"failure": 7, "workspace": None}, "malformed result.json"),
    ],
)
def test_a_worker_that_could_not_run_the_episode_raises(
    tmp_path: Path, knobs: Mapping[str, object], message: str
) -> None:
    with pytest.raises(AgentWorkerError, match=message) as raised:
        run(make_runner(tmp_path, knobs=knobs))
    assert str(episode_dir(tmp_path) / "worker.log") in str(raised.value)


@pytest.mark.unit
def test_a_workspace_the_agent_could_not_hand_back_is_an_outcome_that_grades_zero(tmp_path: Path) -> None:
    outcome = run(make_runner(tmp_path, knobs={"failure": "the workspace is too big", "workspace": None}))
    assert (outcome.exit_code, outcome.workspace_location, outcome.failure) == (0, None, "the workspace is too big")
    grade = asyncio.run(CommandGrader(suite(tmp_path / "grading")).grade(Task("t1", "fix main.py"), outcome))
    assert (grade.score, grade.feedback) == (0.0, "the workspace is too big")


@pytest.mark.unit
def test_an_ordinary_outcome_names_no_failure(tmp_path: Path) -> None:
    assert run(make_runner(tmp_path)).failure is None


# --- settings ----------------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_only_the_default_bridge_network_is_accepted() -> None:
    assert AgentEpisodeSettings(image="i").network == "bridge"
    with pytest.raises(ContractError, match="only Docker's default 'bridge' network, not 'honing-net'"):
        AgentEpisodeSettings(image="i", network="honing-net")


@pytest.mark.unit
def test_the_worker_deadline_adds_every_limit_outside_the_agent_and_the_setup() -> None:
    settings = AgentEpisodeSettings(
        image="i", time_budget_seconds=100, setup_timeout_seconds=10, worker_margin_seconds=7
    )
    assert settings.worker_timeout_seconds == (
        10 + 100 + files.PRE_LAUNCH_SECONDS + files.STOP_AGENT_SECONDS + files.COPY_OUT_SECONDS + 7
    )
    # Each part is the per-call limits, plus what the sandbox adds to every exec.
    grace = files.EXEC_GRACE_SECONDS
    # Slime's steps before the agent's clock starts: five execs, and the launcher's write_file.
    assert files.SLIME_PRE_LAUNCH_EXEC_SECONDS == {
        "sandbox.py:ensure_agent_user": (60,),
        "harness/claude_code.py:ClaudeCodeHarness.write_config": (60,),
        "harness/common.py:run_agent": (30,),
        "sandbox.py:exec_and_wait": (30, 30),
    }
    assert 60 + 60 + 30 + 30 + 30 + 5 * grace + files.FILE_SECONDS == files.PRE_LAUNCH_SECONDS == 910
    # Stopping: the check for ps and pkill, one listing, one pkill, then the polls.
    assert (
        2 * (files.STOP_POLL_SECONDS + grace)
        + (files.STOP_KILL_SECONDS + grace)
        + files.STOP_POLLS * (files.STOP_POLL_SECONDS + grace + files.STOP_POLL_INTERVAL_SECONDS)
        == files.STOP_AGENT_SECONDS
    )
    assert files.FILE_SECONDS + files.COPY_SECONDS + grace == files.COPY_OUT_SECONDS
    assert (files.STOP_AGENT_SECONDS, files.COPY_OUT_SECONDS) == (895, 920)


@pytest.mark.unit
def test_the_sandbox_outlives_the_worker() -> None:
    with pytest.raises(ContractError, match="the sandbox must outlive the worker's deadline"):
        AgentEpisodeSettings(image="i", sandbox_lifetime_seconds=3000)


# --- the worker's deadline ---------------------------------------------------------------------------------------

QUICK = AgentEpisodeSettings(image="i", time_budget_seconds=1, setup_timeout_seconds=1, worker_margin_seconds=1)


@pytest.fixture
def no_fixed_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero the fixed limits (slime's pre-launch steps, stopping the agent, copying out): QUICK's three seconds."""
    monkeypatch.setattr(files, "PRE_LAUNCH_SECONDS", 0)
    monkeypatch.setattr(files, "STOP_AGENT_SECONDS", 0)
    monkeypatch.setattr(files, "COPY_OUT_SECONDS", 0)


def _dead(pid: int) -> bool:
    # A killed orphan is a zombie until the system reaps it, which takes a moment.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.1)
    return False


@pytest.mark.integration  # starts subprocesses and waits on them
@pytest.mark.parametrize("ignore_term", [False, True])
def test_a_worker_past_its_deadline_is_killed_with_its_children_and_its_sandbox_removed(
    tmp_path: Path, docker_log: Path, ignore_term: bool, no_fixed_limits: None
) -> None:
    runner = make_runner(
        tmp_path, knobs={"sleep": 60, "child": True, "ignore_term": ignore_term}, settings=QUICK, grace=1.0
    )
    started = time.monotonic()
    with pytest.raises(AgentWorkerError, match="ran past 3s and was killed"):
        run(runner)
    assert time.monotonic() - started < 20
    directory = episode_dir(tmp_path)
    assert _dead(int((directory / "child.pid").read_text()))
    prefix = json.loads((directory / files.EPISODE_FILE).read_text())["sandbox"]["name_prefix"]
    assert removed(docker_log) == [["rm", "--force", f"{prefix}-fake"]]


@pytest.mark.integration
def test_a_cancelled_episode_kills_the_worker_removes_its_sandbox_and_stays_cancelled(
    tmp_path: Path, docker_log: Path
) -> None:
    runner = make_runner(tmp_path, knobs={"sleep": 60, "child": True})

    async def scenario() -> None:
        task = asyncio.ensure_future(runner.run(EVERY_KIND, Task("t1", "p"), POLICY))
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not list((tmp_path / "episodes").glob("*/child.pid")):
            await asyncio.sleep(0.1)
        task.cancel()
        await task

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(scenario())
    directory = episode_dir(tmp_path)
    assert _dead(int((directory / "child.pid").read_text()))
    assert len(removed(docker_log)) == 1


# --- conformance -------------------------------------------------------------------------------------------------


@pytest.mark.integration
def test_the_runner_passes_the_conformance_cases_with_the_fake_worker(tmp_path: Path) -> None:
    example = load_task_suite(EXAMPLE_SUITE)
    runner = make_runner(tmp_path, task_suite=example)
    seed = spec(("CLAUDE.md", "instructions", "Fix the bug.\n"))
    checks = check_runner(
        runner,
        CommandGrader(example),
        seed=seed,
        tasks=example.tasks("val"),
        snapshot=SNAPSHOT,
        probe=SlimeAgentDeliveryProbe(),
    )
    assert [(check.name, check.passed) for check in checks] == [
        ("support", True),
        ("episodes", True),
        ("delivery", True),
        ("concurrent", True),
        ("repeat", True),
        ("gate", True),
    ], [check.detail for check in checks]


# --- the worker's helpers that need no Slime ---------------------------------------------------------------------


@pytest.mark.unit
def test_the_final_output_is_the_last_result_event_and_other_lines_are_skipped() -> None:
    transcript = "\n".join(
        [
            '{"type": "system", "subtype": "init"}',
            '[claude-code:unrecognized_model] {"model": "slime-actor"}',
            '{"type": "result", "result": "first"}',
            "[1, 2]",
            '{"type": "result", "result": "Done."}',
            "{truncated",
        ]
    )
    assert worker.final_output(transcript) == "Done."
    assert worker.final_output("") == ""
    assert worker.final_output('{"type": "result", "result": 7}') == ""


def _tar(entries: Mapping[str, bytes], links: Mapping[str, str] | None = None) -> str:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, data in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        for name, target in (links or {}).items():
            info = tarfile.TarInfo(name)
            info.type, info.linkname = tarfile.SYMTYPE, target
            archive.addfile(info)
    return base64.b64encode(buffer.getvalue()).decode()


@pytest.mark.unit
def test_the_copied_workspace_skips_what_points_out_of_it(tmp_path: Path) -> None:
    encoded = _tar(
        {"./ok.txt": b"fine", "./sub/b.bin": bytes(range(256))},
        links={"./escape": "/etc/passwd", "./inside": "ok.txt"},
    )
    skipped = worker.unpack_workspace(worker.decode_workspace(encoded, 2**20), tmp_path / "out")
    assert skipped == ["./escape"]
    assert (tmp_path / "out" / "ok.txt").read_text() == "fine"
    assert (tmp_path / "out" / "sub" / "b.bin").read_bytes() == bytes(range(256))
    assert (tmp_path / "out" / "inside").is_symlink()


class _CopyOutSandbox:
    """Answers the copy-out's two calls: the transcript read, and the tar command with a scripted exit and output."""

    def __init__(self, code: int, out: str) -> None:
        self.code, self.out = code, out

    async def read_file(self, path: str) -> str:
        return '{"type": "result", "result": "done"}\n'

    async def exec(self, cmd: str, *, timeout: int) -> tuple[int, str, str]:
        return self.code, self.out, "tar: something failed"


class _StopSandbox:
    """Answers the stop's commands by their first word: ``for`` (the tool check), ``ps`` and ``pkill``."""

    def __init__(self, **answers: tuple[int, str, str]) -> None:
        self.answers = {"for": (0, "", ""), "ps": (0, "", ""), "pkill": (1, "", ""), **answers}
        self.commands: list[str] = []

    async def exec(self, cmd: str, *, timeout: int) -> tuple[int, str, str]:
        self.commands.append(cmd)
        return self.answers[cmd.split()[0]]


@pytest.mark.unit
@pytest.mark.parametrize(("listed", "named"), [("pkill\n", "pkill"), ("ps\n", "ps"), ("ps\npkill\n", "ps or pkill")])
def test_stopping_the_agent_without_ps_or_pkill_is_an_infrastructure_fault(listed: str, named: str) -> None:
    sandbox = _StopSandbox(**{"for": (0, listed, "")})
    with pytest.raises(worker._InfraError, match=f"the sandbox image has no {named}, which stopping the agent needs"):
        asyncio.run(worker._stop_agent(sandbox))  # type: ignore[arg-type]
    assert len(sandbox.commands) == 1  # nothing listed or killed: neither would have worked


@pytest.mark.unit
@pytest.mark.parametrize(
    ("answers", "message"),
    [
        ({"for": (2, "", "boom")}, "cannot check for ps and pkill in the sandbox \\(exit 2\\): boom"),
        ({"ps": (1, "", "ps: bad option")}, "cannot list the agent's processes \\(exit 1\\): ps: bad option"),
        ({"pkill": (2, "", "pkill: bad option")}, "pkill failed \\(exit 2\\): pkill: bad option"),
    ],
)
def test_a_stop_command_that_fails_is_an_infrastructure_fault(
    answers: Mapping[str, tuple[int, str, str]], message: str
) -> None:
    with pytest.raises(worker._InfraError, match=message):
        asyncio.run(worker._stop_agent(_StopSandbox(**answers)))  # type: ignore[arg-type]


@pytest.mark.unit
def test_an_agent_with_nothing_left_running_stops_at_once() -> None:
    # pkill exits 1 when nothing matched: the agent left nothing, which is fine.
    sandbox = _StopSandbox()
    assert asyncio.run(worker._stop_agent(sandbox)) == 0  # type: ignore[arg-type]
    assert [command.split()[0] for command in sandbox.commands] == ["for", "ps", "pkill", "ps"]


def _copy_out(tmp_path: Path, code: int, out: str, max_bytes: int = 2**20) -> dict[str, object]:
    episode = SimpleNamespace(placement=(), copy_out_max_bytes=max_bytes)
    result: dict[str, object] = worker._empty_result()
    asyncio.run(worker._copy_out(tmp_path, episode, _CopyOutSandbox(code, out), result))  # type: ignore[arg-type]
    return result


@pytest.mark.unit
@pytest.mark.parametrize("code", [0, 141])
def test_a_workspace_over_the_cap_is_an_outcome_without_a_workspace(tmp_path: Path, code: int) -> None:
    # The pipeline keeps one byte past the cap; head then closes the pipe, so tar may die of SIGPIPE (141).
    result = _copy_out(tmp_path, code, base64.b64encode(b"x" * 101).decode(), max_bytes=100)
    assert result["workspace"] is None
    assert result["failure"] == "the agent left a workspace larger than 100 bytes as a tar; it was not copied back"
    assert result["final_output"] == "done"
    assert not (tmp_path / files.WORKSPACE_OUT_DIR).exists()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("entries", "pair"),
    [
        ({"./Foo.py": b"1", "./foo.py": b"2"}, "('Foo.py', 'foo.py')"),
        ({"./a/x": b"1", "./A": b"2"}, "('A', 'a')"),
        ({"./caf\u00e9.txt": b"1", "./cafe\u0301.txt": b"2"}, "café"),
        ({"./Stra\u00dfe.txt": b"1", "./strasse.txt": b"2"}, "('Straße.txt', 'strasse.txt')"),
    ],
)
def test_paths_one_case_insensitive_disk_would_merge_are_an_outcome(
    tmp_path: Path, entries: Mapping[str, bytes], pair: str
) -> None:
    result = _copy_out(tmp_path, 0, _tar(entries))
    assert result["workspace"] is None
    failure = str(result["failure"])
    assert failure.startswith("the agent left paths that differ only in case or Unicode normalization"), failure
    assert pair in failure, failure
    assert not (tmp_path / files.WORKSPACE_OUT_DIR).exists()


@pytest.mark.unit
@pytest.mark.parametrize("tar_format", [tarfile.GNU_FORMAT, tarfile.PAX_FORMAT])
def test_names_that_are_not_utf8_are_an_outcome(tmp_path: Path, tar_format: int) -> None:
    # GNU tar on Linux stores the name's bytes as they are; tarfile reads them back with surrogate
    # escapes, and a Mac's disk refuses them on extraction (EILSEQ), which would read as the worker's fault.
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tar_format, errors="surrogateescape") as archive:
        for name in ("./ok.txt", "./sub/\udcff.txt"):
            archive.addfile(tarfile.TarInfo(name), io.BytesIO(b""))
    result = _copy_out(tmp_path, 0, base64.b64encode(buffer.getvalue()).decode())
    assert result["workspace"] is None
    assert result["failure"] == (
        "the agent left paths that are not valid UTF-8, which a Mac's disk cannot hold: b'./sub/\\xff.txt'"
    )
    assert not (tmp_path / files.WORKSPACE_OUT_DIR).exists()


@pytest.mark.unit
def test_paths_that_differ_in_more_than_case_are_kept(tmp_path: Path) -> None:
    result = _copy_out(tmp_path, 0, _tar({"./a/x": b"1", "./a/y": b"2", "./ab": b"3"}))
    assert (result["workspace"], result["failure"]) == (files.WORKSPACE_OUT_DIR, None)
    assert (tmp_path / files.WORKSPACE_OUT_DIR / "a" / "y").read_bytes() == b"2"


@pytest.mark.unit
def test_a_tar_that_failed_under_the_cap_is_an_infrastructure_fault(tmp_path: Path) -> None:
    with pytest.raises(worker._InfraError, match="cannot copy the workspace out \\(exit 2\\): tar: something failed"):
        _copy_out(tmp_path, 2, _tar({"./ok": b"1"}))


@pytest.mark.unit
def test_any_other_error_while_unpacking_is_an_infrastructure_fault(tmp_path: Path) -> None:
    # A file, then a path under it: extraction fails with NotADirectoryError, which is not the agent's doing.
    with pytest.raises(worker._InfraError, match="cannot unpack the workspace"):
        _copy_out(tmp_path, 0, _tar({"./a": b"1", "./a/b": b"2"}))


@pytest.mark.unit
@pytest.mark.parametrize(
    "message",
    [
        # SGLang v0.5.15.post1: tokenizer_manager.py:968-971, :991-997, and managers/utils.py:187-191.
        "The input (40000 tokens) is longer than the model's context length (32768 tokens).",
        "Requested token count exceeds the model's maximum context length of 32768 tokens. You requested a total"
        " of 33000 tokens: 32000 tokens from the input messages and 1000 tokens for the completion. Please reduce"
        " the number of tokens in the input messages or the completion to fit within the limit.",
        "Input length (40000 tokens) exceeds the maximum allowed length (32767 tokens). Use a shorter input or"
        " enable --allow-auto-truncate.",
    ],
)
def test_sglangs_context_overflow_answers_are_recognized(message: str) -> None:
    # Slime raises a 4xx as "sglang upstream <status>: <body>[:400]" (adapters/common.py:495).
    body = json.dumps({"error": {"message": message}})
    assert worker.is_context_overflow(RuntimeError(f"sglang upstream 400: {body[:400]}"))


@pytest.mark.unit
@pytest.mark.parametrize(
    "error",
    [
        RuntimeError('sglang upstream 400: {"error": {"message": "bad sampling params"}}'),
        RuntimeError("sglang upstream 500: the model's context length (32768 tokens)"),
        ValueError("sglang upstream 400: is longer than the model's context length"),
    ],
)
def test_other_policy_server_errors_are_not_overflows(error: Exception) -> None:
    assert not worker.is_context_overflow(error)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("configured", "server_info", "expected"),
    [
        (0, {"max_req_input_len": 32767}, 32767),
        (1000, {"max_req_input_len": 32767}, 1000),
        (0, {"max_req_input_len": None}, 0),
        (0, {"max_req_input_len": True}, 0),
        (0, None, 0),
    ],
)
def test_the_context_limit_is_the_setting_or_the_servers_longest_input(
    configured: int, server_info: object, expected: int
) -> None:
    assert worker.context_limit(configured, server_info) == expected


@pytest.mark.unit
def test_the_copy_out_leaves_out_slimes_directory_and_the_placed_files() -> None:
    placed = (files.PlacedFile("CLAUDE.md", "0" * 64, ("a.md",)), files.PlacedFile(".claude/x y.md", "1", ("b",)))
    command = worker.copy_out_command(placed, 10)
    # pipefail: without it the pipeline's code is base64's, and a tar that failed would go unseen.
    assert command == (
        "set -o pipefail; tar cf - -C /workspace --exclude=./.harness --exclude=./CLAUDE.md"
        " --exclude='./.claude/x y.md' . | head -c 11 | base64 -w0"
    )


@pytest.mark.unit
def test_directories_that_held_only_placed_files_are_removed(tmp_path: Path) -> None:
    (tmp_path / ".claude" / "skills" / "lint").mkdir(parents=True)
    (tmp_path / ".claude" / "commands").mkdir(parents=True)
    (tmp_path / ".claude" / "commands" / "mine.md").write_text("the agent's own")
    placed = (
        files.PlacedFile(".claude/skills/lint/SKILL.md", "0", ("a",)),
        files.PlacedFile(".claude/commands/fix.md", "1", ("b",)),
        files.PlacedFile("CLAUDE.md", "2", ("c",)),
    )
    worker.prune_placed_directories(tmp_path, placed)
    assert sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*")) == [
        ".claude",
        ".claude/commands",
        ".claude/commands/mine.md",
    ]


@pytest.mark.unit
def test_the_copy_in_overlays_the_harness_on_the_workspace(tmp_path: Path) -> None:
    (tmp_path / "workspace" / "src").mkdir(parents=True)
    (tmp_path / "workspace" / "src" / "a.py").write_text("a")
    (tmp_path / "harness" / ".claude" / "commands").mkdir(parents=True)
    (tmp_path / "harness" / ".claude" / "commands" / "fix.md").write_text("fix")
    with tarfile.open(worker.pack_workspace(tmp_path, tmp_path / "in.tar")) as archive:
        names = sorted(member.name for member in archive.getmembers() if member.isfile())
    assert names == [".claude/commands/fix.md", "src/a.py"]


@pytest.mark.unit
def test_an_episode_file_with_a_wrong_field_is_refused(tmp_path: Path) -> None:
    runner = make_runner(tmp_path)
    run(runner)
    path = episode_dir(tmp_path) / files.EPISODE_FILE
    record = json.loads(path.read_text())
    record["agent"]["time_budget_seconds"] = True  # a bool is an int to isinstance
    path.write_text(json.dumps(record))
    with pytest.raises(files.EpisodeFileError, match="time_budget_seconds' must be int, not bool"):
        files.read_episode(path)
    record["agent"]["time_budget_seconds"] = 10
    record["schema_version"] = 2
    path.write_text(json.dumps(record))
    with pytest.raises(files.EpisodeFileError, match="schema_version must be 1"):
        files.read_episode(path)
