"""The command grader: how a command's result becomes a grade, and what the command can see and change."""

from __future__ import annotations

import asyncio
import contextlib
import errno
import gc
import json
import os
import shutil
import signal
import sys
import threading
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from honing.core import ContractError, EpisodeOutcome, Grade, HarnessSpec, PolicyRef, RunnerSupport, Task, TaskGrader
from honing.e2e import check_runner
from honing.harness.graders import (
    FINAL_OUTPUT_ENV,
    METADATA_ENV,
    TASK_ID_ENV,
    WORKSPACE_ENV,
    CommandGrader,
    CommandResult,
    CommandSandbox,
    LocalWorkspace,
    UngradableWorkspaceError,
    local_workspace,
)
from honing.harness.tasks import TaskSuite, load_task_suite
from tests.fakes.harness import SEED_HARNESS
from tests.fakes.model import FakeModelSource, FakeWorld
from tests.fakes.tasks import FakeCommandSandbox, task_line, write_manifest


def _suite(tmp_path: Path, **overrides: object) -> TaskSuite:
    return load_task_suite(write_manifest(tmp_path, [task_line("t1", **overrides)]))


def _outcome(workspace: Path | None, final_output: str = "done") -> EpisodeOutcome:
    return EpisodeOutcome(
        task_id="t1",
        spec_id="h-spec",
        policy_checksum="sha",
        exit_code=0,
        final_output=final_output,
        workspace_location=None if workspace is None else str(workspace),
    )


def _fake_grade(tmp_path: Path, result: CommandResult, **overrides: object) -> tuple[Grade, FakeCommandSandbox]:
    sandbox = FakeCommandSandbox(result)
    grader = CommandGrader(_suite(tmp_path, **overrides), open_sandbox=sandbox.opener(), timeout_seconds=30)
    grade = asyncio.run(grader.grade(Task("t1", "Fix t1."), _outcome(tmp_path / "agent")))
    return grade, sandbox


def _ok(stdout: str, exit_code: int = 0, stderr: str = "") -> CommandResult:
    return CommandResult(exit_code=exit_code, stdout=stdout, stderr=stderr)


def test_the_command_grader_and_the_local_workspace_satisfy_their_protocols(tmp_path: Path) -> None:
    assert isinstance(CommandGrader(_suite(tmp_path)), TaskGrader)
    assert isinstance(LocalWorkspace(tmp_path), CommandSandbox)


# ---------------------------------------------------------------- acceptance: with a fake sandbox


def test_valid_output_becomes_the_grade(tmp_path: Path) -> None:
    stdout = 'collected 4 items\n{"score": 0.5}\n{"score": 0.75, "feedback": "3 of 4 pass"}\n\n'
    grade, sandbox = _fake_grade(tmp_path, _ok(stdout), metadata={"k": 1})
    assert grade == Grade(score=0.75, feedback="3 of 4 pass")
    assert sandbox.opened == [tmp_path / "agent"]
    [(argv, env, cwd)] = sandbox.calls
    assert argv == ("python3", "grade.py")
    assert cwd is None
    assert env == {
        TASK_ID_ENV: "t1",
        FINAL_OUTPUT_ENV: "/sandbox/scratch/final_output.txt",
        METADATA_ENV: "/sandbox/scratch/metadata.json",
    }
    assert sandbox.files == {"final_output.txt": b"done", "metadata.json": b'{"k": 1}'}


def test_an_integer_score_is_accepted(tmp_path: Path) -> None:
    grade, _ = _fake_grade(tmp_path, _ok('{"score": 1, "feedback": ""}'))
    assert grade == Grade(score=1.0, feedback="")


def test_a_nonzero_exit_scores_zero_with_the_exit_code_and_stderr(tmp_path: Path) -> None:
    result = _ok('{"score": 1.0, "feedback": "fine"}', exit_code=2, stderr="Traceback ...\nImportError: no foo\n")
    grade, _ = _fake_grade(tmp_path, result)
    assert grade.score == 0.0
    assert grade.feedback == "the grader exited 2; stderr ends: Traceback ...\nImportError: no foo"


@pytest.mark.parametrize(
    ("stdout", "reason"),
    [
        ("", "the grader printed nothing on stdout"),
        ("{bad json", "the grader's last stdout line is not JSON"),
        ('{"score": 1.0, "feedback": "ok"}\nall done', "the grader's last stdout line is not JSON"),
        ("[1.0]", "is not a JSON object"),
        ('{"feedback": "no score"}', "the grader's score must be a number, got None"),
        ('{"score": "1", "feedback": "x"}', "the grader's score must be a number, got '1'"),
        ('{"score": true, "feedback": "x"}', "the grader's score must be a number, got True"),
        ('{"score": 1.0}', "the grader's feedback must be a string, got None"),
        ('{"score": 1.0, "feedback": 3}', "the grader's feedback must be a string, got 3"),
    ],
)
def test_malformed_output_scores_zero_with_the_reason(tmp_path: Path, stdout: str, reason: str) -> None:
    grade, _ = _fake_grade(tmp_path, _ok(stdout))
    assert grade.score == 0.0
    assert grade.feedback is not None and reason in grade.feedback


@pytest.mark.parametrize("score", ["1.5", "-0.1", "NaN", "Infinity", "-Infinity", "1" + "0" * 400, "-1" + "0" * 400])
def test_a_score_outside_zero_to_one_scores_zero_instead_of_raising(tmp_path: Path, score: str) -> None:
    grade, _ = _fake_grade(tmp_path, _ok(f'{{"score": {score}, "feedback": "x"}}'))
    assert grade.score == 0.0
    assert grade.feedback is not None and "the grader's score must be in [0, 1]" in grade.feedback


def test_a_timeout_reported_by_the_sandbox_scores_zero(tmp_path: Path) -> None:
    grade, _ = _fake_grade(tmp_path, CommandResult(exit_code=-9, stdout="", stderr="", timed_out=True))
    assert grade == Grade(score=0.0, feedback="the grader timed out after 30 s")


def test_the_command_runs_in_the_grader_files_directory(tmp_path: Path) -> None:
    (tmp_path / "hidden").mkdir()
    _, sandbox = _fake_grade(tmp_path, _ok('{"score": 1, "feedback": ""}'), grader_files="hidden")
    assert sandbox.copied == [tmp_path / "hidden"]
    [(_, _, cwd)] = sandbox.calls
    assert cwd == "/sandbox/grader"


def test_long_feedback_is_cut(tmp_path: Path) -> None:
    sandbox = FakeCommandSandbox(_ok(json.dumps({"score": 0.5, "feedback": "x" * 100})))
    grader = CommandGrader(_suite(tmp_path), open_sandbox=sandbox.opener(), max_feedback_chars=10)
    grade = asyncio.run(grader.grade(Task("t1", "Fix t1."), _outcome(tmp_path)))
    assert grade == Grade(score=0.5, feedback="x" * 10 + " [truncated]")


# ---------------------------------------------------------------- infrastructure faults raise


def test_an_outcome_without_a_workspace_raises(tmp_path: Path) -> None:
    grader = CommandGrader(_suite(tmp_path))
    with pytest.raises(ContractError, match="names no workspace"):
        asyncio.run(grader.grade(Task("t1", "Fix t1."), _outcome(None)))


@pytest.mark.parametrize("has_workspace", [False, True])
def test_a_failed_outcome_scores_zero_with_its_reason_and_runs_no_grader(tmp_path: Path, has_workspace: bool) -> None:
    sandbox = FakeCommandSandbox(_ok('{"score": 1.0, "feedback": "solved"}'))
    grader = CommandGrader(_suite(tmp_path), open_sandbox=sandbox.opener(), max_feedback_chars=20)
    outcome = replace(_outcome(tmp_path / "agent" if has_workspace else None), failure="the workspace is " + "x" * 40)
    grade = asyncio.run(grader.grade(Task("t1", "Fix t1."), outcome))
    assert (grade.score, grade.feedback) == (0.0, "the workspace is xxx [truncated]")
    assert sandbox.opened == []


def test_a_sandbox_that_cannot_open_raises(tmp_path: Path) -> None:
    @asynccontextmanager
    async def broken(workspace: Path) -> AsyncGenerator[CommandSandbox, None]:
        raise OSError("no container runtime")
        yield  # pragma: no cover

    grader = CommandGrader(_suite(tmp_path), open_sandbox=broken)
    with pytest.raises(OSError, match="no container runtime"):
        asyncio.run(grader.grade(Task("t1", "Fix t1."), _outcome(tmp_path)))


def test_a_workspace_location_that_does_not_exist_raises(tmp_path: Path) -> None:
    grader = CommandGrader(_suite(tmp_path))
    with pytest.raises(FileNotFoundError):
        asyncio.run(grader.grade(Task("t1", "Fix t1."), _outcome(tmp_path / "gone")))


def test_a_sandbox_that_refuses_the_agents_workspace_scores_zero(tmp_path: Path) -> None:
    @asynccontextmanager
    async def refusing(workspace: Path) -> AsyncGenerator[CommandSandbox, None]:
        raise UngradableWorkspaceError("the agent left a socket")
        yield  # pragma: no cover

    grader = CommandGrader(_suite(tmp_path), open_sandbox=refusing)
    grade = asyncio.run(grader.grade(Task("t1", "Fix t1."), _outcome(tmp_path)))
    assert grade == Grade(score=0.0, feedback="the agent left a socket")


def test_a_task_not_in_the_suite_raises(tmp_path: Path) -> None:
    grader = CommandGrader(_suite(tmp_path))
    with pytest.raises(ContractError, match="not in the task suite"):
        asyncio.run(grader.grade(Task("other", "p"), _outcome(tmp_path)))


# ---------------------------------------------------------------- the local workspace, with real commands


def _python(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def _local_grade(
    tmp_path: Path,
    agent: Path,
    grader: list[str],
    *,
    final_output: str = "",
    line: dict[str, object] | None = None,
    **options: Any,
) -> Grade:
    """Grade ``agent`` (the agent's workspace) with a real ``LocalWorkspace``."""
    suite = load_task_suite(write_manifest(tmp_path / "suite", [task_line("t1", grader=grader, **(line or {}))]))
    command_grader = CommandGrader(suite, **options)
    return asyncio.run(command_grader.grade(Task("t1", "Fix t1."), _outcome(agent, final_output)))


@pytest.fixture
def agent(tmp_path: Path) -> Path:
    (tmp_path / "suite").mkdir()
    workspace = tmp_path / "agent"
    workspace.mkdir()
    (workspace / "answer.txt").write_text("42\n")
    return workspace


REPORT_CWD = (
    "import json, os, pathlib; "
    "pathlib.Path('answer.txt').write_text('overwritten'); pathlib.Path('new.txt').write_text('x'); "
    "print(json.dumps({'score': 1.0, 'feedback': os.getcwd()}))"
)


def test_the_command_runs_on_a_copy_and_never_changes_the_agent_workspace(tmp_path: Path, agent: Path) -> None:
    grade = _local_grade(tmp_path, agent, _python(REPORT_CWD))
    assert grade.score == 1.0
    assert grade.feedback is not None and Path(grade.feedback).resolve() != agent.resolve()
    assert sorted(path.name for path in agent.iterdir()) == ["answer.txt"]
    assert (agent / "answer.txt").read_text() == "42\n"


READ_INPUTS = (
    "import json, os, pathlib; "
    f"out = pathlib.Path(os.environ['{FINAL_OUTPUT_ENV}']).read_text(); "
    f"meta = json.loads(pathlib.Path(os.environ['{METADATA_ENV}']).read_text()); "
    f"print(json.dumps({{'score': 1.0, 'feedback': json.dumps([os.environ['{TASK_ID_ENV}'], out, meta])}}))"
)


def test_the_command_reads_the_final_output_and_the_metadata_from_files(tmp_path: Path, agent: Path) -> None:
    grade = _local_grade(
        tmp_path, agent, _python(READ_INPUTS), final_output="the answer\nis 42", line={"metadata": {"limit": [3]}}
    )
    assert grade.feedback is not None
    assert json.loads(grade.feedback) == ["t1", "the answer\nis 42", {"limit": [3]}]


LIST_ENV = "import json, os; print(json.dumps({'score': 1.0, 'feedback': json.dumps(sorted(os.environ))}))"


def test_the_command_sees_no_host_environment_beyond_path(
    tmp_path: Path, agent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    grade = _local_grade(tmp_path, agent, _python(LIST_ENV))
    assert grade.feedback is not None
    names = set(json.loads(grade.feedback)) - {"LC_CTYPE", "__CF_USER_TEXT_ENCODING"}  # set by the OS on macOS
    assert names == {"PATH", "HOME", "TMPDIR", "LANG", WORKSPACE_ENV, TASK_ID_ENV, FINAL_OUTPUT_ENV, METADATA_ENV}


SHOW_PLACES = (
    "import json, os; workspace = os.environ['WORKSPACE_ENV']; "
    "listed = [sorted(os.listdir('.')), sorted(os.listdir(workspace)), open('answer.txt').read(), "
    "open(os.path.join(workspace, 'answer.txt')).read(), os.path.samefile(os.getcwd(), workspace)]; "
    "print(json.dumps({'score': 1, 'feedback': json.dumps(listed)}))"
).replace("WORKSPACE_ENV", WORKSPACE_ENV)


def test_grader_files_stay_out_of_the_workspace(tmp_path: Path, agent: Path) -> None:
    hidden = tmp_path / "suite" / "hidden"
    (hidden / "cases").mkdir(parents=True)
    (hidden / "cases" / "t1.json").write_text("{}")
    (hidden / "answer.txt").write_text("expected\n")
    (agent / "cases").mkdir()
    (agent / "cases" / "extra.json").write_text("the agent's\n")
    grade = _local_grade(tmp_path, agent, _python(SHOW_PLACES), line={"grader_files": "hidden"})
    assert grade.feedback is not None
    grader_listing, workspace_listing, grader_answer, agent_answer, same = json.loads(grade.feedback)
    # The command runs among the grader's files only; the workspace holds the agent's files only.
    assert grader_listing == ["answer.txt", "cases"]
    assert workspace_listing == ["answer.txt", "cases"]
    assert (grader_answer, agent_answer, same) == ("expected\n", "42\n", False)


def test_grader_files_are_not_a_sibling_of_the_workspace(tmp_path: Path, agent: Path) -> None:
    # Agent code runs from the workspace, so whatever ``..`` holds is in its reach.
    (tmp_path / "suite" / "hidden").mkdir()
    show = (
        "import json, os; parent = os.path.dirname(os.environ['WORKSPACE_ENV']); "
        "print(json.dumps({'score': 1, 'feedback': json.dumps([os.getcwd(), parent])}))"
    ).replace("WORKSPACE_ENV", WORKSPACE_ENV)
    grade = _local_grade(tmp_path, agent, _python(show), line={"grader_files": "hidden"})
    assert grade.feedback is not None
    cwd, parent = map(Path, json.loads(grade.feedback))
    assert not cwd.resolve().is_relative_to(Path(parent).resolve())


def test_the_grader_files_directory_is_removed_after_grading(tmp_path: Path, agent: Path) -> None:
    (tmp_path / "suite" / "hidden").mkdir()
    places: list[Path] = []

    @asynccontextmanager
    async def recording(workspace: Path) -> AsyncGenerator[LocalWorkspace, None]:
        async with local_workspace(workspace) as sandbox:
            yield sandbox
            places.extend(sandbox.grader_directories)

    show_cwd = "import json, os; print(json.dumps({'score': 1, 'feedback': os.getcwd()}))"
    grade = _local_grade(tmp_path, agent, _python(show_cwd), line={"grader_files": "hidden"}, open_sandbox=recording)
    assert grade.feedback is not None
    [place] = places
    assert Path(grade.feedback).resolve().is_relative_to(place.resolve())
    assert not place.exists()


def test_a_command_that_cannot_start_scores_zero(tmp_path: Path, agent: Path) -> None:
    grade = _local_grade(tmp_path, agent, ["honing-no-such-grader"])
    assert grade.score == 0.0
    assert grade.feedback is not None and grade.feedback.startswith("the grader exited 127; stderr ends: cannot run")


def test_a_command_that_runs_too_long_is_killed_and_scores_zero(tmp_path: Path, agent: Path) -> None:
    # The child sleeps too, so the kill has to reach the whole process group for the pipes to close.
    sleeper = (
        "import subprocess, sys, time; "
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); time.sleep(60)"
    )
    started = time.monotonic()
    grade = _local_grade(tmp_path, agent, _python(sleeper), timeout_seconds=1.0)
    assert grade == Grade(score=0.0, feedback="the grader timed out after 1 s")
    assert time.monotonic() - started < 15


def test_only_the_tail_of_a_long_output_is_kept_and_its_last_line_still_grades(tmp_path: Path, agent: Path) -> None:
    noisy = (
        "import json, sys; sys.stdout.write('x' * 1_000_000 + '\\n'); "
        "print(json.dumps({'score': 0.25, 'feedback': 'ok'}))"
    )
    grade = _local_grade(tmp_path, agent, _python(noisy), max_output_bytes=1024)
    assert grade == Grade(score=0.25, feedback="ok")


def test_a_crashing_command_scores_zero_with_its_stderr(tmp_path: Path, agent: Path) -> None:
    grade = _local_grade(tmp_path, agent, _python("raise SystemExit('broken grader')"))
    assert grade == Grade(score=0.0, feedback="the grader exited 1; stderr ends: broken grader")


def test_the_temporary_copy_is_removed_after_grading(tmp_path: Path, agent: Path) -> None:
    roots: list[Path] = []

    @asynccontextmanager
    async def recording(workspace: Path) -> AsyncGenerator[LocalWorkspace, None]:
        async with local_workspace(workspace) as sandbox:
            roots.append(sandbox.workspace.parent)
            yield sandbox

    _local_grade(tmp_path, agent, _python('print(\'{"score": 1, "feedback": ""}\')'), open_sandbox=recording)
    [root] = roots
    assert not root.exists()


PASS = 'print(\'{"score": 1, "feedback": ""}\')'


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs named pipes")
def test_a_fifo_in_the_workspace_scores_zero_without_being_opened(tmp_path: Path, agent: Path) -> None:
    fifo = agent / "logs" / "pipe"
    fifo.parent.mkdir()
    os.mkfifo(fifo)
    grades: list[Grade] = []
    errors: list[BaseException] = []

    def grade() -> None:
        try:
            grades.append(_local_grade(tmp_path, agent, _python(PASS)))
        except BaseException as error:
            errors.append(error)

    # A copy that opened the FIFO would block until something wrote to it, so the
    # grade runs on a thread the test can give up on instead of hanging the run.
    worker = threading.Thread(target=grade, daemon=True)
    worker.start()
    worker.join(timeout=10)
    try:
        assert not worker.is_alive(), "grading blocked on the FIFO"
    finally:
        if worker.is_alive():
            os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))  # release the blocked reader
            worker.join(timeout=10)
    assert errors == []
    refusal = "the workspace holds a FIFO, 'logs/pipe', which the grader does not copy"
    assert grades == [Grade(score=0.0, feedback=refusal)]


NOT_ROOT = pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0, reason="root ignores permissions")


UNREADABLE_FILES = ("zeta", "b10", "b2", "x_y", "secret.txt", "aa", "alpha", "Q")


@NOT_ROOT
def test_files_the_grader_may_not_read_score_zero_and_are_named_in_order(tmp_path: Path, agent: Path) -> None:
    # Enough names that the directory order is unsorted on APFS and on ext4 alike.
    closed = agent / "closed"
    closed.mkdir()
    unreadable = [agent / name for name in UNREADABLE_FILES]
    for path in unreadable:
        path.write_text("x")
    for path in [*unreadable, closed]:
        path.chmod(0)
    try:
        grade = _local_grade(tmp_path, agent, _python(PASS))
    finally:
        for path in unreadable:
            path.chmod(0o644)
        closed.chmod(0o755)
    named = ", ".join(map(repr, sorted([*UNREADABLE_FILES, "closed"])))
    assert grade == Grade(score=0.0, feedback=f"the workspace holds files the grader may not read: {named}")


@pytest.mark.parametrize(
    "reason",
    [
        f"[Errno {errno.ENOSPC}] No space left on device",
        # A permission error on a source the grader can read was the destination's, not the agent's.
        f"[Errno {errno.EACCES}] Permission denied",
    ],
)
def test_any_other_copy_failure_still_raises(
    tmp_path: Path, agent: Path, monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    def failing_copytree(source: Path, destination: Path, **_: object) -> None:
        raise shutil.Error([(str(Path(source) / "answer.txt"), str(Path(destination) / "answer.txt"), reason)])

    monkeypatch.setattr(shutil, "copytree", failing_copytree)
    with pytest.raises(shutil.Error):
        _local_grade(tmp_path, agent, _python(PASS))


def _gone(pid: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def test_a_background_process_the_grader_started_is_killed_when_it_exits(tmp_path: Path, agent: Path) -> None:
    detach = (
        "import json, subprocess, sys; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
        "print(json.dumps({'score': 1, 'feedback': str(child.pid)}))"
    )
    started = time.monotonic()
    grade = _local_grade(tmp_path, agent, _python(detach))
    assert grade.score == 1.0 and grade.feedback is not None
    assert time.monotonic() - started < 15
    child = int(grade.feedback)
    try:
        assert _gone(child, 5), f"background process {child} survived grading"
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(child, signal.SIGKILL)


@NOT_ROOT
def test_a_workspace_root_the_grader_may_not_read_scores_zero_and_is_named(tmp_path: Path, agent: Path) -> None:
    agent.chmod(0)
    try:
        grade = _local_grade(tmp_path, agent, _python(PASS))
    finally:
        agent.chmod(0o755)
    assert grade == Grade(score=0.0, feedback=f"the grader may not read the workspace itself, {str(agent)!r}")


def test_a_permission_error_on_a_readable_root_still_raises(
    tmp_path: Path, agent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_copytree(source: Path, destination: Path, **_: object) -> None:
        raise PermissionError(errno.EACCES, "Permission denied", str(destination))

    monkeypatch.setattr(shutil, "copytree", failing_copytree)
    with pytest.raises(PermissionError):
        _local_grade(tmp_path, agent, _python(PASS))


def test_a_cancelled_grade_leaves_no_process_behind(tmp_path: Path, agent: Path) -> None:
    pids = tmp_path / "pids.txt"
    script = (
        "import os, subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        f"temporary = {str(pids)!r} + '.tmp'; "
        "open(temporary, 'w').write(str(os.getpid()) + ' ' + str(child.pid)); "
        f"os.replace(temporary, {str(pids)!r}); "
        "time.sleep(60)"
    )
    suite = load_task_suite(write_manifest(tmp_path / "suite", [task_line("t1", grader=_python(script))]))
    grader = CommandGrader(suite, timeout_seconds=120)

    async def cancel_once_running() -> None:
        grading = asyncio.ensure_future(grader.grade(Task("t1", "Fix t1."), _outcome(agent)))
        deadline = time.monotonic() + 15
        while not pids.exists():
            assert time.monotonic() < deadline, "the grader never started"
            await asyncio.sleep(0.05)
        grading.cancel()
        with pytest.raises(asyncio.CancelledError):
            await grading

    asyncio.run(cancel_once_running())
    leader, child = map(int, pids.read_text().split())
    try:
        assert _gone(leader, 5), f"the grader {leader} survived a cancelled grade"
        assert _gone(child, 5), f"the grader's child {child} survived a cancelled grade"
    finally:
        for pid in (leader, child):
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)


def test_a_background_process_holding_the_pipes_does_not_turn_a_grade_into_a_timeout(
    tmp_path: Path, agent: Path
) -> None:
    script = (
        "import json, subprocess, sys; "
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        "print(json.dumps({'score': 0.5, 'feedback': 'ok'}), flush=True)"
    )
    started = time.monotonic()
    grade = _local_grade(tmp_path, agent, _python(script), timeout_seconds=10)
    assert grade == Grade(score=0.5, feedback="ok")
    assert time.monotonic() - started < 8


SEPARATORS = "next line\u0085and\u2028paragraph"


def test_unicode_line_separators_inside_the_json_line_do_not_split_it(tmp_path: Path) -> None:
    stdout = "log\r\n" + json.dumps({"score": 1, "feedback": SEPARATORS}, ensure_ascii=False) + "\r\n"
    grade, _ = _fake_grade(tmp_path, _ok(stdout))
    assert grade == Grade(score=1.0, feedback=SEPARATORS)


def test_a_grader_can_echo_text_holding_unicode_line_separators(tmp_path: Path, agent: Path) -> None:
    echo = (
        "import json, os, pathlib; "
        f"text = pathlib.Path(os.environ['{FINAL_OUTPUT_ENV}']).read_text(encoding='utf-8'); "
        "print(json.dumps({'score': 1, 'feedback': text}, ensure_ascii=False))"
    )
    grade = _local_grade(tmp_path, agent, _python(echo), final_output=SEPARATORS)
    assert grade == Grade(score=1.0, feedback=SEPARATORS)


def test_a_score_with_more_digits_than_python_converts_scores_zero(tmp_path: Path) -> None:
    grade, _ = _fake_grade(tmp_path, _ok('{"score": 1' + "0" * 5000 + ', "feedback": "x"}'))
    assert grade.score == 0.0
    assert grade.feedback is not None and grade.feedback.startswith("the grader's last stdout line is not JSON")


def test_a_cancelled_grade_closes_its_subprocess_on_its_own_loop(
    tmp_path: Path, agent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Without it, the transport is closed by the garbage collector after the loop is
    # gone, which fails inside asyncio; about 6 runs in 10 showed it, so 10 runs miss
    # it with odds near 1 in 10,000.
    unraisable: list[str] = []
    monkeypatch.setattr(sys, "unraisablehook", lambda raised: unraisable.append(repr(raised.exc_value)))
    sleeper = task_line("t1", grader=_python("import time; time.sleep(30)"))
    suite = load_task_suite(write_manifest(tmp_path / "suite", [sleeper]))
    grader = CommandGrader(suite, timeout_seconds=60)

    async def cancel_soon() -> None:
        grading = asyncio.ensure_future(grader.grade(Task("t1", "Fix t1."), _outcome(agent)))
        await asyncio.sleep(0.3)
        grading.cancel()
        with pytest.raises(asyncio.CancelledError):
            await grading

    for _ in range(10):
        asyncio.run(cancel_soon())
        gc.collect()
    assert unraisable == []


class _SuiteBlindRunner:
    """A runner that never reads the suite: each episode gets its own empty workspace."""

    support = RunnerSupport(
        agents=frozenset({"claude_code"}), file_kinds=frozenset({"instructions", "skill"}), programmable_loop=False
    )

    def __init__(self, root: Path) -> None:
        self._root = root
        self._count = 0

    async def run(self, spec: HarnessSpec, task: Task, policy: PolicyRef) -> EpisodeOutcome:
        self._count += 1
        workspace = self._root / f"episode-{self._count}"
        workspace.mkdir(parents=True)
        return EpisodeOutcome(
            task_id=task.task_id,
            spec_id=spec.spec_id,
            policy_checksum=policy.checkpoint.checksum,
            exit_code=0,
            workspace_location=str(workspace),
        )


def test_a_command_grader_passes_the_runner_conformance_checks(tmp_path: Path) -> None:
    # The concurrent check once renamed tasks (t1-0, t1-1, ...), which no suite holds.
    (tmp_path / "suite").mkdir()
    grader_argv = _python(PASS)
    lines = [task_line("t1", grader=grader_argv), task_line("t2", grader=grader_argv)]
    suite = load_task_suite(write_manifest(tmp_path / "suite", lines))
    checks = check_runner(
        _SuiteBlindRunner(tmp_path / "episodes"),
        CommandGrader(suite, timeout_seconds=60),
        seed=SEED_HARNESS,
        tasks=suite.tasks("train"),
        snapshot=FakeModelSource(FakeWorld()).materialize(tmp_path / "model"),
    )
    results = {check.name: (check.passed, check.detail) for check in checks}
    # Only delivery fails: this runner writes no harness files and no probe is given.
    assert {name for name, (passed, _) in results.items() if not passed} == {"delivery"}
    assert results["concurrent"] == (True, "4 episodes at once finished")
