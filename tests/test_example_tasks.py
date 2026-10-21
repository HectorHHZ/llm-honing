"""The python-bugfix suite: its grader gives 0 to every unmodified workspace and 1 to its reference fix."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from honing.core import EpisodeOutcome, Grade
from honing.harness.graders import CommandGrader
from honing.harness.tasks import load_task_suite

SUITE = Path(__file__).resolve().parents[1] / "integrations" / "python-bugfix" / "tasks"
TASK_IDS = ("clamp", "mean", "reverse_words")


def _grade(task_id: str, workspace: Path) -> Grade:
    suite = load_task_suite(SUITE / "suite.jsonl")
    entry = suite.entry(task_id)
    outcome = EpisodeOutcome(
        task_id=task_id, spec_id="h-seed", policy_checksum="sha", exit_code=0, workspace_location=str(workspace)
    )
    return asyncio.run(CommandGrader(suite, timeout_seconds=60).grade(entry.task, outcome))


def _agent_copy(task_id: str, tmp_path: Path) -> Path:
    """What an agent would start from: a copy of the task's workspace."""
    workspace = tmp_path / "agent"
    shutil.copytree(SUITE / task_id / "workspace", workspace)
    return workspace


def test_the_suite_has_three_tasks_one_per_split() -> None:
    suite = load_task_suite(SUITE / "suite.jsonl")
    assert tuple(entry.task_id for entry in suite.entries) == TASK_IDS
    assert [suite.split_of(task_id) for task_id in TASK_IDS] == ["train", "val", "heldout"]
    # The hidden tests and the grading script live outside every workspace.
    for task_id in TASK_IDS:
        assert not any(path.name.startswith("test_") for path in (SUITE / task_id / "workspace").rglob("*"))


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_an_unmodified_workspace_scores_zero(task_id: str, tmp_path: Path) -> None:
    grade = _grade(task_id, _agent_copy(task_id, tmp_path))
    assert grade.score == 0.0
    assert grade.feedback is not None and grade.feedback.startswith("0 of 3 tests pass.")


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_the_reference_fix_scores_one(task_id: str, tmp_path: Path) -> None:
    workspace = _agent_copy(task_id, tmp_path)
    shutil.copytree(SUITE / task_id / "reference", workspace, dirs_exist_ok=True)
    assert _grade(task_id, workspace) == Grade(score=1.0, feedback="3 of 3 tests pass.")


def _entries() -> list[tuple[str, str]]:
    """Every task of every manifest in the suite's directory, so a task added later is checked too."""
    return [
        (manifest.name, entry.task_id)
        for manifest in sorted(SUITE.glob("*.jsonl"))
        for entry in load_task_suite(manifest).entries
    ]


@pytest.mark.parametrize(("manifest", "task_id"), _entries())
def test_every_task_is_solvable_and_starts_unsolved(manifest: str, task_id: str, tmp_path: Path) -> None:
    suite = load_task_suite(SUITE / manifest)
    entry = suite.entry(task_id)
    grader = CommandGrader(suite, timeout_seconds=60)

    def grade(workspace: Path) -> Grade:
        outcome = EpisodeOutcome(
            task_id=task_id, spec_id="h-seed", policy_checksum="sha", exit_code=0, workspace_location=str(workspace)
        )
        return asyncio.run(grader.grade(entry.task, outcome))

    start = tmp_path / "start"
    shutil.copytree(entry.workspace, start)
    fixed = tmp_path / "fixed"
    shutil.copytree(entry.workspace, fixed)
    shutil.copytree(entry.workspace.parent / "reference", fixed, dirs_exist_ok=True)

    assert grade(start).score < 1.0
    assert grade(fixed).score == 1.0


def test_a_returned_bool_matches_only_the_same_bool(tmp_path: Path) -> None:
    shutil.copy(SUITE / "grader" / "grade.py", tmp_path / "grade.py")
    (tmp_path / "echo.py").write_text("def echo(value):\n    return value\n")
    cases = [([True], True), ([False], False), ([True], 1), ([1], True), ([1], 1), ([0], False)]
    spec = {"module": "echo", "function": "echo", "cases": [{"args": a, "expected": e} for a, e in cases]}
    (tmp_path / "cases").mkdir()
    (tmp_path / "cases" / "echo.json").write_text(json.dumps(spec))

    result = subprocess.run(
        [sys.executable, "-I", "grade.py", "cases/echo.json"],
        cwd=tmp_path,
        env={"PATH": os.environ.get("PATH", os.defpath), "HONING_WORKSPACE": str(tmp_path)},
        capture_output=True,
        text=True,
        check=True,
    )
    grade = json.loads(result.stdout)

    assert grade["score"] == pytest.approx(3 / 6)
    assert grade["feedback"].split("\n")[1:] == [
        "echo(True) returned True, expected 1",
        "echo(1) returned 1, expected True",
        "echo(0) returned 0, expected False",
    ]


@pytest.mark.parametrize("workspace", [None, ""])
def test_grade_py_run_without_a_workspace_exits_before_deleting_its_cases(
    workspace: str | None, tmp_path: Path
) -> None:
    shutil.copytree(SUITE / "grader", tmp_path / "grader")
    env = {"PATH": os.environ.get("PATH", os.defpath)}
    if workspace is not None:
        env["HONING_WORKSPACE"] = workspace
    result = subprocess.run(
        [sys.executable, "-I", "grade.py", "cases/clamp.json"],
        cwd=tmp_path / "grader",
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert result.stdout == ""
    assert "HONING_WORKSPACE is not set" in result.stderr
    assert sorted(path.name for path in (tmp_path / "grader" / "cases").iterdir()) == [
        "clamp.json",
        "mean.json",
        "reverse_words.json",
    ]


# The agent writes the workspace, so it must not be able to change how its work is graded.

SHADOW = 'import sys\nsys.stdout.write(\'{"score": 1.0, "feedback": "shadowed"}\\n\')\nsys.exit(0)\n'


@pytest.mark.parametrize("module", ["unittest", "json"])
@pytest.mark.parametrize("task_id", TASK_IDS)
def test_a_workspace_module_cannot_shadow_the_standard_library(task_id: str, module: str, tmp_path: Path) -> None:
    workspace = _agent_copy(task_id, tmp_path)
    (workspace / f"{module}.py").write_text(SHADOW)
    grade = _grade(task_id, workspace)
    assert grade.score == 0.0
    assert grade.feedback is not None and grade.feedback.startswith("0 of 3 tests pass.")


EXTRA_TESTS = "import unittest\n\n\nclass Extra(unittest.TestCase):\n" + "".join(
    f"    def test_{n}(self) -> None:\n        pass\n\n" for n in range(20)
)


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_tests_the_agent_adds_beside_the_hidden_ones_do_not_count(task_id: str, tmp_path: Path) -> None:
    workspace = _agent_copy(task_id, tmp_path)
    (workspace / "tests" / task_id).mkdir(parents=True)
    (workspace / "tests" / task_id / "test_extra.py").write_text(EXTRA_TESTS)
    grade = _grade(task_id, workspace)
    assert grade.score == 0.0
    assert grade.feedback is not None and grade.feedback.startswith("0 of 3 tests pass.")


# Agent code that tries to forge the grade instead of fixing the bug.

FORGED = 'print(\'{"score": 1, "feedback": "forged"}\', flush=True)\n'
FORGERIES = {
    "prints a grade at import": FORGED,
    "prints a grade and exits at once": "import os\n" + FORGED + "os._exit(0)\n",
    "patches sys.exit and prints a grade last": (
        "import atexit, sys\n"
        + FORGED
        + "sys.exit = lambda *args: None\n"
        + 'atexit.register(lambda: print(\'{"score": 1, "feedback": "forged"}\', flush=True))\n'
    ),
}


@pytest.mark.parametrize("forgery", list(FORGERIES))
@pytest.mark.parametrize("task_id", TASK_IDS)
def test_agent_code_cannot_forge_a_grade(task_id: str, forgery: str, tmp_path: Path) -> None:
    workspace = _agent_copy(task_id, tmp_path)
    module = workspace / f"{task_id}.py"
    module.write_text(FORGERIES[forgery] + module.read_text())
    grade = _grade(task_id, workspace)
    assert grade.score == 0.0
    assert grade.feedback is not None and grade.feedback.startswith("0 of 3 tests pass.")


# Agent code that answers from the hidden cases instead of computing: the clamp.py from
# the #99 review, one that looks where a sibling directory would be, and one that
# searches the temporary directory. The search is given this process's temporary
# directory, since the child's environment has no TMPDIR and an agent could find it.
ANSWER_FROM = (
    "import glob, json\n"
    "def clamp(*a):\n"
    "    s = json.load(open(sorted(glob.glob({pattern!r}, recursive=True))[0]))\n"
    "    return next(c['expected'] for c in s['cases'] if c['args'] == list(a))\n"
)
HIDDEN_CASE_PATTERNS = {
    "in the working directory": "cases/clamp.json",
    "in a sibling directory": "../grader/cases/clamp.json",
    "anywhere under the temporary directory": str(
        Path(tempfile.gettempdir()) / "honing-grader-*" / "**" / "clamp.json"
    ),
}


@pytest.mark.parametrize("where", list(HIDDEN_CASE_PATTERNS))
def test_relative_and_temp_dir_reads_of_the_hidden_cases_score_zero(where: str, tmp_path: Path) -> None:
    workspace = _agent_copy("clamp", tmp_path)
    (workspace / "clamp.py").write_text(ANSWER_FROM.format(pattern=HIDDEN_CASE_PATTERNS[where]))
    grade = _grade("clamp", workspace)
    assert grade.score == 0.0
    assert grade.feedback is not None and grade.feedback.startswith("0 of 3 tests pass.")
    assert grade.feedback.count("raised IndexError") == 3


# The bypass from the #99 review: walk up the parent processes and read the suite's own
# cases through each one's working directory. The harness process that runs the grade
# (here, pytest) works in the checkout, so its cwd holds the source copy grade.py never
# deletes.
ANSWER_FROM_A_PARENTS_CWD = (
    "import json, os\n"
    "def clamp(*a):\n"
    "    pid = os.getppid()\n"
    "    while pid > 1:\n"
    "        try:\n"
    "            path = f'/proc/{pid}/cwd/integrations/python-bugfix/tasks/grader/cases/clamp.json'\n"
    "            s = json.load(open(path))\n"
    "            return next(c['expected'] for c in s['cases'] if c['args'] == list(a))\n"
    "        except OSError:\n"
    "            pid = int(open(f'/proc/{pid}/stat').read().rsplit(')', 1)[1].split()[1])\n"
)


@pytest.mark.skipif(sys.platform != "linux", reason="follows /proc/<pid>/cwd, which only Linux has")
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#16: LocalWorkspace does not isolate files, so agent code reads the hidden cases through a parent's cwd",
)
def test_agent_code_cannot_read_the_hidden_cases_through_a_parents_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The harness runs from the checkout, as in CI, wherever pytest was started.
    monkeypatch.chdir(SUITE.parents[2])
    workspace = _agent_copy("clamp", tmp_path)
    (workspace / "clamp.py").write_text(ANSWER_FROM_A_PARENTS_CWD)
    grade = _grade("clamp", workspace)
    assert grade.score == 0.0


def _fixed(task_id: str, tmp_path: Path, prefix: str) -> Path:
    """The reference fix, with ``prefix`` run first when the agent's module is imported."""
    workspace = _agent_copy(task_id, tmp_path)
    shutil.copytree(SUITE / task_id / "reference", workspace, dirs_exist_ok=True)
    module = workspace / f"{task_id}.py"
    module.write_text(prefix + module.read_text())
    return workspace


def test_the_agents_code_sees_no_honing_variables(tmp_path: Path) -> None:
    prefix = "import os\nassert not [name for name in os.environ if name.startswith('HONING_')], sorted(os.environ)\n"
    assert _grade("clamp", _fixed("clamp", tmp_path, prefix)) == Grade(score=1.0, feedback="3 of 3 tests pass.")


def test_what_the_agents_code_prints_does_not_break_a_correct_fix(tmp_path: Path) -> None:
    prefix = "import os, sys\nprint('debugging')\nos.write(1, b'{\"value\": 0}\\nnoise\\n')\nsys.stdout.flush()\n"
    assert _grade("clamp", _fixed("clamp", tmp_path, prefix)) == Grade(score=1.0, feedback="3 of 3 tests pass.")


def test_a_value_json_cannot_encode_fails_the_case(tmp_path: Path) -> None:
    workspace = _agent_copy("clamp", tmp_path)
    (workspace / "clamp.py").write_text("def clamp(value, low, high):\n    return {value}\n")
    grade = _grade("clamp", workspace)
    assert grade.score == 0.0
    assert grade.feedback is not None and "clamp(5, 0, 10) raised TypeError: Object of type set" in grade.feedback


def test_a_case_that_runs_past_its_limit_fails(tmp_path: Path) -> None:
    suite_copy = tmp_path / "suite"
    shutil.copytree(SUITE, suite_copy)
    manifest = suite_copy / "suite.jsonl"
    lines = [json.loads(line) for line in manifest.read_text().splitlines()]
    lines[0]["metadata"] = {"case_timeout_seconds": 1}
    manifest.write_text("".join(json.dumps(line) + "\n" for line in lines))
    workspace = _fixed("clamp", tmp_path, "import time\ntime.sleep(60)\n")
    suite = load_task_suite(manifest)
    outcome = EpisodeOutcome(
        task_id="clamp", spec_id="h-seed", policy_checksum="sha", exit_code=0, workspace_location=str(workspace)
    )
    started = time.monotonic()
    grade = asyncio.run(CommandGrader(suite, timeout_seconds=60).grade(suite.entry("clamp").task, outcome))
    assert time.monotonic() - started < 20
    assert grade.score == 0.0
    assert grade.feedback is not None and "clamp(5, 0, 10) took longer than 1 s" in grade.feedback


def test_a_huge_returned_value_does_not_lose_the_rest_of_the_grade(tmp_path: Path) -> None:
    workspace = _fixed("clamp", tmp_path, "")
    module = workspace / "clamp.py"
    module.write_text(
        module.read_text()
        + "\n_correct = clamp\n\n\ndef clamp(value, low, high):\n"
        + "    return 'x' * 1_000_000 if value == 5 else _correct(value, low, high)\n"
    )
    grade = _grade("clamp", workspace)
    assert grade.score == pytest.approx(2 / 3)
    assert grade.feedback is not None
    assert grade.feedback.startswith("2 of 3 tests pass.\nclamp(5, 0, 10) returned 'xxx")
    assert len(grade.feedback) < 400


def test_a_result_line_python_cannot_convert_fails_the_case_instead_of_the_grader(tmp_path: Path) -> None:
    # The child's private copy of stdout is the first free descriptor, 3.
    prefix = "import os\nos.write(3, b'{\"value\": 1' + b'0' * 5000 + b'}\\n')\nos._exit(0)\n"
    grade = _grade("clamp", _fixed("clamp", tmp_path, prefix))
    assert grade.score == 0.0
    assert grade.feedback is not None and grade.feedback.startswith("0 of 3 tests pass.")
    assert "clamp(5, 0, 10) gave a result that is not JSON" in grade.feedback
