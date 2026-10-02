"""The python-bugfix example suite: its grader gives 0 to every unmodified workspace and 1 to its reference fix."""

from __future__ import annotations

import asyncio
import json
import shutil
import time
from pathlib import Path

import pytest

from honing.core import EpisodeOutcome, Grade
from honing.harness.graders import CommandGrader
from honing.harness.tasks import load_task_suite

SUITE = Path(__file__).resolve().parents[1] / "examples" / "tasks" / "python-bugfix"
TASK_IDS = ("clamp", "mean", "reverse_words")


def _grade(task_id: str, workspace: Path) -> Grade:
    suite = load_task_suite(SUITE / "manifest.jsonl")
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
    suite = load_task_suite(SUITE / "manifest.jsonl")
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
    manifest = suite_copy / "manifest.jsonl"
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
