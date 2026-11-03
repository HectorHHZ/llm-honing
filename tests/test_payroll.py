"""The payroll task suite: built from a stand-in upstream checkout, graded by grade.py through CommandGrader.

The stand-in has upstream's module paths and functions (setup_workspace, run_single_instance_eval) and keeps
the ground truth beside the workspace as upstream does, so these tests need neither the real checkout nor
Docker. The live config is checked by building its campaign with that suite and no episodes.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from integrations.payroll.build_suite import build as build_suite

from honing.backends.openhands import Container, OpenHandsEpisodeRunner
from honing.core import EpisodeOutcome, Task
from honing.e2e.pipeline import build
from honing.harness.graders import CommandGrader
from honing.harness.methods.meta_harness import MetaHarnessMethod
from honing.harness.proposers import ClaudeCodeProposer
from honing.harness.tasks import load_task_suite

REPO = Path(__file__).resolve().parents[1]

SETUP = """
import json
from pathlib import Path


def setup_workspace(workspace_dir, log_dir, example):
    documents = Path(workspace_dir, "Documents")
    documents.mkdir(parents=True, exist_ok=True)
    (documents / "attendance.csv").write_text(example["csv"])
    groundtruth = Path(log_dir, "groundtruth")
    groundtruth.mkdir(parents=True)
    (groundtruth / "expected.json").write_text(json.dumps({"total": example["total"]}))
    (groundtruth / "example.json").write_text(json.dumps(example))
"""
EVAL = """
import json
from pathlib import Path


def run_single_instance_eval(workspace_dir, example, lm=None):
    workspace = Path(workspace_dir)
    expected = json.loads((workspace.parent / f"{workspace.name}_logs/groundtruth/expected.json").read_text())
    answer = workspace / "audit.json"
    passed = answer.exists() and json.loads(answer.read_text()).get("total") == expected["total"]
    feedback = "All checkpoints passed." if passed else "Failed checkpoints: total"
    return {"score": 1.0 if passed else 0.0, "feedback": feedback, "example_id": example["id"]}
"""


@pytest.fixture
def upstream(tmp_path: Path) -> Iterator[Path]:
    """A stand-in slm-harness-adaptation checkout with four examples."""
    root = tmp_path / "slm-harness-adaptation"
    for module, source in (
        ("src/task_setups/attendance_payroll_audit_s2l.py", SETUP),
        ("src/task_evals/attendance_payroll_audit_s2l.py", EVAL),
    ):
        (root / module).parent.mkdir(parents=True, exist_ok=True)
        (root / module).write_text(source)
    examples = [
        {"id": f"payroll/{index}", "prompt": f"Audit payroll {index}.", "csv": f"hours\n{index}\n", "total": index}
        for index in range(4)
    ]
    (root / "data").mkdir()
    (root / "data/attendance_payroll_audit_s2l_extra_high.json").write_text(json.dumps(examples))
    (root / "tasks/attendance_payroll_audit_s2l/prompts").mkdir(parents=True)
    (root / "tasks/attendance_payroll_audit_s2l/prompts/default.md").write_text("You audit payroll.\n")
    _forget_upstream()
    yield root
    _forget_upstream()


def _forget_upstream() -> None:
    """Upstream's ``src`` namespace is cached per process; each stand-in checkout needs its own."""
    for name in [name for name in sys.modules if name == "src" or name.startswith("src.")]:
        del sys.modules[name]


@pytest.mark.unit
def test_the_suite_has_upstreams_split_workspaces_and_hidden_ground_truth(tmp_path: Path, upstream: Path) -> None:
    manifest = build_suite(upstream, tmp_path / "suite", train=2, val=1)

    suite = load_task_suite(manifest)
    assert [task.task_id for task in suite.tasks("train")] == ["payroll/0", "payroll/1"]
    assert [task.task_id for task in suite.tasks("val")] == ["payroll/2"]
    entry = suite.entry("payroll/2")
    assert entry.prompt == "Audit payroll 2."
    assert (entry.workspace / "Documents/attendance.csv").read_text() == "hours\n2\n"
    # The ground truth is only in the grader's files, never in what the agent starts from.
    assert entry.grader_files is not None
    assert json.loads((entry.grader_files / "groundtruth/expected.json").read_text()) == {"total": 2}
    assert not list(entry.workspace.rglob("expected.json"))
    assert entry.grader == (sys.executable, "grade.py", "--upstream", str(upstream.resolve()))
    assert (tmp_path / "suite/seed/AGENTS.md").read_text() == "You audit payroll.\n"


def _episode(tmp_path: Path, answer: object | None) -> EpisodeOutcome:
    """An episode directory as the OpenHands runner leaves it, with the agent's files under ``work/``."""
    root = tmp_path / "episode"
    (root / "work").mkdir(parents=True)
    if answer is not None:
        (root / "work/audit.json").write_text(json.dumps(answer))
    return EpisodeOutcome(
        task_id="payroll/1", spec_id="h-x", policy_checksum="ck", exit_code=0, workspace_location=str(root)
    )


@pytest.mark.integration
@pytest.mark.parametrize(("answer", "score"), [({"total": 1}, 1.0), ({"total": 7}, 0.0), (None, 0.0)])
def test_grade_py_scores_the_agents_files_with_upstreams_eval(
    tmp_path: Path, upstream: Path, answer: object | None, score: float
) -> None:
    grader = CommandGrader(load_task_suite(build_suite(upstream, tmp_path / "suite", train=2, val=1)))

    grade = asyncio.run(grader.grade(Task("payroll/1", "Audit payroll 1."), _episode(tmp_path, answer)))

    assert grade.score == score
    assert grade.feedback == ("All checkpoints passed." if score else "Failed checkpoints: total")


@pytest.mark.unit
def test_the_live_config_builds_its_campaign(tmp_path: Path, upstream: Path) -> None:
    suite_dir = tmp_path / "suite"
    manifest = build_suite(upstream, suite_dir, train=2, val=1)
    config = json.loads((REPO / "configs/e2e/payroll_meta_harness.json").read_text())
    config["suite"] = str(manifest)
    config["requires"] = {}
    del config["logging"]
    config["harness_seed"]["args"]["directory"] = str(suite_dir / "seed")
    config["components"]["runner"]["args"]["manifest"] = str(manifest)

    pipeline = build(config, REPO, tmp_path / "run")

    components = pipeline.context.components
    runner, method = components["runner"], components["method"]
    assert isinstance(runner, OpenHandsEpisodeRunner) and runner._container == Container(image="honing-payroll")
    assert isinstance(method, MetaHarnessMethod) and isinstance(method.proposer, ClaudeCodeProposer)
    assert not method.work_dir.is_relative_to(REPO)
    settings = json.loads((tmp_path / "run/proposer-settings.json").read_text())
    assert f"Read(/{REPO}/**)" in settings["permissions"]["deny"]
    head = pipeline.campaign.chain.head()
    assert head.harness is not None and head.harness.get("AGENTS.md") is not None
