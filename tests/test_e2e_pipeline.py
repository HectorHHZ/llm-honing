"""The all-planes pipeline and the runner contract a harness framework (OpenHands, Reef) must pass.

The pipeline runs ``configs/e2e/all_planes.json``: fakes for every plane but the harness runner, which is the
reference workspace runner around a stand-in agent process. The runner checks pass on that runner and fail, by name,
on runners that break one part of the contract each.
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from honing.core import ContractError, EpisodeOutcome, HarnessSpec, PolicyRef, RunnerSupport, Task
from honing.e2e import WorkspaceDeliveryProbe, WorkspaceEpisodeRunner, check_runner, probe_harness
from honing.e2e.check import main as check_main
from honing.e2e.components import PipelineContext
from honing.e2e.pipeline import build, main, run
from tests.fakes.builder import TASKS
from tests.fakes.harness import SEED_HARNESS, FakeEpisodeRunner, FakeTaskGrader
from tests.fakes.model import FakeModelSource, FakeWorld
from tests.fakes.pipeline import FakePolicyEndpoints
from tests.fakes.pipeline import runner as workspace_runner

REPO = Path(__file__).resolve().parents[1]
ALL_PLANES = REPO / "configs/e2e/all_planes.json"
EXPECTED = ["published", "published", "rolled back", "published", "published", "published", "skipped", "published"]


def _config(**changes: Any) -> dict[str, Any]:
    return {**json.loads(ALL_PLANES.read_text()), **changes}


# ---------------------------------------------------------------- the pipeline


@pytest.mark.unit
def test_the_pipeline_crosses_every_plane_and_restores_a_release(tmp_path: Path) -> None:
    code = main(["--config", str(ALL_PLANES), "--root", str(REPO), "--work-dir", str(tmp_path / "out")])

    assert code == 0
    summary = json.loads((tmp_path / "out/summary.json").read_text())
    assert [step["result"] for step in summary["steps"]] == [*EXPECTED, "restored"]
    assert [step["move"] for step in summary["steps"]] == [
        "train", "prune", "quantize", "train", "quantize", "harness", "harness", "train", "rollback",
    ]  # fmt: skip
    assert [r["kind"] for r in summary["releases"]] == [
        "init",
        "train",
        "prune",
        "train",
        "quantize",
        "harness",
        "train",
    ]
    assert summary["head"] == "r0001" and summary["failures"] == []
    pruned = summary["releases"][2]
    assert pruned["parameters"] < summary["releases"][1]["parameters"]
    # The harness episodes ran as agent processes that asked the policy server.
    transcripts = list((tmp_path / "out/workspaces").glob("*/transcript.jsonl"))
    assert transcripts
    request = json.loads(transcripts[0].read_text().splitlines()[1])["request"]
    assert request["messages"][0]["role"] == "system" and request["model"] == "fake-policy"


@pytest.mark.unit
def test_an_unexpected_result_fails_the_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_config(expect=["rolled back", *EXPECTED[1:], "restored"])))

    code = main(["--config", str(path), "--root", str(REPO), "--work-dir", str(tmp_path / "out")])

    assert code == 1
    assert "step 1 (train): expected rolled back, got published" in capsys.readouterr().err


def in_process_runner(context: PipelineContext) -> FakeEpisodeRunner:
    """A runner from somewhere else entirely, swapped in through the config."""
    return FakeEpisodeRunner(context.shared.setdefault("world", FakeWorld()))  # type: ignore[arg-type]


@pytest.mark.unit
def test_a_runner_is_swapped_by_changing_one_config_entry(tmp_path: Path) -> None:
    config = _config()
    config["components"] = {**config["components"], "runner": {"factory": "tests.test_e2e_pipeline:in_process_runner"}}

    summary = run(build(config, REPO, tmp_path / "out"))

    assert summary["failures"] == []
    assert not (tmp_path / "out/workspaces").exists()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"schedule": [{"kind": "serve"}], "expect": []}, "not 'serve'"),
        ({"expect": ["published"]}, "expect needs one of"),
        ({"requires": {"commands": ["no-such-command-xyz"]}}, "command no-such-command-xyz"),
    ],
)
def test_a_bad_config_is_refused_before_anything_runs(tmp_path: Path, change: dict[str, Any], message: str) -> None:
    with pytest.raises(ContractError, match=message):
        build(_config(**change), REPO, tmp_path / "out")


@pytest.mark.unit
def test_every_required_plane_must_be_named(tmp_path: Path) -> None:
    config = _config()
    config["components"] = {k: v for k, v in config["components"].items() if k != "quantizer"}
    with pytest.raises(ContractError, match="names no quantizer component"):
        build(config, REPO, tmp_path / "out")


# ---------------------------------------------------------------- the runner contract


def _setup(tmp_path: Path) -> tuple[PipelineContext, Any]:
    world = FakeWorld()
    context = PipelineContext(root=REPO, work_dir=tmp_path, shared={"world": world})
    context.components["endpoints"] = FakePolicyEndpoints(world)
    snapshot = FakeModelSource(world).materialize(tmp_path)
    return context, snapshot


def _results(checks: list[Any]) -> dict[str, bool]:
    return {check.name: check.passed for check in checks}


@pytest.mark.unit
def test_the_reference_runner_passes_every_check(tmp_path: Path) -> None:
    context, snapshot = _setup(tmp_path)

    checks = check_runner(
        workspace_runner(context), FakeTaskGrader(), seed=SEED_HARNESS, tasks=TASKS, snapshot=snapshot,
        probe=WorkspaceDeliveryProbe(),
    )  # fmt: skip

    assert _results(checks) == dict.fromkeys(["support", "episodes", "delivery", "concurrent", "repeat", "gate"], True)


@pytest.mark.unit
def test_the_probe_harness_has_one_file_of_every_declared_kind() -> None:
    support = RunnerSupport(
        agents=frozenset({"claude_code"}),
        file_kinds=frozenset({"instructions", "skill", "hook"}),
        programmable_loop=False,
    )
    probed = probe_harness(SEED_HARNESS, support)
    assert probed.kinds == {"instructions", "skill", "hook"}
    assert len({file.content for _, file in probed.files}) == len(probed.files)


class DropsSkills(WorkspaceEpisodeRunner):
    """Declares skills but never writes them where the agent reads."""

    def _prepare(self, spec: HarnessSpec, task: Task) -> Path:
        kept = HarnessSpec(agent=spec.agent, files=tuple((p, f) for p, f in spec.files if f.kind != "skill"))
        return super()._prepare(kept, task)


class MisreportsHarness(WorkspaceEpisodeRunner):
    """Reports a different harness than the one it was asked to run."""

    async def run(self, spec: HarnessSpec, task: Task, policy: PolicyRef) -> EpisodeOutcome:
        outcome = await super().run(spec, task, policy)
        return replace(outcome, spec_id="something-else")


class SharesOneWorkspace:
    """Runs every episode in the same place."""

    def __init__(self, inner: WorkspaceEpisodeRunner) -> None:
        self.support = inner.support
        self._inner = inner

    async def run(self, spec: HarnessSpec, task: Task, policy: PolicyRef) -> EpisodeOutcome:
        outcome = await self._inner.run(spec, task, policy)
        return replace(outcome, workspace_location="/shared")


def _subclassed(kind: type, context: PipelineContext) -> WorkspaceEpisodeRunner:
    base = workspace_runner(context)
    runner = kind.__new__(kind)
    runner.__dict__.update(base.__dict__)
    return runner  # type: ignore[no-any-return]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("make", "failing"),
    [
        (lambda c: _subclassed(DropsSkills, c), {"delivery"}),
        (lambda c: _subclassed(MisreportsHarness, c), {"episodes", "delivery", "concurrent", "repeat", "gate"}),
        (lambda c: SharesOneWorkspace(workspace_runner(c)), {"delivery", "concurrent"}),
    ],
    ids=["drops-a-file-kind", "misreports-its-harness", "shares-a-workspace"],
)
def test_a_runner_that_breaks_the_contract_fails_the_check_it_breaks(
    tmp_path: Path, make: Any, failing: set[str]
) -> None:
    context, snapshot = _setup(tmp_path)

    checks = check_runner(
        make(context), FakeTaskGrader(), seed=SEED_HARNESS, tasks=TASKS, snapshot=snapshot,
        probe=WorkspaceDeliveryProbe(),
    )  # fmt: skip

    assert {name for name, passed in _results(checks).items() if not passed} == failing


@pytest.mark.unit
def test_a_runner_without_the_seed_agent_stops_at_support(tmp_path: Path) -> None:
    context, snapshot = _setup(tmp_path)
    checks = check_runner(
        workspace_runner(context, agents=("openhands",)), FakeTaskGrader(), seed=SEED_HARNESS, tasks=TASKS,
        snapshot=snapshot, probe=WorkspaceDeliveryProbe(),
    )  # fmt: skip
    assert [(c.name, c.passed) for c in checks] == [("support", False)]
    assert "no agent 'claude_code'" in checks[0].detail


@pytest.mark.unit
def test_delivery_needs_a_probe_and_says_so(tmp_path: Path) -> None:
    context, snapshot = _setup(tmp_path)
    checks = check_runner(
        workspace_runner(context), FakeTaskGrader(), seed=SEED_HARNESS, tasks=TASKS, snapshot=snapshot
    )
    (delivery,) = [c for c in checks if c.name == "delivery"]
    assert not delivery.passed and "no delivery probe" in delivery.detail


@pytest.mark.unit
def test_an_agent_that_outlives_its_timeout_is_killed_and_reported(tmp_path: Path) -> None:
    context, snapshot = _setup(tmp_path)
    slow = WorkspaceEpisodeRunner(
        command=[sys.executable, "-c", "import time; time.sleep(30)"],
        workspaces=tmp_path / "slow",
        endpoints=context.components["endpoints"],
        support=RunnerSupport(
            agents=frozenset({"claude_code"}), file_kinds=frozenset({"instructions"}), programmable_loop=False
        ),
        timeout_seconds=0.5,
    )
    policy = PolicyRef(checkpoint=snapshot.checkpoint, architecture=snapshot.architecture, variant="master")

    outcome = asyncio.run(slow.run(SEED_HARNESS, TASKS[0], policy))

    assert outcome.exit_code == -9 and outcome.final_output == ""


@pytest.mark.unit
def test_the_check_command_runs_on_a_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = check_main(["--config", str(ALL_PLANES), "--root", str(REPO), "--work-dir", str(tmp_path / "check")])
    assert code == 0
    assert "runner check: ok" in capsys.readouterr().out
