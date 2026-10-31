"""The all-planes pipeline and the runner contract a harness framework (OpenHands, Reef) must pass.

The pipeline runs ``configs/e2e/all_planes.json``: fakes for every plane but the harness runner, which is the
reference workspace runner around a stand-in agent process. The runner checks pass on that runner and fail, by name,
on runners that break one part of the contract each.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from honing.core import (
    ContractError,
    EpisodeOutcome,
    EvolveContext,
    HarnessSpec,
    PolicyRef,
    ProposeResult,
    RunnerSupport,
    Task,
)
from honing.e2e import (
    WorkspaceDeliveryProbe,
    WorkspaceEpisodeRunner,
    check_runner,
    probe_harness,
    resolve_references,
)
from honing.e2e.check import main as check_main
from honing.e2e.components import PipelineContext
from honing.e2e.pipeline import build, main, run
from tests.fakes.builder import TASKS
from tests.fakes.harness import SEED_HARNESS, AppendInstructionMethod, FakeEpisodeRunner, FakeTaskGrader
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
        ({"harness": {"episode_budget": -1}}, "episode_budget must be an integer >= 0"),
        ({"harness_tasks": {"factory": "tests.test_e2e_pipeline:no_tasks"}}, "harness_tasks factory must return"),
    ],
)
def test_a_bad_config_is_refused_before_anything_runs(tmp_path: Path, change: dict[str, Any], message: str) -> None:
    with pytest.raises(ContractError, match=message):
        build(_config(**change), REPO, tmp_path / "out")


class RecordingMethod(AppendInstructionMethod):
    """Remembers the tasks and episode budget each harness move gave it."""

    def __init__(self, seen: list[tuple[tuple[str, ...], int]]) -> None:
        super().__init__()
        self._seen = seen

    def propose(self, context: EvolveContext) -> ProposeResult:
        self._seen.append((tuple(task.task_id for task in context.tasks), context.episode_budget))
        return super().propose(context)


def recording_method(context: PipelineContext) -> RecordingMethod:
    return RecordingMethod(context.shared.setdefault("seen", []))  # type: ignore[arg-type]


def first_task(context: PipelineContext) -> tuple[Task, ...]:
    return TASKS[:1]


def no_tasks(context: PipelineContext) -> tuple[Task, ...]:
    return ()


@pytest.mark.unit
def test_a_harness_move_gets_its_own_tasks_and_budget_while_the_gate_keeps_tasks(tmp_path: Path) -> None:
    config = _config(
        harness_tasks={"factory": "tests.test_e2e_pipeline:first_task"},
        budget=4,
        harness={"episode_budget": {"ref": "budget"}},  # build resolves references first
        schedule=[{"kind": "harness", "label": "verify"}],
        expect=["published"],
    )
    config["components"] = {**config["components"], "method": {"factory": "tests.test_e2e_pipeline:recording_method"}}
    pipeline = build(config, REPO, tmp_path / "out")

    summary = run(pipeline)

    assert summary["failures"] == []
    assert pipeline.context.shared["seen"] == [((TASKS[0].task_id,), 4)]
    assert summary["steps"][0]["episodes"] == 2 * len(TASKS)


@pytest.mark.unit
def test_without_harness_entries_a_harness_move_gets_the_gates_tasks_and_no_budget(tmp_path: Path) -> None:
    config = _config(schedule=[{"kind": "harness", "label": "verify"}], expect=["published"])
    config["components"] = {**config["components"], "method": {"factory": "tests.test_e2e_pipeline:recording_method"}}
    pipeline = build(config, REPO, tmp_path / "out")

    run(pipeline)

    assert pipeline.context.shared["seen"] == [(tuple(task.task_id for task in TASKS), 0)]


@pytest.mark.unit
def test_a_reference_is_replaced_by_the_value_it_names() -> None:
    config = {
        "model": {"provider": "openai", "name": "m-1"},
        "alias": {"ref": "model"},
        "components": {"a": {"args": {"model": {"ref": "alias"}, "names": [{"ref": "model.name"}, "x"]}}},
    }

    resolved = resolve_references(config)

    assert resolved["components"]["a"]["args"] == {"model": config["model"], "names": ["m-1", "x"]}
    assert config["alias"] == {"ref": "model"}  # the input is not changed
    # An object with other keys besides "ref" is an ordinary value.
    assert resolve_references({"x": {"ref": "y", "other": 1}}) == {"x": {"ref": "y", "other": 1}}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"a": {"ref": "missing.path"}}, "no 'missing.path'"),
        ({"a": {"ref": "b"}, "b": {"ref": "a"}}, "form a cycle: b -> a -> b"),
    ],
)
def test_a_reference_to_nothing_or_to_itself_is_refused(config: dict[str, Any], message: str) -> None:
    with pytest.raises(ContractError, match=message):
        resolve_references(config)


@pytest.mark.unit
def test_the_live_config_pins_its_model_in_one_entry() -> None:
    path = REPO / "configs/e2e/gepa_live.json"
    config = json.loads(path.read_text())
    resolved = resolve_references(config)

    pin = config["model"]
    assert resolved["components"]["endpoints"]["args"]["model"] == pin
    assert resolved["components"]["method"]["args"]["reflection"]["model"] == pin
    assert path.read_text().count(pin["name"]) == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"kind": "train"}, "a live run has harness moves and rollbacks, not 'train'"),
        ({"kind": "harness", "repeat": 0}, "repeat must be an integer >= 1"),
        ({"kind": "harness_until_publish"}, "max_moves must be an integer >= 1"),
        ({"kind": "harness_until_publish", "max_moves": True}, "max_moves must be an integer >= 1"),
        ({"kind": "rollback"}, "a rollback names a release id"),
    ],
)
def test_a_live_schedule_step_it_cannot_run_is_refused(entry: dict[str, Any], message: str) -> None:
    from honing.e2e.gepa_live import check_schedule

    check_schedule([{"kind": "harness"}, {"kind": "rollback", "to": "first_publish"}])
    with pytest.raises(ContractError, match=f"schedule step 2: {message}"):
        check_schedule([{"kind": "harness"}, entry])


def _check(name: str, passed: bool = True, not_exercised: bool = False) -> dict[str, Any]:
    return {"name": name, "passed": passed, "detail": "", "not_exercised": not_exercised}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("checks", "skip"),
    [
        ([_check("1 extracted"), _check("6 rollback")], False),
        ([_check("4 rejected stays", False, True)], True),
        ([_check("4 rejected stays", False, True), _check("6 rollback", False, True)], True),
        ([_check("3 gate on real scores", False, True)], False),
        ([_check("6 rollback", False)], False),
        ([_check("4 rejected stays", False, True), _check("5c every episode accounted for", False)], False),
    ],
    ids=["all-pass", "no-rejection", "no-rejection-no-rollback", "no-gate", "real-rollback-failure", "mixed"],
)
def test_only_an_unexercised_rejection_or_rollback_is_a_rerun(checks: list[dict[str, Any]], skip: bool) -> None:
    from honing.e2e.gepa_live import only_not_exercised

    assert only_not_exercised(checks) is skip


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


def _python_agent(tmp_path: Path, context: PipelineContext, code: str, **options: Any) -> WorkspaceEpisodeRunner:
    return WorkspaceEpisodeRunner(
        command=[sys.executable, "-c", code],
        workspaces=tmp_path / "workspaces",
        endpoints=context.components["endpoints"],
        support=RunnerSupport(
            agents=frozenset({"claude_code"}), file_kinds=frozenset({"instructions", "skill"}), programmable_loop=False
        ),
        **options,
    )


def _policy(snapshot: Any) -> PolicyRef:
    return PolicyRef(checkpoint=snapshot.checkpoint, architecture=snapshot.architecture, variant="master")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.unit
def test_a_cancelled_episode_kills_its_agent(tmp_path: Path) -> None:
    context, snapshot = _setup(tmp_path)
    pid_file = tmp_path / "agent.pid"
    code = f"import os, pathlib, time; pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(30)"
    runner = _python_agent(tmp_path, context, code)

    async def fail_once_the_agent_runs() -> None:
        while not pid_file.exists():
            await asyncio.sleep(0.01)
        raise RuntimeError("a sibling episode failed")

    async def gate() -> None:
        # As GEPA's minibatch does: one failure ends the gather, and asyncio.run cancels the rest.
        await asyncio.gather(runner.run(SEED_HARNESS, TASKS[0], _policy(snapshot)), fail_once_the_agent_runs())

    with pytest.raises(RuntimeError, match="a sibling episode failed"):
        asyncio.run(gate())
    pid = int(pid_file.read_text())
    try:
        assert not _alive(pid)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)


@pytest.mark.unit
def test_a_tasks_starting_files_are_copied_in_under_the_harness(tmp_path: Path) -> None:
    context, snapshot = _setup(tmp_path)
    start = tmp_path / "start"
    (start / "pkg").mkdir(parents=True)
    (start / "buggy.py").write_text("x = 1\n")
    (start / "pkg" / "module.py").write_text("y = 2\n")
    runner = _python_agent(
        tmp_path, context, "pass", seed_workspace=lambda task: start if task.task_id == TASKS[0].task_id else None
    )

    seeded = Path(str(asyncio.run(runner.run(SEED_HARNESS, TASKS[0], _policy(snapshot))).workspace_location))
    bare = Path(str(asyncio.run(runner.run(SEED_HARNESS, TASKS[1], _policy(snapshot))).workspace_location))

    assert (seeded / "buggy.py").read_text() == "x = 1\n"
    assert (seeded / "pkg" / "module.py").read_text() == "y = 2\n"
    assert all((seeded / path).read_text() == file.content for path, file in SEED_HARNESS.files)
    assert json.loads((seeded / "task.json").read_text())["task_id"] == TASKS[0].task_id
    assert not (bare / "buggy.py").exists()


@pytest.mark.unit
def test_a_link_inside_the_starting_files_is_copied_as_a_link(tmp_path: Path) -> None:
    context, snapshot = _setup(tmp_path)
    start = tmp_path / "start"
    (start / "pkg").mkdir(parents=True)
    (start / "buggy.py").write_text("x = 1\n")
    (start / "alias.py").symlink_to("buggy.py")
    (start / "pkg" / "up.py").symlink_to("../buggy.py")
    runner = _python_agent(tmp_path, context, "pass", seed_workspace=lambda task: start)

    seeded = Path(str(asyncio.run(runner.run(SEED_HARNESS, TASKS[0], _policy(snapshot))).workspace_location))

    assert (seeded / "alias.py").is_symlink()
    assert os.readlink(seeded / "alias.py") == "buggy.py"
    assert (seeded / "pkg" / "up.py").is_symlink()
    assert (seeded / "pkg" / "up.py").resolve() == (seeded / "buggy.py").resolve()


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["absolute", "climbs_out"])
def test_a_link_that_is_absolute_or_leaves_the_starting_files_is_refused(tmp_path: Path, kind: str) -> None:
    context, snapshot = _setup(tmp_path)
    start = tmp_path / "start"
    (start / "pkg").mkdir(parents=True)
    (start / "buggy.py").write_text("x = 1\n")
    (tmp_path / "secret.json").write_text("{}")
    # The absolute link points inside the starting files, so only its being absolute refuses it.
    target = str((start / "buggy.py").resolve()) if kind == "absolute" else "../../secret.json"
    (start / "pkg" / "link.py").symlink_to(target)
    runner = _python_agent(tmp_path, context, "pass", seed_workspace=lambda task: start)

    with pytest.raises(ContractError, match=r"link that is absolute or leaves them: pkg/link\.py ->"):
        asyncio.run(runner.run(SEED_HARNESS, TASKS[0], _policy(snapshot)))


@pytest.mark.unit
def test_a_dangling_link_that_clashes_with_a_runner_file_is_refused(tmp_path: Path) -> None:
    context, snapshot = _setup(tmp_path)
    start = tmp_path / "start"
    start.mkdir()
    # Nothing exists behind the link, but the runner would still write task.json through it.
    (start / "task.json").symlink_to("missing.json")
    runner = _python_agent(tmp_path, context, "pass", seed_workspace=lambda task: start)

    with pytest.raises(ContractError, match=r"starting files use names .*task\.json"):
        asyncio.run(runner.run(SEED_HARNESS, TASKS[0], _policy(snapshot)))


@pytest.mark.unit
@pytest.mark.parametrize("clash", ["task.json", "result.json", "harness"])
def test_starting_files_that_clash_with_runner_or_harness_files_are_refused(tmp_path: Path, clash: str) -> None:
    context, snapshot = _setup(tmp_path)
    start = tmp_path / "start"
    name = SEED_HARNESS.files[0][0] if clash == "harness" else clash
    (start / name).parent.mkdir(parents=True, exist_ok=True)
    (start / name).write_text("stale")
    runner = _python_agent(tmp_path, context, "pass", seed_workspace=lambda task: start)

    with pytest.raises(ContractError, match=f"starting files use names .*{name}"):
        asyncio.run(runner.run(SEED_HARNESS, TASKS[0], _policy(snapshot)))


@pytest.mark.unit
def test_the_agent_gets_only_the_base_and_forwarded_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, snapshot = _setup(tmp_path)
    monkeypatch.setenv("E2E_TEST_SECRET", "not-for-the-agent")
    monkeypatch.setenv("E2E_TEST_FORWARDED", "for-the-agent")
    code = "import json, os; open('env.json', 'w').write(json.dumps(dict(os.environ)))"
    runner = _python_agent(
        tmp_path, context, code, forward_environment=["E2E_TEST_FORWARDED"], environment={"E2E_TEST_EXTRA": "1"}
    )

    outcome = asyncio.run(runner.run(SEED_HARNESS, TASKS[0], _policy(snapshot)))
    seen = json.loads((Path(str(outcome.workspace_location)) / "env.json").read_text())

    assert "E2E_TEST_SECRET" not in seen
    assert seen["E2E_TEST_FORWARDED"] == "for-the-agent" and seen["E2E_TEST_EXTRA"] == "1"
    assert seen["PATH"] == os.environ["PATH"]
    assert {"HONING_POLICY_URL", "HONING_POLICY_MODEL", "HONING_POLICY_CHECKSUM", "HONING_WORKSPACE"} <= set(seen)


@pytest.mark.unit
def test_the_check_command_runs_on_a_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = check_main(["--config", str(ALL_PLANES), "--root", str(REPO), "--work-dir", str(tmp_path / "check")])
    assert code == 0
    assert "runner check: ok" in capsys.readouterr().out
