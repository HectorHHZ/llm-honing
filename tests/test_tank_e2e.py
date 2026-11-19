"""The tank campaign script: the dry-run config, its moves, startup checks, and the full dry run (with Node)."""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from honing.backends.tank.e2e import _move, build, main, tasks
from honing.backends.tank.scenarios import SEATS, TankScenario, build_suite, write_suite
from honing.backends.tank.traces import TankTraceSource
from honing.core import ContractError

REPO = Path(__file__).resolve().parents[1]
DRY_RUN = REPO / "configs/e2e/tank_dry_run.json"
SGLANG = REPO / "configs/e2e/tank_campaign_sglang.json"
NODE = shutil.which("node")
# CI's test job sets this on every Python version but one, because its e2e job plays the same config.
SKIP_DRY_RUN = "HONING_SKIP_TANK_DRY_RUN"

sys.path.insert(0, str(REPO / "scripts/e2e"))
from check_campaign import problems  # noqa: E402


def _config(**changes: object) -> dict[str, object]:
    return {**json.loads(DRY_RUN.read_text()), **changes}


@pytest.mark.unit
def test_grpo_plays_rollouts_under_the_heads_harness_and_other_moves_are_refused() -> None:
    assert _move({"kind": "train", "algorithm": "grpo"}).train.rollout == "harness"  # type: ignore[union-attr]
    assert _move({"kind": "train", "algorithm": "sft"}).train.rollout == "none"  # type: ignore[union-attr]
    harness = _move({"kind": "harness", "label": "gepa", "episode_budget": 6}).harness
    assert (harness.label, harness.episode_budget) == ("gepa", 6)  # type: ignore[union-attr]
    with pytest.raises(ContractError, match="not 'prune'"):
        _move({"kind": "prune"})


@pytest.mark.unit
def test_a_config_is_refused_before_any_round_when_the_machine_or_the_config_lacks_something(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(ContractError, match="command node"):
        build(_config(), REPO, tmp_path / "a")
    monkeypatch.undo()
    components = _config()["components"]
    without_method = {k: v for k, v in components.items() if k != "method"}  # type: ignore[union-attr]
    with pytest.raises(ContractError, match="names no method"):
        build(_config(requires={}, components=without_method), REPO, tmp_path / "b")
    with pytest.raises(ContractError, match="unknown task split 'dev'"):
        build(_config(requires={}, validation_tasks={"split": "dev"}), REPO, tmp_path / "c")
    with pytest.raises(ContractError, match="unknown gate settings \\['size'\\]"):
        build(_config(requires={}, gate={"size": 8}), REPO, tmp_path / "d")


@pytest.mark.unit
def test_a_config_that_still_names_gate_tasks_is_refused_with_the_settings_that_replace_it(tmp_path: Path) -> None:
    config = _config(gate_tasks={"split": "val", "limit": 16})
    del config["validation_tasks"]
    with pytest.raises(
        ContractError, match=r"gate_tasks is replaced by two settings: validation_tasks, .*, and gate, "
    ):
        build(config, REPO, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.unit
def test_a_suite_that_trains_on_a_gate_pool_seed_is_refused_and_the_committed_suite_builds(tmp_path: Path) -> None:
    # The #197 review's suite: the committed one with its first train pair moved to the first val pool seed,
    # which the first gate would otherwise play.
    records = build_suite()
    for record, seat in zip(records[:2], SEATS, strict=True):
        record.update(task_id=f"tank-train-s100000-{seat}", tank=asdict(TankScenario(100_000, seat)))
    suite = tmp_path / "suite.jsonl"
    write_suite(records, suite)
    with pytest.raises(
        ContractError,
        match=r"val pool, seeds 100000 to 199999, .*: tank-train-s100000-blue \(seed 100000\), "
        r"tank-train-s100000-red \(seed 100000\)$",
    ):
        build(_config(requires={}, suite=str(suite)), REPO, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert build(_config(requires={}), REPO, tmp_path / "committed").campaign.chain.head().release_id == "r0000"


@pytest.mark.unit
def test_no_task_setting_or_factory_can_request_heldout(tmp_path: Path) -> None:
    for setting in ("validation_tasks", "harness_tasks"):
        with pytest.raises(ContractError, match="experiment scripts"):
            build(_config(requires={}, **{setting: {"split": "heldout"}}), REPO, tmp_path / setting)
    pipeline = SimpleNamespace(root=REPO, work_dir=tmp_path, shared={})
    with pytest.raises(ContractError, match="experiment scripts"):
        tasks(pipeline, split="heldout")


@pytest.mark.unit
def test_the_sglang_campaign_builds_gepa_reading_the_rounds_traces_as_the_dry_run_does(tmp_path: Path) -> None:
    """The config is built as written, except that the slots needing a GPU take the dry run's stand-ins."""
    pytest.importorskip("gepa", reason="requires the optional 'gepa' extra")
    sglang, dry_run = json.loads(SGLANG.read_text()), json.loads(DRY_RUN.read_text())
    method = sglang["components"]["method"]
    assert method["args"]["traces"] == dry_run["components"]["method"]["args"]["traces"]
    on_cpu = {slot: dry_run["components"][slot] for slot in ("model_source", "endpoints", "trainer")}
    chat = dry_run["components"]["method"]["args"]["chat"]
    components = {**sglang["components"], **on_cpu, "method": {**method, "args": {**method["args"], "chat": chat}}}

    run = build({**sglang, "requires": {}, "components": components}, REPO, tmp_path / "out")

    assert isinstance(run.context.components["method"].traces, TankTraceSource)  # type: ignore[attr-defined]
    assert run.context.integration_dir == REPO / "integrations/tank-war"


@pytest.mark.unit
def test_main_reports_a_refused_config_without_a_traceback(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps(_config(requires={"paths": ["/nonexistent/tank"]})))
    assert main(["--config", str(config), "--root", str(REPO), "--work-dir", str(tmp_path / "out")]) == 2
    assert "cannot start: this config needs path /nonexistent/tank" in capsys.readouterr().err


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(NODE is None, reason="needs Node 22")
@pytest.mark.skipif(
    bool(os.environ.get(SKIP_DRY_RUN)), reason=f"{SKIP_DRY_RUN} is set: CI's e2e job plays this config"
)
def test_the_dry_run_publishes_the_gepa_harness_and_rolls_back_both_train_moves_on_it(tmp_path: Path) -> None:
    work = tmp_path / "out"
    assert main(["--config", str(DRY_RUN), "--root", str(REPO), "--work-dir", str(work)]) == 0

    summary = json.loads((work / "summary.json").read_text())
    assert [(s["move"], s["result"]) for s in summary["steps"]] == [
        ("harness", "published"),
        ("train", "rolled back"),
        ("train", "rolled back"),
    ]
    assert summary["head"] == "r0001"
    assert problems(work, []) == []
    records = [json.loads(line) for line in (work / "releases/commits.jsonl").read_text().splitlines()]
    # The published harness is GEPA's rewrite of the weapon instructions, and both train moves start from it.
    assert "whenever" in json.dumps(records[1])
    assert [record["candidate"]["parent_id"] for record in records[1:]] == ["r0000", "r0001", "r0001"]
    decisions = [record["decision"] for record in records[1:]]
    assert [decision["policy"] for decision in decisions] == ["gepa+paired_interval"] + ["paired_interval"] * 2
    assert [
        [decision["metrics"][name] for name in ("seed_first", "seed_last", "tasks", "rounds", "stopped_early")]
        for decision in decisions
    ] == [[100_000, 100_003, 4, 8, 0], [100_004, 100_007, 4, 8, 0], [100_008, 100_011, 4, 8, 0]]
    # The harness beats a head that never fires, and its 30% interval clears zero. Under the published instructions
    # the sft candidate ties the head on every seed and the grpo candidate does worse on one.
    assert [decision["metrics"]["difference_mean"] for decision in decisions] == [0.625, 0.0, -0.125]
    assert [decision["metrics"]["difference_low"] > 0.0 for decision in decisions] == [True, False, False]
    kinds = {
        json.loads((p / "manifest.json").read_text())["kind"]
        for p in (work / "runs").iterdir()
        if (p / "manifest.json").exists()
    }
    assert kinds == {"evaluation", "rollout", "teacher-demonstration"}
