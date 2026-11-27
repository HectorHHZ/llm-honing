"""The tank campaign script: the dry-run configs, their moves, startup checks, and the full dry runs (with Node)."""

from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from honing.backends.tank.e2e import _move, build, main, tasks
from honing.backends.tank.scenarios import SEATS, TankScenario, build_suite, write_suite
from honing.backends.tank.traces import TankTraceSource
from honing.campaign import ScriptedScheduler
from honing.core import ContractError, DirectionEditSpec, method_states
from tests.fakes.scheduler import HistoryScheduler

REPO = Path(__file__).resolve().parents[1]
DRY_RUN = REPO / "configs/e2e/tank_dry_run.json"
LEVERS = REPO / "configs/e2e/tank_levers_dry_run.json"
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
    with pytest.raises(ContractError, match="first_seed must be a seed of the val pool"):
        build(_config(requires={}, gate={"rounds": 8, "first_seed": 2001}), REPO, tmp_path / "e")


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
def test_the_schedule_runs_through_the_scheduler_slot_unless_the_config_names_a_scheduler(tmp_path: Path) -> None:
    pytest.importorskip("gepa", reason="the dry run's method is GEPA")
    scripted = build(_config(requires={}), REPO, tmp_path / "scripted")
    assert isinstance(scripted.scheduler, ScriptedScheduler)
    exported = scripted.scheduler.export_state()
    assert (exported["schema_version"], exported["next"], exported["moves"]) == (2, 0, 3)

    components = {
        **_config()["components"],  # type: ignore[dict-item]
        "scheduler": {"factory": "tests.fakes.scheduler:history_scheduler", "args": {"kinds": ["train"]}},
    }
    with pytest.warns(
        UserWarning,
        match=r"the 'history' scheduler chooses every move, so the schedule's moves \(harness, train, train\)",
    ):
        named = build(_config(requires={}, components=components), REPO, tmp_path / "named")
    assert isinstance(named.scheduler, HistoryScheduler)


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
    # Every move is charged for every purpose. The gate is the fresh-seed gate alone, 8 rounds on both sides for
    # every move. GEPA's method episodes are its 3 + 3 minibatch episodes and its own validation run, 4 val tasks
    # on both sides, plus one priced reflection. sft plays 2 teacher rounds; grpo samples 4 tasks x 2.
    costs = [step["cost"] for step in summary["steps"]]
    assert [cost["episodes"] for cost in costs] == [
        {"gate": 16, "method": 14, "rollout": 0, "demonstration": 0},
        {"gate": 16, "method": 0, "rollout": 0, "demonstration": 2},
        {"gate": 16, "method": 0, "rollout": 8, "demonstration": 0},
    ]
    # The summary's episodes are the gate's, so they match the gate count, not the method's validation run.
    assert [step["episodes"] for step in summary["steps"]] == [cost["episodes"]["gate"] for cost in costs]
    assert [cost["api_usd"] for cost in costs] == [0.002, 0.0, 0.0]
    # Training GPU time is exact; serving is the fake decision servers' (1 GPU each) time over every window
    # whose episodes played them, so each move, the harness one included, is charged some.
    assert [cost["gpu_seconds"] - cost["serving_gpu_seconds"] for cost in costs] == pytest.approx([0.0, 1.5, 1.5])
    assert all(cost["serving_gpu_seconds"] > 0 for cost in costs)
    assert summary["budget"]["unit"] == "usd" and summary["budget"]["stopped"] is None
    kinds = {
        json.loads((p / "manifest.json").read_text())["kind"]
        for p in (work / "runs").iterdir()
        if (p / "manifest.json").exists()
    }
    assert kinds == {"evaluation", "rollout", "teacher-demonstration"}


@pytest.mark.unit
def test_a_step_names_its_trainer_its_method_and_its_direction_edit() -> None:
    edit = _move(
        {
            "kind": "train",
            "algorithm": "direction_edit",
            "trainer": "direction_edit",
            "direction_edit": {"question": "weapon", "target": "fire", "strength": 0.5},
        }
    )
    assert edit.trainer == "direction_edit"
    assert edit.train.direction_edit == DirectionEditSpec("weapon", "fire", 0.5)  # type: ignore[union-attr]
    assert _move({"kind": "harness", "method": "meta_harness"}).harness.method == "meta_harness"  # type: ignore[union-attr]
    plain = _move({"kind": "train"})
    assert plain.trainer is None and plain.train.direction_edit is None  # type: ignore[union-attr]
    assert _move({"kind": "harness"}).harness.method is None  # type: ignore[union-attr]
    whole = {"question": None, "target": "label", "strength": 2.0}
    every = _move({"kind": "train", "algorithm": "direction_edit", "direction_edit": whole})
    assert every.train.direction_edit == DirectionEditSpec(None, "label", 2.0)  # type: ignore[union-attr]
    # A step that states only part of an edit would reset the rest of the trainer's settings without saying so.
    for partial in ({"strength": 0.5}, {"target": "fire", "strength": 0.5}, {**whole, "steps": 1}, "fire"):
        with pytest.raises(ContractError, match=r"states exactly \['question', 'strength', 'target'\]"):
            _move({"kind": "train", "algorithm": "direction_edit", "direction_edit": partial})
    with pytest.raises(ContractError, match="belongs to a direction_edit job"):
        _move({"kind": "train", "algorithm": "sft", "direction_edit": whole})


@pytest.mark.unit
def test_the_levers_config_builds_two_methods_and_two_trainers_and_refuses_a_step_for_another(
    tmp_path: Path,
) -> None:
    levers = {**json.loads(LEVERS.read_text()), "requires": {}}

    run = build(levers, REPO, tmp_path / "a")

    assert sorted(method_states(run.campaign.chain.head().method_state)) == ["gepa", "meta_harness"]
    assert [(m.kind, m.trainer, m.harness.method if m.harness else None) for m in run.moves] == [
        ("harness", None, "gepa"),
        ("train", None, None),
        ("train", "direction_edit", None),
        ("harness", None, "meta_harness"),
        ("train", None, None),
    ]
    with pytest.raises(ContractError, match="trainer 'grpo', but the campaign has 'direction_edit'"):
        build({**levers, "schedule": [{"kind": "train", "trainer": "grpo"}]}, REPO, tmp_path / "b")
    with pytest.raises(ContractError, match="method 'reflexion', but the campaign runs 'gepa', 'meta_harness'"):
        build({**levers, "schedule": [{"kind": "harness", "method": "reflexion"}]}, REPO, tmp_path / "c")
    broken = {**levers["components"], "trainers": ["direction_edit"]}
    with pytest.raises(ContractError, match="trainers must be an object"):
        build({**levers, "components": broken}, REPO, tmp_path / "d")
    assert not list(tmp_path.glob("*/runs/*"))  # refused before any round


@pytest.mark.integration
def test_the_levers_config_runs_both_methods_and_both_trainers_in_one_campaign(tmp_path: Path) -> None:
    """The whole config on the stand-in match process, where every round runs out the clock.

    Nothing wins, so nothing is published: this checks what each move was given, not the results. CI's e2e job
    plays the same config on real rounds and pins its results with ``check_campaign.py``.
    """
    levers = json.loads(LEVERS.read_text())
    levers["requires"] = {}
    components = levers["components"]
    components["runner"]["args"] = {"command": ["{python}", "{root}/tests/fakes/tank_match.py"]}
    # The stand-in match records no hashes of its outputs, so its rounds cannot be read back as decision traces.
    del components["method"]["args"]["traces"], components["methods"][0]["args"]["traces"]
    components["method"]["args"]["chat"] = {"factory": "tests.fakes.tank:chat"}

    run = build(levers, REPO, tmp_path / "out")
    outcomes = run.campaign.run(ScriptedScheduler(run.moves))

    assert [outcome.skipped_reason for outcome in outcomes] == [
        "gepa proposed nothing",
        None,
        None,
        "meta_harness proposed nothing",
        None,
    ]
    assert all(outcome.release is None for outcome in outcomes)
    records = [json.loads(line) for line in run.campaign.chain.record_path.read_text().splitlines()]
    assert [record["op"] for record in records] == ["publish", "skip", "reject", "reject", "skip", "reject"]
    assert [record.get("method") for record in records] == [None, "gepa", None, None, "meta_harness", None]
    # Every line carries both methods' states, and a method's move changes only its own.
    states = [
        method_states(r["candidate"]["method_state"] if "candidate" in r else r["method_state"]) for r in records
    ]
    assert all(sorted(state) == ["gepa", "meta_harness"] for state in states)
    assert states[1]["gepa"] != states[0]["gepa"] and states[1]["meta_harness"] == states[0]["meta_harness"]
    assert states[4]["meta_harness"] != states[3]["meta_harness"] and states[4]["gepa"] == states[3]["gepa"]
    assert states[2] == states[1] and states[5] == states[4]  # a train move changes neither
    # The default trainer ran both sft steps, and the named one the edit, with the step's own parameters.
    default, edit = run.context.components["trainer"], run.campaign._trainers["direction_edit"]
    assert [job.algorithm for job in default.jobs] == ["sft", "sft"]  # type: ignore[attr-defined]
    assert [job.direction_edit for job in edit.jobs] == [DirectionEditSpec("weapon", "fire", 1.0)]  # type: ignore[attr-defined]
    # Each of the three steps drew its own teacher rounds, the later two on seeds no task uses.
    batches = [batch.batch_id for batch in (default.batches[0], edit.batches[0], default.batches[1])]  # type: ignore[attr-defined]
    assert batches == [f"tank-teacher-demonstrations-{n}" for n in (1, 2, 3)]
    teacher_seeds = sorted(
        json.loads((p / "manifest.json").read_text())["seed"]
        for p in (tmp_path / "out/runs").iterdir()
        if (p / "manifest.json").exists() and "--teacher--" in p.name
    )
    assert teacher_seeds == [1001, 1001, 300000, 300000, 300001, 300001]


@pytest.mark.unit
def test_the_real_weights_levers_config_schedules_one_nanojev_sft_step_and_one_direction_edit() -> None:
    config = json.loads((REPO / "configs/e2e/tank_nanojev_levers.json").read_text())

    moves = [_move(entry) for entry in config["schedule"]]

    assert [(m.train.algorithm, m.trainer) for m in moves] == [("sft", None), ("direction_edit", "direction_edit")]  # type: ignore[union-attr]
    assert moves[1].train.direction_edit == DirectionEditSpec("weapon", "fire", 1.0)  # type: ignore[union-attr]
    components = config["components"]
    entries = [components[slot] for slot in components if slot != "trainers"] + list(components["trainers"].values())
    for entry in entries:
        module, _, name = entry["factory"].partition(":")
        assert callable(getattr(importlib.import_module(module), name)), entry["factory"]
    # Held out for validation: the teacher rounds whose seed ends in 0 or 5, in the first batch and in later ones.
    held_out = re.compile(components["trainer"]["args"]["validation_groups"])
    seeds = [*range(1001, 1009), *range(300_000, 300_008)]
    assert [seed for seed in seeds if held_out.search(f"tank:{seed}:blue")] == [1005, 300_000, 300_005]
    with pytest.raises(ContractError, match="this config needs environment variable HONING_NANOJEV_SOURCE"):
        build({**config, "requires": {"env": ["HONING_NANOJEV_SOURCE"]}}, REPO, REPO / "runs/never-created")
