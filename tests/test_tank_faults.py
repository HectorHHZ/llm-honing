"""The Tank War fault library (#171): the catalog, the builders on fakes, the CLI's reproducibility, and the gate."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from honing.backends.tank.faults import FAULTS, Fault, build, build_library, checksums, relabel
from honing.backends.tank.harness import CONTROLLER_PATH, QUESTIONS_PATH
from honing.backends.tank.scenarios import SEED_POOLS
from honing.core import ContractError, DataContract, DirectionEditSpec, TrainingBatch
from tests.conftest import load_script
from tests.fakes import tank as fakes

REPO = Path(__file__).resolve().parents[1]
INTEGRATION = REPO / "integrations/tank-war"
DRY_RUN = REPO / "configs/e2e/tank_faults_dry_run.json"
NANOJEV = REPO / "configs/e2e/tank_faults_nanojev.json"
FREEZE = json.loads((INTEGRATION / "exp0/v0/freeze.json").read_text())
MIRRORED_TURNS = next(dict(f.relabel[1]) for f in FAULTS if f.relabel and f.relabel[0] == "motion")
CONTROLLER_KINDS = [fault.controller_kind for fault in FAULTS if fault.controller_kind]


def _batch() -> TrainingBatch:
    batch = fakes.recorded_decisions(None).next_batch(DataContract("typed_decisions"))
    assert batch is not None
    return batch


def _chosen(batch: TrainingBatch, question: str) -> Counter[str]:
    return Counter(option for decision in batch.decisions for key, option in decision.chosen if key == question)


def _library(out: Path, *, earlier: bool = True, faults: tuple[Fault, ...] = FAULTS) -> tuple[Any, Any, Any]:
    """A library on the stand-in model: a clean release one fire edit above the model source."""
    context = SimpleNamespace(shared={})
    source, trainer, editor = (
        fakes.model_source(context),
        fakes.trainer(context),
        fakes.direction_edit_trainer(context),
    )
    base = source.materialize(out / "model")
    clean = editor.world.snapshot(base.architecture, editor.world.quality[base.checkpoint.checksum] + 0.1)
    library = build_library(
        out / "library",
        integration_dir=INTEGRATION,
        clean=clean,
        earlier=base if earlier else None,
        batch=_batch(),
        trainer=trainer,
        edit_trainer=editor,
        faults=faults,
    )
    return library, trainer, editor


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mirror(option: str) -> str:
    return option.replace("left", "?").replace("right", "left").replace("?", "right")


@pytest.mark.unit
def test_the_catalog_covers_every_cause_with_kinds_the_builders_and_the_match_know() -> None:
    assert len({fault.fault_id for fault in FAULTS}) == len(FAULTS)
    assert {fault.cause for fault in FAULTS} == {"harness", "weights", "controller", "chance"}
    harness_kinds = load_script("integrations/tank-war/scripts/build-harness-faults.py").KINDS
    controller_kinds = load_script("integrations/tank-war/scripts/build-controller-faults.py").KINDS
    runtime = (INTEGRATION / "scripts/controller-faults.mjs").read_text()
    match = re.search(r"CONTROLLER_FAULTS = Object\.freeze\(\[(.*?)\]\)", runtime, re.S)
    assert match is not None
    playable = set(re.findall(r'"(\w+)"', match.group(1)))
    for fault in FAULTS:
        recipes = [fault.harness_kind, fault.controller_kind, fault.edit, fault.relabel, fault.earlier or None]
        assert sum(recipe is not None for recipe in recipes) == (0 if fault.cause == "chance" else 1), fault
        assert bool(fault.questions) == (fault.cause != "chance") and set(fault.questions) <= {"motion", "weapon"}
        assert fault.harness_kind is None or fault.harness_kind in harness_kinds
        # A controller kind must be one #186's builder writes and one the match can play.
        assert fault.controller_kind is None or {fault.controller_kind} <= set(controller_kinds) & playable
    # The Exp 0 v0 selected faults are in the catalog.
    kinds = {fault.harness_kind for fault in FAULTS} | {fault.controller_kind for fault in FAULTS}
    assert {
        FREEZE["selected_faults"]["harness"]["builder_kind"],
        FREEZE["selected_faults"]["controller"]["builder_kind"],
    } <= kinds
    assert DirectionEditSpec("weapon", "fire", FREEZE["selected_faults"]["weights"][0]["builder_strength"]) in {
        fault.edit for fault in FAULTS
    }


@pytest.mark.unit
def test_relabel_maps_one_questions_labels_and_refuses_a_batch_it_would_not_change() -> None:
    batch = _batch()
    assert _chosen(batch, "weapon") == Counter({"hold": 21, "fire": 8})

    flipped = relabel(batch, "weapon", {"fire": "hold", "hold": "fire"})
    assert _chosen(flipped, "weapon") == Counter({"fire": 21, "hold": 8})
    assert _chosen(flipped, "motion") == _chosen(batch, "motion")
    assert [d.parts for d in flipped.decisions] == [d.parts for d in batch.decisions]
    assert flipped.batch_id == "tank-recorded-s3114-blue-weapon-relabeled"

    mirrored = relabel(batch, "motion", MIRRORED_TURNS)
    before, after = _chosen(batch, "motion"), _chosen(mirrored, "motion")
    assert after == Counter({_mirror(option): count for option, count in before.items()}) != before

    with pytest.raises(
        ContractError, match=r"relabeling drive through .* changes no label in batch tank-recorded-s3114-blue"
    ):
        relabel(batch, "drive", {"forward": "reverse"})
    with pytest.raises(ContractError, match=r"relabeling weapon through .* changes no label"):
        relabel(batch, "weapon", {"fire": "fire"})


@pytest.mark.unit
def test_each_fault_changes_only_its_own_causes_part_of_the_clean_release(tmp_path: Path) -> None:
    library, trainer, editor = _library(tmp_path)
    manifest = json.loads(library.path.read_text())
    assert manifest["format"] == "tank-fault-library" and manifest["skipped"] == []
    assert [entry["id"] for entry in manifest["faults"]] == [fault.fault_id for fault in FAULTS]
    clean = manifest["clean"]["artifacts"]
    assert set(clean) == {CONTROLLER_PATH, QUESTIONS_PATH, "weights", "controller_fault"}
    assert clean["controller_fault"] is None
    parts = {"harness": {CONTROLLER_PATH, QUESTIONS_PATH}, "weights": {"weights"}}
    parts |= {"controller": {"controller_fault"}, "chance": set()}
    for entry in manifest["faults"]:
        changed = {key for key, value in entry["artifacts"].items() if value != clean[key]}
        assert changed <= parts[entry["cause"]] and len(changed) == (entry["cause"] != "chance"), entry["id"]
        assert set(entry) == {
            "id",
            "cause",
            "affected_questions",
            "expected_symptom",
            "artifacts",
            "recipe",
            "harness_dir",
            "controller_fault_file",
            "weights_location",
        }
        assert entry["expected_symptom"] and entry["affected_questions"] == list(
            next(f.questions for f in FAULTS if f.fault_id == entry["id"])
        )
        # What the manifest points at is what it hashed.
        harness = library.path.parent / entry["harness_dir"]
        for path in (CONTROLLER_PATH, QUESTIONS_PATH):
            assert _sha256(harness / path) == entry["artifacts"][path]
        if entry["controller_fault_file"]:
            assert (
                _sha256(library.path.parent / entry["controller_fault_file"]) == entry["artifacts"]["controller_fault"]
            )
        release = library.releases[entry["id"]]
        assert release.weights.checkpoint.checksum == entry["artifacts"]["weights"]
    # The trainers ran from the clean weights: the reverse edit as stated, and sft on relabeled teacher labels.
    assert [job.direction_edit for job in editor.jobs] == [DirectionEditSpec("weapon", "fire", -3.0)]
    assert [job.algorithm for job in trainer.jobs] == ["sft", "sft"]
    weapon, motion = trainer.batches
    assert _chosen(weapon, "weapon") == Counter({"fire": 21, "hold": 8})
    assert _chosen(motion, "motion") == Counter({_mirror(o): n for o, n in _chosen(_batch(), "motion").items()})


@pytest.mark.unit
def test_harness_and_controller_faults_are_the_exp0_v0_builders_outputs_and_reproduce_the_freeze(
    tmp_path: Path,
) -> None:
    library, _, _ = _library(tmp_path)
    by_id = {entry["id"]: entry["artifacts"] for entry in library.manifest["faults"]}
    load_script("integrations/tank-war/scripts/build-harness-faults.py").build(tmp_path / "harness")
    load_script("integrations/tank-war/scripts/build-controller-faults.py").build(tmp_path / "controller")
    for fault in FAULTS:
        if fault.harness_kind:
            for path in (CONTROLLER_PATH, QUESTIONS_PATH):
                built = tmp_path / "harness" / fault.harness_kind / path
                assert by_id[fault.fault_id][path] == _sha256(built), (fault.fault_id, path)
        if fault.controller_kind:
            built = tmp_path / "controller" / f"{fault.controller_kind}.json"
            assert by_id[fault.fault_id]["controller_fault"] == _sha256(built)
    # check-v0-faults.mjs's controller identity: the game sources, the controller settings, and the switch.
    sources = FREEZE["game"]["sources_sha256"]

    def identity(settings: str, switch: str | None) -> str:
        parts = {name: sources[f"web/game/{name}.js"] for name in ("engine", "planner", "adapter")}
        text = json.dumps({**parts, "settings": settings, "fault": switch}, separators=(",", ":"))
        return hashlib.sha256(text.encode()).hexdigest()

    clean = library.manifest["clean"]["artifacts"][CONTROLLER_PATH]
    standoff = by_id["harness-extreme-standoff"][CONTROLLER_PATH]
    assert identity(clean, None) == FREEZE["clean_harness"]["recorded_controller_sha256"]
    assert identity(standoff, None) == FREEZE["selected_faults"]["harness"]["recorded_controller_sha256"]
    drop = by_id["controller-drop-fire"]["controller_fault"]
    assert identity(clean, drop) == FREEZE["selected_faults"]["controller"]["recorded_controller_sha256"]


@pytest.mark.unit
def test_a_fault_that_changes_another_part_is_refused_and_a_missing_earlier_checkpoint_is_skipped(
    tmp_path: Path,
) -> None:
    earlier = next(fault for fault in FAULTS if fault.earlier)
    library, _, _ = _library(tmp_path / "a", earlier=False, faults=(earlier, FAULTS[-1]))
    assert library.manifest["skipped"] == [
        {"id": "weights-earlier-checkpoint", "reason": "the clean release was not edited from an earlier one"}
    ]
    assert [entry["id"] for entry in library.manifest["faults"]] == ["chance-clean"]
    mislabeled = replace(FAULTS[0], fault_id="mislabeled", cause="controller")
    with pytest.raises(
        ContractError, match=r"mislabeled is a controller fault, but it changes \['tank/controller.json'\]"
    ):
        _library(tmp_path / "b", faults=(mislabeled,))
    with pytest.raises(ContractError, match="silent is a weights fault, but it changes nothing"):
        _library(tmp_path / "c", faults=(Fault("silent", "weights", ("weapon",), "None."),))


@pytest.mark.unit
def test_a_config_is_refused_before_anything_is_built(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = json.loads(DRY_RUN.read_text())
    with pytest.raises(ContractError, match=r"unknown settings \['clean_edit'\]"):
        build({**config, "clean_edit": 1.0}, REPO, tmp_path / "a")
    with pytest.raises(ContractError, match=r"unknown settings \['round'\]"):
        build({**config, "gate": {"round": 8}}, REPO, tmp_path / "b")
    monkeypatch.delenv("HONING_NANOJEV_SOURCE", raising=False)
    with pytest.raises(ContractError, match="this config needs environment variable HONING_NANOJEV_SOURCE"):
        build(json.loads(NANOJEV.read_text()), REPO, tmp_path / "c")
    no_edit = {**config, "components": {k: v for k, v in config["components"].items() if k != "trainers"}}
    with pytest.raises(ContractError, match=r"names no trainers\.direction_edit"):
        build(no_edit, REPO, tmp_path / "d")
    assert not (tmp_path / "a").exists() and not (tmp_path / "c").exists()


def _cli(config: Path, work_dir: Path) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "scripts/e2e/tank_faults.py", f"--config={config}", f"--work-dir={work_dir}"]
    return subprocess.run(command, cwd=REPO, capture_output=True, text=True, timeout=300)


@pytest.mark.functional
def test_the_dry_run_builds_every_fault_from_a_clean_release_twice_with_the_same_checksums(tmp_path: Path) -> None:
    runs = [_cli(DRY_RUN, tmp_path / name) for name in ("first", "second")]
    for run in runs:
        assert run.returncode == 0, run.stderr
        assert run.stdout.startswith(f"{len(FAULTS)} faults in ")
    first, second = (
        json.loads((tmp_path / name / "library/library.json").read_text()) for name in ("first", "second")
    )
    assert checksums(first) == checksums(second)
    assert len(checksums(first)) == len(FAULTS) + 1
    # Every fault's checksums are its own, but for the chance control, which is the clean release.
    distinct = {json.dumps(artifacts, sort_keys=True) for artifacts in checksums(first).values()}
    assert len(distinct) == len(FAULTS)


def _manifests(runs: Path) -> list[Mapping[str, Any]]:
    return [json.loads(path.read_text()) for path in sorted(runs.glob("*/manifest.json"))]


@pytest.mark.functional
def test_the_gate_keeps_the_faults_the_clean_release_beats_on_fresh_val_seeds(tmp_path: Path) -> None:
    """The stand-in match ignores harness and controller, so only weaker weights can lose here."""
    config = json.loads(DRY_RUN.read_text())
    config["components"]["endpoints"] = {"factory": "tests.fakes.tank:ranked_endpoints"}
    config["gate"] = {"rounds": 8, "confidence": 0.5}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))

    run = _cli(path, tmp_path / "out")

    assert run.returncode == 0, run.stderr
    rows = json.loads((tmp_path / "out/library/gate.json").read_text())
    assert [row["id"] for row in rows] == [fault.fault_id for fault in FAULTS]
    # The clean release (quality 0.6) is strong; the model source (0.5) and the reverse edit (0.3) are weak.
    assert {row["id"] for row in rows if row["kept"]} == {"weights-earlier-checkpoint", "weights-reverse-fire-edit"}
    for row in rows:
        assert row["drop_high"] < 0 if row["kept"] else row["drop_low"] <= 0 <= row["drop_high"]
    seeds = [range(row["seeds"][0], row["seeds"][1] + 1) for row in rows]
    assert [len(block) for block in seeds] == [4] * len(FAULTS)
    assert seeds[0].start == SEED_POOLS["val"].start and all(a.stop == b.start for a, b in pairwise(seeds))
    assert "| weights-reverse-fire-edit | weights | 1.000 | 0.000 |" in run.stdout
    # Each controller fault played its 8 rounds with its switch, and nothing else had one.
    switched = Counter(m["arguments"].get("controller-fault-file") for m in _manifests(tmp_path / "out/runs"))
    library = (tmp_path / "out/library").resolve()
    assert switched == {
        None: 8 * (2 * len(FAULTS) - len(CONTROLLER_KINDS)),
        **{str(library / f"controller-candidates/{kind}.json"): 8 for kind in CONTROLLER_KINDS},
    }
