"""Checks that the registered Tank War design cannot silently accept bad pairs."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "integrations/tank-war/scripts/exp0_protocol.py"
SPEC = importlib.util.spec_from_file_location("tank_exp0_protocol", SCRIPT)
assert SPEC and SPEC.loader
exp0 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exp0)


def _run(root: Path, seed: int, seat: str, outcome: int) -> None:
    run = root / f"s{seed}-{seat}"
    run.mkdir(parents=True)
    manifest = {
        "kind": "evaluation",
        "seed": seed,
        "seat": seat,
        "opponent": "planner",
        "seconds": 30,
        "interval_ms": 200,
        "opponent_interval_ms": 300,
        "opponent_seat": {"controllerMode": "plans"},
        "sample_seed": None,
        "code_commit": "clean-build",
        "sources": {"engine": "clean-build"},
        "decision_server": "http://localhost:18731",
        "model": {"weights_sha256": "clean-weights"},
        "harness_sha256": "clean-harness",
        "controller_sha256": "clean-controller",
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    (run / "result.json").write_text(json.dumps({"seed": seed, "seat": seat, "outcome": {seat: outcome}}))


def test_seed_groups_are_disjoint_and_reproducible() -> None:
    first = exp0.seed_plan()
    assert first == exp0.seed_plan()
    values = [seed for group in first["groups"].values() for seed in group]
    assert len(values) == len(set(values))
    assert len(first["groups"]["fault_validation"]) == 128


def test_fault_screen_uses_both_seats_and_rejects_missing_pairs(tmp_path: Path) -> None:
    clean, fault = tmp_path / "clean", tmp_path / "fault"
    seeds = exp0.seed_plan()["groups"]["fault_validation"]
    for seed in seeds:
        for seat in exp0.SEATS:
            _run(clean, seed, seat, 1)
            _run(fault, seed, seat, -1)
    result = exp0.validate_fault(clean, fault, seeds)
    assert result["retained"] is True
    assert result["mean_fault_minus_clean"] == -1
    assert result["ci95"] == [-1, -1]
    (fault / f"s{seeds[0]}-red" / "result.json").unlink()
    with pytest.raises(FileNotFoundError):
        exp0.validate_fault(clean, fault, seeds)


def test_fault_screen_rejects_build_change_between_seed_blocks(tmp_path: Path) -> None:
    clean, fault = tmp_path / "clean", tmp_path / "fault"
    seeds = exp0.seed_plan()["groups"]["fault_validation"]
    for seed in seeds:
        for seat in exp0.SEATS:
            _run(clean, seed, seat, 1)
            _run(fault, seed, seat, -1)
    changed = fault / f"s{seeds[1]}-red" / "manifest.json"
    manifest = json.loads(changed.read_text())
    manifest["sources"]["engine"] = "changed-mid-screen"
    changed.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="changed across seed blocks"):
        exp0.validate_fault(clean, fault, seeds)


def test_fault_screen_accepts_equivalent_servers_on_different_gpus(tmp_path: Path) -> None:
    clean, fault = tmp_path / "clean", tmp_path / "fault"
    seeds = exp0.seed_plan()["groups"]["fault_validation"]
    for index, seed in enumerate(seeds):
        gpu = index % 2
        for seat in exp0.SEATS:
            _run(clean, seed, seat, 1)
            _run(fault, seed, seat, -1)
            for root in (clean, fault):
                path = root / f"s{seed}-{seat}" / "manifest.json"
                manifest = json.loads(path.read_text())
                manifest["decision_server"] = f"http://localhost:{18731 + gpu}"
                manifest["model"].update(gpu=str(gpu), gpuUuid=f"gpu-{gpu}")
                path.write_text(json.dumps(manifest))
    assert exp0.validate_fault(clean, fault, seeds)["retained"] is True
    changed = fault / f"s{seeds[0]}-blue" / "manifest.json"
    manifest = json.loads(changed.read_text())
    manifest["model"]["weights_sha256"] = "different-weights"
    changed.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="changed across seed blocks"):
        exp0.validate_fault(clean, fault, seeds)


@pytest.mark.parametrize(
    "layer,digest",
    [("harness", "harness_sha256"), ("weights", "weights_sha256"), ("controller", "controller_sha256")],
)
def test_single_layer_provenance_accepts_only_named_change(layer: str, digest: str) -> None:
    clean = {
        "sources": {"engine": "build"},
        "code_commit": "build",
        "opponent_seat": {"controllerMode": "plans"},
        "sample_seed": None,
        "decision_server": "http://localhost:18731",
        "model": {"weights_sha256": "clean-weights"},
        "harness_sha256": "clean-harness",
        "controller_sha256": "clean-controller",
    }
    fault = json.loads(json.dumps(clean))
    if digest == "weights_sha256":
        fault["model"][digest] = "fault-weights"
        fault["decision_server"] = "http://localhost:18732"
    else:
        fault[digest] = "fault-" + digest
    exp0._validate_single_layer(clean, fault, layer, "synthetic")
    fault["sources"]["engine"] = "other-build"
    with pytest.raises(ValueError, match="changed sources"):
        exp0._validate_single_layer(clean, fault, layer, "synthetic")
