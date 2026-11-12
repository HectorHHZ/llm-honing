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
