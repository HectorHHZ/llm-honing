"""The Exp 0 screen must refuse a partial replay gate and preserve candidates."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "integrations/tank-war/scripts"


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / file)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(SCRIPTS))
screen = _load("tank_screen_fault", "screen-fault.py")
builder = _load("tank_controller_fault_builder", "build-controller-faults.py")


def test_controller_candidates_are_separate_and_named(tmp_path: Path) -> None:
    records = builder.build(tmp_path / "candidates")
    assert len(records) == 3
    assert len({record["kind"] for record in records}) == 3
    for record in records:
        path = tmp_path / "candidates" / f"{record['kind']}.json"
        assert json.loads(path.read_text()) == record
        assert record["layer"] == "controller"


def test_screen_requires_full_independent_gate(tmp_path: Path) -> None:
    log = tmp_path / "gate.jsonl"
    seeds = list(range(64))
    log.write_text("".join(json.dumps({"seed": seed, "seat": "blue", "passed": True}) + "\n" for seed in seeds))
    with pytest.raises(ValueError, match="incomplete"):
        screen.require_gate(log, seeds)
    with log.open("a") as stream:
        for seed in seeds:
            stream.write(json.dumps({"seed": seed, "seat": "red", "passed": True}) + "\n")
    screen.require_gate(log, seeds)
    rows = log.read_text().splitlines()
    rows[-1] = json.dumps({"seed": seeds[-1], "seat": "red", "passed": False})
    log.write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match="failed"):
        screen.require_gate(log, seeds)


def test_screen_refuses_incomplete_prior_match(tmp_path: Path) -> None:
    run = tmp_path / "partial"
    run.mkdir()
    (run / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="incomplete"):
        screen._completed(run)
