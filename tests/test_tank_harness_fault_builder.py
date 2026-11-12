"""Harness candidate construction preserves all clean files outside one changed file."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "integrations/tank-war/scripts/build-harness-faults.py"
SPEC = importlib.util.spec_from_file_location("tank_harness_faults", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


@pytest.mark.unit
def test_each_harness_candidate_changes_one_file_and_keeps_shot_guard_off(tmp_path: Path) -> None:
    records = module.build(tmp_path / "faults")
    assert len(records) >= 5
    for record in records:
        root = tmp_path / "faults" / record["family"]
        assert len(record["changed_paths"]) == 1
        controller = json.loads((root / "tank/controller.json").read_text())
        assert controller["fireGuardEnabled"] is False
        assert record["clean_sha256"] != record["fault_sha256"]
