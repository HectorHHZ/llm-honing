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
def test_each_harness_candidate_changes_one_file_and_only_guard_fault_enables_guard(tmp_path: Path) -> None:
    records = module.build(tmp_path / "faults")
    assert len(records) >= 5
    for record in records:
        root = tmp_path / "faults" / record["family"]
        assert len(record["changed_paths"]) == 1
        controller = json.loads((root / "tank/controller.json").read_text())
        assert controller["fireGuardEnabled"] is (record["family"] == "enable_shot_guard")
        assert record["clean_sha256"] != record["fault_sha256"]


@pytest.mark.unit
@pytest.mark.parametrize("family", ["stop_and_hold", "ignore_shell_threat"])
def test_instruction_only_fault_keeps_execution_settings(tmp_path: Path, family: str) -> None:
    module.build(tmp_path / "faults")
    root = tmp_path / "faults" / family
    assert (root / "tank/controller.json").read_bytes() == (module.ROOT / "config/controller.json").read_bytes()
    record = json.loads((root / "fault.json").read_text())
    assert record["changed_paths"] == ["tank/questions.json"]
