"""Incomplete clean pairings must not create a verified fault label."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "integrations/tank-war/scripts/assemble-screened-fault-progress.py"
SPEC = importlib.util.spec_from_file_location("tank_screened_progress", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


@pytest.mark.unit
def test_unpaired_loss_is_retained_without_cause(tmp_path: Path) -> None:
    fault = tmp_path / "s42-blue"
    fault.mkdir()
    (fault / "result.json").write_text(json.dumps({"seed": 42, "seat": "blue", "outcome": {"blue": -1}}))
    (fault / "manifest.json").write_text("{}")
    record = module.unpaired_record(fault, "controller", "suppress-selected-turn-v1")
    assert record["fault_loss"] is True
    assert record["cause_label"] is None
    assert record["technical_reference"]["reason"] == "missing same-build clean counterpart"


@pytest.mark.unit
def test_conflicting_clean_counterparts_are_rejected(tmp_path: Path) -> None:
    roots = [tmp_path / "a", tmp_path / "b"]
    for root, score in zip(roots, (1, -1)):
        run = root / "s42-blue"
        run.mkdir(parents=True)
        (run / "result.json").write_text(json.dumps({"outcome": {"blue": score}}))
    with pytest.raises(ValueError, match="conflicting clean result"):
        module.clean_index(roots)
