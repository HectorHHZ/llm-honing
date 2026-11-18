"""Replay routing keeps each Tank War fault instance's files isolated."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "integrations/tank-war/scripts/replay-multifamily-blind.py"
SPEC = importlib.util.spec_from_file_location("tank_multifamily_replay", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


@pytest.mark.unit
def test_routes_harness_and_controller_by_instance_and_checks_hashes(tmp_path: Path) -> None:
    harness = tmp_path / "harness"
    (harness / "tank").mkdir(parents=True)
    (harness / "tank/controller.json").write_text("{}\n")
    (harness / "tank/questions.json").write_text("{}\n")
    controller = tmp_path / "controller.json"
    controller.write_text("{}\n")
    mapping = {
        "h1": {
            "layer": "harness",
            "path": str(harness),
            "controller_sha256": module.sha(harness / "tank/controller.json"),
            "questions_sha256": module.sha(harness / "tank/questions.json"),
        },
        "c1": {"layer": "controller", "path": str(controller), "fault_sha256": module.sha(controller)},
    }
    assert module.candidate_paths({"technical_label": "harness", "fault_instance": "h1"}, mapping)[0] == harness
    assert module.candidate_paths({"technical_label": "controller", "fault_instance": "c1"}, mapping)[1] == controller
    with pytest.raises(ValueError, match="missing controller replay files"):
        module.candidate_paths({"technical_label": "controller", "fault_instance": "h1"}, mapping)
    controller.write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match="controller candidate changed"):
        module.candidate_paths({"technical_label": "controller", "fault_instance": "c1"}, mapping)
