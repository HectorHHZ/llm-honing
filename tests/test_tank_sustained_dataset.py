"""Sustained-fault labels require a same-state, layer-specific local effect."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "integrations/tank-war/scripts/assemble-sustained-fault-dataset.py"
SPEC = importlib.util.spec_from_file_location("tank_sustained_dataset", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def write_pair(
    tmp_path: Path,
    *,
    request_changed: bool,
    selected_changed: bool,
    command_changed: bool,
    reply_changed: bool = False,
    state_changed: bool = False,
) -> tuple[Path, Path]:
    clean, fault = tmp_path / "clean", tmp_path / "fault"
    for directory, is_fault in ((clean, False), (fault, True)):
        directory.mkdir()
        state = {"decision_index": 0, "state_hash": "different" if is_fault and state_changed else "same"}
        action = {
            "selected_action": "left" if is_fault and selected_changed else "right",
            "acknowledged_command": "stop" if is_fault and command_changed else "forward",
        }
        call = {
            "request": "altered" if is_fault and request_changed else "original",
            "reply": {
                "answers": {
                    "motion": {
                        "choice": "left" if is_fault and reply_changed else "right",
                        "probabilities": {"left": 0.5, "right": 0.5},
                    }
                },
                "latency_ms": 10 if is_fault else 20,
            },
        }
        for name, row in (("states.jsonl", state), ("execution.jsonl", action), ("model_calls.jsonl", call)):
            (directory / name).write_text(json.dumps(row) + "\n")
    return clean, fault


@pytest.mark.unit
@pytest.mark.parametrize(
    ("layer", "request_changed", "selected_changed", "command_changed"),
    [
        ("harness", True, True, True),
        ("weights", False, True, False),
        ("controller", False, False, True),
    ],
)
def test_local_effect_is_attributed_to_its_layer(
    tmp_path: Path, layer: str, request_changed: bool, selected_changed: bool, command_changed: bool
) -> None:
    clean, fault = write_pair(
        tmp_path,
        request_changed=request_changed,
        selected_changed=selected_changed,
        command_changed=command_changed,
    )
    evidence = module.first_local_difference(clean, fault, layer)
    assert evidence["verified"] is True
    assert evidence["decision_index"] == 0
    assert evidence["state_hash"] == "same"


@pytest.mark.unit
def test_harness_command_only_change_is_uncertain(tmp_path: Path) -> None:
    clean, fault = write_pair(tmp_path, request_changed=True, selected_changed=False, command_changed=True)
    evidence = module.first_local_difference(clean, fault, "harness")
    assert evidence["verified"] is False


@pytest.mark.unit
def test_different_raw_state_cannot_support_local_attribution(tmp_path: Path) -> None:
    clean, fault = write_pair(
        tmp_path, request_changed=True, selected_changed=True, command_changed=True, state_changed=True
    )
    evidence = module.first_local_difference(clean, fault, "harness")
    assert evidence["verified"] is False
    assert evidence["reason"] == "states diverged before a local action difference"
