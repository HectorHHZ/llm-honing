"""Edited checkpoints are audited separately from no-injection controls."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "integrations/tank-war/scripts/analyze-multifamily-weight-probes.py"
SPEC = importlib.util.spec_from_file_location("tank_multifamily_weight_probe_analysis", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


@pytest.mark.unit
def test_splits_checkpoint_instances_and_rejects_tampered_probe(tmp_path: Path) -> None:
    cases = [
        {
            "episode_id": "w1",
            "technical_label": "weights",
            "fault_instance": "edit-1",
            "technical_reference_index": 0,
            "decisions": 1,
        },
        {
            "episode_id": "w2",
            "technical_label": "weights",
            "fault_instance": "edit-2",
            "technical_reference_index": 0,
            "decisions": 1,
        },
        {"episode_id": "c1", "technical_label": "no_injection", "decisions": 1},
    ]
    source = tmp_path / "blind"
    write(source / "sample-freeze.json", {"status": "frozen_before_model_answers", "cases": cases})
    records = []
    for case in cases:
        control = case["technical_label"] == "no_injection"
        write(
            source / "probes" / case["episode_id"] / "probes.json",
            {
                "indices": [
                    {
                        "decision_index": 0,
                        "same_model_input": True,
                        "selected_action_changed": not control,
                        "max_abs_probability_change": 0.0 if control else 0.3,
                    }
                ]
            },
        )
        records.append(
            {
                "episode_id": case["episode_id"],
                "probe_sha256": module.sha(source / "probes" / case["episode_id"] / "probes.json"),
            }
        )
    write(
        source / "probes/summary.json",
        {
            "status": "complete",
            "cases": 3,
            "sample_freeze_sha256": module.sha(source / "sample-freeze.json"),
            "records": records,
        },
    )
    write(source / "probe-audit.json", {"valid": 3})
    result = module.analyze(source)
    assert set(result["edited_checkpoint_instances"]) == {"edit-1", "edit-2"}
    assert result["edited_checkpoint_instances"]["edit-1"]["reference_delta_gt_0p01"] == 1
    assert result["control_decisions"]["delta_gt_0p01"] == 0
    (source / "probes/w2/probes.json").write_text("{}\n")
    with pytest.raises(ValueError, match="probe changed"):
        module.analyze(source)
