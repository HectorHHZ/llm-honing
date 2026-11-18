"""The grouped diagnosis score preserves pairing and class balance."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "integrations/tank-war/scripts/score-multifamily-diagnosis.py"
SPEC = importlib.util.spec_from_file_location("tank_multifamily_diagnosis_score", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


@pytest.mark.unit
def test_pairing_and_instance_reset_groups() -> None:
    rows = []
    for layer in module.LAYERS:
        for instance in ("a", "b"):
            for seed in (101, 202):
                rows.append(
                    {
                        "label": layer,
                        "instance": f"{layer}-{instance}",
                        "seed": seed,
                        "read": {"cause": "uncertain"},
                        "probe": {"cause": layer, "confidence": 1.0},
                    }
                )
    result = module.paired_interval(rows, "read", "probe", draws=100)
    assert result["point"] == 1
    assert result["instance_and_reset_group_bootstrap_95pct"] == [1, 1]
    assert result["independent_seed_blocks"] == 2
    assert module.macro_f1(rows, "read") == 0
    assert module.macro_f1(rows, "probe") == 1
    assert module.calibration_error(rows, "probe") == 0
    with pytest.raises(ValueError, match="no eligible instance"):
        module.paired_interval([row for row in rows if row["label"] != "harness"], "read", "probe", draws=100)
