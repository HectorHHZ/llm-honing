"""Four-mode Tank War diagnosis freezes reject changed anonymous evidence."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "integrations/tank-war/scripts/freeze-multifamily-diagnosis.py"
SPEC = importlib.util.spec_from_file_location("tank_multifamily_diagnosis_freeze", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


@pytest.mark.unit
def test_freezes_two_models_four_modes_and_rejects_changed_probe(tmp_path: Path) -> None:
    source = tmp_path / "source"
    bare, fewshot, schema = (tmp_path / name for name in ("bare.txt", "fewshot.txt", "schema.json"))
    bare.write_text("same instruction\n")
    fewshot.write_text("same instruction with development examples\n")
    schema.write_text("{}\n")
    registration = tmp_path / "registration.json"
    write(
        registration,
        {
            "prompt_sha256": {"bare": module.sha(bare), "fewshot": module.sha(fewshot)},
            "schema_sha256": module.sha(schema),
            "primary_model": {"id": "gpt-6-sol", "effort": "high"},
            "secondary_model": {"id": "gpt-6-astra", "effort": "medium"},
            "modes": ["read", "read+fewshot", "probe", "probe+fewshot"],
            "limits": {
                "maximum_input_tokens_per_model_mode": 4_000_000,
                "maximum_output_tokens_per_model_mode": 100_000,
            },
        },
    )
    write(source / "inputs/e1.json", {"timeline": []})
    write(source / "probes/e1/probes.json", {"indices": []})
    write(
        source / "sample-freeze.json",
        {
            "status": "frozen_before_model_answers",
            "pre_registration_sha256": module.sha(registration),
            "cases": [{"episode_id": "e1", "input_sha256": module.sha(source / "inputs/e1.json")}],
        },
    )
    write(source / "replays/summary.json", {"valid": 1})
    write(
        source / "probes/summary.json",
        {
            "status": "complete",
            "cases": 1,
            "records": [{"episode_id": "e1", "probe_sha256": module.sha(source / "probes/e1/probes.json")}],
        },
    )
    write(source / "probe-audit.json", {"valid": 1})
    out = tmp_path / "diagnosis"
    manifest = module.freeze(source, registration, bare, fewshot, schema, out)
    assert manifest["maximum_calls"] == 8
    assert set(manifest["runs"]) == {"sol-bare", "sol-fewshot", "astra-bare", "astra-fewshot"}
    assert module.read(out / "sol-bare/freeze.json")["modes"] == ["read", "probe"]
    assert module.read(out / "astra-fewshot/freeze.json")["modes"] == ["read+fewshot", "probe+fewshot"]
    (source / "probes/e1/probes.json").write_text("{}\n")
    with pytest.raises(ValueError, match="controlled checks changed"):
        module.freeze(source, registration, bare, fewshot, schema, tmp_path / "new-diagnosis")
