"""Determinism gate ignores timing metadata while retaining complete option probabilities."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "integrations/tank-war/scripts/reproducibility-gate.py"
SPEC = importlib.util.spec_from_file_location("tank_reproducibility_gate", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


@pytest.mark.unit
def test_probability_signature_ignores_latency_but_keeps_branches() -> None:
    reply = {
        "answers": {"motion": {"choice": "left", "probabilities": {"left": 0.8, "right": 0.2}}},
        "latencyMs": 51,
        "branches": {"left": {"answers": {"weapon": {"choice": "hold", "probabilities": {"hold": 0.6}}}}},
    }
    signature = module.probabilities(reply)
    reply["latencyMs"] = 900
    assert module.probabilities(reply) == signature
    reply["branches"]["left"]["answers"]["weapon"]["probabilities"]["hold"] = 0.5
    assert module.probabilities(reply) != signature
