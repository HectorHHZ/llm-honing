"""The Tank War identity gate compares every returned option, including branch questions."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "integrations/tank-war/scripts/identity-probe.py"
SPEC = importlib.util.spec_from_file_location("tank_identity_probe", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


@pytest.mark.unit
def test_probability_paths_cover_all_questions_and_branches() -> None:
    reply = {
        "answers": {"motion": {"probabilities": {"left": 0.2, "right": 0.8}}},
        "branches": {"left": {"answers": {"weapon": {"probabilities": {"hold": 0.7, "fire": 0.3}}}}},
    }
    assert module.probabilities(reply) == {
        "answers/motion/left": 0.2,
        "answers/motion/right": 0.8,
        "branches/left/answers/weapon/hold": 0.7,
        "branches/left/answers/weapon/fire": 0.3,
    }
