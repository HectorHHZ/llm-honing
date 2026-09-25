"""Independent reconstruction detects altered native request packing without models."""

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from examples.ra2.integration.models.choice_protocol import pack_request

LEAGUE = Path(__file__).resolve().parents[1] / "examples/ra2/league"
sys.path.insert(0, str(LEAGUE))
SPEC = importlib.util.spec_from_file_location("ra2_native_encoding_verify", LEAGUE / "verify_native_encoding.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def measure(state, questions):
    return {
        key: {
            "stateTokens": len(state),
            "inputTokens": len(state) + len(json.dumps(question)),
            "encoding": "fixture-only",
            "promptSha256": hashlib.sha256((state + json.dumps(question)).encode()).hexdigest(),
        }
        for key, question in questions.items()
    }


def test_reencoding_compares_context_options_instructions_and_native_token_audit():
    parts = {"essential": ["normal state"], "logs": ["old log"]}
    questions = {"next_action": {"type": "choice", "instructions": "Choose", "criteria": {"A": "Wait", "B": "Attack"}}}
    state, audit, trimmed = pack_request(parts, questions, measure, 2048)
    prediction = {"stateText": state, "tokenAudit": audit, "trimmed": trimmed}
    MODULE.verify_prediction(parts, questions, prediction, measure, 2048)
    changed = copy.deepcopy(questions)
    changed["next_action"]["instructions"] = "Other instructions"
    with pytest.raises(ValueError, match="reconstructed"):
        MODULE.verify_prediction(parts, changed, prediction, measure, 2048)
    changed = copy.deepcopy(questions)
    changed["next_action"]["criteria"]["B"] = "Retreat"
    with pytest.raises(ValueError, match="reconstructed"):
        MODULE.verify_prediction(parts, changed, prediction, measure, 2048)
    changed_prediction = copy.deepcopy(prediction)
    changed_prediction["tokenAudit"]["next_action"]["promptSha256"] = "0" * 64
    with pytest.raises(ValueError, match="reconstructed"):
        MODULE.verify_prediction(parts, questions, changed_prediction, measure, 2048)
