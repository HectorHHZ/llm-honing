"""Portable receiver tests; these fixtures are not game or model evidence."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest

SPEC = importlib.util.spec_from_file_location(
    "hurrican_run", Path(__file__).parents[1] / "integrations/hurrican/run.py"
)
assert SPEC is not None and SPEC.loader is not None
run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run)
pytestmark = pytest.mark.unit


class Native:
    def __init__(self):
        self.calls = 0
        self.tick = 0

    def step(self, observation, action, frames):
        self.calls += 1
        self.tick += frames
        return {"id": observation["id"] + 1, "tick": self.tick}


class Decision:
    def __init__(self, response):
        self.response = response
        self.raw = json.dumps(response).encode()

    def decide(self, request, *, raw_path):
        raw_path.write_bytes(self.raw)
        return self.response


def fixture():
    observation = {"id": 0, "tick": 0, "player": {"x": 12, "energy": 100}}
    request = run.request_for(observation, 40)
    response = {
        "id": 0,
        "stateText": "\n".join(request["parts"]["essential"]),
        "trimmed": [],
        "tokenAudit": {
            "action": {
                "stateTokens": 200,
                "stateRoom": 1930,
                "inputTokens": 300,
                "options": 10,
                "stateTruncated": False,
            }
        },
        "answers": {
            "action": {
                "type": "choice",
                "choice": "right",
                "probabilities": {key: 1.0 if key == "right" else 0.0 for key in run.ACTIONS},
            }
        },
    }
    return observation, request, response


def mutate(response, case):
    probabilities = response["answers"]["action"]["probabilities"]
    audit = response["tokenAudit"]["action"]
    if case in ("zero", "two", "boolean", "nan", "infinity", "negative", "above_one"):
        value = {
            "zero": 0,
            "two": 2,
            "boolean": True,
            "nan": float("nan"),
            "infinity": float("inf"),
            "negative": -0.1,
            "above_one": 1.1,
        }[case]
        probabilities.update(dict.fromkeys(probabilities, value))
    elif case == "id":
        response["id"] = 2
    elif case == "boolean_id":
        response["id"] = False
    elif case == "argmax":
        response["answers"]["action"]["choice"] = "left"
    elif case == "missing_option":
        del probabilities["wait"]
    elif case == "extra_option":
        probabilities["invented"] = 0
    elif case == "extra_question":
        response["answers"]["invented"] = {}
    elif case == "missing_question":
        response["answers"] = {}
    elif case == "state":
        response["stateText"] = "missing current state"
    elif case == "trimmed":
        response["trimmed"] = [{"section": "essential", "index": 1}]
    elif case == "missing_trimmed":
        del response["trimmed"]
    elif case == "truncated":
        audit["stateTruncated"] = True
    elif case == "missing_audit":
        response["tokenAudit"] = {}
    elif case == "missing_audit_field":
        del audit["inputTokens"]
    elif case == "options":
        audit["options"] = 9
    elif case == "tokens":
        audit["stateRoom"] = 199
    elif case == "boolean_tokens":
        audit["stateTokens"] = True
    elif case == "zero_tokens":
        audit["stateTokens"] = 0
    elif case == "answer_type":
        response["answers"]["action"]["type"] = "text"
    else:
        raise AssertionError(case)


@pytest.mark.parametrize(
    "case",
    [
        "zero",
        "two",
        "boolean",
        "nan",
        "infinity",
        "negative",
        "above_one",
        "id",
        "boolean_id",
        "argmax",
        "missing_option",
        "extra_option",
        "extra_question",
        "missing_question",
        "state",
        "trimmed",
        "missing_trimmed",
        "truncated",
        "missing_audit",
        "missing_audit_field",
        "options",
        "tokens",
        "boolean_tokens",
        "zero_tokens",
        "answer_type",
    ],
)
def test_rejection_preserves_bytes_and_never_advances(tmp_path, case):
    observation, _, response = fixture()
    mutate(response, case)
    native = Native()
    decision = Decision(response)
    with pytest.raises(ValueError):
        run.decide_step(native, decision, observation, 40, tmp_path)
    assert native.calls == 0
    assert native.tick == 0
    assert next(tmp_path.glob("*.response.raw")).read_bytes() == decision.raw
    rejection = json.loads(next(tmp_path.glob("*.rejection.json")).read_text())
    assert rejection["request"]["id"] == observation["id"]
    assert rejection["reason"]


def test_valid_choice_rounding_and_tie(tmp_path):
    observation, request, response = fixture()
    response["answers"]["action"]["probabilities"] = dict.fromkeys(run.ACTIONS, 0.1000)
    assert run.checked_action(response, request) == ("right", 0.1)
    native = Native()
    result = run.decide_step(native, Decision(response), observation, 40, tmp_path)
    assert native.calls == 1 and result["tick"] == 40


def test_rounding_bound_rejects_above_precision():
    _, request, response = fixture()
    probabilities = response["answers"]["action"]["probabilities"]
    probabilities.update(dict.fromkeys(probabilities, 0.1001))
    with pytest.raises(ValueError, match="sum"):
        run.checked_action(response, request)
    probabilities.update(dict.fromkeys(probabilities, 0.10004))
    assert run.checked_action(response, request)[0] == "right"


def test_no_optional_context_is_submitted():
    _, request, _ = fixture()
    assert set(request["parts"]) == {"essential"}
    assert set(request["questions"]) == {"action"}
    assert len(request["questions"]["action"]["criteria"]) == 10


def test_missing_audit_counts_do_not_look_like_nullable_counts():
    _, request, response = fixture()
    for field in ("stateTokens", "stateRoom", "inputTokens"):
        response["tokenAudit"]["action"][field] = None
    assert run.checked_action(response, request)[0] == "right"


@pytest.mark.parametrize("raw", [b"not JSON", b"[]", b'{"id":NaN}', b'{"id":1e999}'])
def test_http_raw_bytes_survive_parse_failure(tmp_path, raw):
    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return raw

    path = tmp_path / "response.raw"
    with patch.object(run.urllib.request, "urlopen", return_value=Reply()), pytest.raises(ValueError):
        run.DecisionClient("http://localhost:1").decide({}, raw_path=path)
    assert path.read_bytes() == raw


def test_separate_attempts_do_not_overwrite_rejections(tmp_path):
    observation, _, response = fixture()
    response["id"] = 2
    for _ in range(2):
        with pytest.raises(ValueError):
            run.decide_step(Native(), Decision(copy.deepcopy(response)), observation, 40, tmp_path)
    assert len(list(tmp_path.glob("*.rejection.json"))) == 2
