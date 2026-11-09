"""Portable fixtures validate the decision gate; these are not native/model evidence."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def adapter(repo_root: Path, monkeypatch):
    path = repo_root / "integrations/vvvvvv"
    monkeypatch.syspath_prepend(str(path))
    spec = importlib.util.spec_from_file_location("vvvvvv_run", path / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("decision", None)


def response_for(request: dict) -> dict:
    options = request["questions"]["action"]["criteria"]
    return {
        "id": request["id"],
        "trimmed": [],
        "stateText": "\n".join(
            text
            for section in ("essential", "details", "memory", "logs")
            for text in request["parts"].get(section, [])
        ),
        "tokenAudit": {
            "action": {
                "options": len(options),
                "stateTruncated": False,
                "stateTokens": 100,
                "stateRoom": 1000,
                "inputTokens": 200,
            }
        },
        "answers": {"action": {"choice": "right", "probabilities": {key: float(key == "right") for key in options}}},
    }


PROBLEMS = [
    "id",
    "bool_id",
    "missing_option",
    "extra_option",
    "unknown_choice",
    "argmax",
    "zero",
    "two",
    "bool",
    "negative",
    "nan",
    "inf",
    "too_large",
    "bad_sum",
    "missing_state",
    "state_changed",
    "required_removed",
    "missing_trimmed",
    "bad_trimmed",
    "missing_audit",
    "missing_question_audit",
    "truncated",
    "bad_options",
    "bool_options",
    "over_budget",
    "bad_count",
    "extra_answer",
    "missing_answer",
]


def damage(response: dict, problem: str) -> None:
    answer = response["answers"]["action"]
    probabilities = answer["probabilities"]
    audit = response["tokenAudit"]["action"]
    if problem == "id":
        response["id"] += 1
    elif problem == "bool_id":
        response["id"] = True
    elif problem == "missing_option":
        del probabilities["wait"]
    elif problem == "extra_option":
        probabilities["invented"] = 0
    elif problem == "unknown_choice":
        answer["choice"] = "invented"
    elif problem == "argmax":
        answer["choice"] = "left"
    elif problem in {"zero", "two", "bool"}:
        answer["probabilities"] = dict.fromkeys(probabilities, {"zero": 0, "two": 2, "bool": True}[problem])
    elif problem in {"negative", "nan", "inf", "too_large"}:
        probabilities["wait"] = {"negative": -1, "nan": float("nan"), "inf": float("inf"), "too_large": 1.1}[problem]
    elif problem == "bad_sum":
        probabilities["wait"] = 0.1
    elif problem == "missing_state":
        del response["stateText"]
    elif problem == "state_changed":
        response["stateText"] += "changed"
    elif problem == "required_removed":
        response["trimmed"] = [{"section": "essential", "text": "current state"}]
    elif problem == "missing_trimmed":
        del response["trimmed"]
    elif problem == "bad_trimmed":
        response["trimmed"] = ["essential"]
    elif problem == "missing_audit":
        del response["tokenAudit"]
    elif problem == "missing_question_audit":
        response["tokenAudit"] = {}
    elif problem == "truncated":
        audit["stateTruncated"] = True
    elif problem == "bad_options":
        audit["options"] = 5
    elif problem == "bool_options":
        audit["options"] = True
    elif problem == "over_budget":
        audit["stateTokens"] = 1001
    elif problem == "bad_count":
        audit["stateRoom"] = -1
    elif problem == "extra_answer":
        response["answers"]["other"] = copy.deepcopy(answer)
    elif problem == "missing_answer":
        response["answers"] = {}
    else:
        raise AssertionError(problem)


@pytest.mark.parametrize("problem", PROBLEMS)
def test_rejection_does_not_step(adapter, tmp_path, problem):
    state = {"id": 7, "tick": 30}

    class Native:
        calls = 0

        def step(self, *_args):
            self.calls += 1
            state["tick"] += 15
            return state

    class Client:
        def decide(self, request, *, raw_path):
            response = response_for(request)
            damage(response, problem)
            raw_path.write_bytes(json.dumps(response).encode())
            return response

    native = Native()
    with pytest.raises((ValueError, TypeError)):
        adapter.decide_and_step(Client(), native, state, 15, tmp_path)
    assert native.calls == 0
    assert state == {"id": 7, "tick": 30}
    artifact = json.loads((tmp_path / "000007.rejected.json").read_text())
    assert artifact["nativeCalls"] == 0 and artifact["pausedState"] == state
    assert (tmp_path / artifact["responseRawFile"]).read_bytes()


def test_rounding_ties_optional_removal(adapter):
    request = adapter.request_for({"id": 7}, 15)
    request["parts"]["memory"] = ["optional old history"]
    response = response_for(request)
    response["trimmed"] = [{"section": "memory", "text": "optional old history"}]
    response["stateText"] = "\n".join(request["parts"]["essential"])
    response["answers"]["action"].update(choice="right", probabilities=dict.fromkeys(adapter.ACTIONS, 0.1667))
    assert adapter.checked_action(response, request) == ("right", 0.1667)


@pytest.mark.parametrize("raw", [b"not json", b"[]", b'{"id":NaN}', b'{"id":1e999}'])
def test_raw_invalid_json_preserved(adapter, tmp_path, monkeypatch, raw):
    from decision import DecisionClient

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return raw

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: Reply())
    output = tmp_path / "response.raw"
    with pytest.raises(ValueError):
        DecisionClient("http://127.0.0.1:1").decide({"id": 7}, raw_path=output)
    assert output.read_bytes() == raw
