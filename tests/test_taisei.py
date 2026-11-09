"""Portable response rejection and native command-parser checks; no game assets."""

from __future__ import annotations

import ctypes
import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.conftest import load_script


@pytest.fixture
def demo():
    return load_script("integrations/taisei/run.py")


def answer(request: dict) -> dict:
    return {
        "id": request["id"],
        "answers": {
            "action": {
                "choice": "right",
                "probabilities": {key: float(key == "right") for key in request["questions"]["action"]["criteria"]},
            }
        },
        "stateText": "\n".join(request["parts"]["essential"]),
        "trimmed": [],
        "tokenAudit": {
            "action": {
                "stateTruncated": False,
                "stateTokens": 100,
                "stateRoom": 1500,
                "inputTokens": 200,
                "maxInputTokens": 2048,
                "options": 9,
            }
        },
    }


@pytest.mark.parametrize(
    "problem",
    [
        "zero",
        "two",
        "boolean",
        "nan",
        "overflow",
        "negative",
        "bad_id",
        "boolean_id",
        "missing_option",
        "extra_option",
        "bad_argmax",
        "missing_state",
        "clipped",
        "trimmed",
        "bad_audit",
        "extra_question",
        "boolean_tokens",
        "over_budget",
        "malformed_json",
    ],
)
def test_rejection_preserves_bytes_without_native_input(demo, tmp_path: Path, problem: str) -> None:
    observation = {"id": 1, "frame": 42}
    request = demo.request_for(observation, 60)
    response = answer(request)
    probabilities = response["answers"]["action"]["probabilities"]
    if problem in {"zero", "two", "boolean"}:
        value = {"zero": 0, "two": 2, "boolean": True}[problem]
        response["answers"]["action"]["probabilities"] = dict.fromkeys(probabilities, value)
    elif problem in {"nan", "overflow", "negative"}:
        probabilities["stay"] = {"nan": float("nan"), "overflow": float("inf"), "negative": -1}[problem]
    elif problem == "bad_id":
        response["id"] = 2
    elif problem == "boolean_id":
        response["id"] = True
    elif problem == "missing_option":
        probabilities.pop("stay")
    elif problem == "extra_option":
        probabilities["bomb"] = 0
    elif problem == "bad_argmax":
        response["answers"]["action"]["choice"] = "stay"
    elif problem == "missing_state":
        response.pop("stateText")
    elif problem == "clipped":
        response["tokenAudit"]["action"]["stateTruncated"] = True
    elif problem == "trimmed":
        response["trimmed"] = ["essential"]
    elif problem == "bad_audit":
        response["tokenAudit"] = {}
    elif problem == "extra_question":
        response["answers"]["unexpected"] = {}
    elif problem == "boolean_tokens":
        response["tokenAudit"]["action"]["stateTokens"] = True
    elif problem == "over_budget":
        response["tokenAudit"]["action"]["stateTokens"] = 1600
    raw = b"{broken" if problem == "malformed_json" else json.dumps(response).encode()

    class Client:
        def decide(self, _request, *, raw_path):
            raw_path.write_bytes(raw)
            return json.loads(raw, parse_constant=demo.reject_json_constant)

    calls = []
    native = SimpleNamespace(session=tmp_path, step=lambda *args: calls.append(args))
    paused = deepcopy(observation)
    with pytest.raises((ValueError, TypeError)):
        demo.decision_step(Client(), native, observation, 60)
    assert calls == []
    assert observation == paused
    assert (tmp_path / "response-000001.raw").read_bytes() == raw
    assert "rejected" in json.loads((tmp_path / "decisions.jsonl").read_text())


def test_rounded_distribution_and_tied_argmax(demo) -> None:
    request = demo.request_for({"id": 1, "frame": 42}, 60)
    response = answer(request)
    response["answers"]["action"]["probabilities"] = dict.fromkeys(demo.ACTIONS, 0.1111)
    assert demo.checked_action(response, request) == ("right", 0.1111)


def test_all_submitted_context_is_required(demo) -> None:
    request = demo.request_for({"id": 1, "frame": 42}, 60)
    assert set(request["parts"]) == {"essential"}
    assert len(request["questions"]["action"]["criteria"]) <= 16


@pytest.fixture(scope="module")
def native_parser(tmp_path_factory, repo_root: Path):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("C compiler unavailable")
    directory = tmp_path_factory.mktemp("taisei-parser")
    source = directory / "parser.c"
    source.write_text(
        '#include "honing_command.h"\n'
        "int parse(const char *s) { HoningCommand c; return honing_parse_command(s, 1, &c); }\n"
    )
    library = directory / "parser.so"
    subprocess.run(
        [
            compiler,
            "-shared",
            "-fPIC",
            "-Wall",
            "-Werror",
            "-I",
            str(repo_root / "integrations/taisei"),
            str(source),
            "-o",
            str(library),
        ],
        check=True,
    )
    loaded = ctypes.CDLL(str(library))
    loaded.parse.argtypes = [ctypes.c_char_p]
    loaded.parse.restype = ctypes.c_int
    return loaded.parse


@pytest.mark.parametrize(
    "text",
    [
        '{"id":1,"action":"left","frames":60}',
        '{"id":1,"action":"stop","frames":1}',
    ],
)
def test_native_accepts_canonical_commands(native_parser, text: str) -> None:
    assert native_parser(text.encode()) == 1


@pytest.mark.parametrize(
    "text",
    [
        '{"id":0,"action":"left","frames":60}',
        '{"id":1,"action":"left","frames":0}',
        '{"id":1,"action":"left","frames":121}',
        '{"id":1,"action":"bomb","frames":60}',
        '{"id":1,"action":"left","frames":60,"extra":1}',
        '{"id":4294967297,"action":"left","frames":60}',
        '{"id":1,"action":"left","frames":-4294967236}',
        '{"id":true,"action":"left","frames":60}',
        '{"id":1,"action":"left","frames":60}garbage',
    ],
)
def test_native_rejects_stale_out_of_range_and_noncanonical_commands(native_parser, text: str) -> None:
    assert native_parser(text.encode()) == 0
