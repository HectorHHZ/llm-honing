"""One portable contract suite for the shared receiver; no game, server model or ML packages."""

from __future__ import annotations

import copy
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest

from integrations._shared.decision import DecisionClient, checked_action, strict_json_loads

ACTIONS = {"left": "Move left.", "right": "Move right.", "wait": "Stay still."}
ROOT = Path(__file__).resolve().parents[1]


def request_and_response():
    request = {
        "id": 7,
        "parts": {"essential": ["required instructions", "required current state"]},
        "questions": {"action": {"type": "choice", "instructions": "Choose.", "criteria": dict(ACTIONS)}},
    }
    response = {
        "id": 7,
        "stateText": "\n".join(request["parts"]["essential"]),
        "trimmed": [],
        "tokenAudit": {
            "action": {
                "stateTokens": 20,
                "stateRoom": 100,
                "inputTokens": 40,
                "options": len(ACTIONS),
                "stateTruncated": False,
            }
        },
        "answers": {
            "action": {"type": "choice", "choice": "right", "probabilities": {"left": 0, "right": 1, "wait": 0}}
        },
    }
    return request, response


class SharedChoiceContractTests(unittest.TestCase):
    @pytest.mark.unit
    def test_argmax_and_ties_keep_the_original_values(self):
        request, response = request_and_response()
        before = copy.deepcopy(response)
        self.assertEqual(checked_action(response, request, ACTIONS), ("right", 1))
        self.assertEqual(response, before)
        response["answers"]["action"].update(choice="left", probabilities=dict.fromkeys(ACTIONS, 0.3333))
        self.assertEqual(checked_action(response, request, ACTIONS), ("left", 0.3333))

    @pytest.mark.unit
    def test_missing_token_counts_reject_but_explicit_null_is_accepted(self):
        for field in ("stateTokens", "stateRoom", "inputTokens"):
            with self.subTest(field=field):
                request, response = request_and_response()
                del response["tokenAudit"]["action"][field]
                with self.assertRaisesRegex(ValueError, "tokenAudit"):
                    checked_action(response, request, ACTIONS)
        request, response = request_and_response()
        response["tokenAudit"]["action"].update(stateTokens=None, stateRoom=None, inputTokens=None)
        self.assertEqual(checked_action(response, request, ACTIONS), ("right", 1))

    @pytest.mark.unit
    def test_audited_counts_and_clipping_are_checked(self):
        damages = [
            {"stateTruncated": True},
            {"stateTruncated": 0},
            {"stateTruncated": "false"},
            {"options": 2},
            {"options": True},
            {"stateTokens": -1},
            {"inputTokens": True},
            {"stateRoom": 1.5},
            {"stateTokens": 101},
            {"stateTokens": 41},
            {"maxInputTokens": 39},
            {"stateTokens": None, "stateRoom": 0},
            {"inputTokens": None, "maxInputTokens": 0},
        ]
        for damage in damages:
            with self.subTest(damage=damage):
                request, response = request_and_response()
                response["tokenAudit"]["action"].update(damage)
                with self.assertRaisesRegex(ValueError, "tokenAudit"):
                    checked_action(response, request, ACTIONS)

    @pytest.mark.unit
    def test_required_context_never_disappears(self):
        for field, value in (
            ("trimmed", None),
            ("trimmed", {}),
            ("trimmed", ["essential"]),
            ("trimmed", [{"section": "essential", "text": "required current state"}]),
            ("stateText", "required instructions"),
            ("stateText", None),
        ):
            with self.subTest(field=field, value=value):
                request, response = request_and_response()
                response[field] = value
                with self.assertRaises(ValueError):
                    checked_action(response, request, ACTIONS)

    @pytest.mark.unit
    def test_whole_optional_removals_follow_the_existing_server_order(self):
        request, response = request_and_response()
        request["parts"].update(logs=["old log", "new log"], memory=["old memory", "new memory"], details=["detail"])
        removals = [
            {"section": "logs", "text": "old log"},
            {"section": "logs", "text": "new log"},
            {"section": "memory", "text": "new memory"},
        ]
        response["trimmed"] = removals
        response["stateText"] += "\ndetail\nold memory"
        self.assertEqual(
            checked_action(response, request, ACTIONS, optional_sections=("logs", "memory", "details")), ("right", 1)
        )
        for changed in (
            removals[::-1],
            [{"section": "logs", "text": "new log"}],
            [{"section": "logs", "text": "unsubmitted log"}],
        ):
            with self.subTest(removals=changed):
                response["trimmed"] = changed
                with self.assertRaises(ValueError):
                    checked_action(response, request, ACTIONS, optional_sections=("logs", "memory", "details"))
        response["trimmed"] = removals
        with self.assertRaises(ValueError):
            checked_action(response, request, ACTIONS)
        with self.assertRaises(ValueError):
            checked_action(response, request, ACTIONS, optional_sections=("essential",))

    @pytest.mark.unit
    def test_exact_questions_ids_and_answer_type_are_required(self):
        for problem in (
            "id",
            "bool_id",
            "missing_answer",
            "extra_answer",
            "missing_audit",
            "extra_audit",
            "missing_type",
            "bad_type",
            "list_answer",
            "list_response",
        ):
            with self.subTest(problem=problem):
                request, response = request_and_response()
                if problem == "id":
                    response["id"] = 6
                elif problem == "bool_id":
                    response["id"] = True
                elif problem == "missing_answer":
                    response["answers"] = {}
                elif problem == "extra_answer":
                    response["answers"]["extra"] = response["answers"]["action"]
                elif problem == "missing_audit":
                    response["tokenAudit"] = {}
                elif problem == "extra_audit":
                    response["tokenAudit"]["extra"] = response["tokenAudit"]["action"]
                elif problem == "missing_type":
                    del response["answers"]["action"]["type"]
                elif problem == "bad_type":
                    response["answers"]["action"]["type"] = "score"
                elif problem == "list_answer":
                    response["answers"]["action"] = []
                else:
                    response = []
                with self.assertRaises(ValueError):
                    checked_action(response, request, ACTIONS)

    @pytest.mark.unit
    def test_every_submitted_choice_is_validated(self):
        request, response = request_and_response()
        request["questions"]["second"] = copy.deepcopy(request["questions"]["action"])
        response["tokenAudit"]["second"] = copy.deepcopy(response["tokenAudit"]["action"])
        response["answers"]["second"] = copy.deepcopy(response["answers"]["action"])
        self.assertEqual(checked_action(response, request, ACTIONS), ("right", 1))
        response["answers"]["second"]["choice"] = "left"
        with self.assertRaisesRegex(ValueError, "argmax"):
            checked_action(response, request, ACTIONS)

    @pytest.mark.unit
    def test_invalid_distributions_are_never_normalized(self):
        for problem in (
            "zero",
            "two",
            "bool",
            "negative",
            "above_one",
            "nan",
            "infinity",
            "huge",
            "string",
            "null",
            "sum",
            "missing",
            "extra",
            "unoffered",
            "argmax",
        ):
            with self.subTest(problem=problem):
                request, response = request_and_response()
                answer = response["answers"]["action"]
                probabilities = answer["probabilities"]
                if problem in {"zero", "two", "bool"}:
                    probabilities.update(dict.fromkeys(probabilities, {"zero": 0, "two": 2, "bool": True}[problem]))
                elif problem in {"negative", "above_one", "nan", "infinity", "huge", "string", "null"}:
                    probabilities["right"] = {
                        "negative": -1,
                        "above_one": 1.1,
                        "nan": float("nan"),
                        "infinity": float("inf"),
                        "huge": 10**1000,
                        "string": "1",
                        "null": None,
                    }[problem]
                elif problem == "sum":
                    probabilities["right"] = 0.9
                elif problem == "missing":
                    del probabilities["wait"]
                elif problem == "extra":
                    probabilities["invented"] = 0
                elif problem == "unoffered":
                    answer["choice"] = "invented"
                else:
                    answer["choice"] = "left"
                with self.assertRaises(ValueError):
                    checked_action(response, request, ACTIONS)


class SharedHttpAndJsonTests(unittest.TestCase):
    @pytest.mark.unit
    def test_strict_json_rejects_ambiguous_or_nonfinite_responses(self):
        for raw in (
            b"broken",
            b"\xff",
            b'{"id": 7, "id": 8}',
            b'{"answers":{"action":{"type":"choice","type":"score"}}}',
            b'{"id":NaN}',
            b'{"id":Infinity}',
            b'{"id":-Infinity}',
            b'{"id":1e309}',
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                strict_json_loads(raw)

    @pytest.mark.unit
    def test_raw_response_precedes_json_and_object_validation(self):
        class Reply:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self):
                return raw

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "response.raw"
            for raw in (b"broken", b"[]", b'{"id":7,"id":8}', b'{"id":1e309}', b"\xff"):
                with self.subTest(raw=raw), patch("urllib.request.urlopen", return_value=Reply()):
                    with self.assertRaises(ValueError):
                        DecisionClient("http://127.0.0.1:1").decide({"id": 7}, raw_path=output)
                    self.assertEqual(output.read_bytes(), raw)

    @pytest.mark.unit
    def test_http_error_bytes_are_retained_before_the_error_is_raised(self):
        raw = b"\xffupstream failed"
        error = urllib.error.HTTPError("http://localhost/v1/decide", 503, "unavailable", {}, io.BytesIO(raw))
        with tempfile.TemporaryDirectory() as directory, patch("urllib.request.urlopen", side_effect=error):
            output = Path(directory) / "response.raw"
            with self.assertRaises(urllib.error.HTTPError):
                DecisionClient("http://localhost").decide({"id": 7}, raw_path=output)
            self.assertEqual(output.read_bytes(), raw)

    @pytest.mark.unit
    def test_routes_body_and_readiness(self):
        requests = []

        class Reply:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self):
                return b'{"ready":true}'

        def open_request(request, *, timeout):
            requests.append((request, timeout))
            return Reply()

        with patch("urllib.request.urlopen", side_effect=open_request):
            client = DecisionClient("http://localhost:8049/", 5)
            self.assertEqual(client.health(), {"ready": True})
            self.assertEqual(client.decide({"id": 7}), {"ready": True})
        self.assertEqual(
            [request.full_url for request, _ in requests],
            ["http://localhost:8049/v1/health", "http://localhost:8049/v1/decide"],
        )
        self.assertEqual(json.loads(requests[1][0].data), {"id": 7})
        self.assertTrue(all(timeout == 5 for _, timeout in requests))
        with patch.object(client, "_call", return_value={"ready": 1}), self.assertRaises(ValueError):
            client.health()

    @pytest.mark.unit
    def test_scripts_share_the_receiver_and_preserve_their_context_policy(self):
        paths = {
            "hollow-knight/run.py": (),
            "taisei/run.py": (),
            "hurrican/run.py": (),
            "celeste-classic/run.py": ("logs",),
            "hallowdale/run.py": (),
            "vvvvvv/decision.py": ("details", "memory", "logs"),
        }
        imported = 0
        for relative, optional in paths.items():
            path = ROOT / "integrations" / relative
            if not path.exists():
                continue
            imported += 1
            spec = importlib.util.spec_from_file_location("contract_" + relative.split("/")[0], path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            with self.subTest(adapter=relative):
                self.assertIs(module.DecisionClient, sys.modules["_shared.decision"].DecisionClient)
                self.assertEqual(module.checked_decision_action.__module__, "_shared.decision")
                request, response = request_and_response()
                request["questions"]["action"]["criteria"] = dict(module.ACTIONS)
                response["answers"]["action"]["probabilities"] = {key: float(key == "right") for key in module.ACTIONS}
                response["tokenAudit"]["action"]["options"] = len(module.ACTIONS)
                self.assertEqual(module.checked_action(response, request), ("right", 1))
                request["parts"]["logs"] = ["optional log"]
                response["trimmed"] = [{"section": "logs", "text": "optional log"}]
                if optional:
                    self.assertEqual(module.checked_action(response, request), ("right", 1))
                else:
                    with self.assertRaises(ValueError):
                        module.checked_action(response, request)
        self.assertGreater(imported, 0)

    @pytest.mark.functional
    def test_standalone_entrypoints_need_only_the_standard_library(self):
        paths = (
            "hollow-knight/run.py",
            "taisei/run.py",
            "hurrican/run.py",
            "celeste-classic/run.py",
            "hallowdale/run.py",
            "vvvvvv/run.py",
        )
        with tempfile.TemporaryDirectory() as directory:
            for relative in paths:
                path = ROOT / "integrations" / relative
                if not path.exists():
                    continue
                with self.subTest(adapter=relative):
                    result = subprocess.run(
                        [sys.executable, "-S", str(path), "--help"],
                        cwd=directory,
                        capture_output=True,
                        text=True,
                        timeout=15,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("usage:", result.stdout)

    @pytest.mark.functional
    def test_repeated_wrapper_imports_preserve_runtime_discovery_and_the_import_path(self):
        paths = (
            "hollow-knight/run.py",
            "taisei/run.py",
            "hurrican/run.py",
            "celeste-classic/run.py",
            "hallowdale/run.py",
            "vvvvvv/decision.py",
        )
        available = [
            str(ROOT / "integrations" / relative) for relative in paths if (ROOT / "integrations" / relative).exists()
        ]
        # A fresh -S interpreter has no installed Griddly runtime. Leaking the
        # integrations parent creates a false-positive namespace package instead.
        program = f"""
import importlib.util
import sys
paths = {available!r}
before = list(sys.path)
assert importlib.util.find_spec('griddly') is None
shared_client = None
for iteration in range(2):
    for index, path in enumerate(paths):
        spec = importlib.util.spec_from_file_location('wrapper_%s_%s' % (iteration, index), path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert sys.path == before, (path, before, sys.path)
        assert importlib.util.find_spec('griddly') is None, path
        if shared_client is None:
            shared_client = module.DecisionClient
        assert module.DecisionClient is shared_client
# Preserve a caller's preexisting identical path entry as well.
existing = {str(ROOT / 'integrations')!r}
sys.path.append(existing)
with_existing = list(sys.path)
spec = importlib.util.spec_from_file_location('wrapper_existing', paths[0])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert sys.path == with_existing
assert any(entry is existing for entry in sys.path)
print('scoped import path, optional-runtime discovery and shared identity passed')
"""
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-S", "-c", program],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=15,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
