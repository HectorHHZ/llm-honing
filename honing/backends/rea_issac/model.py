"""REA-Issac actions through Honing's existing typed decision HTTP routes."""

from __future__ import annotations

import json
import math
import os
import urllib.request
from typing import Any
from urllib.parse import urlsplit

from honing.backends.rea_issac.catalog import OPTIONS, action_for, criteria
from honing.backends.rea_issac.harness import ReaIssacHarness
from honing.backends.rea_issac.scenarios import ReaIssacScenario


def make_request(
    observation: dict[str, Any], scenario: ReaIssacScenario, harness: ReaIssacHarness, history: list[dict[str, Any]]
) -> dict[str, Any]:
    recent = history[-harness.history_steps :] if harness.history_steps else []
    frames = min(scenario.decision_frames, scenario.max_ticks - observation["tick"])
    state = "\n".join(
        [
            "Play REA-Issac. Defeat the Warden and survive before the fixed tick budget expires.",
            f"The chosen movement, shooting and bomb actions will be held for {frames} ticks at 30 ticks/second.",
            "Movement and shooting are independent. Inference does not advance game time.",
            "Coordinates start at the top-left: x grows east, y grows south. Diagonal vectors are normalized.",
            "Clear enemies to unseal doors. Gold doors lead to upgrades. Bombs damage nearby enemies, rocks and you.",
            "Bombs trigger on a NO-to-YES transition; release the bomb action before placing another.",
            "Only the current room and discovered map are observed. Unvisited room contents and RNG state are hidden.",
            "Public observation: " + json.dumps(observation, sort_keys=True, separators=(",", ":")),
            "Recent actions and outcomes: " + json.dumps(recent, sort_keys=True, separators=(",", ":")),
            harness.notes,
        ]
    )
    return {
        "id": f"rea-issac-s{scenario.seed}-t{observation['tick']}",
        "parts": {"essential": [state], "details": [], "memory": [], "logs": []},
        "questions": {
            name: {"type": "choice", "instructions": harness.instructions[name], "criteria": criteria(name)}
            for name in OPTIONS
        },
    }


def validate_reply(request: dict[str, Any], reply: dict[str, Any]) -> dict[str, Any]:
    audit = reply.get("tokenAudit")
    if (
        reply.get("id") != request["id"]
        or reply.get("stateText") != request["parts"]["essential"][0]
        or reply.get("trimmed") != []
        or not isinstance(audit, dict)
        or any(not isinstance(entry, dict) or entry.get("stateTruncated") is True for entry in audit.values())
    ):
        raise ValueError("Model did not preserve complete decision context")
    answers = reply.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(OPTIONS):
        raise ValueError("Model must answer move, shoot and bomb")
    choices = {}
    for question, options in OPTIONS.items():
        answer = answers[question]
        if not isinstance(answer, dict):
            raise ValueError("Model answer must be an object")
        probabilities, choice = answer.get("probabilities"), answer.get("choice")
        if not isinstance(probabilities, dict) or set(probabilities) != set(options) or choice not in options:
            raise ValueError("Model returned incomplete probabilities or an invalid choice")
        if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()):
            raise ValueError("Model probabilities must be finite and within 0..1")
        if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-5):
            raise ValueError("Model probabilities must sum to one")
        if probabilities[choice] < max(probabilities.values()):
            raise ValueError("Model choice must maximize its probabilities")
        choices[question] = choice
    return action_for(choices)


class DecisionClient:
    def __init__(self, base_url: str, timeout_seconds: float = 120) -> None:
        parsed = urlsplit(base_url)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("decision server must be an HTTP(S) URL without credentials, query or fragment")
        if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")
        self.base_url, self.timeout_seconds = base_url.rstrip("/"), timeout_seconds

    def _request(self, route: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if os.environ.get("HONING_TOKEN"):
            headers["Authorization"] = "Bearer " + os.environ["HONING_TOKEN"]
        request = urllib.request.Request(
            self.base_url + route,
            headers=headers,
            data=None if payload is None else json.dumps(payload, allow_nan=False).encode(),
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            body = response.read(4 * 1024 * 1024 + 1)
        if len(body) > 4 * 1024 * 1024:
            raise ValueError("decision response exceeds 4 MiB")
        result = json.loads(body)
        if not isinstance(result, dict):
            raise ValueError("decision server must return a JSON object")
        return result

    def health(self) -> dict[str, Any]:
        health = self._request("/v1/health")
        if health.get("ready") is not True or not isinstance(health.get("metadata"), dict):
            raise ValueError("decision server is not ready or has no model metadata")
        return health

    def decide(self, request: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        response = self._request("/v1/decide", request)
        return validate_reply(request, response), response
