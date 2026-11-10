"""Connect full Craftax to the existing typed decision HTTP protocol."""

from __future__ import annotations

import json
import math
import os
import urllib.request
from typing import Any
from urllib.parse import urlsplit

from honing.backends.craftax.harness import CraftaxHarness
from honing.backends.craftax.scenarios import CraftaxScenario
from integrations.craftax.catalog import ACTIONS, action_description


def make_request(
    frame: dict[str, Any], scenario: CraftaxScenario, harness: CraftaxHarness, history: list[dict[str, Any]]
) -> dict[str, Any]:
    """One essential state string, so silent context trimming can be detected exactly."""
    recent = history[-harness.history_steps :] if harness.history_steps else []
    state = "\n".join(
        [
            "Play full Craftax. Task goals: " + ", ".join(scenario.goals) + ".",
            "Execute one action per decision; unavailable actions may do nothing. Inference pauses game time.",
            "The map is an agent-centered observed view, not the hidden world. Null cells are unknown.",
            "DO interacts with the faced tile; moving changes facing. Native crafting prerequisites apply.",
            "Public observation: " + json.dumps(frame["observation"], sort_keys=True, separators=(",", ":")),
            "Achievements: " + ", ".join(frame["achievements"]),
            "Recent actions and outcomes: " + json.dumps(recent, sort_keys=True, separators=(",", ":")),
            harness.notes,
        ]
    )
    return {
        "id": f"craftax-s{scenario.seed}-step{frame['step']}",
        "parts": {"essential": [state], "details": [], "memory": [], "logs": []},
        "questions": {
            "action": {
                "type": "choice",
                "instructions": harness.instructions,
                "criteria": {name: action_description(name) for name in ACTIONS},
            }
        },
    }


def validate_reply(request: dict[str, Any], reply: dict[str, Any]) -> str:
    """Require the complete context, all native probabilities, and an argmax choice."""
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
    if not isinstance(answers, dict) or set(answers) != {"action"} or not isinstance(answers["action"], dict):
        raise ValueError("Model returned incomplete answers")
    answer = answers["action"]
    probabilities, choice = answer.get("probabilities"), answer.get("choice")
    if not isinstance(probabilities, dict) or set(probabilities) != set(ACTIONS) or choice not in ACTIONS:
        raise ValueError("Model returned invalid native probabilities or choice")
    if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()):
        raise ValueError("Model returned invalid native probabilities or choice")
    if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-5):
        raise ValueError("Model probabilities must sum to one")
    if probabilities[choice] < max(probabilities.values()):
        raise ValueError("Model choice must maximize its probabilities")
    return str(choice)


class DecisionClient:
    """Credentials are read from the environment and never written into run records."""

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
        self.base_url, self.timeout_seconds = base_url.rstrip("/"), timeout_seconds

    def _request(self, route: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if os.environ.get("HONING_TOKEN"):
            headers["Authorization"] = "Bearer " + os.environ["HONING_TOKEN"]
        request = urllib.request.Request(
            self.base_url + route, headers=headers, data=None if payload is None else json.dumps(payload).encode()
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            body = response.read(4 * 1024 * 1024 + 1)
        if len(body) > 4 * 1024 * 1024:
            raise ValueError("decision response exceeds 4 MiB")
        value = json.loads(body)
        if not isinstance(value, dict):
            raise ValueError("decision server must return a JSON object")
        return value

    def health(self) -> dict[str, Any]:
        health = self._request("/v1/health")
        if health.get("ready") is not True or not isinstance(health.get("metadata"), dict):
            raise ValueError("decision server is not ready or has no model metadata")
        return health

    def decide(self, request: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        reply = self._request("/v1/decide", request)
        return validate_reply(request, reply), reply
