"""Honing's existing typed HTTP decision envelope over a full kitchen snapshot."""

from __future__ import annotations

import math
import os
import urllib.request
from typing import Any
from urllib.parse import urlsplit

from integrations.overcooked.engine import ACTIONS, canonical

DESCRIPTIONS = {
    "NORTH": "Move up one tile and face north",
    "SOUTH": "Move down one tile and face south",
    "EAST": "Move right one tile and face east",
    "WEST": "Move left one tile and face west",
    "STAY": "Wait one game tick",
    "INTERACT": "Interact with the adjacent tile you face",
}
DEFAULT_INSTRUCTIONS = "Choose one native action to cooperate with your teammate and deliver the allowed soups."


def make_request(
    frame: dict[str, Any],
    player: int,
    instructions: str = DEFAULT_INSTRUCTIONS,
    history: list[dict[str, Any]] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    if frame["status"] != "running":
        raise ValueError("a finished game has no decision")
    state = "\n".join(
        [
            "Overcooked-AI: you control chef " + str(player) + ". Both chefs share delivery rewards.",
            "Observation: full public kitchen; teammate's pending choice is not revealed. Inference pauses game time.",
            "Rules: " + canonical(frame["rules"]),
            "Map: " + canonical(frame["map"]),
            "State: " + canonical(frame["state"]),
            f"Score {frame['score']}; deliveries {frame['deliveries']}; step {frame['step']}/{frame['max_steps']}.",
            "Recent joint actions and outcomes: " + canonical(history or []),
            notes,
        ]
    )
    return {
        "id": f"overcooked-p{player}-t{frame['step']}-{frame['frame_hash'][:12]}",
        "parts": {"essential": [state], "details": [], "memory": [], "logs": []},
        "questions": {"action": {"type": "choice", "instructions": instructions, "criteria": DESCRIPTIONS.copy()}},
    }


def validate_reply(request: dict[str, Any], reply: dict[str, Any]) -> dict[str, float]:
    if not isinstance(reply, dict):
        raise ValueError("decision reply must be an object")
    if (
        reply.get("id") != request["id"]
        or reply.get("stateText") != request["parts"]["essential"][0]
        or reply.get("trimmed") != []
    ):
        raise ValueError("decision reply changed the ID or required context")
    audit = reply.get("tokenAudit")
    entry = audit.get("action") if isinstance(audit, dict) else None
    if (
        not isinstance(entry, dict)
        or set(audit) != {"action"}
        or entry.get("stateTruncated") is not False
        or type(entry.get("options")) is not int
        or entry["options"] != len(ACTIONS)
    ):
        raise ValueError("decision reply must audit all six options and preserve context")
    for key in ("stateTokens", "stateRoom", "inputTokens"):
        if key not in entry or (entry[key] is not None and (type(entry[key]) is not int or entry[key] <= 0)):
            raise ValueError("decision reply has missing or invalid token counts")
    size, room, total = (entry[key] for key in ("stateTokens", "stateRoom", "inputTokens"))
    limit = entry.get("maxInputTokens")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("decision audit has an invalid token limit")
    if (
        (size is not None and room is not None and size > room)
        or (size is not None and total is not None and size > total)
        or (total is not None and limit is not None and total > limit)
        or (room is not None and limit is not None and room > limit)
    ):
        raise ValueError("decision token audit contradicts complete context")
    answers = reply.get("answers")
    if not isinstance(answers, dict) or set(answers) != {"action"}:
        raise ValueError("decision reply must answer action")
    answer = answers["action"]
    if not isinstance(answer, dict):
        raise ValueError("action answer must be an object")
    probs, chosen = answer.get("probabilities"), answer.get("choice")
    if not isinstance(probs, dict) or set(probs) != set(ACTIONS) or chosen not in ACTIONS:
        raise ValueError("decision reply must cover exactly the native actions")
    if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in probs.values()):
        raise ValueError("decision probabilities must be finite and in [0,1]")
    total = math.fsum(probs.values())
    if not math.isclose(total, 1.0, rel_tol=0, abs_tol=len(ACTIONS) * 0.5e-4 + 1e-12):
        raise ValueError("decision probabilities do not sum to one")
    if probs[chosen] < max(probs.values()):
        raise ValueError("server choice is not an argmax")
    return {name: probs[name] / total for name in ACTIONS}


class DecisionClient:
    def __init__(self, base_url: str, timeout: float = 120) -> None:
        url = urlsplit(base_url)
        if (
            url.scheme not in ("http", "https")
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("decision endpoint must be HTTP(S) without credentials, query or fragment")
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self.base_url, self.timeout = base_url.rstrip("/"), timeout

    def _request(self, route: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if os.environ.get("HONING_TOKEN"):
            headers["Authorization"] = "Bearer " + os.environ["HONING_TOKEN"]
        req = urllib.request.Request(
            self.base_url + route,
            headers=headers,
            data=None if payload is None else canonical(payload).encode("utf-8"),
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            data = response.read(4 * 1024 * 1024 + 1)
        if len(data) > 4 * 1024 * 1024:
            raise ValueError("decision response exceeds 4 MiB")
        import json

        value = json.loads(data)
        if not isinstance(value, dict):
            raise ValueError("server response must be an object")
        return value

    def health(self) -> dict[str, Any]:
        value = self._request("/v1/health")
        if value.get("ready") is not True or not isinstance(value.get("metadata"), dict):
            raise ValueError("decision server is not ready")
        return value

    def decide(self, request: dict[str, Any]) -> dict[str, Any]:
        reply = self._request("/v1/decide", request)
        validate_reply(request, reply)
        return reply
