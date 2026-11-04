"""Typed decisions over public objects and native legal actions."""

from __future__ import annotations

import json
import math
import os
import urllib.request
from collections import deque
from typing import Any
from urllib.parse import urlsplit

from honing.backends.griddly.catalog import DESCRIPTIONS
from honing.backends.griddly.engine import canonical
from honing.backends.griddly.harness import GriddlyHarness
from honing.backends.griddly.scenarios import GriddlyScenario


def make_request(
    frame: dict[str, Any], scenario: GriddlyScenario, harness: GriddlyHarness, history: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(frame["legal_actions"]) < 2:
        raise ValueError("forced actions do not become model questions")
    state = "\n".join(
        [
            "Spiders: reach the gem; touching a spider kills you. Inference pauses game time.",
            "Choose one legal relative action per tick. NONE facing uses the native UP default. "
            "Enemy motion is random every three ticks.",
            "This is a global object observation, not the original local view. Internal queued actions are hidden.",
            f"Task budget: {scenario.max_steps} ticks; current step: {frame['step']}.",
            "Public observation: " + canonical(frame["observation"]),
            "Recent outcomes: " + canonical(history[-harness.history_steps :] if harness.history_steps else []),
            harness.notes,
        ]
    )
    return {
        "id": f"spiders-l{scenario.level_id}-s{scenario.seed}-step{frame['step']}",
        "parts": {"essential": [state], "details": [], "memory": [], "logs": []},
        "questions": {
            "action": {
                "type": "choice",
                "instructions": harness.instructions,
                "criteria": {name: DESCRIPTIONS[name] for name in frame["legal_actions"]},
            }
        },
    }


def validate_reply(request: dict[str, Any], reply: dict[str, Any]) -> dict[str, float]:
    """Validate context and rounded server probabilities; return the normalized acting distribution."""
    audit = reply.get("tokenAudit")
    if (
        reply.get("id") != request["id"]
        or reply.get("stateText") != request["parts"]["essential"][0]
        or reply.get("trimmed") != []
        or not isinstance(audit, dict)
        or any(not isinstance(v, dict) or v.get("stateTruncated") is True for v in audit.values())
    ):
        raise ValueError("model did not preserve complete decision context")
    answers = reply.get("answers")
    if not isinstance(answers, dict) or set(answers) != {"action"} or not isinstance(answers["action"], dict):
        raise ValueError("model returned incomplete answers")
    answer = answers["action"]
    probabilities, choice = answer.get("probabilities"), answer.get("choice")
    options = request["questions"]["action"]["criteria"]
    if not isinstance(probabilities, dict) or set(probabilities) != set(options) or choice not in options:
        raise ValueError("model returned incomplete native probabilities or choice")
    if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()):
        raise ValueError("model returned invalid probabilities")
    total = math.fsum(probabilities.values())
    # Pinned Laya rounds each option to four decimal places: at most half a unit per option.
    tolerance = len(options) * 0.5e-4 + 1e-12
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError("model probabilities must sum to one")
    if probabilities[choice] < max(probabilities.values()):
        raise ValueError("server choice must maximize its probabilities")
    return {name: probabilities[name] / total for name in options}


class DecisionClient:
    def __init__(self, base_url: str, timeout_seconds: float = 120) -> None:
        url = urlsplit(base_url)
        if (
            url.scheme not in ("http", "https")
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("decision server must be an HTTP(S) URL without credentials, query or fragment")
        self.base_url, self.timeout_seconds = base_url.rstrip("/"), timeout_seconds

    def _request(self, route: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if os.environ.get("HONING_TOKEN"):
            headers["Authorization"] = "Bearer " + os.environ["HONING_TOKEN"]
        request = urllib.request.Request(
            self.base_url + route,
            headers=headers,
            data=None if payload is None else canonical(payload).encode("utf-8"),
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            body = response.read(4 * 1024 * 1024 + 1)
        if len(body) > 4 * 1024 * 1024:
            raise ValueError("decision response exceeds 4 MiB")
        value = json.loads(body)
        if not isinstance(value, dict):
            raise ValueError("decision server must return an object")
        return value

    def health(self) -> dict[str, Any]:
        value = self._request("/v1/health")
        if value.get("ready") is not True or not isinstance(value.get("metadata"), dict):
            raise ValueError("decision server is not ready or has no model metadata")
        return value

    def decide(self, request: dict[str, Any]) -> dict[str, Any]:
        reply = self._request("/v1/decide", request)
        validate_reply(request, reply)
        return reply


def baseline(frame: dict[str, Any]) -> str:
    """A visible-map shortest-path heuristic, avoiding current enemy cells without future simulation."""
    objects, legal = frame["observation"]["objects"], frame["legal_actions"]
    player = next(obj for obj in objects if obj["Name"] == "gnome")
    goal = tuple(next(obj["Location"] for obj in objects if obj["Name"] == "gem"))
    blocked = {tuple(obj["Location"]) for obj in objects if obj["Name"] in ("wall", "spider")}
    start = tuple(player["Location"])
    queue: deque[tuple[tuple[int, int], tuple[int, int] | None]] = deque([(start, None)])
    seen = {start}
    grid = frame["observation"]["grid"]
    directions = ((0, -1), (1, 0), (0, 1), (-1, 0))
    first = None
    while queue:
        location, step = queue.popleft()
        if location == goal:
            first = step
            break
        for dx, dy in directions:
            nxt = (location[0] + dx, location[1] + dy)
            if 0 <= nxt[0] < grid["Width"] and 0 <= nxt[1] < grid["Height"] and nxt not in blocked and nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, nxt if step is None else step))
    if first is None:
        return "WAIT"
    desired = (first[0] - start[0], first[1] - start[1])
    heading = ("UP", "RIGHT", "DOWN", "LEFT").index("UP" if player["Orientation"] == "NONE" else player["Orientation"])
    forward = directions[heading]
    if desired == forward and "FORWARD" in legal:
        return "FORWARD"
    if desired == (-forward[0], -forward[1]) and "BACKWARD" in legal:
        return "BACKWARD"
    turn = "TURN_RIGHT" if desired == directions[(heading + 1) % 4] else "TURN_LEFT"
    return turn if turn in legal else "WAIT"
