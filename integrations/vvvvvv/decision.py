"""Validate the existing Honing decision contract before native gameplay input."""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ACTIONS = {
    "wait": "Release movement and flip.",
    "left": "Move left.",
    "right": "Move right.",
    "flip": "Flip gravity once if touching ground or roof.",
    "flip_left": "Move left and flip gravity once.",
    "flip_right": "Move right and flip gravity once.",
}


def checked_action(response: dict[str, Any], request: dict[str, Any]) -> tuple[str, float]:
    """Require the complete submitted state and audited choices before sending input to VVVVVV."""
    if type(response.get("id")) is not int or response["id"] != request["id"]:
        raise ValueError("Decision response id does not match the observation")
    # Current native state is essential. Future optional details/history may be removed whole.
    sections = ("essential", "details", "memory", "logs")
    retained = {section: list(request["parts"].get(section, [])) for section in sections}
    trimmed = response.get("trimmed")
    if not isinstance(trimmed, list):
        raise ValueError("Decision response omitted its trimmed audit")
    for removal in trimmed:
        if not isinstance(removal, dict) or set(removal) != {"section", "text"}:
            raise ValueError("Invalid trimmed context entry")
        section, text = removal["section"], removal["text"]
        if section not in {"details", "memory", "logs"} or text not in retained[section]:
            raise ValueError("Required or unknown context was removed")
        retained[section].remove(text)
    state = "\n".join(text for section in sections for text in retained[section])
    if response.get("stateText") != state:
        raise ValueError("Decision response stateText differs from the complete submitted context")
    audits = response.get("tokenAudit")
    if not isinstance(audits, dict) or set(audits) != set(request["questions"]):
        raise ValueError("Decision response tokenAudit does not cover the submitted questions")
    for key, question in request["questions"].items():
        audit = audits[key]
        if not isinstance(audit, dict) or audit.get("stateTruncated") is not False:
            raise ValueError(f"Decision tokenAudit reports truncated or unaudited context for {key}")
        if type(audit.get("options")) is not int or audit["options"] != len(question["criteria"]):
            raise ValueError(f"Decision tokenAudit has the wrong option count for {key}")
        # DJEV reports null token counts. Validate all available counts without inventing them.
        for field in ("stateTokens", "stateRoom", "inputTokens", "maxInputTokens"):
            value = audit.get(field)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"Decision tokenAudit has an invalid {field} for {key}")
        size, room, total, limit = (
            audit.get(field) for field in ("stateTokens", "stateRoom", "inputTokens", "maxInputTokens")
        )
        if (
            (size is not None and room is not None and size > room)
            or (size is not None and total is not None and size > total)
            or (total is not None and limit is not None and total > limit)
            or (state and any(value == 0 for value in (size, room, total, limit)))
        ):
            raise ValueError(f"Decision tokenAudit contradicts complete context for {key}")
    if not isinstance(response.get("answers"), dict) or set(response["answers"]) != set(request["questions"]):
        raise ValueError("Decision answers do not cover exactly the submitted questions")
    try:
        answer = response["answers"]["action"]
        action, probabilities = answer["choice"], answer["probabilities"]
    except (KeyError, TypeError) as error:
        raise ValueError("Decision response has no action distribution") from error
    if (
        not isinstance(probabilities, dict)
        or set(probabilities) != set(ACTIONS)
        or not isinstance(action, str)
        or action not in ACTIONS
    ):
        raise ValueError("Decision response does not cover the offered actions")
    values = list(probabilities.values())
    if any(
        isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1
        for p in values
    ):
        raise ValueError("Invalid model probabilities")
    # Laya returns four decimal places, allowing at most 0.00005 rounding error per option.
    if abs(math.fsum(values) - 1) > len(ACTIONS) * 0.00005 + 1e-8:
        raise ValueError("Model probabilities do not sum to one within rounding precision")
    if probabilities[action] != max(values):
        raise ValueError("Chosen input is not the model's argmax")
    return action, float(probabilities[action])


def reject_json_constant(value: str) -> Any:
    """Keep NaN/Infinity out of the JSON trace as well as the game input."""
    raise ValueError(f"Nonstandard JSON numeric constant: {value}")


def finite_json_float(value: str) -> float:
    """Reject exponent overflow before a decoded response can enter the JSON trace."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite JSON number: {value}")
    return result


class DecisionClient:
    """Use Honing's existing health and decide routes without loading a model in the game client."""

    def __init__(self, url: str, timeout_seconds: float = 180) -> None:
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError("decision-url must be an HTTP(S) server base URL")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout must be positive and finite")
        self.url = url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _call(
        self, route: str, payload: dict[str, Any] | None = None, *, raw_path: Path | None = None
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        request = urllib.request.Request(self.url + route, body, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            if raw_path is not None:
                raw_path.write_bytes(error.read())
            raise
        if raw_path is not None:
            raw_path.write_bytes(raw)
        result = json.loads(raw, parse_constant=reject_json_constant, parse_float=finite_json_float)
        if not isinstance(result, dict):
            raise ValueError("Decision server response must be an object")
        return result

    def health(self) -> dict[str, Any]:
        result = self._call("/v1/health")
        if result.get("ready") is not True:
            raise ValueError("Decision server is not ready")
        return result

    def decide(self, request: dict[str, Any], *, raw_path: Path | None = None) -> dict[str, Any]:
        return self._call("/v1/decide", request, raw_path=raw_path)
