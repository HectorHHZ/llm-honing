"""Check Honing's existing choice response before any standalone game's native input.

This is a receiver for the current serving contract, not a new wire or trace format.
It deliberately imports no Honing model, ML package, or third-party dependency.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

SECTIONS = ("essential", "details", "memory", "logs")
OPTIONAL_SECTIONS = ("logs", "memory", "details")
REQUIRED_AUDIT = {"stateTokens", "stateRoom", "inputTokens", "options", "stateTruncated"}


def reject_json_constant(value: str) -> Any:
    """Reject JavaScript numeric constants that are not JSON numbers."""
    raise ValueError(f"Nonstandard JSON numeric constant: {value}")


def finite_json_float(value: str) -> float:
    """Reject exponent overflow before a decoded response enters a JSON trace."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite JSON number: {value}")
    return result


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate keys at every JSON object depth rather than selecting a value."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key}")
        result[key] = value
    return result


def strict_json_loads(raw: str | bytes | bytearray) -> Any:
    """Decode ordinary JSON without duplicate keys or nonfinite numeric values."""
    return json.loads(
        raw, parse_constant=reject_json_constant, parse_float=finite_json_float, object_pairs_hook=unique_object
    )


def _retained_state(request: Mapping[str, Any], response: Mapping[str, Any], optional: set[str]) -> str:
    parts = request.get("parts")
    if not isinstance(parts, dict) or set(parts) - set(SECTIONS):
        raise ValueError("Submitted context must contain known state sections")
    retained = {}
    for section in SECTIONS:
        entries = parts.get(section, [])
        if not isinstance(entries, list) or any(not isinstance(text, str) for text in entries):
            raise ValueError("Submitted context sections must be lists of text")
        retained[section] = list(entries)
    trimmed = response.get("trimmed")
    if not isinstance(trimmed, list):
        raise ValueError("Decision response omitted its trimmed context audit")
    for removal in trimmed:
        if not isinstance(removal, dict) or set(removal) != {"section", "text"}:
            raise ValueError("Invalid trimmed context entry")
        section, text = removal["section"], removal["text"]
        if not isinstance(section, str) or section not in optional or not isinstance(text, str):
            raise ValueError("Required or unknown context was removed")
        # Existing serving order: oldest logs, then newest memory, then newest details.
        next_section = next((key for key in OPTIONAL_SECTIONS if key in optional and retained[key]), None)
        if section != next_section or text != retained[section][0 if section == "logs" else -1]:
            raise ValueError("Trimmed context differs from the submitted optional entries or serving order")
        retained[section].pop(0 if section == "logs" else -1)
    state = "\n".join(text for section in SECTIONS for text in retained[section])
    if response.get("stateText") != state:
        raise ValueError("Decision response stateText differs from the complete retained context")
    return state


def checked_action(
    response: Any,
    request: dict[str, Any],
    actions: Mapping[str, str],
    *,
    optional_sections: Iterable[str] = (),
) -> tuple[str, float]:
    """Validate every submitted choice question, then return the native action and its probability.

    All context is required unless an adapter explicitly permits whole optional entries.
    Explicit null token counts are supported; missing token-count fields are not.
    The four-decimal Laya rounding bound permits ties without normalizing any value.
    """
    optional = set(optional_sections)
    if not optional <= set(OPTIONAL_SECTIONS):
        raise ValueError("Only details, memory and logs may be optional context")
    if not isinstance(response, dict):
        raise ValueError("Decision response must be an object")
    if type(request.get("id")) is not int or type(response.get("id")) is not int or response["id"] != request["id"]:
        raise ValueError("Decision response id does not match the observation")
    questions = request.get("questions")
    if not isinstance(questions, dict) or not questions or "action" not in questions:
        raise ValueError("Submitted questions must include an action choice")
    for key, question in questions.items():
        if not isinstance(question, dict) or question.get("type") != "choice":
            raise ValueError(f"Submitted question {key} must be a choice")
        criteria = question.get("criteria")
        if not isinstance(criteria, dict) or not criteria or any(not isinstance(option, str) for option in criteria):
            raise ValueError(f"Submitted question {key} has no offered choices")
    if set(questions["action"]["criteria"]) != set(actions):
        raise ValueError("Submitted action choices differ from the native adapter's actions")
    state = _retained_state(request, response, optional)
    audits = response.get("tokenAudit")
    if not isinstance(audits, dict) or set(audits) != set(questions):
        raise ValueError("Decision response tokenAudit does not cover the submitted questions")
    for key, question in questions.items():
        audit = audits[key]
        if not isinstance(audit, dict) or not set(audit) >= REQUIRED_AUDIT or audit["stateTruncated"] is not False:
            raise ValueError(f"Decision tokenAudit reports truncated or unaudited context for {key}")
        if type(audit["options"]) is not int or audit["options"] != len(question["criteria"]):
            raise ValueError(f"Decision tokenAudit has the wrong option count for {key}")
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
    answers = response.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise ValueError("Decision answers do not cover exactly the submitted questions")
    checked = {}
    for key, question in questions.items():
        answer = answers[key]
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise ValueError(f"Decision answer {key} is not a choice")
        choice, probabilities = answer.get("choice"), answer.get("probabilities")
        if (
            not isinstance(probabilities, dict)
            or set(probabilities) != set(question["criteria"])
            or not isinstance(choice, str)
            or choice not in question["criteria"]
        ):
            raise ValueError(f"Decision response does not cover the offered actions for {key}")
        values = list(probabilities.values())
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 <= value <= 1
            or not math.isfinite(value)
            for value in values
        ):
            raise ValueError("Invalid model probabilities")
        # Laya rounds each probability to four decimal places: at most 0.00005 per option.
        if abs(math.fsum(values) - 1) > len(values) * 0.00005 + 1e-8:
            raise ValueError("Model probabilities do not sum to one within rounding precision")
        if probabilities[choice] != max(values):
            raise ValueError("Chosen input is not the model's argmax")
        checked[key] = (choice, float(probabilities[choice]))
    return checked["action"]


class DecisionClient:
    """Call the existing health/decide routes and save response bytes before decoding."""

    def __init__(self, url: str, timeout_seconds: float = 180) -> None:
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError("decision-url must be an HTTP(S) server base URL")
        if isinstance(timeout_seconds, bool) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
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
            try:
                if raw_path is not None:
                    raw_path.write_bytes(error.read())
            finally:
                error.close()
            raise
        if raw_path is not None:
            raw_path.write_bytes(raw)
        result = strict_json_loads(raw)
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
