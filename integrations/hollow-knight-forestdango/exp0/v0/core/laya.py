"""Client for the frozen Laya decision service plus the response guard.

The guard is `run.py::checked_action` from code/work/viewport-hollow-knight/integrations/hollow-knight
with one generalisation: the offered option set is taken from the request (skills change per state)
instead of the fixed 11 native actions. Every other check is kept: id match, `trimmed == []`, full
`stateText` echo, `tokenAudit` per question (not truncated, option count, token-count consistency),
probabilities over exactly the offered options, finite, in [0,1], summing to one within rounding, and
the returned choice must be the argmax. A rejection raises: the caller must not execute anything and
must keep the raw response bytes (they are written to `raw_path` before parsing).

The service at 127.0.0.1:8809 is owned by another process: this module only does GET /v1/health and
POST /v1/decide. It never starts, stops or reconfigures the service.

Exp 0 v0 copy (code/exp0/hk_v9/core, 2026-11-25): copied from code/hk-hitl/core/laya.py; the only change is that
there is no default URL and 8808/8809 are refused: the collector passes 8811 (clean) or 8812 (alternate).
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

URL = None  # Exp 0 copy: no default service; the collector passes 8811 (clean) or 8812 (alternate)
MAX_OPTIONS = 16


class GuardRejection(ValueError):  # noqa: N818
    """The response failed the guard; nothing may be executed for this decision."""


def build_request(identity: int, essential: list[str], instructions: str, criteria: dict[str, str]) -> dict[str, Any]:
    if not 1 <= len(criteria) <= MAX_OPTIONS:
        raise ValueError("Laya accepts 1..16 options")
    return {
        "id": int(identity),
        "parts": {"essential": list(essential)},
        "questions": {"action": {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}},
    }


def checked_choice(response: dict[str, Any], request: dict[str, Any]) -> tuple[str, float, dict[str, float]]:
    """Return (choice, probability, probabilities) or raise GuardRejection."""
    try:
        if type(response.get("id")) is not int or response["id"] != request["id"]:
            raise GuardRejection("Decision response id does not match the request")
        if response.get("trimmed") != []:
            raise GuardRejection("Decision response trimmed context or omitted its trimmed audit")
        state = "\n".join(
            text
            for section in ("essential", "details", "memory", "logs")
            for text in request["parts"].get(section, [])
        )
        if response.get("stateText") != state:
            raise GuardRejection("Decision response stateText differs from the complete submitted context")
        audits = response.get("tokenAudit")
        if not isinstance(audits, dict) or set(audits) != set(request["questions"]):
            raise GuardRejection("Decision response tokenAudit does not cover the submitted questions")
        for key, question in request["questions"].items():
            audit = audits[key]
            if not isinstance(audit, dict) or audit.get("stateTruncated") is not False:
                raise GuardRejection(f"tokenAudit reports truncated or unaudited context for {key}")
            if type(audit.get("options")) is not int or audit["options"] != len(question["criteria"]):
                raise GuardRejection(f"tokenAudit has the wrong option count for {key}")
            for field in ("stateTokens", "stateRoom", "inputTokens", "maxInputTokens"):
                value = audit.get(field)
                if value is not None and (type(value) is not int or value < 0):
                    raise GuardRejection(f"tokenAudit has an invalid {field} for {key}")
            size, room, total, limit = (
                audit.get(f) for f in ("stateTokens", "stateRoom", "inputTokens", "maxInputTokens")
            )
            if (
                (size is not None and room is not None and size > room)
                or (size is not None and total is not None and size > total)
                or (total is not None and limit is not None and total > limit)
                or (state and any(value == 0 for value in (size, room, total, limit)))
            ):
                raise GuardRejection(f"tokenAudit contradicts complete context for {key}")
        offered = request["questions"]["action"]["criteria"]
        try:
            answer = response["answers"]["action"]
            choice, probabilities = answer["choice"], answer["probabilities"]
        except (KeyError, TypeError) as error:
            raise GuardRejection("Decision response has no action distribution") from error
        if (
            not isinstance(probabilities, dict)
            or set(probabilities) != set(offered)
            or not isinstance(choice, str)
            or choice not in offered
        ):
            raise GuardRejection("Decision response does not cover exactly the offered options")
        values = list(probabilities.values())
        if any(
            isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1
            for p in values
        ):
            raise GuardRejection("Invalid model probabilities")
        if abs(math.fsum(values) - 1) > len(offered) * 0.00005 + 1e-8:
            raise GuardRejection("Model probabilities do not sum to one within rounding precision")
        if probabilities[choice] != max(values):
            raise GuardRejection("Chosen option is not the model's argmax")
        return choice, float(probabilities[choice]), {k: float(v) for k, v in probabilities.items()}
    except GuardRejection:
        raise
    except Exception as error:  # malformed structure of any other kind is also a rejection
        raise GuardRejection(f"Malformed decision response: {type(error).__name__}: {error}") from error


def _reject_constant(value: str) -> Any:
    raise ValueError(f"Nonstandard JSON numeric constant: {value}")


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite JSON number: {value}")
    return result


class LayaClient:
    def __init__(self, url: str = URL, timeout: float = 240.0):
        if not url or url.rstrip("/").endswith((":8808", ":8809")):
            raise ValueError(
                "Exp 0 copy: pass the service URL explicitly; 8808/8809 belong to Codex / the Demo session"
            )
        self.url = url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict[str, Any]:
        with urllib.request.urlopen(self.url + "/v1/health", timeout=30) as response:
            result = json.loads(response.read())
        if result.get("ready") is not True:
            raise RuntimeError("Decision server is not ready")
        return result

    def decide(self, request: dict[str, Any], raw_path: Path | None = None) -> tuple[dict[str, Any], float]:
        """POST the request; the raw bytes are saved before any parsing. Returns (response, seconds)."""
        body = json.dumps(request, allow_nan=False).encode()
        http = urllib.request.Request(self.url + "/v1/decide", body, {"Content-Type": "application/json"})
        started = time.monotonic()
        try:
            with urllib.request.urlopen(http, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            if raw_path is not None:
                raw_path.write_bytes(error.read())
            raise
        seconds = time.monotonic() - started
        if raw_path is not None:
            raw_path.write_bytes(raw)
        try:
            result = json.loads(raw, parse_constant=_reject_constant, parse_float=_finite_float)
        except Exception as error:
            raise GuardRejection(f"Unparseable decision response: {error}") from error
        if not isinstance(result, dict):
            raise GuardRejection("Decision server response must be an object")
        return result, seconds
