# fmt: off
# isort: skip_file
# ruff: noqa
"""Client for the frozen Laya decision service plus the response guard.

The guard is the repository's shared receiver, `integrations/_shared/decision.py`, called with the options this
request offered (skills change per state) instead of a fixed native action set. It checks the id, `trimmed == []`,
the full `stateText` echo, `tokenAudit` per question (required keys present, not truncated, option count, token-count
consistency), probabilities over exactly the offered options, finite, in [0,1], summing to one within rounding, and
that the returned choice is the argmax. A rejection raises GuardRejection: the caller must not execute anything. The
raw response bytes are written to `raw_path` before parsing, also for an HTTP error body.

The service is owned by another process: this module only does GET /v1/health and POST /v1/decide. It never starts,
stops or reconfigures the service. HONING_DECISION_URL overrides its address.
"""
from __future__ import annotations

import http.client
import os
import sys
import time
from pathlib import Path
from typing import Any

# Resolve the sibling stdlib-only receiver without exposing integrations as importable runtimes.
_shared_import_path = str(Path(__file__).resolve().parents[4])
sys.path.insert(0, _shared_import_path)
try:
    from _shared.decision import DecisionClient, checked_action
finally:
    # Remove only our inserted object, preserving preexisting equal path entries.
    for _shared_index, _shared_entry in enumerate(sys.path):
        if _shared_entry is _shared_import_path:
            del sys.path[_shared_index]
            break

URL = os.environ.get("HONING_DECISION_URL", "http://127.0.0.1:8809")
MAX_OPTIONS = 16


class GuardRejection(ValueError):
    """The response failed the guard; nothing may be executed for this decision."""


def build_request(identity: int, essential: list[str], instructions: str, criteria: dict[str, str]) -> dict[str, Any]:
    if not 1 <= len(criteria) <= MAX_OPTIONS:
        raise ValueError("Laya accepts 1..16 options")
    return {"id": int(identity), "parts": {"essential": list(essential)},
            "questions": {"action": {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}}}


def checked_choice(response: dict[str, Any], request: dict[str, Any]) -> tuple[str, float, dict[str, float]]:
    """Return (choice, probability, probabilities) or raise GuardRejection."""
    try:
        choice, probability = checked_action(response, request, request["questions"]["action"]["criteria"])
        probabilities = response["answers"]["action"]["probabilities"]
    except Exception as error:  # the receiver raises ValueError; malformed structure of any other kind also rejects
        raise GuardRejection(str(error)) from error
    return choice, probability, {key: float(value) for key, value in probabilities.items()}


class LayaClient:
    def __init__(self, url: str = URL, timeout: float = 240.0):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self._client = DecisionClient(self.url, timeout)

    def health(self) -> dict[str, Any]:
        return self._client.health()

    def decide(self, request: dict[str, Any], raw_path: Path | None = None) -> tuple[dict[str, Any], float]:
        """POST the request; the raw bytes are saved before any parsing. Returns (response, seconds)."""
        started = time.monotonic()
        try:
            result = self._client.decide(request, raw_path=raw_path)
        except (OSError, http.client.HTTPException) as error:  # includes HTTP error statuses and timeouts
            raise GuardRejection(f"Decision request failed: {error}") from error
        except ValueError as error:  # undecodable, duplicate keys, nonfinite numbers, not an object
            raise GuardRejection(f"Unparseable decision response: {error}") from error
        return result, time.monotonic() - started
