"""An in-process fake of the pinned Laya decision service, for tests only (no model, no network beyond 127.0.0.1).

It speaks the same API as the Exp 0 services (8811-8814): ``GET /v1/health`` and ``POST /v1/decide``, with
``branches``, and builds its reply like ``honing.backends.jev.laya`` + ``token_budget.prepare_request``:
``stateText`` is the newline join of ``essential, details, memory, logs``, ``trimmed`` is empty, the token
audit is present, probabilities are rounded to four decimals, ``choice`` is the argmax.

Probabilities are a deterministic function of the request text and the checkpoint role, so the same
request always gets the same answer (probe identity holds exactly) and the alternate checkpoint answers
differently (weights evidence exists). Tests start it on a free port and stop it themselves.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

SECTIONS = ("essential", "details", "memory", "logs")
CLEAN_SHA = "c" * 64
ALTERNATE_SHA = "a" * 64

PREFERENCES = {
    # option key -> (phrase in the state that raises it, bonus)
    "fight_nearest_enemy": ("enemy is 1 step", 3.0),
    "attack": ("enemy is 1 step", 3.0),
    "jump_right": ("pit is 1 step", 3.5),
    "right": ("way is open", 1.2),
    "advance_right": ("way is open", 1.2),
    "strike": ("enemy is 1 step", 3.0),  # the two-question game (contract 0.2.0): attack question
}


def _noise(text: str, option: str) -> float:
    digest = hashlib.sha256(f"{text}\x00{option}".encode()).digest()
    return (int.from_bytes(digest[:4], "big") / 2**32 - 0.5) * 0.6


def probabilities(state: str, options: list[str], role: str) -> dict[str, float]:
    scores = {}
    for position, option in enumerate(options):  # order-sensitive, as Laya is
        score = _noise(state, f"{position}:{option}")
        phrase, bonus = PREFERENCES.get(option, ("", 0.0))
        if phrase and phrase in state:
            score += bonus
        if role == "alternate":
            score = 0.55 * score + (0.9 if option.startswith("wait") else 0.0)
        scores[option] = score
    top = max(scores.values())
    weights = {option: math.exp(score - top) for option, score in scores.items()}
    total = sum(weights.values())
    rounded = {option: round(value / total, 4) for option, value in weights.items()}
    return rounded  # noqa: RET504


def answer(request: dict[str, Any], role: str) -> dict[str, Any]:
    parts = request.get("parts", {})
    state = "\n".join(text for key in SECTIONS for text in parts.get(key, []))
    answers, audit = {}, {}
    for key, question in request["questions"].items():
        options = list(question["criteria"])
        probs = probabilities(state + "\n" + json.dumps(question), options, role)
        choice = max(options, key=lambda option: (probs[option], -options.index(option)))
        answers[key] = {
            "type": "choice",
            "choice": choice,
            "probabilities": probs,
            "confidence": 0.0,
            "answer_confidence": probs[choice],
        }
        tokens = len(state.split())
        audit[key] = {
            "stateTokens": tokens,
            "stateRoom": 1900,
            "inputTokens": tokens + 40,
            "options": len(options),
            "stateTruncated": False,
        }
    reply = {
        "answers": answers,
        "modelResult": {"model": "fake-laya", "answers": answers},
        "stateText": state,
        "tokenAudit": audit,
        "trimmed": [],
        "latencyMs": 1.0,
    }
    if "id" in request:
        reply["id"] = request["id"]
    return reply


class FakeLaya:
    """``with FakeLaya("clean") as server: server.url`` -- a thread serving the fake API on a free port."""

    def __init__(self, role: str, *, sha: str | None = None, corrupt_every: int = 0) -> None:
        self.role = role
        self.sha = sha or (CLEAN_SHA if role == "clean" else ALTERNATE_SHA)
        self.calls = 0
        self.corrupt_every = corrupt_every
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # quiet
                pass

            def _send(self, value: Any, raw: bytes | None = None) -> None:
                body = raw if raw is not None else json.dumps(value).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                self._send(
                    {
                        "ready": True,
                        "maxBranches": 4,
                        "metadata": {
                            "repo": "fake/laya",
                            "revision": "0" * 40,
                            "role": outer.role,
                            "weightsSha256": outer.sha,
                            "path": "/nonexistent/local/path",
                            "backend": "fake",
                        },
                    }
                )

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                outer.calls += 1
                if outer.corrupt_every and outer.calls % outer.corrupt_every == 0:
                    self._send(None, raw=b'{"answers": {"action": {"probabilities": {}}}}')
                    return
                branches = body.pop("branches", None)
                reply = answer(body, outer.role)
                if branches:
                    reply["branches"] = {key: answer(value, outer.role) for key, value in branches.items()}
                self._send(reply)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> FakeLaya:
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._server.shutdown()
        self._server.server_close()
