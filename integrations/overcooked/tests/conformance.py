"""Labelled protocol stand-in for the existing runner check, never model evidence."""

from __future__ import annotations

import threading
from typing import Any, ClassVar

from honing.backends.jev.server import DecisionServer
from honing.core import PolicyEndpoint
from tests.fakes.model import FakeModelSource, FakeWorld


class StayModel:
    metadata: ClassVar[dict[str, Any]] = {"family": "protocol-stand-in", "real_model": False}

    def predict(self, parts: Any, questions: Any) -> dict[str, Any]:
        options = questions["action"]["criteria"]
        return {
            "answers": {
                "action": {
                    "type": "choice",
                    "choice": "STAY",
                    "probabilities": {name: float(name == "STAY") for name in options},
                }
            },
            "stateText": "\n".join(parts["essential"]),
            "trimmed": [],
            "tokenAudit": {
                "action": {
                    "stateTruncated": False,
                    "options": len(options),
                    "stateTokens": None,
                    "stateRoom": None,
                    "inputTokens": None,
                }
            },
        }


class Endpoints:
    def __init__(self) -> None:
        self.server = DecisionServer(("127.0.0.1", 0), StayModel())
        self.thread = threading.Thread(target=self.server.serve, daemon=True)
        self.thread.start()

    def resolve(self, policy: Any) -> PolicyEndpoint:
        return PolicyEndpoint(self.server.url, "stand-in", "stand-in", "jev_decision")

    def close(self) -> None:
        self.server.stop()
        self.thread.join(timeout=5)


def endpoints(context: Any) -> Endpoints:
    return Endpoints()


def model_source(context: Any) -> FakeModelSource:
    return FakeModelSource(context.shared.setdefault("overcooked_test_world", FakeWorld()))
