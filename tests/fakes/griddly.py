"""CPU protocol stand-in reading only the same public state as a decision model."""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from typing import Any, ClassVar

from honing.backends.griddly.model import baseline
from honing.backends.jev.server import DecisionServer
from honing.core import PolicyEndpoint, PolicyRef
from tests.fakes.model import FakeModelSource, FakeWorld


class PublicStateModel:
    """A rule-based stand-in, never evidence of learned game capability."""

    metadata: ClassVar[dict[str, str]] = {
        "family": "stand-in",
        "model": "spiders-public-state-rule",
        "revision": "test",
    }

    def predict(self, parts: Mapping[str, object], questions: Mapping[str, object]) -> Mapping[str, object]:
        state = "\n".join(parts["essential"])  # type: ignore[arg-type]
        line = next(line for line in state.splitlines() if line.startswith("Public observation: "))
        observation = json.loads(line.removeprefix("Public observation: "))
        question: Any = questions["action"]
        options = list(question["criteria"])
        preferred = baseline({"observation": observation, "legal_actions": options})
        probabilities = {name: 0.8 if name == preferred else 0.2 / (len(options) - 1) for name in options}
        return {
            "answers": {"action": {"type": "choice", "choice": preferred, "probabilities": probabilities}},
            "stateText": state,
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


def model_source(context: Any) -> FakeModelSource:
    world = context.shared.setdefault("griddly_world", FakeWorld())
    return FakeModelSource(world)


class StandInEndpoints:
    def __init__(self) -> None:
        self.server = DecisionServer(("127.0.0.1", 0), PublicStateModel())
        self.thread = threading.Thread(target=self.server.serve, daemon=True)
        self.thread.start()

    def resolve(self, policy: PolicyRef) -> PolicyEndpoint:
        return PolicyEndpoint(self.server.url, "test-tokenizer", "stand-in", runtime="jev_decision")

    def close(self) -> None:
        self.server.stop()
        self.thread.join(timeout=5)


def endpoints(context: Any) -> StandInEndpoints:
    return StandInEndpoints()
