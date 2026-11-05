"""The real match CLI with a stand-in engine/client; it needs no game or model."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from honing.backends.craftax import match
from tests.fakes.craftax import FakeCraftaxEngine, fake_versions


class Client:
    def __init__(self, url: str) -> None:
        self.url = url

    def health(self) -> dict[str, Any]:
        return {"ready": True, "metadata": {"model": "test-stand-in"}}

    def decide(self, request: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        productive = "strong" in self.url or ("partial" in self.url and request["id"].endswith("step0"))
        choice = "DO" if productive else "NOOP"
        probabilities = {a: float(a == choice) for a in request["questions"]["action"]["criteria"]}
        return choice, {
            "id": request["id"],
            "stateText": request["parts"]["essential"][0],
            "trimmed": [],
            "tokenAudit": {},
            "answers": {"action": {"choice": choice, "probabilities": probabilities}},
        }


match.CraftaxEngine = FakeCraftaxEngine  # type: ignore[assignment]
match.DecisionClient = Client  # type: ignore[assignment]
match.runtime_versions = fake_versions
if os.environ.get("FAKE_CRAFTAX_CRASH"):
    raise RuntimeError("stand-in process crash")
match.main()
