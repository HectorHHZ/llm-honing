"""Serve a deterministic stand-in decision model through Honing's real DecisionServer, for the serving check.

Run: python scripts/e2e/fake_decision_server.py [--port 0] [--family laya]

The model packs the four state sections as the real families do and gives each option a probability proportional to
its description's length plus one, so a client can compute the expected answer on its own. It prints the same ready
line as ``python -m honing.backends.jev.server``.
"""

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from honing.backends.jev.server import DecisionServer
from honing.backends.jev.token_budget import SECTIONS


class LongestOptionModel:
    """Satisfies DecisionModel: option probabilities in proportion to description length plus one."""

    def __init__(self, family: str) -> None:
        self.metadata = {"family": family, "model": "longest-option", "revision": "e2e"}

    def predict(self, parts: Mapping[str, object], questions: Mapping[str, object]) -> Mapping[str, object]:
        state = "\n".join(str(line) for key in SECTIONS for line in parts.get(key, []))  # type: ignore[attr-defined]
        answers, audit = {}, {}
        for key, question in questions.items():
            criteria = question["criteria"]  # type: ignore[index]
            weights = {option: len(text) + 1 for option, text in criteria.items()}
            total = sum(weights.values())
            probabilities = {option: weight / total for option, weight in weights.items()}
            choice = max(probabilities, key=lambda option: probabilities[option])
            answers[key] = {"type": "choice", "choice": choice, "probabilities": probabilities}
            state_tokens = len(state.split())
            audit[key] = {
                "stateTokens": state_tokens,
                "stateRoom": 1_000_000,
                "inputTokens": state_tokens + sum(len(t.split()) for t in criteria.values()),
                "options": len(criteria),
                "stateTruncated": False,
            }
        return {
            "answers": answers,
            "modelResult": {"rule": "longest option", "answers": answers},
            "stateText": state,
            "tokenAudit": audit,
            "trimmed": [],
            "latencyMs": 0.0,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--family", default="laya")
    args = parser.parse_args()
    model = LongestOptionModel(args.family)
    server = DecisionServer(("127.0.0.1", args.port), model)
    print(json.dumps({"ready": True, "url": server.url, "metadata": model.metadata}), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
