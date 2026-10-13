"""Fake components for the tank end-to-end dry run (``configs/e2e/tank_dry_run.json``).

The runner, grader, rollouts, demonstrations, match process, decision server,
and GEPA are the real ones; only the decision model, the trainer, and the chat
model GEPA reflects with are stand-ins.

:class:`ForecastModel` reads the forecasts the match sends and answers as a
model of a given quality would: it prefers motions that avoid every shell and
keep a safe shot, misjudging them by more the lower its quality (deterministic
coins per state), and takes a permitted shot only from :data:`FIRING_QUALITY`
up. The starting model never fires, so it can at best run out the clock.
Training raises quality, so a train move starts firing. Weapon instructions
that tell it to fire whenever the shot is permitted make any quality fire, so
a harness edit changes the score only through what the model is told, as the
all-planes dry run's verify bonus does.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from honing.backends.jev.server import DecisionServer
from honing.backends.jev.token_budget import SECTIONS
from honing.backends.tank.e2e import E2EContext
from honing.core import ArchitectureSpec, BackendSupport, PolicyEndpoint, PolicyRef
from tests.fakes.harness import FakeChatModel
from tests.fakes.model import FakeModelSource, FakeTrainer, FakeWorld

INTEGRATION = Path(__file__).resolve().parents[2] / "integrations/tank-duel"
START_QUALITY = 0.5
# Below this quality the stand-in hesitates: it takes no permitted shot unless told to.
FIRING_QUALITY = 0.7
EAGER_WORDS = "whenever"
DECISION_TRAINER = BackendSupport(
    name="fake-jev",
    architectures="uniform",
    train_numerics=frozenset({"bf16"}),
    weight_formats=frozenset({"hf_safetensors"}),
    algorithms=(("sft", "native"), ("grpo", "native")),
    contracts=frozenset({"typed_decisions"}),
)
# The reflection GEPA's scripted chat model returns: the default question text with weapon instructions
# that tell the model to take every permitted shot.
EAGER_QUESTIONS = (
    "```\n"
    + json.dumps(
        {
            **json.loads((INTEGRATION / "config/questions.json").read_text(encoding="utf-8")),
            "weapon": "Decide whether to fire. Fire whenever the selected motion's shot is permitted: a hesitant"
            " tank loses the round. Otherwise hold.",
        },
        indent=2,
    )
    + "\n```"
)


def _coin(state: str) -> float:
    return int(hashlib.sha256(state.encode()).hexdigest()[:8], 16) / 2**32


def _motion_utility(description: str) -> float:
    """Higher for a motion that avoids every shell and keeps a safe shot."""
    utility = 0.0
    if "existing shell collision none" not in description:
        utility -= 10.0
    if "enemy hit none" not in description and "self hit none" in description:
        utility += 3.0
    if "self hit none" not in description:
        utility -= 3.0
    return utility


class ForecastModel:
    """Satisfies DecisionModel: answers tank questions from the forecasts in the state."""

    def __init__(self, quality: float) -> None:
        self.quality = quality
        self.metadata = {"family": "laya", "model": f"forecast-q{quality:.3f}", "revision": "e2e"}

    def predict(self, parts: Mapping[str, object], questions: Mapping[str, object]) -> Mapping[str, object]:
        state = "\n".join(str(line) for key in SECTIONS for line in parts.get(key, []))  # type: ignore[attr-defined]
        forecasts = {
            line.split(": ", 1)[0]: line.split(": ", 1)[1]
            for line in state.split("OPTION FORECASTS:\n", 1)[-1].splitlines()
            if ": " in line
        }
        answers, audit = {}, {}
        for key, question in questions.items():
            q: Any = question
            options = list(q["criteria"])
            if key == "weapon":
                permitted = "Fire permitted=true" in state
                eager = EAGER_WORDS in q["instructions"].lower()
                fire = permitted and (eager or self.quality >= FIRING_QUALITY)
                probabilities = {"fire": 0.9 if fire else 0.1, "hold": 0.1 if fire else 0.9}
            else:
                # A weaker model misjudges motions: each option's utility is off by up to 8 * (1 - quality).
                weights = {
                    o: math.exp(
                        _motion_utility(forecasts.get(f"{key}/{o}", "")) + 8 * (1 - self.quality) * _coin(state + o)
                    )
                    for o in options
                }
                total = sum(weights.values())
                probabilities = {o: w / total for o, w in weights.items()}
            choice = max(options, key=lambda o: probabilities[o])
            answers[key] = {"type": "choice", "choice": choice, "probabilities": probabilities}
            tokens = len(state.split())
            audit[key] = {
                "stateTokens": tokens,
                "stateRoom": 1_000_000,
                "inputTokens": tokens,
                "options": len(options),
                "stateTruncated": False,
            }
        return {
            "answers": answers,
            "modelResult": {"answers": answers},
            "stateText": state,
            "tokenAudit": audit,
            "trimmed": [],
            "latencyMs": 0.0,
        }


def _world(context: Any) -> FakeWorld:
    world = context.shared.setdefault("world", FakeWorld())
    if not isinstance(world, FakeWorld):
        raise TypeError("context.shared['world'] is not a FakeWorld")
    return world


def model_source(context: Any) -> FakeModelSource:
    architecture = ArchitectureSpec.uniform_layers(num_layers=4, hidden_size=64, attention_heads=4, ffn_size=128)
    return FakeModelSource(_world(context), architecture=architecture, quality=START_QUALITY)


class QualityServers:
    """Satisfies PolicyEndpoints: one in-process decision server per model quality, started on first use."""

    def __init__(self, world: FakeWorld) -> None:
        self._world = world
        self._servers: dict[float, DecisionServer] = {}
        self._lock = threading.Lock()

    def resolve(self, policy: PolicyRef) -> PolicyEndpoint:
        quality = self._world.quality[policy.checkpoint.checksum]
        with self._lock:
            if quality not in self._servers:
                server = DecisionServer(("127.0.0.1", 0), ForecastModel(quality))
                threading.Thread(target=server.serve, daemon=True).start()
                self._servers[quality] = server
            url = self._servers[quality].url
        return PolicyEndpoint(url, "fake-tokenizer", f"forecast-q{quality:.3f}", runtime="jev_decision")


def endpoints(context: Any) -> QualityServers:
    return QualityServers(_world(context))


def trainer(context: E2EContext, gain: float = 0.2) -> FakeTrainer:
    return FakeTrainer(_world(context), support=DECISION_TRAINER, gain=gain)


def chat(context: object, replies: tuple[str, ...] = (EAGER_QUESTIONS,)) -> FakeChatModel:
    """GEPA's reflection model: returns ``replies`` in order."""
    return FakeChatModel(list(replies))
