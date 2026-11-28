"""Fake components for the tank end-to-end dry runs (``configs/e2e/tank_dry_run.json`` and
``configs/e2e/tank_levers_dry_run.json``), and System 2's scripted diagnoses for the corrected-decision one.

The runner, grader, rollouts, demonstrations, match process, decision server,
GEPA and Meta-Harness are the real ones; only the decision model, the
trainers, the chat model GEPA reflects with, and the agent Meta-Harness
proposes through are stand-ins.

:class:`ForecastModel` reads the forecasts the match sends and answers as a
model of a given quality would: it prefers motions that avoid every shell and
keep a safe shot, misjudging them by more the lower its quality (deterministic
coins per state), and takes a permitted shot only from :data:`FIRING_QUALITY`
up. The starting model never fires, so it can at best run out the clock.
Training raises quality, so a train move starts firing. Weapon instructions
that tell it to fire whenever the shot is permitted make any quality fire, so
a harness edit changes the score only through what the model is told, as the
all-planes dry run's verify bonus does.

GEPA's scripted chat model can also check what it is shown: with
``expect_decisions`` it refuses a reflection prompt in which an example does
not hold the round's rendered decisions with every option's probability.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from honing.backends.jev.server import DecisionRecorder, DecisionServer
from honing.backends.jev.token_budget import SECTIONS
from honing.backends.tank.corrected import CorrectedDecisionProvider
from honing.backends.tank.e2e import E2EContext
from honing.backends.tank.harness import CONTROLLER_PATH, seed_harness
from honing.core import (
    ArchitectureSpec,
    BackendSupport,
    ChatMessage,
    ChatReply,
    ContractError,
    ModelSnapshot,
    PolicyEndpoint,
    PolicyRef,
    TrainingBatch,
    TrainJobSpec,
    TrainResult,
)
from honing.harness.diagnosis import Diagnosis
from tests.fakes.harness import FakeChatModel, FakeProposerAgent
from tests.fakes.model import FakeModelSource, FakeTrainer, FakeWorld

INTEGRATION = Path(__file__).resolve().parents[2] / "integrations/tank-war"
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
DIRECTION_EDITOR = BackendSupport(
    name="fake-direction-edit",
    architectures="uniform",
    train_numerics=frozenset({"bf16"}),
    weight_formats=frozenset({"hf_safetensors"}),
    algorithms=(("direction_edit", "native"),),
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
    """Satisfies PolicyEndpoints: one in-process decision server per model quality, started on first use.

    With ``recorder``, which makes a recorder for each server, every server records its answered decisions.
    """

    def __init__(self, world: FakeWorld, recorder: Callable[[], DecisionRecorder] | None = None) -> None:
        self._world = world
        self._recorder = recorder
        self._servers: dict[float, tuple[DecisionServer, threading.Thread]] = {}
        self._lock = threading.Lock()

    def resolve(self, policy: PolicyRef) -> PolicyEndpoint:
        quality = self._world.quality[policy.checkpoint.checksum]
        with self._lock:
            if quality not in self._servers:
                recorder = self._recorder() if self._recorder is not None else None
                server = DecisionServer(("127.0.0.1", 0), ForecastModel(quality), recorder)
                thread = threading.Thread(target=server.serve, daemon=True)
                thread.start()
                self._servers[quality] = (server, thread)
            url = self._servers[quality][0].url
        return PolicyEndpoint(url, "fake-tokenizer", f"forecast-q{quality:.3f}", runtime="jev_decision", gpus=1)

    def close(self) -> None:
        """Stop every server. A server that records sends what is still waiting first, as at its exit."""
        with self._lock:
            servers, self._servers = list(self._servers.values()), {}
        for server, thread in servers:
            server.stop()
            thread.join(timeout=60)
            server.server_close()


def endpoints(context: Any, reef_url: str | None = None, reef_scenario: str | None = None) -> QualityServers:
    """The stand-in decision servers. With ``reef_url`` and ``reef_scenario``, each one records in that Reef scenario,
    as the decision server's ``--reef-url`` and ``--reef-scenario`` make it do. The token is ``REEF_TOKEN``."""
    if (reef_url is None) != (reef_scenario is None):
        raise ContractError("the endpoints' reef_url and reef_scenario go together")
    if reef_url is None:
        return QualityServers(_world(context))
    # Imported only here: reef_client is the optional `reef` extra.
    from honing.backends.reef.recorder import ReefRecorder

    token = os.environ.get("REEF_TOKEN")
    return QualityServers(_world(context), lambda: ReefRecorder(reef_url, reef_scenario, token=token))


def trainer(context: E2EContext, gain: float = 0.2) -> FakeTrainer:
    return FakeTrainer(_world(context), support=DECISION_TRAINER, gain=gain)


# A rendered motion and weapon decision (honing/harness/traces.py): each option's probability, then the choice.
RENDERED_MOTION = re.compile(r"^  motion: (?:\w+ [01]\.\d{3}, ){8}\w+ [01]\.\d{3}; chose \w+$", re.MULTILINE)
RENDERED_WEAPON = re.compile(r"^  weapon: fire [01]\.\d{3}, hold [01]\.\d{3}; chose (?:fire|hold)$", re.MULTILINE)


class DecisionReadingChatModel(FakeChatModel):
    """A scripted chat model that refuses a reflection prompt unless every example shows rendered decisions."""

    def complete(self, messages: Sequence[ChatMessage], *, max_tokens: int, temperature: float) -> ChatReply:
        examples = "\n".join(message.content for message in messages).split("# Example ")[1:]
        if not examples or not all(RENDERED_MOTION.search(e) and RENDERED_WEAPON.search(e) for e in examples):
            raise ContractError("the reflection prompt does not show each round's decisions with their probabilities")
        return super().complete(messages, max_tokens=max_tokens, temperature=temperature)


def chat(
    context: object,
    replies: tuple[str, ...] = (EAGER_QUESTIONS,),
    expect_decisions: bool = False,
    cost_per_call: float | None = None,
) -> FakeChatModel:
    """GEPA's reflection model: returns ``replies`` in order, each priced at ``cost_per_call``.

    With ``expect_decisions`` it checks each prompt shows the rounds' decisions.
    """
    model = DecisionReadingChatModel if expect_decisions else FakeChatModel
    return model(list(replies), cost_per_call=cost_per_call)


class FakeDirectionEditTrainer(FakeTrainer):
    """A direction edit that moves quality by ``gain`` for each unit of the job's strength.

    It refuses a job that does not say which edit to make, so a run that
    publishes or rejects its candidate shows the move's own question, target
    and strength reached the trainer.
    """

    def train(self, job: TrainJobSpec, base: ModelSnapshot, batch: TrainingBatch | None) -> TrainResult:
        spec = job.direction_edit
        if spec is None:
            raise ContractError("the stand-in direction edit needs the move's question, target and strength")
        self.jobs.append(job)
        self.batches.append(batch)
        quality = self.world.quality[base.checkpoint.checksum] + self.gain * spec.strength
        snapshot = self.world.snapshot(base.architecture, quality)
        return TrainResult(snapshot=snapshot, metrics={"strength": spec.strength})


def scripted_corrections(
    context: E2EContext, runs: Mapping[str, str], causes: Mapping[str, str]
) -> CorrectedDecisionProvider:
    """Corrected decisions from a scripted System 2: ``causes`` gives each episode's diagnosis.

    Every episode is diagnosed under release 0 and the seed harness, the head
    a campaign's first move trains. ``runs`` maps episodes to their run
    directories.
    """
    harness = seed_harness(context.integration_dir).spec_id
    diagnoses = [
        Diagnosis(episode, cause, 1.0, f"scripted: {cause}", release_id="r0000", harness_id=harness)  # type: ignore[arg-type]
        for episode, cause in causes.items()
    ]
    return CorrectedDecisionProvider(diagnoses, {episode: Path(path) for episode, path in runs.items()})


def direction_edit_trainer(context: Any, gain: float = 0.1) -> FakeDirectionEditTrainer:
    return FakeDirectionEditTrainer(_world(context), support=DIRECTION_EDITOR, gain=gain)


def proposer(
    context: object, settings: Sequence[Mapping[str, object]] = ({"assistProfile": "standard"},)
) -> FakeProposerAgent:
    """A Meta-Harness proposer that follows the skill's workflow with scripted controller settings.

    Each call copies the served candidate, merges the next entry of
    ``settings`` into its ``tank/controller.json``, and writes
    ``pending_eval.json``; with no entry left it exits 1.
    """
    queue = [dict(change) for change in settings]

    def script(workdir: Path) -> int:
        if not queue:
            return 1
        change = queue.pop(0)
        rows = [json.loads(row) for row in (workdir / "evolution_summary.jsonl").read_text().splitlines()]
        parent = next(row["name"] for row in rows if row["served"])
        name = f"controller_{len(rows)}"
        folder = workdir / "candidates" / name
        shutil.copytree(workdir / "candidates" / parent, folder)
        controller = folder / CONTROLLER_PATH
        controller.write_text(json.dumps({**json.loads(controller.read_text()), **change}, indent=2) + "\n")
        entry = {
            "name": name,
            "parent": parent,
            "hypothesis": f"controller settings {change} win more rounds",
            "changes": f"set {change} in {CONTROLLER_PATH}",
        }
        (workdir / "pending_eval.json").write_text(json.dumps({"iteration": len(rows), "candidates": [entry]}))
        return 0

    return FakeProposerAgent(script)
