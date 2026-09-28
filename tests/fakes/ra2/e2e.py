"""Fake components for the RA2 end-to-end dry run (``configs/e2e/ra2_dry_run.json``).

The runner and grader in the dry run are the real ones; only their match and
check processes are stand-ins (``fake_match.py``, ``fake_check.py``). A fake
match's result is decided by the decision server URL, so the fake endpoints
map a model's quality to a URL: a model better than the starting one wins,
anything else draws. Training raises quality, so a train move wins its gate;
a harness change leaves the policy as it is, so it ties and is rolled back.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

from honing.backends.ra2.demonstrations import RA2DemonstrationProvider
from honing.backends.ra2.e2e import E2EContext
from honing.backends.ra2.harness import QUESTIONS_PATH
from honing.core import (
    ArchitectureSpec,
    BackendSupport,
    EvaluationResult,
    EvolveContext,
    HarnessFile,
    HarnessProposal,
    MethodRequirements,
    PolicyEndpoint,
    PolicyRef,
    ProposeResult,
    SelectionDecision,
)
from honing.release import WinLossSelector
from tests.fakes.model import FakeModelSource, FakeTrainer, FakeWorld
from tests.fakes.ra2.demos import PROTOCOL, write_match

START_QUALITY = 0.5
DECISION_TRAINER = BackendSupport(
    name="fake-jev",
    architectures="uniform",
    train_numerics=frozenset({"bf16"}),
    weight_formats=frozenset({"hf_safetensors"}),
    algorithms=(("sft", "native"), ("grpo", "native")),
    contracts=frozenset({"typed_decisions"}),
)


def _world(context: E2EContext) -> FakeWorld:
    world = context.shared.setdefault("world", FakeWorld())
    if not isinstance(world, FakeWorld):
        raise TypeError("context.shared['world'] is not a FakeWorld")
    return world


def model_source(context: E2EContext) -> FakeModelSource:
    architecture = ArchitectureSpec.uniform_layers(num_layers=4, hidden_size=64, attention_heads=4, ffn_size=128)
    return FakeModelSource(_world(context), architecture=architecture, quality=START_QUALITY)


class QualityEndpoints:
    """Satisfies PolicyEndpoints: a model better than the starting one is served as a winning server."""

    def __init__(self, world: FakeWorld) -> None:
        self._world = world

    def resolve(self, policy: PolicyRef) -> PolicyEndpoint:
        better = self._world.quality[policy.checkpoint.checksum] > START_QUALITY
        url = "http://strong:1" if better else "http://even:1"
        return PolicyEndpoint(url, "fake-tokenizer", "laya", runtime="jev_decision")


def endpoints(context: E2EContext) -> QualityEndpoints:
    return QualityEndpoints(_world(context))


def trainer(context: E2EContext, gain: float = 0.1) -> FakeTrainer:
    return FakeTrainer(_world(context), support=DECISION_TRAINER, gain=gain)


def demonstrations(context: E2EContext, seeds: tuple[int, ...] = (100, 101, 102)) -> RA2DemonstrationProvider:
    """Labelled matches in #49's format, written into the run's work directory."""
    matches = [write_match(context.work_dir / "demonstrations", seed) for seed in seeds]
    return RA2DemonstrationProvider(matches, PROTOCOL, batch_id="dry-run-demonstrations")


class RewordRulesMethod:
    """Satisfies EvolveMethod: rewords the operating rules once, then has nothing more to propose."""

    name = "reword_rules"
    state_schema_version = 1
    requires = MethodRequirements(file_kinds=frozenset({"instructions"}))
    selector = WinLossSelector()

    def __init__(self, addition: str) -> None:
        self._addition = addition

    def initial_state(self, seed: object) -> Mapping[str, object]:
        return {"schema_version": self.state_schema_version, "proposals": 0}

    def propose(self, context: EvolveContext) -> ProposeResult:
        current = context.current.get(QUESTIONS_PATH)
        questions = json.loads(current.content) if current is not None else {}
        if self._addition in questions.get("intent.rules", ""):
            return ProposeResult(proposal=None, state=context.state)
        questions["intent.rules"] = questions.get("intent.rules", "") + " " + self._addition
        spec = context.current.with_file(QUESTIONS_PATH, HarnessFile("instructions", json.dumps(questions, indent=2)))
        return ProposeResult(
            proposal=HarnessProposal(spec=spec, note="reword the operating rules"), state=context.state
        )

    def record(
        self,
        state: Mapping[str, object],
        proposal: HarnessProposal,
        evaluation: EvaluationResult,
        decision: SelectionDecision,
    ) -> Mapping[str, object]:
        return {**state, "proposals": int(str(state["proposals"])) + 1}


def method(context: E2EContext, addition: str = "Scout before attacking.") -> RewordRulesMethod:
    return RewordRulesMethod(addition)
