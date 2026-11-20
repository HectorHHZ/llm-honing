"""Several trainers and several harness methods in one campaign.

A move names its trainer or its method; a move that names none runs the
campaign's default. Each method keeps its own state inside the one
``method_state`` a release carries, so a rollback restores them together.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest

from honing.campaign import HarnessComponents, ScriptedScheduler
from honing.core import (
    METHOD_STATES_KEY,
    BackendSupport,
    CampaignRuleError,
    ContractError,
    DataContract,
    DecisionQuestion,
    DecisionSample,
    DirectionEditSpec,
    EvaluationResult,
    EvolveContext,
    HarnessFile,
    HarnessMoveConfig,
    HarnessProposal,
    HarnessSpec,
    MethodRequirements,
    Move,
    ProposeResult,
    SelectionDecision,
    TrainingBatch,
    TrainJobSpec,
    UpdateCandidate,
    keyed_method_state,
    method_states,
)
from tests.fakes.builder import TASKS, build_campaign
from tests.fakes.harness import INSTRUCTIONS_PATH, SEED_HARNESS, FakeEpisodeRunner, FakeTaskGrader
from tests.fakes.model import SLIME_LIKE, FakeModelSource, FakeTrainer, FakeWorld, small_moe

DECISIONS = BackendSupport(
    name="fake-decisions",
    architectures="heterogeneous",
    train_numerics=frozenset({"bf16"}),
    weight_formats=frozenset({"hf_safetensors"}),
    algorithms=(("sft", "native"), ("direction_edit", "native")),
    contracts=frozenset({"typed_decisions"}),
)


class _ScriptedSelector:
    """Selects or rejects as told, and keeps the method state each candidate showed it."""

    def __init__(self, name: str, select: bool) -> None:
        self.name = name
        self.select = select
        self.shown: list[Mapping[str, object]] = []

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        self.shown.append(candidate.method_state)
        return SelectionDecision(outcome="select" if self.select else "reject", policy=self.name, reason="scripted")


class NotingMethod:
    """Appends a line naming itself to the instructions. Its state counts its proposals and keeps every verdict."""

    state_schema_version = 1

    def __init__(self, name: str, *, select: bool = True, proposes: bool = True) -> None:
        self.name = name
        self.requires = MethodRequirements(file_kinds=frozenset({"instructions"}))
        self.selector = _ScriptedSelector(name, select)
        self.proposes = proposes
        self.contexts: list[EvolveContext] = []

    def initial_state(self, seed: HarnessSpec) -> Mapping[str, object]:
        return {"schema_version": 1, "method": self.name, "proposals": 0, "verdicts": []}

    def propose(self, context: EvolveContext) -> ProposeResult:
        self.contexts.append(context)
        count = int(str(context.state["proposals"])) + 1
        state = {**context.state, "proposals": count}
        if not self.proposes:
            return ProposeResult(proposal=None, state=state, episodes_spent=2)
        file = context.current.get(INSTRUCTIONS_PATH)
        assert file is not None
        text = file.content + f"{self.name} {count}\n"
        spec = context.current.with_file(INSTRUCTIONS_PATH, HarnessFile(kind="instructions", content=text))
        return ProposeResult(proposal=HarnessProposal(spec=spec, note=self.name), state=state)

    def record(
        self,
        state: Mapping[str, object],
        proposal: HarnessProposal,
        evaluation: EvaluationResult,
        decision: SelectionDecision,
    ) -> Mapping[str, object]:
        verdicts = state["verdicts"]
        assert isinstance(verdicts, list)
        return {**state, "verdicts": [*verdicts, decision.selected]}


class _CountingData:
    """Serves typed decisions and counts how often a batch was asked for."""

    def __init__(self) -> None:
        self.calls = 0

    def next_batch(self, contract: DataContract) -> TrainingBatch | None:
        self.calls += 1
        question = DecisionQuestion(kind="choice", instructions="Fire?", options=(("fire", "Fire"), ("hold", "Hold")))
        decision = DecisionSample(
            parts=(("essential", ("state",)),),
            questions=(("weapon", question),),
            chosen=(("weapon", "hold"),),
            behavior_probs=None,
            reward=None,
            group=None,
            source_release=None,
            source_harness=None,
        )
        return TrainingBatch(f"decisions-{self.calls}", DataContract("typed_decisions"), decisions=(decision,))


def _harness(method: str | None = None) -> Move:
    return Move(kind="harness", harness=HarnessMoveConfig(method=method))


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _own(state: Mapping[str, object], name: str) -> Mapping[str, object]:
    return method_states(state)[name]


# ------------------------------------------------------------------ trainers


@pytest.mark.unit
def test_a_move_runs_through_the_trainer_it_names_and_the_default_trainer_runs_the_rest(tmp_path: Path) -> None:
    named: dict[str, FakeTrainer] = {}

    def trainers(world: FakeWorld) -> dict[str, FakeTrainer]:
        named["second"] = FakeTrainer(world, gain=0.2)
        return named

    fixture = build_campaign(tmp_path, make_trainers=trainers)
    job = TrainJobSpec(algorithm="sft", steps=3)

    outcomes = fixture.campaign.run(
        ScriptedScheduler([Move(kind="train", train=job, trainer="second"), Move(kind="train")])
    )

    assert [outcome.release is not None for outcome in outcomes] == [True, True]
    assert named["second"].jobs == [job]
    assert [trained.steps for trained in fixture.trainer.jobs] == [10]  # the campaign's default job


@pytest.mark.unit
def test_a_prune_move_recovers_through_the_trainer_it_names(tmp_path: Path) -> None:
    named: dict[str, FakeTrainer] = {}

    def trainers(world: FakeWorld) -> dict[str, FakeTrainer]:
        named["recovery"] = FakeTrainer(world)
        return named

    fixture = build_campaign(tmp_path, make_trainers=trainers)

    fixture.campaign.advance(Move(kind="prune", trainer="recovery"))

    assert [job.algorithm for job in named["recovery"].jobs] == ["distill"]
    assert fixture.trainer.jobs == []


@pytest.mark.unit
def test_a_trainer_the_campaign_does_not_have_is_refused_before_a_batch_is_drawn(tmp_path: Path) -> None:
    data = _CountingData()
    fixture = build_campaign(tmp_path, data=data, make_trainers=lambda world: {"second": FakeTrainer(world)})
    before = _records(fixture.campaign.chain.record_path)

    with pytest.raises(ContractError, match="the move is for trainer 'third', but the campaign has 'second'"):
        fixture.campaign.advance(Move(kind="train", trainer="third"))
    with pytest.raises(ContractError, match="trainer 'third'"):
        fixture.campaign.check_move(Move(kind="prune", trainer="third"))

    assert data.calls == 0 and fixture.trainer.jobs == []
    assert _records(fixture.campaign.chain.record_path) == before


@pytest.mark.unit
def test_a_campaign_without_named_trainers_says_so(tmp_path: Path) -> None:
    fixture = build_campaign(tmp_path)

    with pytest.raises(ContractError, match="trainer 'second', but the campaign has no named trainers"):
        fixture.campaign.advance(Move(kind="train", trainer="second"))


@pytest.mark.unit
def test_a_moves_direction_edit_reaches_the_trainer_it_names(tmp_path: Path) -> None:
    named: dict[str, FakeTrainer] = {}

    def trainers(world: FakeWorld) -> dict[str, FakeTrainer]:
        named["edit"] = FakeTrainer(world, support=DECISIONS)
        return named

    fixture = build_campaign(tmp_path, data=_CountingData(), make_trainers=trainers)
    weapon = DirectionEditSpec(question="weapon", target="fire", strength=1.5)
    motion = DirectionEditSpec(question="motion", strength=-0.5)

    for spec in (weapon, motion):
        job = TrainJobSpec("direction_edit", contract="typed_decisions", direction_edit=spec)
        fixture.campaign.advance(Move(kind="train", train=job, trainer="edit"))

    assert [job.direction_edit for job in named["edit"].jobs] == [weapon, motion]
    assert [batch.batch_id for batch in named["edit"].batches if batch is not None] == ["decisions-1", "decisions-2"]


@pytest.mark.unit
def test_a_trainer_name_belongs_to_a_train_or_a_prune_move_and_a_direction_to_its_algorithm() -> None:
    assert Move(kind="train", trainer="second").trainer == "second"
    assert Move(kind="prune", trainer="second").trainer == "second"
    with pytest.raises(ContractError, match="a trainer belongs to a train move"):
        Move(kind="harness", trainer="second")
    with pytest.raises(ContractError, match="must name a trainer or be None"):
        Move(kind="train", trainer="")
    with pytest.raises(ContractError, match="belongs to a direction_edit job"):
        TrainJobSpec("sft", contract="typed_decisions", direction_edit=DirectionEditSpec())
    with pytest.raises(ContractError, match="finite, nonzero"):
        DirectionEditSpec(strength=0.0)
    with pytest.raises(ContractError, match="'label' or an option id"):
        DirectionEditSpec(target="")
    with pytest.raises(ContractError, match="name a question or be None"):
        DirectionEditSpec(question="")
    assert DirectionEditSpec() == DirectionEditSpec(question=None, target="label", strength=1.0)


# ------------------------------------------------------------------ methods


@pytest.mark.unit
def test_each_method_proposes_from_its_own_state_and_a_release_carries_every_methods(tmp_path: Path) -> None:
    first, second = NotingMethod("first"), NotingMethod("second")
    fixture = build_campaign(tmp_path, method=first, methods=[second])
    chain = fixture.campaign.chain
    assert chain.head().method_state == keyed_method_state(
        {"first": first.initial_state(chain.head().harness), "second": second.initial_state(chain.head().harness)}  # type: ignore[arg-type]
    )

    outcomes = fixture.campaign.run(ScriptedScheduler([_harness("second"), _harness(), _harness("first")]))

    assert [outcome.release is not None for outcome in outcomes] == [True, True, True]
    assert [context.state["method"] for context in second.contexts] == ["second"]
    assert [context.state["proposals"] for context in first.contexts] == [0, 1]
    state = chain.head().method_state
    assert set(state) == {"schema_version", METHOD_STATES_KEY}
    assert _own(state, "first") == {"schema_version": 1, "method": "first", "proposals": 2, "verdicts": [True, True]}
    assert _own(state, "second") == {"schema_version": 1, "method": "second", "proposals": 1, "verdicts": [True]}
    text = chain.head().harness.get(INSTRUCTIONS_PATH).content  # type: ignore[union-attr]
    assert text.splitlines()[1:] == ["second 1", "first 1", "first 2"]


@pytest.mark.unit
def test_a_methods_selector_is_shown_its_own_state_not_every_methods(tmp_path: Path) -> None:
    first, second = NotingMethod("first"), NotingMethod("second")
    fixture = build_campaign(tmp_path, method=first, methods=[second])

    fixture.campaign.advance(_harness("second"))

    assert second.selector.shown == [{"schema_version": 1, "method": "second", "proposals": 1, "verdicts": []}]
    assert first.selector.shown == []


@pytest.mark.unit
def test_a_rejected_move_keeps_its_methods_state_for_the_next_release_and_leaves_the_others(tmp_path: Path) -> None:
    first, second = NotingMethod("first"), NotingMethod("second", select=False)
    fixture = build_campaign(tmp_path, method=first, methods=[second])
    chain = fixture.campaign.chain

    rejected, published = fixture.campaign.run(ScriptedScheduler([_harness("second"), _harness("first")]))

    assert rejected.release is None and published.release is not None
    reject = _records(chain.record_path)[1]
    assert reject["op"] == "reject"
    carried = reject["candidate"]["method_state"]  # type: ignore[index]
    assert _own(carried, "second")["verdicts"] == [False] and _own(carried, "first")["proposals"] == 0
    state = chain.head().method_state
    assert _own(state, "second") == {"schema_version": 1, "method": "second", "proposals": 1, "verdicts": [False]}
    assert _own(state, "first")["verdicts"] == [True]


@pytest.mark.unit
def test_a_skipped_move_records_every_methods_state_under_the_method_that_ran(tmp_path: Path) -> None:
    first, second = NotingMethod("first"), NotingMethod("second", proposes=False)
    fixture = build_campaign(tmp_path, method=first, methods=[second])

    outcome = fixture.campaign.advance(_harness("second"))
    again = fixture.campaign.advance(_harness("second"))

    assert outcome.skipped_reason == "second proposed nothing" and again.release is None
    skip = _records(fixture.campaign.chain.record_path)[1]
    assert (skip["op"], skip["method"], skip["episodes_spent"]) == ("skip", "second", 2)
    assert _own(skip["method_state"], "second")["proposals"] == 1  # type: ignore[arg-type]
    assert _own(skip["method_state"], "first")["proposals"] == 0  # type: ignore[arg-type]
    assert [context.state["proposals"] for context in second.contexts] == [0, 1]  # carried in memory


@pytest.mark.unit
def test_a_rollback_restores_every_methods_state_together(tmp_path: Path) -> None:
    first, second = NotingMethod("first"), NotingMethod("second")
    fixture = build_campaign(tmp_path, method=first, methods=[second])

    fixture.campaign.run(ScriptedScheduler([_harness("first"), _harness("second"), _harness("first")]))
    fixture.campaign.rollback("r0001")
    fixture.campaign.advance(_harness("second"))

    assert [context.state["proposals"] for context in second.contexts] == [0, 0]
    state = fixture.campaign.chain.head().method_state
    assert _own(state, "first")["proposals"] == 1 and _own(state, "second")["proposals"] == 1


@pytest.mark.unit
def test_a_method_the_campaign_does_not_run_is_refused_before_any_method_proposes(tmp_path: Path) -> None:
    first, second = NotingMethod("first"), NotingMethod("second")
    fixture = build_campaign(tmp_path, method=first, methods=[second])
    before = _records(fixture.campaign.chain.record_path)

    with pytest.raises(ContractError, match="method 'third', but the campaign runs 'first', 'second'"):
        fixture.campaign.advance(_harness("third"))
    with pytest.raises(ContractError, match="method 'third'"):
        fixture.campaign.check_move(_harness("third"))

    assert first.contexts == [] and second.contexts == []
    assert _records(fixture.campaign.chain.record_path) == before


@pytest.mark.unit
def test_a_campaign_with_one_method_publishes_that_methods_state_as_it_is(tmp_path: Path) -> None:
    method = NotingMethod("only")
    fixture = build_campaign(tmp_path, method=method)

    fixture.campaign.run(ScriptedScheduler([_harness(), _harness("only")]))

    expected = {"schema_version": 1, "method": "only", "proposals": 2, "verdicts": [True, True]}
    assert fixture.campaign.chain.head().method_state == expected
    assert _records(fixture.campaign.chain.record_path)[-1]["candidate"]["method_state"] == expected  # type: ignore[index]
    assert method.selector.shown[-1] == {**expected, "verdicts": [True]}


@pytest.mark.unit
def test_two_methods_cannot_share_a_name() -> None:
    world = FakeWorld()
    with pytest.raises(ContractError, match=r"names each harness method once, got \['same', 'same'\]"):
        HarnessComponents(
            runner=FakeEpisodeRunner(world),
            grader=FakeTaskGrader(),
            method=NotingMethod("same"),
            tasks=TASKS,
            methods=(NotingMethod("same"),),
        )


@pytest.mark.unit
def test_a_method_whose_state_schema_changed_is_refused_by_its_own_name(tmp_path: Path) -> None:
    first, second = NotingMethod("first"), NotingMethod("second")
    fixture = build_campaign(tmp_path, method=first, methods=[second])
    second.state_schema_version = 2

    with pytest.raises(
        Exception, match="second reads method state schema version 2, but the head's state is version 1"
    ):
        fixture.campaign.advance(_harness("second"))

    assert second.contexts == []
    assert fixture.campaign.advance(_harness("first")).release is not None


@pytest.mark.unit
def test_keyed_method_states_round_trip_and_refuse_another_shape() -> None:
    own = {"gepa": {"schema_version": 3, "served": 0}, "meta_harness": {"schema_version": 2, "candidates": []}}

    state = keyed_method_state(own)

    assert state == {"schema_version": 1, "methods": own} and method_states(state) == own
    assert json.loads(json.dumps(state)) == state
    with pytest.raises(ContractError, match="keyed by method name under schema version 1, got version 3"):
        method_states(own["gepa"])
    with pytest.raises(ContractError, match="got version 2"):
        method_states({"schema_version": 2, "methods": own})
    with pytest.raises(ContractError, match="must map each method's name to its state"):
        method_states({"schema_version": 1, "methods": ["gepa"]})
    with pytest.raises(ContractError, match="keyed by its name"):
        keyed_method_state({"": {}})


@pytest.mark.unit
def test_a_named_trainer_that_cannot_build_the_first_model_is_refused_at_the_start(tmp_path: Path) -> None:
    any_shape = replace(SLIME_LIKE, name="any-shape", architectures="heterogeneous")
    fixture = build_campaign(
        tmp_path, support=any_shape, make_trainers=lambda world: {"narrow": FakeTrainer(world, support=SLIME_LIKE)}
    )
    arch = small_moe()
    uneven = replace(arch, layers=(arch.layers[0], replace(arch.layers[1], num_experts=6), *arch.layers[2:]))

    with pytest.raises(
        CampaignRuleError, match="architecture_supported: fake-slime builds uniform architectures only"
    ):
        fixture.campaign.start(FakeModelSource(fixture.world, architecture=uneven), tmp_path / "again", SEED_HARNESS)
