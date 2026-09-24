"""Every plane protocol has a fake that satisfies it, and value objects refuse invalid values."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Protocol

import pytest

from honing.campaign import ScriptedScheduler
from honing.core import (
    ChatReply,
    CheckpointRef,
    CompressionStep,
    ContractError,
    CostDelta,
    DataContract,
    EpisodeOutcome,
    EvaluationResult,
    Grade,
    HarnessFile,
    HarnessPhaseConfig,
    HarnessSpec,
    HoneRelease,
    ImportanceMap,
    ModelSnapshot,
    Move,
    NumericsPolicy,
    PolicyEndpoint,
    PolicySample,
    ProposeResult,
    PrunePlan,
    QuantizePlan,
    QuantScheme,
    RecoverySpec,
    TeacherRef,
    TrainingBatch,
    TrainJobSpec,
    UpdateCandidate,
    UpdatePolicy,
    protocols,
)
from honing.release import AcceptAll, EpisodeEvaluator, NoRegressionSelector, WinLossSelector
from tests.fakes.builder import INT4_G128, TASKS
from tests.fakes.compression import (
    ActivationCountScorer,
    FakeQuantizer,
    FixedTelemetryStore,
    KeepTopKExpertSelector,
    UniformExpertEditor,
)
from tests.fakes.data import FakeRolloutProvider, ListDatasetProvider, token_batch
from tests.fakes.harness import (
    AppendInstructionMethod,
    FakeChatModel,
    FakeEpisodeRunner,
    FakeTaskGrader,
    StaticPolicyEndpoints,
)
from tests.fakes.model import FakeModelSource, FakeTrainer, FakeWorld, IdentityConverter, parameters_of, small_moe


def implementations() -> list[tuple[object, type]]:
    world = FakeWorld()
    runner = FakeEpisodeRunner(world)
    grader = FakeTaskGrader()
    return [
        (FakeModelSource(world), protocols.ModelSource),
        (ListDatasetProvider([token_batch()]), protocols.DataProvider),
        (FakeRolloutProvider(), protocols.RolloutProvider),
        (FakeTrainer(world), protocols.Trainer),
        (IdentityConverter(world), protocols.WeightConverter),
        (ActivationCountScorer(), protocols.ImportanceScorer),
        (KeepTopKExpertSelector(), protocols.PruneSelector),
        (UniformExpertEditor(world), protocols.ModelEditor),
        (FakeQuantizer(world), protocols.Quantizer),
        (FixedTelemetryStore(None), protocols.TelemetryStore),
        (runner, protocols.EpisodeRunner),
        (grader, protocols.TaskGrader),
        (FakeChatModel(["reply"]), protocols.ChatModel),
        (StaticPolicyEndpoints({}), protocols.PolicyEndpoints),
        (AppendInstructionMethod(), protocols.EvolveMethod),
        (WinLossSelector(), protocols.CandidateSelector),
        (NoRegressionSelector(0.1), protocols.CandidateSelector),
        (AcceptAll(), protocols.CandidateSelector),
        (EpisodeEvaluator(runner, grader, TASKS), protocols.Evaluator),
        (ScriptedScheduler([]), protocols.CoAdaptScheduler),
    ]


@pytest.mark.unit
def test_every_implementation_satisfies_its_protocol() -> None:
    for instance, protocol in implementations():
        assert isinstance(instance, protocol), f"{type(instance).__name__} does not satisfy {protocol.__name__}"


@pytest.mark.unit
def test_every_protocol_has_an_implementation_checked_here() -> None:
    """A new protocol must arrive with a fake, or the skeleton cannot exercise its seam."""
    declared = {
        member
        for _, member in inspect.getmembers(protocols, inspect.isclass)
        if member.__module__ == protocols.__name__ and Protocol in member.__mro__
    }
    covered = {protocol for _, protocol in implementations()}
    assert (
        declared - covered == set()
    ), f"protocols without an implementation: {sorted(p.__name__ for p in declared - covered)}"


def _snapshot(weight_format: str = "hf_safetensors") -> ModelSnapshot:
    architecture = small_moe()
    ref = CheckpointRef(location="fake://a", weight_format=weight_format, checksum="a")
    return ModelSnapshot(ref, architecture, parameters_of(architecture))


INVALID: list[tuple[str, Callable[[], object]]] = [
    ("a group scheme without a group size", lambda: QuantScheme(weight_format="int4", granularity="group")),
    ("a per-channel scheme with a group size", lambda: QuantScheme("int8", "per_channel", group_size=8)),
    ("quantization-aware training without a scheme", lambda: NumericsPolicy(mode="fake_quant_qat")),
    ("a bf16 job with a scheme", lambda: NumericsPolicy(mode="bf16", scheme=INT4_G128)),
    ("a harness path that leaves the root", lambda: HarnessSpec("claude_code", (("../x", HarnessFile("skill", "")),))),
    ("a harness path given twice", lambda: HarnessSpec("a", (("x", HarnessFile("skill", "")),) * 2)),
    ("a loss mask of the wrong length", lambda: PolicySample(tokens=(1, 2), loss_mask=(1,))),
    (
        "a preference batch with samples",
        lambda: TrainingBatch("b", DataContract("preference_pairs"), token_batch().samples),
    ),
    (
        "a rewarded contract with unrewarded samples",
        lambda: TrainingBatch("b", DataContract("token_sequences", requires_reward=True), token_batch().samples),
    ),
    ("lora without a rank", lambda: UpdatePolicy(kind="lora")),
    (
        "a parameter pattern that does not compile",
        lambda: UpdatePolicy(kind="param_subset", trainable_patterns=("(",)),
    ),
    (
        "an external teacher loaded in process",
        lambda: TeacherRef(kind="external", ref="http://t", placement="in_process"),
    ),
    ("an unknown algorithm", lambda: TrainJobSpec(algorithm="rlhf")),
    (
        "an empty prune plan",
        lambda: PrunePlan((), small_moe(), small_moe(), ImportanceMap("weights", ()), CostDelta(0)),
    ),
    (
        "recovery after a quantize step",
        lambda: CompressionStep(QuantizePlan(INT4_G128, _snapshot().checkpoint), RecoverySpec("sft", 1)),
    ),
    ("an evaluation with missing scores", lambda: EvaluationResult(("t",), (1.0,), (), "master")),
    (
        "a candidate other than init without a parent",
        lambda: UpdateCandidate(kind="train", parent_id=None, master=_snapshot()),
    ),
    (
        "a release whose master is not safetensors",
        lambda: HoneRelease("r0000", UpdateCandidate("init", None, _snapshot("megatron_torch_dist")), None),
    ),
    ("a move carrying another kind's payload", lambda: Move(kind="train", scheme=INT4_G128)),
    ("a phase on a move that is not a harness move", lambda: Move(kind="train", phase="pre_training")),
    ("an unknown harness phase", lambda: Move(kind="harness", phase="mid_training")),
    ("a grade above one", lambda: Grade(score=1.5)),
    ("a grade below zero", lambda: Grade(score=-0.1)),
    ("an outcome with no task", lambda: EpisodeOutcome("", "h", "c", exit_code=0)),
    ("a negative episode count", lambda: ProposeResult(proposal=None, state={}, episodes_spent=-1)),
    ("a phase with no tasks", lambda: HarnessPhaseConfig(tasks=(), max_proposals=1, max_episodes=1)),
    ("a reply with a negative cost", lambda: ChatReply("x", 1, 1, cost_usd=-0.01)),
    ("an endpoint with no URL", lambda: PolicyEndpoint(base_url="", tokenizer_location="t", served_model="m")),
    (
        "method state holding a set",
        lambda: UpdateCandidate("init", None, _snapshot(), method_state={"schema_version": 1, "front": {"a"}}),
    ),
    (
        "method state holding a tuple",
        lambda: UpdateCandidate("init", None, _snapshot(), method_state={"schema_version": 1, "scores": (1, 2)}),
    ),
    (
        "method state with a non-string key",
        lambda: UpdateCandidate("init", None, _snapshot(), method_state={"schema_version": 1, "by_task": {1: 0.5}}),
    ),
    (
        "method state without a schema version",
        lambda: UpdateCandidate("init", None, _snapshot(), method_state={"proposals": 0}),
    ),
    (
        "method state with a boolean schema version",
        lambda: UpdateCandidate("init", None, _snapshot(), method_state={"schema_version": True}),
    ),
]


@pytest.mark.unit
@pytest.mark.parametrize(("case", "build"), INVALID, ids=[case for case, _ in INVALID])
def test_an_invalid_value_is_refused_at_construction(case: str, build: Callable[[], object]) -> None:
    with pytest.raises(ContractError):
        build()
