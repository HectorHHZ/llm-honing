"""The plane contracts: every seam where one component is swapped for another.

Each protocol is the whole surface a plane exposes. A fake that satisfies one
is enough to run a campaign end to end, which is how the walking skeleton
exercises every seam before any real backend exists.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from honing.core.backend import BackendSupport
from honing.core.candidate import EvaluationResult, PolicyRef, SelectionDecision, UpdateCandidate
from honing.core.chat import ChatMessage, ChatReply
from honing.core.compression import CompressionBudget, ImportanceMap, PrunePlan, QuantizePlan, SignalKind
from honing.core.data import DataContract, TrainingBatch
from honing.core.endpoints import PolicyEndpoint
from honing.core.errors import ContractError, require
from honing.core.harness import (
    EpisodeOutcome,
    EpisodeRecord,
    Grade,
    HarnessProposal,
    HarnessSpec,
    MethodRequirements,
    ProposeResult,
    RunnerSupport,
    Task,
)
from honing.core.release import HoneRelease
from honing.core.schedule import CampaignState, Move
from honing.core.telemetry import UtilizationTelemetry
from honing.core.training import TrainJobSpec, TrainResult
from honing.core.weights import CheckpointRef, ModelSnapshot, ServingWeights, WeightFormat

# ---------------------------------------------------------------- model source and data


@runtime_checkable
class ModelSource(Protocol):
    """Produces the initial model, once, at the start of a campaign."""

    def materialize(self, work_dir: Path) -> ModelSnapshot: ...


@runtime_checkable
class DataProvider(Protocol):
    """Offline datasets or online feedback. Returns ``None`` when it holds nothing for the contract."""

    def next_batch(self, contract: DataContract) -> TrainingBatch | None: ...


@runtime_checkable
class RolloutProvider(Protocol):
    """Samples from a policy, with or without a harness.

    ``release_id`` is stamped on every sample so a batch can be traced to the
    release that produced it; it is ``None`` for weights that are not published.
    """

    def collect(
        self,
        tasks: Sequence[Task],
        policy: PolicyRef,
        harness: HarnessSpec | None,
        release_id: str | None,
        samples_per_task: int,
    ) -> TrainingBatch: ...


# ---------------------------------------------------------------- training


@runtime_checkable
class Trainer(Protocol):
    """Runs one training job and returns master-format weights."""

    support: BackendSupport

    def train(self, job: TrainJobSpec, base: ModelSnapshot, batch: TrainingBatch | None) -> TrainResult: ...


@runtime_checkable
class WeightConverter(Protocol):
    """Converts between the master format and a backend's own format."""

    def convert(self, checkpoint: CheckpointRef, target: WeightFormat) -> CheckpointRef: ...


# ---------------------------------------------------------------- compression


@runtime_checkable
class ImportanceScorer(Protocol):
    """Scores structures from one signal. ``telemetry`` is ``None`` for signals that do not read it."""

    signal: SignalKind

    def score(self, model: ModelSnapshot, telemetry: UtilizationTelemetry | None) -> ImportanceMap: ...


@runtime_checkable
class PruneSelector(Protocol):
    """Chooses what to remove within a budget and states the architecture that remains."""

    def select(self, importance: ImportanceMap, model: ModelSnapshot, budget: CompressionBudget) -> PrunePlan: ...


@runtime_checkable
class ModelEditor(Protocol):
    """Applies a plan to master weights."""

    def apply(self, model: ModelSnapshot, plan: PrunePlan) -> ModelSnapshot: ...


@runtime_checkable
class Quantizer(Protocol):
    """Builds serving weights from master weights. Never modifies the master."""

    def quantize(self, model: ModelSnapshot, plan: QuantizePlan) -> ServingWeights: ...


@runtime_checkable
class TelemetryStore(Protocol):
    """Aggregates utilization over a window."""

    def observe(self, stats: UtilizationTelemetry) -> None: ...

    def window(self) -> UtilizationTelemetry | None: ...


# ---------------------------------------------------------------- harness


@runtime_checkable
class EpisodeRunner(Protocol):
    """Runs one agent episode under a harness against a policy, and does not judge it.

    Asynchronous because episodes are long and a gate runs many of them at once.
    """

    support: RunnerSupport

    async def run(self, spec: HarnessSpec, task: Task, policy: PolicyRef) -> EpisodeOutcome: ...


@runtime_checkable
class TaskGrader(Protocol):
    """Judges one episode's outcome. The score is in ``[0, 1]``; the feedback is for reflection.

    Asynchronous like the runner, since a grader may run a command in the
    episode's own sandbox.
    """

    async def grade(self, task: Task, outcome: EpisodeOutcome) -> Grade: ...


@runtime_checkable
class ChatModel(Protocol):
    """A language model a method calls to reflect or propose. Synchronous, like ``propose``."""

    def complete(self, messages: Sequence[ChatMessage], *, max_tokens: int, temperature: float) -> ChatReply: ...


@runtime_checkable
class ProposerAgent(Protocol):
    """A coding agent a method runs in a working directory, where it reads and writes files.

    ``instructions`` is the method's skill text; how the agent receives it, and
    how the session is logged, is the implementation's concern. Returns the
    agent's exit code. Synchronous, like ``propose``.
    """

    def run(self, prompt: str, workdir: Path, *, instructions: str) -> int: ...


@runtime_checkable
class PolicyEndpoints(Protocol):
    """Finds the engine serving a policy. Raises ``ContractError`` for a policy no engine serves."""

    def resolve(self, policy: PolicyRef) -> PolicyEndpoint: ...


async def run_graded(
    runner: EpisodeRunner, grader: TaskGrader, spec: HarnessSpec, task: Task, policy: PolicyRef
) -> EpisodeRecord:
    """Run one episode and grade it: the only way the gate and methods obtain an episode record."""
    outcome = await runner.run(spec, task, policy)
    if (outcome.task_id, outcome.spec_id, outcome.policy_checksum) != (
        task.task_id,
        spec.spec_id,
        policy.checkpoint.checksum,
    ):
        raise ContractError(
            f"the runner reported task {outcome.task_id!r}, spec {outcome.spec_id!r}, and policy "
            f"{outcome.policy_checksum!r} for an episode it was asked to run on task {task.task_id!r}, "
            f"spec {spec.spec_id!r}, and policy {policy.checkpoint.checksum!r}"
        )
    return EpisodeRecord(outcome=outcome, grade=await grader.grade(task, outcome))


@dataclass(frozen=True)
class EvolveContext:
    """Everything a harness evolution method may read when it proposes.

    ``runner``, ``grader``, and ``episode_budget`` let a method run and score
    its own episodes, for example a minibatch against a candidate that is not
    the current head. :meth:`episode` does both in one call.
    """

    current: HarnessSpec
    policy: PolicyRef
    tasks: tuple[Task, ...]
    records: tuple[EpisodeRecord, ...]
    state: Mapping[str, object]
    runner: EpisodeRunner
    grader: TaskGrader
    episode_budget: int

    def __post_init__(self) -> None:
        require(self.episode_budget >= 0, "episode_budget must be >= 0")

    async def episode(self, spec: HarnessSpec, task: Task, policy: PolicyRef) -> EpisodeRecord:
        return await run_graded(self.runner, self.grader, spec, task, policy)


# ---------------------------------------------------------------- gate and campaign


@runtime_checkable
class CandidateSelector(Protocol):
    """Decides whether an evaluated candidate is published."""

    name: str

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision: ...


@runtime_checkable
class Evaluator(Protocol):
    """Scores a candidate and the release it would replace on the same tasks."""

    def evaluate(self, candidate: UpdateCandidate, incumbent: HoneRelease) -> EvaluationResult: ...


@runtime_checkable
class EvolveMethod(Protocol):
    """A harness evolution algorithm.

    ``propose`` returns the method's new state whether or not it proposes
    anything, because an attempt that stops early has still spent episodes.
    ``record`` is called after every gate decision, accepted or rejected,
    because archive-based methods learn from rejected candidates too. The state
    either returns is published with the next release, and must carry
    ``state_schema_version``.
    """

    name: str
    requires: MethodRequirements
    selector: CandidateSelector
    state_schema_version: int

    def initial_state(self, seed: HarnessSpec) -> Mapping[str, object]: ...

    def propose(self, context: EvolveContext) -> ProposeResult: ...

    def record(
        self,
        state: Mapping[str, object],
        proposal: HarnessProposal,
        evaluation: EvaluationResult,
        decision: SelectionDecision,
    ) -> Mapping[str, object]: ...


@runtime_checkable
class CoAdaptScheduler(Protocol):
    """Chooses the next move from the campaign's state."""

    def next_move(self, state: CampaignState) -> Move: ...
