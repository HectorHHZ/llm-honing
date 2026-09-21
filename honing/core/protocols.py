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
from honing.core.compression import CompressionBudget, ImportanceMap, PrunePlan, QuantizePlan, SignalKind
from honing.core.data import DataContract, TrainingBatch
from honing.core.errors import require
from honing.core.harness import EpisodeRecord, HarnessProposal, HarnessSpec, MethodRequirements, RunnerSupport, Task
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
    """Runs one agent episode under a harness against a policy.

    Asynchronous because episodes are long and a gate runs many of them at once.
    """

    support: RunnerSupport

    async def run(self, spec: HarnessSpec, task: Task, policy: PolicyRef) -> EpisodeRecord: ...


@dataclass(frozen=True)
class EvolveContext:
    """Everything a harness evolution method may read when it proposes.

    ``runner`` and ``episode_budget`` let a method run its own episodes, for
    example a minibatch against a candidate that is not the current head.
    """

    current: HarnessSpec
    policy: PolicyRef
    tasks: tuple[Task, ...]
    records: tuple[EpisodeRecord, ...]
    state: Mapping[str, object]
    runner: EpisodeRunner
    episode_budget: int

    def __post_init__(self) -> None:
        require(self.episode_budget >= 0, "episode_budget must be >= 0")


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

    ``record`` is called after every gate decision, accepted or rejected,
    because archive-based methods learn from rejected candidates too. The state
    it returns is published with the next release.
    """

    name: str
    requires: MethodRequirements
    selector: CandidateSelector

    def initial_state(self, seed: HarnessSpec) -> Mapping[str, object]: ...

    def propose(self, context: EvolveContext) -> HarnessProposal | None: ...

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
