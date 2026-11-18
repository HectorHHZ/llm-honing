"""Wires every fake into a started campaign."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from honing.campaign import Campaign, CampaignDefaults, CompressionComponents, HarnessComponents
from honing.campaign.cost import CampaignBudget
from honing.core import (
    BackendSupport,
    CandidateSelector,
    CompressionBudget,
    DataProvider,
    EvolveMethod,
    PruneSelector,
    QuantScheme,
    RecoverySpec,
    RolloutProvider,
    Task,
    TeacherRef,
    TrainJobSpec,
)
from honing.release import EpisodeEvaluator, Gate, NoRegressionSelector, ReleaseChain, WinLossSelector
from tests.fakes.compression import (
    ActivationCountScorer,
    FakeQuantizer,
    FixedTelemetryStore,
    KeepTopKExpertSelector,
    UniformExpertEditor,
    skewed_expert_telemetry,
)
from tests.fakes.data import ListDatasetProvider, token_batch
from tests.fakes.harness import SEED_HARNESS, AppendInstructionMethod, FakeEpisodeRunner, FakeTaskGrader
from tests.fakes.model import SLIME_LIKE, FakeModelSource, FakeTrainer, FakeWorld, small_moe

TASKS = (
    Task(task_id="sum", prompt="Add the numbers."),
    Task(task_id="sort", prompt="Sort the list."),
    Task(task_id="parse", prompt="Parse the date."),
)

INT4_G128 = QuantScheme(weight_format="int4", granularity="group", group_size=128)
INT4_G32 = QuantScheme(weight_format="int4", granularity="group", group_size=32)

ENGINE_LINEAGE_TEACHER = TeacherRef(kind="lineage", ref="parent", placement="engine")


@dataclass
class CampaignFixture:
    campaign: Campaign
    world: FakeWorld
    trainer: FakeTrainer
    editor: UniformExpertEditor
    quantizer: FakeQuantizer
    runner: FakeEpisodeRunner
    grader: FakeTaskGrader
    method: EvolveMethod
    data: DataProvider


def build_campaign(
    work_dir: Path,
    *,
    trainer_gain: float = 0.1,
    support: BackendSupport = SLIME_LIKE,
    prune_selector: PruneSelector | None = None,
    recovery: RecoverySpec | None = None,
    rollout: RolloutProvider | None = None,
    method: EvolveMethod | None = None,
    data: DataProvider | None = None,
    make_runner: Callable[[FakeWorld], FakeEpisodeRunner] = FakeEpisodeRunner,
    gate_tasks: Sequence[Task] = TASKS,
    harness_selector: CandidateSelector | None = None,
    method_tasks: Sequence[Task] | None = None,
    budget: CampaignBudget | None = None,
    gpu_seconds_per_step: float | None = 1.5,
) -> CampaignFixture:
    """A started campaign: release 0 is published and every plane is a fake.

    ``gate_tasks`` are the gate's tasks and the harness plane's default tasks.
    ``harness_selector`` is the gate's selector for harness candidates. With
    ``method_tasks``, a harness candidate must pass the method's selector on
    those tasks and then the gate's (``HarnessComponents.method_evaluator``).
    """
    world = FakeWorld()
    trainer = FakeTrainer(world, support=support, gain=trainer_gain, gpu_seconds_per_step=gpu_seconds_per_step)
    runner = make_runner(world)
    grader = FakeTaskGrader()
    method = method if method is not None else AppendInstructionMethod()
    editor = UniformExpertEditor(world)
    quantizer = FakeQuantizer(world)
    data = data if data is not None else ListDatasetProvider([token_batch()])

    gate = Gate(
        EpisodeEvaluator(runner, grader, gate_tasks),
        default_selector=WinLossSelector(),
        # Cost-driven changes need only avoid regressing; quality-driven ones must win.
        selectors={
            "prune": NoRegressionSelector(0.05),
            "quantize": NoRegressionSelector(0.05),
            **({"harness": harness_selector} if harness_selector is not None else {}),
        },
    )
    method_evaluator = EpisodeEvaluator(runner, grader, method_tasks) if method_tasks is not None else None
    campaign = Campaign(
        chain=ReleaseChain(work_dir / "releases"),
        gate=gate,
        trainer=trainer,
        data=data,
        compression=CompressionComponents(
            scorer=ActivationCountScorer(),
            selector=prune_selector if prune_selector is not None else KeepTopKExpertSelector(),
            editor=editor,
            quantizer=quantizer,
            telemetry=FixedTelemetryStore(skewed_expert_telemetry(small_moe())),
        ),
        harness=HarnessComponents(
            runner=runner, grader=grader, method=method, tasks=tuple(gate_tasks), method_evaluator=method_evaluator
        ),
        defaults=CampaignDefaults(
            train=TrainJobSpec(algorithm="sft", steps=10),
            budget=CompressionBudget(keep_ratio=0.75),
            recovery=(
                recovery
                if recovery is not None
                else RecoverySpec(algorithm="distill", steps=10, teacher=ENGINE_LINEAGE_TEACHER)
            ),
            scheme=INT4_G128,
        ),
        rollout=rollout,
        budget=budget,
    )
    campaign.start(FakeModelSource(world), work_dir, harness_seed=SEED_HARNESS)
    return CampaignFixture(campaign, world, trainer, editor, quantizer, runner, grader, method, data)
