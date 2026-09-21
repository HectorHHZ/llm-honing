"""Model-side fakes: a checkpoint is a checksum, and its quality is a number."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from honing.core import (
    ArchitectureSpec,
    BackendSupport,
    CheckpointRef,
    ModelSnapshot,
    QuantScheme,
    TrainingBatch,
    TrainJobSpec,
    TrainResult,
    WeightFormat,
)

SHARED_PARAMETERS = 1_000
LAYER_PARAMETERS = 500
EXPERT_PARAMETERS = 100


def parameters_of(architecture: ArchitectureSpec) -> int:
    return SHARED_PARAMETERS + sum(
        LAYER_PARAMETERS + layer.num_experts * EXPERT_PARAMETERS for layer in architecture.layers
    )


def small_moe() -> ArchitectureSpec:
    return ArchitectureSpec.uniform_layers(
        num_layers=4, hidden_size=64, attention_heads=4, ffn_size=128, num_experts=8
    )


@dataclass
class FakeWorld:
    """Stands in for storage and accelerators: each checkpoint checksum maps to a model quality."""

    quality: dict[str, float] = field(default_factory=dict)
    trained_scheme: dict[str, QuantScheme] = field(default_factory=dict)
    created: int = 0

    def checkpoint(self, quality: float, weight_format: WeightFormat = "hf_safetensors") -> CheckpointRef:
        self.created += 1
        checksum = f"sha-{self.created:04d}"
        self.quality[checksum] = round(min(1.0, max(0.0, quality)), 6)
        return CheckpointRef(location=f"fake://{checksum}", weight_format=weight_format, checksum=checksum)

    def snapshot(
        self, architecture: ArchitectureSpec, quality: float, weight_format: WeightFormat = "hf_safetensors"
    ) -> ModelSnapshot:
        return ModelSnapshot(self.checkpoint(quality, weight_format), architecture, parameters_of(architecture))


SLIME_LIKE = BackendSupport(
    name="fake-slime",
    architectures="uniform",
    train_numerics=frozenset({"bf16", "fp8", "fake_quant_qat"}),
    weight_formats=frozenset({"hf_safetensors", "megatron_torch_dist"}),
    algorithms=(
        ("sft", "native"),
        ("distill", "hook"),
        ("opd", "native"),
        ("gkd", "hook"),
        ("ppo", "native"),
        ("grpo", "native"),
    ),
)


class FakeModelSource:
    def __init__(
        self,
        world: FakeWorld,
        architecture: ArchitectureSpec | None = None,
        quality: float = 0.5,
        weight_format: WeightFormat = "hf_safetensors",
    ) -> None:
        self.world = world
        self.architecture = architecture if architecture is not None else small_moe()
        self.quality = quality
        self.weight_format = weight_format

    def materialize(self, work_dir: Path) -> ModelSnapshot:
        return self.world.snapshot(self.architecture, self.quality, self.weight_format)


class FakeTrainer:
    """Every job moves quality by ``gain`` and remembers which scheme a QAT job trained for."""

    def __init__(self, world: FakeWorld, support: BackendSupport = SLIME_LIKE, gain: float = 0.1) -> None:
        self.world = world
        self.support = support
        self.gain = gain
        self.jobs: list[TrainJobSpec] = []

    def train(self, job: TrainJobSpec, base: ModelSnapshot, batch: TrainingBatch | None) -> TrainResult:
        self.jobs.append(job)
        snapshot = self.world.snapshot(base.architecture, self.world.quality[base.checkpoint.checksum] + self.gain)
        if job.numerics.mode == "fake_quant_qat" and job.numerics.scheme is not None:
            self.world.trained_scheme[snapshot.checkpoint.checksum] = job.numerics.scheme
        return TrainResult(snapshot=snapshot, metrics={"train_steps": float(job.steps)})


class IdentityConverter:
    """Pretends to convert by relabeling the format; enough to satisfy the protocol."""

    def __init__(self, world: FakeWorld) -> None:
        self.world = world

    def convert(self, checkpoint: CheckpointRef, target: WeightFormat) -> CheckpointRef:
        return self.world.checkpoint(self.world.quality[checkpoint.checksum], target)
