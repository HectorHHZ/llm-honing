"""Where weights live, in what format, and what they were built from.

The master weights of every release are Hugging Face safetensors
(:data:`MASTER_WEIGHT_FORMAT`). Compression editors, post-training quantizers,
and the local preference trainer all read that format. A training backend that
needs another format converts before its job and converts back after it; the
conversion never leaks into a release.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, get_args

from honing.core.errors import require, require_choice
from honing.core.numerics import QuantScheme
from honing.core.structure import ArchitectureSpec

WeightFormat = Literal["hf_safetensors", "megatron_torch_dist"]

MASTER_WEIGHT_FORMAT: Final[WeightFormat] = "hf_safetensors"


@dataclass(frozen=True)
class CheckpointRef:
    """A checkpoint on storage, identified by its checksum."""

    location: str
    weight_format: WeightFormat
    checksum: str

    def __post_init__(self) -> None:
        require(bool(self.location), "CheckpointRef.location must be set")
        require(bool(self.checksum), "CheckpointRef.checksum must be set")
        require_choice(self.weight_format, get_args(WeightFormat), "CheckpointRef.weight_format")


@dataclass(frozen=True)
class ModelSnapshot:
    """A checkpoint together with the architecture it has and its parameter count."""

    checkpoint: CheckpointRef
    architecture: ArchitectureSpec
    parameters: int

    def __post_init__(self) -> None:
        require(self.parameters > 0, f"parameters must be > 0, got {self.parameters}")


@dataclass(frozen=True)
class ServingWeights:
    """Quantized weights built from one master checkpoint under one scheme.

    ``source`` records which master they came from. A candidate whose master
    changed but whose serving weights did not is refused, because those serving
    weights were never quantized from the weights being released.
    """

    checkpoint: CheckpointRef
    scheme: QuantScheme
    source: CheckpointRef
