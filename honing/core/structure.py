"""How a model is shaped, and how to name one piece of it.

A :data:`StructureAddress` names something a compression step can remove or
mask: a layer, an expert, an attention head, FFN channels, hidden channels, or
an element mask over one parameter. Pruning plans and the ``sparse_mask``
update policy use the same addresses, which is what lets masked training
rehearse a structural cut before the cut is made.

:class:`ArchitectureSpec` describes the model a plan leaves behind. It keeps a
shape per layer even though v1 accepts uniform architectures only, so adding
heterogeneous layers later is a change to a backend, not to this type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from honing.core.errors import require


def _non_negative(name: str, value: int) -> None:
    require(value >= 0, f"{name} must be >= 0, got {value}")


def _channel_indices(name: str, channels: tuple[int, ...]) -> None:
    require(len(channels) > 0, f"{name} must name at least one channel")
    require(all(channel >= 0 for channel in channels), f"{name} indices must be >= 0")
    require(len(set(channels)) == len(channels), f"{name} indices must be unique")


@dataclass(frozen=True)
class Layer:
    """A whole transformer block."""

    index: int

    def __post_init__(self) -> None:
        _non_negative("Layer.index", self.index)


@dataclass(frozen=True)
class Expert:
    """One routed expert inside one layer."""

    layer: int
    expert: int

    def __post_init__(self) -> None:
        _non_negative("Expert.layer", self.layer)
        _non_negative("Expert.expert", self.expert)


@dataclass(frozen=True)
class AttentionHead:
    """One attention head inside one layer."""

    layer: int
    head: int

    def __post_init__(self) -> None:
        _non_negative("AttentionHead.layer", self.layer)
        _non_negative("AttentionHead.head", self.head)


@dataclass(frozen=True)
class FFNChannels:
    """Intermediate channels of one FFN; ``expert`` is set when the FFN belongs to an expert."""

    layer: int
    channels: tuple[int, ...]
    expert: int | None = None

    def __post_init__(self) -> None:
        _non_negative("FFNChannels.layer", self.layer)
        _channel_indices("FFNChannels.channels", self.channels)
        if self.expert is not None:
            _non_negative("FFNChannels.expert", self.expert)


@dataclass(frozen=True)
class HiddenChannels:
    """Residual-stream channels.

    Removing them reshapes every layer, the embedding, and the output head at
    once, which is why no per-layer variant exists.
    """

    channels: tuple[int, ...]

    def __post_init__(self) -> None:
        _channel_indices("HiddenChannels.channels", self.channels)


@dataclass(frozen=True)
class ParamMask:
    """An element mask over one parameter. The parameter keeps its shape."""

    param: str
    mask_location: str

    def __post_init__(self) -> None:
        require(bool(self.param), "ParamMask.param must name a parameter")
        require(bool(self.mask_location), "ParamMask.mask_location must be set")


StructureAddress: TypeAlias = Layer | Expert | AttentionHead | FFNChannels | HiddenChannels | ParamMask


@dataclass(frozen=True)
class LayerShape:
    """The shape of one transformer block. ``num_experts`` is 0 for a dense FFN."""

    attention_heads: int
    ffn_size: int
    num_experts: int = 0

    def __post_init__(self) -> None:
        require(self.attention_heads > 0, f"attention_heads must be > 0, got {self.attention_heads}")
        require(self.ffn_size > 0, f"ffn_size must be > 0, got {self.ffn_size}")
        _non_negative("LayerShape.num_experts", self.num_experts)


@dataclass(frozen=True)
class ArchitectureSpec:
    """The shape of a whole model: its residual width and one shape per layer."""

    hidden_size: int
    layers: tuple[LayerShape, ...]

    def __post_init__(self) -> None:
        require(self.hidden_size > 0, f"hidden_size must be > 0, got {self.hidden_size}")
        require(len(self.layers) > 0, "an architecture needs at least one layer")

    @classmethod
    def uniform_layers(
        cls, *, num_layers: int, hidden_size: int, attention_heads: int, ffn_size: int, num_experts: int = 0
    ) -> ArchitectureSpec:
        """An architecture whose every layer has the same shape."""
        require(num_layers > 0, f"num_layers must be > 0, got {num_layers}")
        shape = LayerShape(attention_heads=attention_heads, ffn_size=ffn_size, num_experts=num_experts)
        return cls(hidden_size=hidden_size, layers=(shape,) * num_layers)

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def uniform(self) -> bool:
        """Every layer has the same shape.

        Which experts or heads each layer keeps may still differ: a uniform
        expert drop keeps 48 experts in every layer, but not the same 48.
        """
        return len(set(self.layers)) == 1
