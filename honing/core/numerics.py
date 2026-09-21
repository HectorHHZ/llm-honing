"""Numeric precision: how master weights are trained and how serving weights are stored.

One :class:`QuantScheme` is referenced from three places: fake-quant training
(:class:`NumericsPolicy`), post-training quantization (``QuantizePlan``), and
the engine that loads the serving weights. Sharing one object is what lets a
campaign rule refuse a quantization-aware run whose scheme differs from the
one it exports. Left unchecked, that mismatch fails silently: the weights
adapt to one quantization grid and are served on another.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

from honing.core.errors import require, require_choice

Granularity = Literal["per_tensor", "per_channel", "group", "block"]
NumericsMode = Literal["bf16", "fp8", "fake_quant_qat"]


@dataclass(frozen=True)
class QuantScheme:
    """A quantization format.

    ``exclude`` names modules kept at the master precision; embeddings and the
    output head are the usual ones, and Slime's FP8 training excludes both.
    """

    weight_format: str
    granularity: Granularity
    group_size: int | None = None
    activation_format: str | None = None
    kv_cache_format: str | None = None
    exclude: tuple[str, ...] = ("embed_tokens", "lm_head")

    def __post_init__(self) -> None:
        require(bool(self.weight_format), "QuantScheme.weight_format must be named")
        require_choice(self.granularity, get_args(Granularity), "QuantScheme.granularity")
        if self.granularity in ("group", "block"):
            require(
                self.group_size is not None and self.group_size > 0,
                f"a {self.granularity} scheme needs a positive group_size",
            )
        else:
            require(self.group_size is None, f"a {self.granularity} scheme takes no group_size")


@dataclass(frozen=True)
class NumericsPolicy:
    """The precision a training job runs its forward and backward passes in.

    ``fake_quant_qat`` keeps BF16 master weights but quantizes them in the
    forward pass under ``scheme``, so the result tolerates that exact scheme at
    serving time.
    """

    mode: NumericsMode = "bf16"
    scheme: QuantScheme | None = None

    def __post_init__(self) -> None:
        require_choice(self.mode, get_args(NumericsMode), "NumericsPolicy.mode")
        if self.mode == "fake_quant_qat":
            require(self.scheme is not None, "quantization-aware training needs the scheme it trains for")
        if self.mode == "bf16":
            require(self.scheme is None, "a bf16 job takes no quantization scheme")
