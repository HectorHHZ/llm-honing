"""The contract a harness evolution method uses to call a language model.

Methods reflect on transcripts or propose edits by calling a model. The
protocol lives here so a method depends on nothing but ``honing.core``; the
client that implements it, and whatever library it wraps, lives outside core.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, get_args

from honing.core.errors import require, require_choice

ChatRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    """One turn of a conversation sent to a model."""

    role: ChatRole
    content: str

    def __post_init__(self) -> None:
        require_choice(self.role, get_args(ChatRole), "ChatMessage.role")


@dataclass(frozen=True)
class ChatReply:
    """A model's reply and what it cost.

    ``cost_usd`` is ``None`` when the price is unknown, as it is for a model
    served locally.
    """

    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None = None

    def __post_init__(self) -> None:
        require(self.input_tokens >= 0 and self.output_tokens >= 0, "token counts must be >= 0")
        require(
            self.cost_usd is None or (math.isfinite(self.cost_usd) and self.cost_usd >= 0.0),
            f"cost_usd must be a finite amount >= 0, got {self.cost_usd}",
        )
