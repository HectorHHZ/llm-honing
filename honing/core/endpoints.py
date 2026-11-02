"""Where a policy is served, for runners that talk to a live engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

from honing.core.errors import require, require_choice

EndpointRuntime = Literal["sglang", "openai_chat", "jev_decision"]


@dataclass(frozen=True)
class PolicyEndpoint:
    """An engine serving one policy.

    ``base_url`` is the engine's HTTP root, ``tokenizer_location`` is where the
    tokenizer for the served weights can be loaded from, and ``served_model``
    is the name the engine answers to. ``runtime`` says which protocol the engine
    speaks: an SGLang server for language models, any other OpenAI-compatible chat
    API (a provider, a router such as OpenRouter, or a vLLM server), or a Jev
    decision server that answers typed questions with a probability per option.
    """

    base_url: str
    tokenizer_location: str
    served_model: str
    runtime: EndpointRuntime = "sglang"

    def __post_init__(self) -> None:
        require(bool(self.base_url), "PolicyEndpoint.base_url must be set")
        require(bool(self.tokenizer_location), "PolicyEndpoint.tokenizer_location must be set")
        require(bool(self.served_model), "PolicyEndpoint.served_model must be set")
        require_choice(self.runtime, get_args(EndpointRuntime), "PolicyEndpoint.runtime")
