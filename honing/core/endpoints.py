"""Where a policy is served, for runners that talk to a live engine."""

from __future__ import annotations

from dataclasses import dataclass

from honing.core.errors import require


@dataclass(frozen=True)
class PolicyEndpoint:
    """An engine serving one policy.

    ``base_url`` is the engine's HTTP root, ``tokenizer_location`` is where the
    tokenizer for the served weights can be loaded from, and ``served_model``
    is the name the engine answers to.
    """

    base_url: str
    tokenizer_location: str
    served_model: str

    def __post_init__(self) -> None:
        require(bool(self.base_url), "PolicyEndpoint.base_url must be set")
        require(bool(self.tokenizer_location), "PolicyEndpoint.tokenizer_location must be set")
        require(bool(self.served_model), "PolicyEndpoint.served_model must be set")
