"""Any model behind an OpenAI-compatible URL, as a Honing policy.

The URL can be a provider's API, a router such as OpenRouter, or a local vLLM or
SGLang server. Honing calls the model but holds no weights for it, so it gets a
stand-in snapshot whose checksum comes from the URL and model name. That is
enough to run and gate harness moves against it; training, pruning, or
quantizing it means nothing. Core has no weight format for such models, so the
stand-in reuses the master format and marks its location ``gateway://``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from honing.core import (
    MASTER_WEIGHT_FORMAT,
    ArchitectureSpec,
    CheckpointRef,
    ContractError,
    ModelSnapshot,
    PolicyEndpoint,
    PolicyRef,
    require,
)

_PLACEHOLDER = ArchitectureSpec.uniform_layers(num_layers=1, hidden_size=1, attention_heads=1, ffn_size=1)


@dataclass(frozen=True)
class GatewayModel:
    """One model at one OpenAI-compatible URL.

    Attributes:
        base_url: The endpoint's root, ending in ``/v1`` for most providers.
        model: The name the endpoint serves the model under.
        api_key_env: The environment variable holding the key; the key itself is never stored.
        input_usd_per_million: The price of input tokens, if known.
        output_usd_per_million: The price of output tokens, if known.
    """

    base_url: str
    model: str
    api_key_env: str | None = None
    input_usd_per_million: float | None = None
    output_usd_per_million: float | None = None

    def __post_init__(self) -> None:
        require(
            self.base_url.startswith(("http://", "https://")), f"base_url must be an HTTP URL, got {self.base_url}"
        )
        require(bool(self.model), "GatewayModel.model must name the served model")
        for price in (self.input_usd_per_million, self.output_usd_per_million):
            require(price is None or price >= 0.0, f"prices must be >= 0, got {price}")

    @property
    def checksum(self) -> str:
        """The stand-in checksum: stable for one endpoint and model name."""
        digest = hashlib.sha256(f"{self.base_url.rstrip('/')}|{self.model}".encode()).hexdigest()
        return f"gateway-{digest[:16]}"

    def snapshot(self) -> ModelSnapshot:
        """Returns the stand-in snapshot a release holds for this model."""
        checkpoint = CheckpointRef(
            location=f"gateway://{self.base_url.split('://', 1)[1].rstrip('/')}/{self.model}",
            weight_format=MASTER_WEIGHT_FORMAT,
            checksum=self.checksum,
        )
        # parameters=1 is a placeholder, so a campaign's honing curve says nothing about a gateway model's size.
        return ModelSnapshot(checkpoint=checkpoint, architecture=_PLACEHOLDER, parameters=1)

    def endpoint(self) -> PolicyEndpoint:
        """Returns the endpoint; ``sglang`` is core's runtime for OpenAI-compatible chat."""
        return PolicyEndpoint(
            base_url=self.base_url.rstrip("/"), tokenizer_location="gateway", served_model=self.model, runtime="sglang"
        )


class GatewayModelSource:
    """Satisfies :class:`~honing.core.ModelSource` with a gateway model's stand-in snapshot.

    Args:
        model: The gateway model.
    """

    def __init__(self, model: GatewayModel) -> None:
        self._model = model

    def materialize(self, work_dir: Path) -> ModelSnapshot:
        """Returns the stand-in snapshot.

        Args:
            work_dir: Unused; nothing is downloaded.
        """
        return self._model.snapshot()


class GatewayEndpoints:
    """Satisfies :class:`~honing.core.PolicyEndpoints` for gateway models.

    Args:
        *models: The gateway models a campaign may run against.

    Raises:
        ContractError: No model was given.
    """

    def __init__(self, *models: GatewayModel) -> None:
        require(len(models) > 0, "GatewayEndpoints needs at least one model")
        self._models = {model.checksum: model for model in models}

    def resolve(self, policy: PolicyRef) -> PolicyEndpoint:
        """Returns the endpoint serving ``policy``.

        Args:
            policy: A gateway model's stand-in policy.

        Raises:
            ContractError: No gateway model has that checksum.
        """
        return self.model(policy).endpoint()

    def model(self, policy: PolicyRef) -> GatewayModel:
        """Returns the gateway model behind ``policy``, for a runner that needs its key and prices.

        Args:
            policy: A gateway model's stand-in policy.

        Raises:
            ContractError: No gateway model has that checksum.
        """
        try:
            return self._models[policy.checkpoint.checksum]
        except KeyError:
            raise ContractError(f"no gateway model serves checkpoint {policy.checkpoint.checksum!r}") from None


def model_source(context: object, **model: object) -> GatewayModelSource:
    """Builds the model source for a pipeline config.

    Args:
        context: The pipeline context.
        **model: :class:`GatewayModel` fields.

    Returns:
        The model source.
    """
    return GatewayModelSource(_shared_model(context, model))


def endpoints(context: object, **model: object) -> GatewayEndpoints:
    """Builds the endpoints for a pipeline config, from the same fields as :func:`model_source`.

    Args:
        context: The pipeline context.
        **model: :class:`GatewayModel` fields.

    Returns:
        The endpoints.
    """
    return GatewayEndpoints(_shared_model(context, model))


def _shared_model(context: object, fields: Mapping[str, object]) -> GatewayModel:
    model = GatewayModel(**fields)  # type: ignore[arg-type]
    shared = getattr(context, "shared", None)
    if isinstance(shared, dict) and shared.setdefault(f"gateway:{model.checksum}", model) != model:
        raise ContractError(f"the config describes gateway model {model.model} two different ways")
    return model
