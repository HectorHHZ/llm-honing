"""A model behind a hosted OpenAI-compatible API (OpenRouter, a provider's endpoint) as a Honing policy.

Honing holds no weights for such a model, so it gets a stand-in snapshot whose
checksum comes from the endpoint and model name. That is enough to run and gate
harness moves against it; training, pruning, or quantizing it means nothing.
Core has no weight format for hosted models, so the stand-in reuses the master
format and marks its location ``hosted://``.
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
class HostedModel:
    """One model at one OpenAI-compatible endpoint.

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
        require(bool(self.model), "HostedModel.model must name the served model")
        for price in (self.input_usd_per_million, self.output_usd_per_million):
            require(price is None or price >= 0.0, f"prices must be >= 0, got {price}")

    @property
    def checksum(self) -> str:
        """The stand-in checksum: stable for one endpoint and model name."""
        digest = hashlib.sha256(f"{self.base_url.rstrip('/')}|{self.model}".encode()).hexdigest()
        return f"hosted-{digest[:16]}"

    def snapshot(self) -> ModelSnapshot:
        """Returns the stand-in snapshot a release holds for this model."""
        checkpoint = CheckpointRef(
            location=f"hosted://{self.base_url.split('://', 1)[1].rstrip('/')}/{self.model}",
            weight_format=MASTER_WEIGHT_FORMAT,
            checksum=self.checksum,
        )
        return ModelSnapshot(checkpoint=checkpoint, architecture=_PLACEHOLDER, parameters=1)

    def endpoint(self) -> PolicyEndpoint:
        """Returns the endpoint; ``sglang`` is core's runtime for OpenAI-compatible chat."""
        return PolicyEndpoint(
            base_url=self.base_url.rstrip("/"), tokenizer_location="hosted", served_model=self.model, runtime="sglang"
        )


class HostedModelSource:
    """Satisfies :class:`~honing.core.ModelSource` with a hosted model's stand-in snapshot.

    Args:
        model: The hosted model.
    """

    def __init__(self, model: HostedModel) -> None:
        self._model = model

    def materialize(self, work_dir: Path) -> ModelSnapshot:
        """Returns the stand-in snapshot.

        Args:
            work_dir: Unused; nothing is downloaded.
        """
        return self._model.snapshot()


class HostedEndpoints:
    """Satisfies :class:`~honing.core.PolicyEndpoints` for hosted models.

    Args:
        *models: The hosted models a campaign may run against.

    Raises:
        ContractError: No model was given.
    """

    def __init__(self, *models: HostedModel) -> None:
        require(len(models) > 0, "HostedEndpoints needs at least one model")
        self._models = {model.checksum: model for model in models}

    def resolve(self, policy: PolicyRef) -> PolicyEndpoint:
        """Returns the endpoint serving ``policy``.

        Args:
            policy: A hosted model's stand-in policy.

        Raises:
            ContractError: No hosted model has that checksum.
        """
        return self.hosted(policy).endpoint()

    def hosted(self, policy: PolicyRef) -> HostedModel:
        """Returns the hosted model behind ``policy``, for a runner that needs its key and prices.

        Args:
            policy: A hosted model's stand-in policy.

        Raises:
            ContractError: No hosted model has that checksum.
        """
        try:
            return self._models[policy.checkpoint.checksum]
        except KeyError:
            raise ContractError(f"no hosted model serves checkpoint {policy.checkpoint.checksum!r}") from None


def model_source(context: object, **model: object) -> HostedModelSource:
    """Builds the model source for a pipeline config.

    Args:
        context: The pipeline context.
        **model: :class:`HostedModel` fields.

    Returns:
        The model source.
    """
    return HostedModelSource(_shared_model(context, model))


def endpoints(context: object, **model: object) -> HostedEndpoints:
    """Builds the endpoints for a pipeline config, from the same fields as :func:`model_source`.

    Args:
        context: The pipeline context.
        **model: :class:`HostedModel` fields.

    Returns:
        The endpoints.
    """
    return HostedEndpoints(_shared_model(context, model))


def _shared_model(context: object, fields: Mapping[str, object]) -> HostedModel:
    model = HostedModel(**fields)  # type: ignore[arg-type]
    shared = getattr(context, "shared", None)
    if isinstance(shared, dict) and shared.setdefault(f"hosted:{model.checksum}", model) != model:
        raise ContractError(f"the config describes hosted model {model.model} two different ways")
    return model
