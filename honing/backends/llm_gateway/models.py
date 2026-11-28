"""Any model behind an OpenAI-compatible URL, as a Honing policy.

The URL can be a provider's API, a router such as OpenRouter, or a local vLLM or
SGLang server. Honing calls the model but holds no weights for it, so it gets a
stand-in snapshot whose checksum comes from the URL and model name. That is
enough to run and gate harness moves against it; training, pruning, or
quantizing it means nothing. Core has no weight format for such models, so the
stand-in reuses the master format and marks its location ``gateway://``.

With a Reef route, the endpoints send the policy's calls through Reef, which
records each one and forwards it to the model (``honing/backends/reef/chat.py``,
#224). The snapshot and its checksum stay the same, since the model does.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from honing.backends.reef.chat import ReefChatRoute
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
        """Returns the endpoint, an OpenAI-compatible chat API."""
        return PolicyEndpoint(
            base_url=self.base_url.rstrip("/"),
            tokenizer_location="gateway",
            served_model=self.model,
            runtime="openai_chat",
            gpus=0,  # a remote API; its calls are priced by tokens
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

    With ``reef``, a policy's endpoint is the route's URL, and its key is Reef's
    token: Reef forwards each call to the model its deployment names, with the
    provider's key, and records it in the route's scenario. Reef's deployment
    names one model, so the endpoints then hold one. The agents that call a
    policy add no headers, so the route carries no tags, and its scenario
    names the calls. This follows the model binding Reef gives an installed harness
    (``scenario_url``, reef/control/operations/harness.py, Reef bc8c150): the
    scenario's URL, called with the client's own token. Differs: the URL ends
    in ``/v1``, as the agents' OpenAI clients expect.

    Args:
        *models: The gateway models a campaign may run against.
        reef: The Reef route the calls go through, or None to call each model's URL.

    Raises:
        ContractError: No model was given, or a route holds tags or comes with more than one model.
    """

    def __init__(self, *models: GatewayModel, reef: ReefChatRoute | None = None) -> None:
        require(len(models) > 0, "GatewayEndpoints needs at least one model")
        if reef is not None:
            require(len(models) == 1, "Reef forwards every call to the one model its deployment names")
            require(not reef.tags, "an agent sends no Reef tags, so a gateway's Reef route takes none")
        self._models = {model.checksum: model for model in models}
        self._reef = reef

    def resolve(self, policy: PolicyRef) -> PolicyEndpoint:
        """Returns the endpoint serving ``policy``.

        Args:
            policy: A gateway model's stand-in policy.

        Raises:
            ContractError: No gateway model has that checksum.
        """
        endpoint = self.model(policy).endpoint()
        return endpoint if self._reef is None else replace(endpoint, base_url=self._reef.base_url)

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

    def api_key_env(self, policy: PolicyRef) -> str | None:
        """Returns the name of the variable holding the key for ``policy``'s model, or None if it needs none.

        Runners find this method with ``hasattr``; it is not part of :class:`~honing.core.PolicyEndpoints`.

        Args:
            policy: A gateway model's stand-in policy.

        Raises:
            ContractError: No gateway model has that checksum.
        """
        model = self.model(policy)
        return model.api_key_env if self._reef is None else self._reef.token_env


def model_source(context: object, **model: object) -> GatewayModelSource:
    """Builds the model source for a pipeline config.

    Args:
        context: The pipeline context.
        **model: :class:`GatewayModel` fields.

    Returns:
        The model source.
    """
    return GatewayModelSource(_shared_model(context, model))


def endpoints(context: object, reef: Mapping[str, object] | None = None, **model: object) -> GatewayEndpoints:
    """Builds the endpoints for a pipeline config, from the same fields as :func:`model_source`.

    Args:
        context: The pipeline context.
        reef: :class:`~honing.backends.reef.chat.ReefChatRoute` fields, to send the calls through Reef.
        **model: :class:`GatewayModel` fields.

    Returns:
        The endpoints.
    """
    route = None if reef is None else ReefChatRoute(**reef)  # type: ignore[arg-type]
    return GatewayEndpoints(_shared_model(context, model), reef=route)


def _shared_model(context: object, fields: Mapping[str, object]) -> GatewayModel:
    model = GatewayModel(**fields)  # type: ignore[arg-type]
    shared = getattr(context, "shared", None)
    if isinstance(shared, dict) and shared.setdefault(f"gateway:{model.checksum}", model) != model:
        raise ContractError(f"the config describes gateway model {model.model} two different ways")
    return model
