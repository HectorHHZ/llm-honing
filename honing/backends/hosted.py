"""A model behind a hosted OpenAI-compatible API, as a policy a harness runner can call.

Honing's releases are built around weights it holds: a release's master is a
Hugging Face safetensors checkpoint. A model reached through an API (OpenRouter,
a provider's own endpoint) has no weights Honing can read, so this module gives
it a stand-in snapshot whose checksum is derived from the endpoint and the model
name. That is enough to run and gate harness moves against the model, which is
what the harness plane needs from it. It cannot be trained, pruned, or quantized,
and nothing here pretends otherwise: the snapshot's location is ``hosted://...``.

Core has no weight format for a hosted model yet; the stand-in reuses the master
format so a release can hold it. Representing hosted models properly is a core
change for its own issue.

The key is never stored. ``api_key_env`` names the environment variable a runner
reads it from.
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

# One layer of size one: the snapshot is an identifier, not a description of the model's shape.
_PLACEHOLDER = ArchitectureSpec.uniform_layers(num_layers=1, hidden_size=1, attention_heads=1, ffn_size=1)


@dataclass(frozen=True)
class HostedModel:
    """One model at one OpenAI-compatible endpoint, with its prices in US dollars per million tokens."""

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
        """Stable for one endpoint and model name; a different model is a different policy."""
        digest = hashlib.sha256(f"{self.base_url.rstrip('/')}|{self.model}".encode()).hexdigest()
        return f"hosted-{digest[:16]}"

    def snapshot(self) -> ModelSnapshot:
        checkpoint = CheckpointRef(
            location=f"hosted://{self.base_url.split('://', 1)[1].rstrip('/')}/{self.model}",
            weight_format=MASTER_WEIGHT_FORMAT,
            checksum=self.checksum,
        )
        return ModelSnapshot(checkpoint=checkpoint, architecture=_PLACEHOLDER, parameters=1)

    def endpoint(self) -> PolicyEndpoint:
        # "sglang" is the runtime that speaks OpenAI-compatible chat completions; core has no other.
        return PolicyEndpoint(
            base_url=self.base_url.rstrip("/"),
            tokenizer_location="hosted",
            served_model=self.model,
            runtime="sglang",
        )


class HostedModelSource:
    """Satisfies :class:`~honing.core.ModelSource`: the hosted model's stand-in snapshot."""

    def __init__(self, model: HostedModel) -> None:
        self.model = model

    def materialize(self, work_dir: Path) -> ModelSnapshot:
        return self.model.snapshot()


class HostedEndpoints:
    """Satisfies :class:`~honing.core.PolicyEndpoints` for hosted models, keyed by their stand-in checksums."""

    def __init__(self, *models: HostedModel) -> None:
        require(len(models) > 0, "HostedEndpoints needs at least one model")
        self.models = {model.checksum: model for model in models}

    def resolve(self, policy: PolicyRef) -> PolicyEndpoint:
        return self.hosted(policy).endpoint()

    def hosted(self, policy: PolicyRef) -> HostedModel:
        """The hosted model behind a policy, for a runner that also needs its key and prices."""
        try:
            return self.models[policy.checkpoint.checksum]
        except KeyError:
            raise ContractError(f"no hosted model serves checkpoint {policy.checkpoint.checksum!r}") from None


# ---------------------------------------------------------------- factories for a pipeline config


def _model(context: object, args: Mapping[str, object]) -> HostedModel:
    """One HostedModel per run, shared by the model source and the endpoints built from the same args."""
    shared = getattr(context, "shared", None)
    model = HostedModel(**args)  # type: ignore[arg-type]
    if isinstance(shared, dict):
        existing = shared.setdefault(f"hosted:{model.checksum}", model)
        if existing != model:
            raise ContractError(f"two configs name hosted model {model.model} differently")
    return model


def model_source(context: object, **model: object) -> HostedModelSource:
    """``{"base_url": ..., "model": ..., "api_key_env": ..., "input_usd_per_million": ..., ...}``"""
    return HostedModelSource(_model(context, model))


def endpoints(context: object, **model: object) -> HostedEndpoints:
    """The same args as :func:`model_source`."""
    return HostedEndpoints(_model(context, model))
