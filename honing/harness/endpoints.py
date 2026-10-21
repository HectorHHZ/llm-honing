"""Find the engine serving a policy when Honing did not start it.

An SGLang server that is already running, or a remote API, is declared up front
as a fixed map from checkpoint checksum to endpoint. Starting, stopping and
health-checking servers is not done here.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import cast, get_args

from honing.core import ContractError, EndpointRuntime, PolicyEndpoint, PolicyRef, require, require_choice


@dataclass(frozen=True)
class StaticEndpoint:
    """One declared endpoint.

    ``tokenizer_location`` has no default: an SGLang entry needs a real one, and
    a remote API with no local tokenizer writes something like ``"remote"``.
    ``api_key_env`` is the *name* of the environment variable holding the key,
    never the key itself; :class:`~honing.core.PolicyEndpoint` has no key field,
    so runners read the name through :meth:`StaticPolicyEndpoints.api_key_env`.
    """

    base_url: str
    served_model: str
    tokenizer_location: str
    runtime: EndpointRuntime = "sglang"
    api_key_env: str | None = None

    def __post_init__(self) -> None:
        require(
            self.base_url.startswith(("http://", "https://")),
            f"StaticEndpoint.base_url must start with http:// or https://, got {self.base_url!r}",
        )
        require(bool(self.served_model), "StaticEndpoint.served_model must be set")
        require(bool(self.tokenizer_location), "StaticEndpoint.tokenizer_location must be set")
        require_choice(self.runtime, get_args(EndpointRuntime), "StaticEndpoint.runtime")
        require(
            self.api_key_env is None or bool(self.api_key_env), "StaticEndpoint.api_key_env must be None or a name"
        )

    def endpoint(self) -> PolicyEndpoint:
        return PolicyEndpoint(
            base_url=self.base_url,
            tokenizer_location=self.tokenizer_location,
            served_model=self.served_model,
            runtime=self.runtime,
        )


class StaticPolicyEndpoints:
    """A fixed map from checkpoint checksum to the engine serving it.

    Satisfies :class:`~honing.core.PolicyEndpoints`. ``default``, when set,
    serves every checkpoint the map does not name, for the case where all
    policies go to one remote API. Each key variable is checked when the map is
    built, so a missing key fails before any episode, and again on every
    resolve. Only ``environ`` (``os.environ`` unless given) is read.
    """

    def __init__(
        self,
        endpoints: Mapping[str, StaticEndpoint],
        *,
        default: StaticEndpoint | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self._endpoints = dict(endpoints)
        self._default = default
        self._environ = os.environ if environ is None else environ
        for checksum, entry in self._endpoints.items():
            self._check_key(entry, f"the endpoint for checkpoint {checksum!r}")
        if default is not None:
            self._check_key(default, "the default endpoint")

    def resolve(self, policy: PolicyRef) -> PolicyEndpoint:
        entry = self._entry(policy)
        self._check_key(entry, f"the endpoint for checkpoint {policy.checkpoint.checksum!r}")
        return entry.endpoint()

    def api_key_env(self, policy: PolicyRef) -> str | None:
        """The name of the variable holding the key for ``policy``'s endpoint, or None if it needs none."""
        return self._entry(policy).api_key_env

    def _entry(self, policy: PolicyRef) -> StaticEndpoint:
        checksum = policy.checkpoint.checksum
        entry = self._endpoints.get(checksum, self._default)
        if entry is None:
            raise ContractError(
                f"no endpoint is configured for checkpoint {checksum!r} at {policy.checkpoint.location}"
            )
        return entry

    def _check_key(self, entry: StaticEndpoint, label: str) -> None:
        # The message names the variable only; the value never appears in an error.
        if entry.api_key_env is not None and not self._environ.get(entry.api_key_env):
            raise ContractError(f"{label} needs the environment variable {entry.api_key_env}, which is unset or empty")


_ENTRY_KEYS = frozenset(field.name for field in fields(StaticEndpoint))


def static_endpoints(
    context: object,
    endpoints: Mapping[str, Mapping[str, object]],
    default: Mapping[str, object] | None = None,
) -> StaticPolicyEndpoints:
    """Pipeline factory: build a :class:`StaticPolicyEndpoints` from a config's JSON entries.

    ``endpoints`` maps checkpoint checksum to an entry with the fields of
    :class:`StaticEndpoint`. ``context`` is unused.
    """
    del context
    if not isinstance(endpoints, Mapping):
        raise ContractError("endpoints must be an object")
    return StaticPolicyEndpoints(
        {checksum: _entry_from_json(entry, f"endpoint {checksum!r}") for checksum, entry in endpoints.items()},
        default=None if default is None else _entry_from_json(default, "the default endpoint"),
    )


def _entry_from_json(entry: object, label: str) -> StaticEndpoint:
    if not isinstance(entry, Mapping):
        raise ContractError(f"{label} must be an object")
    if "api_key" in entry:
        raise ContractError(
            f"{label} holds an api_key; store the key in an environment variable and name it with api_key_env"
        )
    unknown = sorted(set(entry) - _ENTRY_KEYS)
    require(not unknown, f"{label} has unknown keys {unknown}")
    missing = sorted({"base_url", "served_model", "tokenizer_location"} - set(entry))
    require(not missing, f"{label} is missing {missing}")
    for name, value in entry.items():
        require(
            isinstance(value, str) or (name == "api_key_env" and value is None),
            f"{label}'s {name} must be a string",
        )
    api_key_env = entry.get("api_key_env")
    return StaticEndpoint(
        base_url=str(entry["base_url"]),
        served_model=str(entry["served_model"]),
        tokenizer_location=str(entry["tokenizer_location"]),
        runtime=cast(EndpointRuntime, entry.get("runtime", "sglang")),
        api_key_env=None if api_key_env is None else str(api_key_env),
    )
