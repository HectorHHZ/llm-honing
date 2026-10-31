"""Find the engine serving a policy when Honing did not start it.

An SGLang server that is already running, or a remote API, is declared up front
as a fixed map from checkpoint checksum to endpoint. An entry with a
``health_url`` is checked on resolve (:func:`server_answers`), at most once per
TTL; a server that does not answer fails the lookup. Honing did not start these
servers, so it never starts or stops them.
"""

from __future__ import annotations

import http.client
import os
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from typing import cast, get_args

from honing.core import ContractError, EndpointRuntime, PolicyEndpoint, PolicyRef, require, require_choice

# How long a server that passed its health check is trusted without another, so a step resolving once per episode
# does not probe once per episode.
HEALTH_TTL_SECONDS = 30.0
# Above SGLang's own limit: at 0.5.21 its /health runs a one-token generation (SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION
# is on by default), takes at least 1 s, and answers 503 by itself after SGLANG_HEALTH_CHECK_TIMEOUT (default 20 s)
# without scheduler progress. A shorter client timeout would count a busy but healthy engine as down; this one leaves
# the slow cases to SGLang's 503.
HEALTH_TIMEOUT_SECONDS = 30.0


def server_answers(url: str, timeout_seconds: float) -> bool:
    """True if ``GET url`` answers 200 within ``timeout_seconds``; a refused connection, a timeout, a dropped
    connection, a URL that cannot be parsed or any other status is False.

    ``timeout_seconds`` bounds each socket operation, not the whole request: a server that keeps sending a byte at a
    time can hold the probe longer.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            return bool(response.status == 200)
    except (OSError, http.client.HTTPException, ValueError):
        return False


def _is_http_url(url: str) -> bool:
    """An http or https URL with a host and a port that parses, so probing it cannot fail on the URL itself."""
    try:
        parts = urllib.parse.urlsplit(url)  # ValueError on a malformed IPv6 host
        _ = parts.port  # ValueError on a port that is not a number or is out of range
    except ValueError:
        return False
    return parts.scheme in ("http", "https") and bool(parts.hostname)


@dataclass(frozen=True)
class StaticEndpoint:
    """One declared endpoint.

    ``tokenizer_location`` has no default: an SGLang entry needs a real one, and
    a remote API with no local tokenizer writes something like ``"remote"``.
    ``api_key_env`` is the *name* of the environment variable holding the key,
    never the key itself; :class:`~honing.core.PolicyEndpoint` has no key field,
    so runners read the name through :meth:`StaticPolicyEndpoints.api_key_env`.
    ``health_url``, when set, is the route checked on resolve (an SGLang
    server's ``/health``, a decision server's ``/v1/health``); a remote API
    without one leaves it unset and is never checked.
    """

    base_url: str
    served_model: str
    tokenizer_location: str
    runtime: EndpointRuntime = "sglang"
    api_key_env: str | None = None
    health_url: str | None = None

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
        require(
            self.health_url is None or _is_http_url(self.health_url),
            f"StaticEndpoint.health_url must be None or an http:// or https:// URL with a host and a valid port, "
            f"got {self.health_url!r}",
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

    An entry with a ``health_url`` is probed on resolve unless that URL passed
    less than ``health_ttl_seconds`` ago; one that does not answer 200 within
    ``health_timeout_seconds`` raises :class:`~honing.core.ContractError`
    naming the endpoint. A failure is not remembered: Honing does not run the
    server, which may come back. ``probe`` and ``clock`` stand in for the
    network and the time in tests.
    """

    def __init__(
        self,
        endpoints: Mapping[str, StaticEndpoint],
        *,
        default: StaticEndpoint | None = None,
        environ: Mapping[str, str] | None = None,
        health_ttl_seconds: float = HEALTH_TTL_SECONDS,
        health_timeout_seconds: float = HEALTH_TIMEOUT_SECONDS,
        probe: Callable[[str, float], bool] = server_answers,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        require(health_ttl_seconds >= 0, f"health_ttl_seconds must be >= 0, got {health_ttl_seconds}")
        require(health_timeout_seconds > 0, f"health_timeout_seconds must be > 0, got {health_timeout_seconds}")
        self._endpoints = dict(endpoints)
        self._default = default
        self._environ = os.environ if environ is None else environ
        self._ttl = health_ttl_seconds
        self._timeout = health_timeout_seconds
        self._probe = probe
        self._clock = clock
        # Keyed by health URL, so a default entry serving many checkpoints is checked once per TTL.
        self._checked: dict[str, float] = {}
        for checksum, entry in self._endpoints.items():
            self._check_key(entry, f"the endpoint for checkpoint {checksum!r}")
        if default is not None:
            self._check_key(default, "the default endpoint")

    def resolve(self, policy: PolicyRef) -> PolicyEndpoint:
        entry = self._entry(policy)
        self._check_key(entry, f"the endpoint for checkpoint {policy.checkpoint.checksum!r}")
        if entry.health_url is not None:
            self._check_health(entry, entry.health_url, policy)
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

    def _check_health(self, entry: StaticEndpoint, health_url: str, policy: PolicyRef) -> None:
        checked = self._checked.get(health_url)
        if checked is not None and self._clock() - checked < self._ttl:
            return
        if not self._probe(health_url, self._timeout):
            raise ContractError(
                f"the endpoint {entry.base_url} for checkpoint {policy.checkpoint.checksum!r} did not answer "
                f"{health_url} with 200 within {self._timeout:g} s"
            )
        self._checked[health_url] = self._clock()

    def _check_key(self, entry: StaticEndpoint, label: str) -> None:
        # The message names the variable only; the value never appears in an error.
        if entry.api_key_env is not None and not self._environ.get(entry.api_key_env):
            raise ContractError(f"{label} needs the environment variable {entry.api_key_env}, which is unset or empty")


_ENTRY_KEYS = frozenset(field.name for field in fields(StaticEndpoint))
_OPTIONAL_KEYS = frozenset({"api_key_env", "health_url"})


def static_endpoints(
    context: object,
    endpoints: Mapping[str, Mapping[str, object]],
    default: Mapping[str, object] | None = None,
    health_ttl_seconds: float = HEALTH_TTL_SECONDS,
    health_timeout_seconds: float = HEALTH_TIMEOUT_SECONDS,
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
        health_ttl_seconds=health_ttl_seconds,
        health_timeout_seconds=health_timeout_seconds,
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
            isinstance(value, str) or (name in _OPTIONAL_KEYS and value is None),
            f"{label}'s {name} must be a string",
        )
    api_key_env = entry.get("api_key_env")
    health_url = entry.get("health_url")
    return StaticEndpoint(
        base_url=str(entry["base_url"]),
        served_model=str(entry["served_model"]),
        tokenizer_location=str(entry["tokenizer_location"]),
        runtime=cast(EndpointRuntime, entry.get("runtime", "sglang")),
        api_key_env=None if api_key_env is None else str(api_key_env),
        health_url=None if health_url is None else str(health_url),
    )
