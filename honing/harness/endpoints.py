"""Find the SGLang server serving a policy when Honing did not start it.

An SGLang server that is already running for a real checkpoint is declared up
front as a fixed map from checkpoint checksum to endpoint. A hosted API goes
through :class:`~honing.backends.llm_gateway.GatewayEndpoints` instead. An
entry with a ``health_url`` is checked on resolve (:func:`answers_twice`), at
most once per TTL; a server that fails two probes in a row fails the lookup,
and every lookup of it until the TTL has passed.
Honing did not start these servers, so it never starts or stops them.

A server that needs a key names its variable through ``api_key_env(policy)``,
a method a runner finds with ``hasattr``: it is not part of
:class:`~honing.core.PolicyEndpoints`.

A runner resolves through :func:`resolve_endpoint`, which calls ``resolve`` on a
thread, so a probe or a server start never holds up the episodes on its event
loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import http.client
import os
import threading
import time
import urllib.parse
import urllib.request
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields

from honing.core import ContractError, PolicyEndpoint, PolicyEndpoints, PolicyRef, require

# How long a health check's result is kept: a server that passed is trusted without another probe, and one that
# failed fails again at once. A probe takes at least 1 s from a healthy SGLang server and about two
# HEALTH_TIMEOUT_SECONDS from a hung one; it runs off the event loop (resolve_endpoint), but the episodes that ask
# while it runs wait for it. A server Honing started that crashes is still found on the next resolve, by its exit
# (honing.backends.sglang.endpoints), so the long TTL only delays finding a hung one.
HEALTH_TTL_SECONDS = 300.0
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


def answers_twice(probe: Callable[[str, float], bool], url: str, timeout_seconds: float) -> bool:
    """True unless two probes of ``url`` in a row fail: a busy engine can leave a single probe waiting past the
    timeout, and counting it down stops or fails a server other episodes are using. A hung server is therefore found
    within about twice ``timeout_seconds``."""
    return probe(url, timeout_seconds) or probe(url, timeout_seconds)


async def resolve_endpoint(endpoints: PolicyEndpoints, policy: PolicyRef) -> PolicyEndpoint:
    """``endpoints.resolve(policy)`` off the running event loop: what a runner calls instead of ``resolve``.

    A resolve can start, probe or restart a server, minutes for a cold start, so it runs on a thread while the
    loop goes on with the other episodes. Calls on one loop for the same map and policy while one runs share it,
    so a gate's gathered episodes take one thread per policy, not one each. The thread is a daemon: a process that
    exits, on a Ctrl-C too, does not wait for a start in progress, as it would for a pool's worker (joined at
    exit) or the loop's default executor (joined when ``asyncio.run`` ends). A caller cancelled meanwhile stops
    waiting, but the resolve runs on, and whatever it starts stays with the map, which stops it on close.
    """
    loop = asyncio.get_running_loop()
    with _IN_FLIGHT_LOCK:
        in_flight = _IN_FLIGHT.setdefault(loop, {})
    # The map's id is stable while the call is in flight: the thread holds a reference to it.
    key = (id(endpoints), policy)
    future = in_flight.get(key)
    if future is None:
        future = in_flight[key] = loop.create_future()
        future.add_done_callback(_retrieve)
        threading.Thread(
            target=_resolve_on_thread,
            args=(endpoints, policy, loop, future, in_flight, key),
            name="honing-resolve",
            daemon=True,
        ).start()
    return await asyncio.shield(future)


_InFlight = dict[tuple[int, PolicyRef], "asyncio.Future[PolicyEndpoint]"]
# Per event loop, the resolves running now: a campaign runs each gate in its own asyncio.run, and a future belongs
# to one loop.
_IN_FLIGHT: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _InFlight] = weakref.WeakKeyDictionary()
_IN_FLIGHT_LOCK = threading.Lock()


def _resolve_on_thread(
    endpoints: PolicyEndpoints,
    policy: PolicyRef,
    loop: asyncio.AbstractEventLoop,
    future: asyncio.Future[PolicyEndpoint],
    in_flight: _InFlight,
    key: tuple[int, PolicyRef],
) -> None:
    answer: PolicyEndpoint | BaseException
    try:
        answer = endpoints.resolve(policy)
    except BaseException as error:  # handed to the callers on the loop
        answer = error

    def settle() -> None:
        in_flight.pop(key, None)
        if future.done():
            return
        if isinstance(answer, BaseException):
            future.set_exception(answer)
        else:
            future.set_result(answer)

    # A loop that has closed (its asyncio.run ended with every caller cancelled) has no one left to tell.
    with contextlib.suppress(RuntimeError):
        loop.call_soon_threadsafe(settle)


def _retrieve(future: asyncio.Future[PolicyEndpoint]) -> None:
    """Mark a failure retrieved, so a resolve whose callers were all cancelled logs nothing when it fails."""
    if not future.cancelled():
        future.exception()


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
    """One declared SGLang server; its endpoint's runtime is always ``"sglang"``.

    ``tokenizer_location`` is the checkpoint's tokenizer. ``api_key_env`` is
    the *name* of the environment variable holding the key of a server started
    with ``--api-key``, never the key itself; :class:`~honing.core.PolicyEndpoint`
    has no key field, so runners read the name through
    :meth:`StaticPolicyEndpoints.api_key_env`. ``health_url``, when set, is the
    route checked on resolve (the server's ``/health``); an entry without one is
    never checked.
    """

    base_url: str
    served_model: str
    tokenizer_location: str
    api_key_env: str | None = None
    health_url: str | None = None

    def __post_init__(self) -> None:
        require(
            self.base_url.startswith(("http://", "https://")),
            f"StaticEndpoint.base_url must start with http:// or https://, got {self.base_url!r}",
        )
        require(bool(self.served_model), "StaticEndpoint.served_model must be set")
        require(bool(self.tokenizer_location), "StaticEndpoint.tokenizer_location must be set")
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
            runtime="sglang",
        )


class StaticPolicyEndpoints:
    """A fixed map from checkpoint checksum to the SGLang server serving it.

    Satisfies :class:`~honing.core.PolicyEndpoints`. Each key variable is
    checked when the map is built, so a missing key fails before any episode,
    and again on every resolve. Only ``environ`` (``os.environ`` unless given)
    is read.

    An entry with a ``health_url`` is probed on resolve unless that URL was
    checked less than ``health_ttl_seconds`` ago, and the result is kept that
    long. One that fails two probes in a row, each given
    ``health_timeout_seconds`` to answer 200, raises
    :class:`~honing.core.ContractError` naming the endpoint, and so does every
    resolve against that URL until the TTL has passed, without probing again:
    each of the episodes a runner gathers would otherwise wait out two
    timeouts. One URL is probed by one thread at a time; a resolve that comes
    meanwhile waits and takes its result. After the TTL it is probed
    again, since Honing does not run the server and it may have come back.
    ``probe`` and ``clock`` stand in for the network and the time in tests.
    """

    def __init__(
        self,
        endpoints: Mapping[str, StaticEndpoint],
        *,
        environ: Mapping[str, str] | None = None,
        health_ttl_seconds: float = HEALTH_TTL_SECONDS,
        health_timeout_seconds: float = HEALTH_TIMEOUT_SECONDS,
        probe: Callable[[str, float], bool] = server_answers,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        require(health_ttl_seconds >= 0, f"health_ttl_seconds must be >= 0, got {health_ttl_seconds}")
        require(health_timeout_seconds > 0, f"health_timeout_seconds must be > 0, got {health_timeout_seconds}")
        self._endpoints = dict(endpoints)
        self._environ = os.environ if environ is None else environ
        self._ttl = health_ttl_seconds
        self._timeout = health_timeout_seconds
        self._probe = probe
        self._clock = clock
        # Health URL to when it was last checked and whether it answered. Keyed by URL, so one server declared for
        # several checkpoints is checked once per TTL.
        self._checked: dict[str, tuple[float, bool]] = {}
        # One lock per health URL: runners resolve on threads, and two checks of one URL at once would probe it twice.
        self._probing = {entry.health_url: threading.Lock() for entry in self._endpoints.values() if entry.health_url}
        for checksum, entry in self._endpoints.items():
            self._check_key(entry, f"the endpoint for checkpoint {checksum!r}")

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
        entry = self._endpoints.get(checksum)
        if entry is None:
            raise ContractError(
                f"no endpoint is configured for checkpoint {checksum!r} at {policy.checkpoint.location}"
            )
        return entry

    def _check_health(self, entry: StaticEndpoint, health_url: str, policy: PolicyRef) -> None:
        with self._probing[health_url]:
            last = self._checked.get(health_url)
            if last is None or self._clock() - last[0] >= self._ttl:
                answered = answers_twice(self._probe, health_url, self._timeout)
                last = self._checked[health_url] = (self._clock(), answered)
        if not last[1]:
            raise ContractError(
                f"the endpoint {entry.base_url} for checkpoint {policy.checkpoint.checksum!r} did not answer "
                f"{health_url} with 200 within {self._timeout:g} s, twice, {self._clock() - last[0]:.0f} s ago "
                f"(checked again {self._ttl:g} s after that)"
            )

    def _check_key(self, entry: StaticEndpoint, label: str) -> None:
        # The message names the variable only; the value never appears in an error.
        if entry.api_key_env is not None and not self._environ.get(entry.api_key_env):
            raise ContractError(f"{label} needs the environment variable {entry.api_key_env}, which is unset or empty")


_ENTRY_KEYS = frozenset(field.name for field in fields(StaticEndpoint))
_OPTIONAL_KEYS = frozenset({"api_key_env", "health_url"})


def static_endpoints(
    context: object,
    endpoints: Mapping[str, Mapping[str, object]],
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
        api_key_env=None if api_key_env is None else str(api_key_env),
        health_url=None if health_url is None else str(health_url),
    )
