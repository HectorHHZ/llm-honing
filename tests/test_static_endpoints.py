"""StaticPolicyEndpoints: a fixed map from checkpoint checksum to an engine Honing did not start."""

from __future__ import annotations

import email.message
import http.client
import socket
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from honing.core import ArchitectureSpec, CheckpointRef, ContractError, PolicyEndpoint, PolicyRef
from honing.harness.endpoints import (
    HEALTH_TIMEOUT_SECONDS,
    StaticEndpoint,
    StaticPolicyEndpoints,
    server_answers,
    static_endpoints,
)

SECRET = "sk-test-value-that-must-not-leak"
ARCHITECTURE = ArchitectureSpec.uniform_layers(num_layers=1, hidden_size=8, attention_heads=1, ffn_size=16)


def policy(checksum: str) -> PolicyRef:
    return PolicyRef(
        checkpoint=CheckpointRef(location=f"/ckpt/{checksum}", weight_format="hf_safetensors", checksum=checksum),
        architecture=ARCHITECTURE,
        variant="master",
    )


SGLANG = StaticEndpoint(base_url="http://127.0.0.1:30000", served_model="qwen", tokenizer_location="/tok/qwen")
REMOTE = StaticEndpoint(
    base_url="https://api.example.com/v1",
    served_model="gpt-4.1-mini-2025-04-14",
    tokenizer_location="remote",
    api_key_env="TEST_POLICY_KEY",
)


@pytest.mark.unit
def test_a_configured_checkpoint_resolves_to_its_endpoint() -> None:
    endpoints = StaticPolicyEndpoints({"abc": SGLANG}, environ={})
    assert endpoints.resolve(policy("abc")) == PolicyEndpoint(
        base_url="http://127.0.0.1:30000", tokenizer_location="/tok/qwen", served_model="qwen", runtime="sglang"
    )


@pytest.mark.unit
def test_a_jev_decision_entry_keeps_its_runtime() -> None:
    entry = StaticEndpoint(
        base_url="http://127.0.0.1:8100", served_model="jev", tokenizer_location="/tok/jev", runtime="jev_decision"
    )
    assert StaticPolicyEndpoints({"abc": entry}, environ={}).resolve(policy("abc")).runtime == "jev_decision"


@pytest.mark.unit
def test_an_unknown_checkpoint_raises_naming_its_checksum() -> None:
    endpoints = StaticPolicyEndpoints({"abc": SGLANG}, environ={})
    with pytest.raises(ContractError, match="no endpoint is configured for checkpoint 'missing'"):
        endpoints.resolve(policy("missing"))


@pytest.mark.unit
def test_the_default_serves_unknown_checkpoints_but_not_configured_ones() -> None:
    endpoints = StaticPolicyEndpoints({"abc": SGLANG}, default=REMOTE, environ={"TEST_POLICY_KEY": SECRET})
    assert endpoints.resolve(policy("other")).served_model == "gpt-4.1-mini-2025-04-14"
    assert endpoints.resolve(policy("abc")).served_model == "qwen"


@pytest.mark.unit
@pytest.mark.parametrize("as_default", [False, True])
def test_a_missing_key_variable_fails_at_construction_without_a_value(as_default: bool) -> None:
    # The variable is set to an empty string, and another variable holds the secret.
    environ = {"TEST_POLICY_KEY": "", "OTHER": SECRET}
    with pytest.raises(ContractError, match="TEST_POLICY_KEY") as raised:
        if as_default:
            StaticPolicyEndpoints({}, default=REMOTE, environ=environ)
        else:
            StaticPolicyEndpoints({"abc": REMOTE}, environ=environ)
    assert SECRET not in str(raised.value)


@pytest.mark.unit
def test_a_key_variable_removed_after_construction_fails_on_resolve() -> None:
    environ = {"TEST_POLICY_KEY": SECRET}
    endpoints = StaticPolicyEndpoints({"abc": REMOTE}, environ=environ)
    del environ["TEST_POLICY_KEY"]
    with pytest.raises(ContractError, match="TEST_POLICY_KEY") as raised:
        endpoints.resolve(policy("abc"))
    assert SECRET not in str(raised.value)


@pytest.mark.unit
def test_api_key_env_names_the_variable_and_never_the_value() -> None:
    endpoints = StaticPolicyEndpoints({"abc": SGLANG, "key": REMOTE}, environ={"TEST_POLICY_KEY": SECRET})
    assert endpoints.api_key_env(policy("key")) == "TEST_POLICY_KEY"
    assert endpoints.api_key_env(policy("abc")) is None
    with pytest.raises(ContractError, match="'missing'"):
        endpoints.api_key_env(policy("missing"))


@pytest.mark.unit
def test_the_environment_defaults_to_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEST_POLICY_KEY", raising=False)
    with pytest.raises(ContractError, match="TEST_POLICY_KEY"):
        StaticPolicyEndpoints({"abc": REMOTE})
    monkeypatch.setenv("TEST_POLICY_KEY", SECRET)
    assert StaticPolicyEndpoints({"abc": REMOTE}).resolve(policy("abc")).base_url == "https://api.example.com/v1"


@pytest.mark.unit
@pytest.mark.parametrize("url", ["127.0.0.1:30000", "ftp://host", "", "HTTP://host"])
def test_a_base_url_without_an_http_scheme_is_refused(url: str) -> None:
    with pytest.raises(ContractError, match="base_url"):
        StaticEndpoint(base_url=url, served_model="m", tokenizer_location="t")


@pytest.mark.unit
def test_the_factory_builds_from_json_entries() -> None:
    endpoints = static_endpoints(
        None,
        endpoints={
            "abc": {"base_url": "http://127.0.0.1:30000", "served_model": "qwen", "tokenizer_location": "/tok"}
        },
        default={
            "base_url": "https://api.example.com/v1",
            "served_model": "remote-model",
            "tokenizer_location": "remote",
            "api_key_env": None,
        },
    )
    assert endpoints.resolve(policy("abc")).base_url == "http://127.0.0.1:30000"
    assert endpoints.resolve(policy("other")).served_model == "remote-model"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"base_url": "http://h", "served_model": "m", "tokenizer_location": "t", "model": "x"}, "unknown keys"),
        ({"base_url": "http://h", "served_model": "m", "tokenizer_location": "t", "api_key": SECRET}, "api_key_env"),
        ({"base_url": "host:1", "served_model": "m", "tokenizer_location": "t"}, "base_url"),
        ({"base_url": "http://h", "served_model": "m"}, "missing"),
        ({"base_url": "http://h", "served_model": 3, "tokenizer_location": "t"}, "must be a string"),
        ({"base_url": "http://h", "served_model": "m", "tokenizer_location": "t", "runtime": "vllm"}, "runtime"),
        ("http://h", "must be an object"),
    ],
)
def test_the_factory_refuses_a_malformed_entry(entry: object, message: str) -> None:
    with pytest.raises(ContractError, match=message) as raised:
        static_endpoints(None, endpoints={"abc": entry})  # type: ignore[dict-item]
    assert SECRET not in str(raised.value)


@pytest.mark.unit
@pytest.mark.parametrize("endpoints", [[{"base_url": "http://h"}], "http://h"])
def test_the_factory_refuses_endpoints_that_are_not_an_object(endpoints: object) -> None:
    with pytest.raises(ContractError, match="endpoints must be an object"):
        static_endpoints(None, endpoints=endpoints)  # type: ignore[arg-type]


# ---------------------------------------------------------------- health checks


class FakeProbe:
    """Answers health probes in place of HTTP: up unless ``up`` is False; records each call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, float]] = []
        self.up = True

    def __call__(self, url: str, timeout_seconds: float) -> bool:
        self.calls.append((url, timeout_seconds))
        return self.up


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


CHECKED = StaticEndpoint(
    base_url="http://127.0.0.1:30000/v1",
    served_model="qwen",
    tokenizer_location="/tok/qwen",
    health_url="http://127.0.0.1:30000/health",
)


def checked(**entries: StaticEndpoint) -> tuple[StaticPolicyEndpoints, FakeProbe, FakeClock]:
    probe, clock = FakeProbe(), FakeClock()
    default = entries.pop("default", None)
    return StaticPolicyEndpoints(entries, default=default, environ={}, probe=probe, clock=clock), probe, clock


@pytest.mark.unit
def test_an_entry_without_a_health_url_is_never_probed() -> None:
    endpoints, probe, clock = checked(abc=SGLANG)
    endpoints.resolve(policy("abc"))
    clock.now += 1e6
    endpoints.resolve(policy("abc"))
    assert probe.calls == []


@pytest.mark.unit
def test_an_entry_is_probed_on_first_resolve_then_trusted_for_the_ttl() -> None:
    endpoints, probe, clock = checked(abc=CHECKED)
    endpoints.resolve(policy("abc"))
    assert probe.calls == [("http://127.0.0.1:30000/health", 30.0)]
    clock.now += 29.9
    endpoints.resolve(policy("abc"))
    assert len(probe.calls) == 1
    clock.now += 0.1
    endpoints.resolve(policy("abc"))
    assert len(probe.calls) == 2


@pytest.mark.unit
def test_an_endpoint_that_is_down_fails_naming_it_and_is_probed_again_next_time() -> None:
    endpoints, probe, _clock = checked(abc=CHECKED)
    probe.up = False
    with pytest.raises(ContractError, match=r"the endpoint http://127.0.0.1:30000/v1 for checkpoint 'abc'") as raised:
        endpoints.resolve(policy("abc"))
    assert "http://127.0.0.1:30000/health" in str(raised.value) and "30 s" in str(raised.value)
    probe.up = True  # Honing does not run it, so it may come back
    assert endpoints.resolve(policy("abc")).base_url == "http://127.0.0.1:30000/v1"
    assert len(probe.calls) == 2


@pytest.mark.unit
def test_an_endpoint_that_passed_and_then_went_down_fails_after_the_ttl() -> None:
    endpoints, probe, clock = checked(abc=CHECKED)
    endpoints.resolve(policy("abc"))
    probe.up = False
    clock.now += 30.0
    with pytest.raises(ContractError, match="did not answer"):
        endpoints.resolve(policy("abc"))


@pytest.mark.unit
def test_a_default_serving_many_checkpoints_is_probed_once_per_ttl() -> None:
    endpoints, probe, _clock = checked(default=CHECKED)
    endpoints.resolve(policy("a"))
    endpoints.resolve(policy("b"))
    assert len(probe.calls) == 1


@pytest.mark.unit
def test_health_settings_are_checked() -> None:
    with pytest.raises(ContractError, match="health_timeout_seconds"):
        StaticPolicyEndpoints({}, environ={}, health_timeout_seconds=0)
    with pytest.raises(ContractError, match="health_ttl_seconds"):
        StaticPolicyEndpoints({}, environ={}, health_ttl_seconds=-1)
    with pytest.raises(ContractError, match="health_url"):
        StaticEndpoint(base_url="http://h", served_model="m", tokenizer_location="t", health_url="h:1/health")


@pytest.mark.unit
@pytest.mark.parametrize(
    "url",
    ["ftp://h/health", "http:///health", "http://[::1/health", "http://h:99999/health", "http://h:port/health"],
    ids=["scheme", "no-host", "bad-ipv6", "port-out-of-range", "port-not-a-number"],
)
def test_a_health_url_the_probe_could_not_parse_is_refused_at_construction(url: str) -> None:
    with pytest.raises(ContractError, match="health_url"):
        StaticEndpoint(base_url="http://h", served_model="m", tokenizer_location="t", health_url=url)


@pytest.mark.unit
@pytest.mark.parametrize("url", ["http://127.0.0.1:30000/health", "https://h/v1/health", "http://[::1]:30000/health"])
def test_a_well_formed_health_url_is_accepted(url: str) -> None:
    entry = StaticEndpoint(base_url="http://h", served_model="m", tokenizer_location="t", health_url=url)
    assert entry.health_url == url


@pytest.mark.unit
def test_the_factory_reads_health_url_and_refuses_a_bad_one() -> None:
    base = {"base_url": "http://h", "served_model": "m", "tokenizer_location": "t"}
    endpoints = static_endpoints(
        None, endpoints={"a": {**base, "health_url": "http://h/health"}, "b": {**base, "health_url": None}}
    )
    assert endpoints._entry(policy("a")).health_url == "http://h/health"
    assert endpoints._entry(policy("b")).health_url is None
    with pytest.raises(ContractError, match="health_url must be a string"):
        static_endpoints(None, endpoints={"a": {**base, "health_url": 3}})
    with pytest.raises(ContractError, match="health_url"):
        static_endpoints(None, endpoints={"a": {**base, "health_url": "h/health"}})


# ---------------------------------------------------------------- the default probe


class _Answer:
    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self) -> _Answer:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


@pytest.mark.unit
def test_the_default_timeout_leaves_slow_answers_to_sglang_s_own_20_s_limit() -> None:
    assert HEALTH_TIMEOUT_SECONDS > 20


@pytest.mark.unit
@pytest.mark.parametrize(("status", "up"), [(200, True), (204, False)])
def test_the_probe_is_up_only_on_200_and_passes_its_timeout(
    monkeypatch: pytest.MonkeyPatch, status: int, up: bool
) -> None:
    seen: list[tuple[str, float]] = []

    def urlopen(url: str, timeout: float) -> _Answer:
        seen.append((url, timeout))
        return _Answer(status)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assert server_answers("http://h/health", 7.5) is up
    assert seen == [("http://h/health", 7.5)]


@pytest.mark.unit
@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("timed out"),
        TimeoutError("timed out"),
        urllib.error.URLError(ConnectionRefusedError()),
        urllib.error.HTTPError("http://h/health", 503, "Service Unavailable", email.message.Message(), None),
        http.client.RemoteDisconnected("closed"),
        http.client.BadStatusLine("garbage"),
    ],
    ids=["timeout", "socket-timeout", "refused", "503", "disconnected", "bad-status"],
)
def test_the_probe_counts_a_timeout_or_any_failure_as_down(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    def urlopen(url: str, timeout: float) -> _Answer:
        raise error

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assert server_answers("http://h/health", 1.0) is False


@pytest.mark.unit
def test_the_probe_counts_a_url_it_cannot_parse_as_down() -> None:
    # urlsplit raises ValueError here, which is neither an OSError nor an HTTPException.
    assert server_answers("http://[::1/health", 1.0) is False


@pytest.fixture
def stub() -> Iterator[str]:
    """A loopback server: /up answers 200, /sick 503, /hang only once the test ends."""
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def do_GET(self) -> None:
            if self.path == "/hang":
                release.wait(10)
            self.send_response({"/up": 200, "/hang": 200}.get(self.path, 503))
            self.send_header("Content-Length", "0")
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.unit
def test_the_probe_against_a_loopback_stub_that_is_up_sick_hung_or_gone(stub: str) -> None:
    assert server_answers(f"{stub}/up", 5.0) is True
    assert server_answers(f"{stub}/sick", 5.0) is False
    started = time.monotonic()
    assert server_answers(f"{stub}/hang", 0.2) is False
    assert time.monotonic() - started < 5.0
    with socket.socket() as closed:
        closed.bind(("127.0.0.1", 0))
        port = closed.getsockname()[1]
    assert server_answers(f"http://127.0.0.1:{port}/health", 1.0) is False
