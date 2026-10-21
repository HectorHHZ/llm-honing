"""StaticPolicyEndpoints: a fixed map from checkpoint checksum to an engine Honing did not start."""

from __future__ import annotations

import pytest

from honing.core import ArchitectureSpec, CheckpointRef, ContractError, PolicyEndpoint, PolicyRef
from honing.harness.endpoints import StaticEndpoint, StaticPolicyEndpoints, static_endpoints

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
