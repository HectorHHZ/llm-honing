"""The agent gateway: the proxy config Honing writes, and the real proxy in front of a chat-only upstream.

The integration tests run LiteLLM's proxy from its own environment when it exists, against a loopback
upstream that answers chat completions only (tests/fakes/chat_upstream.py), as SGLang does.
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

from honing.backends.agent_gateway import PUBLIC_MODEL, AgentGateway, Upstream, default_python, proxy_config
from honing.core import ContractError, PolicyEndpoint
from tests.fakes.chat_upstream import ChatUpstream

REPO = Path(__file__).resolve().parents[1]
PYTHON = default_python(REPO)


def _post(url: str, body: dict[str, object], key: str | None = "gateway-key") -> tuple[int, str]:
    headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
    request = urllib.request.Request(url, json.dumps(body).encode(), headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


@pytest.mark.unit
def test_every_model_name_reaches_the_upstream_and_no_key_is_written(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UPSTREAM_KEY", "secret-upstream")
    monkeypatch.setenv("HONING_GATEWAY_KEY", "secret-gateway")

    config = proxy_config(Upstream("http://gpu:30000/v1", "honed-7b", "UPSTREAM_KEY"))

    names = [entry["model_name"] for entry in config["model_list"]]  # type: ignore[attr-defined]
    params = {json.dumps(entry["litellm_params"]) for entry in config["model_list"]}  # type: ignore[attr-defined]
    assert names == [PUBLIC_MODEL, "*"] and len(params) == 1
    assert json.loads(params.pop()) == {
        "model": "hosted_vllm/honed-7b",
        "api_base": "http://gpu:30000/v1",
        "api_key": "os.environ/UPSTREAM_KEY",
        "use_chat_completions_api": True,
    }
    assert config["general_settings"] == {"master_key": "os.environ/HONING_GATEWAY_KEY"}
    assert "secret" not in json.dumps(config)


@pytest.mark.unit
def test_an_upstream_without_a_key_gets_a_placeholder() -> None:
    config = proxy_config(Upstream("http://127.0.0.1:30000/v1", "honed-7b"))

    assert config["model_list"][0]["litellm_params"]["api_key"] == "unused"  # type: ignore[index]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("runtime", "base_url", "expected"),
    [
        ("sglang", "http://gpu:30000", "http://gpu:30000/v1"),
        ("sglang", "http://127.0.0.1:8000/abc/v1/", "http://127.0.0.1:8000/abc/v1"),
        ("openai_chat", "https://openrouter.ai/api/v1", "https://openrouter.ai/api/v1"),
    ],
)
def test_a_policy_endpoint_becomes_an_upstream(runtime: str, base_url: str, expected: str) -> None:
    endpoint = PolicyEndpoint(base_url=base_url, tokenizer_location="t", served_model="honed-7b", runtime=runtime)  # type: ignore[arg-type]

    assert Upstream.from_endpoint(endpoint, "KEY") == Upstream(expected, "honed-7b", "KEY")


@pytest.mark.unit
def test_a_decision_server_is_not_an_upstream() -> None:
    endpoint = PolicyEndpoint(
        base_url="http://gpu:8011", tokenizer_location="t", served_model="m", runtime="jev_decision"
    )

    with pytest.raises(ContractError, match="OpenAI-compatible endpoint, not a jev_decision one"):
        Upstream.from_endpoint(endpoint)


@pytest.mark.unit
@pytest.mark.parametrize("unset", ["HONING_GATEWAY_KEY", "UPSTREAM_KEY"])
def test_a_missing_key_is_refused_before_the_proxy_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unset: str
) -> None:
    monkeypatch.setenv("HONING_GATEWAY_KEY", "g")
    monkeypatch.setenv("UPSTREAM_KEY", "u")
    monkeypatch.delenv(unset)

    with (
        pytest.raises(ContractError, match=f"needs the environment variable {unset}"),
        AgentGateway.launch(
            Upstream("http://127.0.0.1:1/v1", "m", "UPSTREAM_KEY"), python=_installed(tmp_path), work_dir=tmp_path
        ),
    ):
        pass
    assert not (tmp_path / "litellm.yaml").exists()


@pytest.mark.unit
def test_a_missing_environment_is_refused_with_where_to_install_it(tmp_path: Path) -> None:
    with (
        pytest.raises(ContractError, match=r"no gateway environment at .*integrations/agent_gateway/README\.md"),
        AgentGateway.launch(Upstream("http://127.0.0.1:1/v1", "m"), python=tmp_path / "python", work_dir=tmp_path),
    ):
        pass


@pytest.mark.unit
def test_a_port_already_in_use_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HONING_GATEWAY_KEY", "g")
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]

        with (
            pytest.raises(ContractError, match=f"port {port} is already in use"),
            AgentGateway.launch(
                Upstream("http://127.0.0.1:1/v1", "m"), python=_installed(tmp_path), work_dir=tmp_path, port=port
            ),
        ):
            pass


def _installed(tmp_path: Path) -> Path:
    """A stand-in for the environment's interpreter, for refusals that come before the proxy starts."""
    python = tmp_path / "python"
    python.touch()
    return python


@pytest.fixture(scope="module")
def served(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[AgentGateway, ChatUpstream]]:
    with pytest.MonkeyPatch.context() as env, ChatUpstream() as upstream:
        env.setenv("HONING_GATEWAY_KEY", "gateway-key")
        env.setenv("UPSTREAM_KEY", "upstream-key")
        with AgentGateway.launch(
            Upstream(upstream.url, "honed-7b", "UPSTREAM_KEY"),
            python=PYTHON,
            work_dir=tmp_path_factory.mktemp("gateway"),
            port=_free_port(),
        ) as gateway:
            yield gateway, upstream


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port: int = probe.getsockname()[1]
        return port


requires_proxy = pytest.mark.skipif(not PYTHON.exists(), reason="no agent gateway environment")


@pytest.mark.integration
@pytest.mark.slow
@requires_proxy
@pytest.mark.parametrize(
    ("route", "body"),
    [
        (
            "/v1/messages",
            {"model": "claude-sonnet-4-5", "max_tokens": 20, "messages": [{"role": "user", "content": "hi"}]},
        ),
        ("/v1/responses", {"model": "gpt-5-codex", "input": "hi"}),
        ("/v1/chat/completions", {"model": PUBLIC_MODEL, "messages": [{"role": "user", "content": "hi"}]}),
    ],
)
@pytest.mark.parametrize("stream", [False, True])
def test_each_agent_api_reaches_the_upstream_as_chat_completions(
    served: tuple[AgentGateway, ChatUpstream], route: str, body: dict[str, object], stream: bool
) -> None:
    gateway, upstream = served
    seen = len(upstream.requests)

    status, text = _post(gateway.url + route, {**body, "stream": stream})

    assert status == 200, text
    assert "pong from honed-7b" in text
    (request,) = upstream.requests[seen:]
    assert (request["path"], request["model"], request["authorization"]) == (
        "/v1/chat/completions",
        "honed-7b",
        "Bearer upstream-key",
    )


@pytest.mark.integration
@pytest.mark.slow
@requires_proxy
@pytest.mark.parametrize("key", [None, "wrong-key"])
def test_a_request_without_the_gateway_key_is_refused(
    served: tuple[AgentGateway, ChatUpstream], key: str | None
) -> None:
    gateway, upstream = served
    seen = len(upstream.requests)

    status, _ = _post(gateway.url + "/v1/messages", {"model": "x", "max_tokens": 5, "messages": []}, key=key)

    # No key is a 401. A wrong one is a 400: LiteLLM looks up any key but the master key in a database of
    # virtual keys, which the gateway does not run.
    assert status == (401 if key is None else 400)
    assert len(upstream.requests) == seen


@pytest.mark.integration
@pytest.mark.slow
@requires_proxy
def test_leaving_the_block_stops_the_proxy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HONING_GATEWAY_KEY", "gateway-key")

    with AgentGateway.launch(
        Upstream("http://127.0.0.1:1/v1", "m"), python=PYTHON, work_dir=tmp_path, port=_free_port()
    ) as gateway:
        assert not gateway.exited()

    assert gateway.exited()
    with pytest.raises(OSError):
        urllib.request.urlopen(f"{gateway.url}/health/liveliness", timeout=5)
