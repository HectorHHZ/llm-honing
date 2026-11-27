"""Chat calls through Reef (#224): the route's URL and headers, the clients that take it, and a real Reef.

The integration tests run Reef in its own Python 3.12 environment, in front of the stand-in OpenAI-compatible
server ``tests/fakes/chat_upstream.py``, the way ``reef serve --inference.upstream-url`` fronts a provider.
Set ``HONING_REEF_BIN`` to the ``reef`` executable of an environment with Reef bc8c150 (the CI ``reef`` job does).
Without it they are skipped. The ``ChatModel`` tests also need the ``litellm`` extra.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from honing.backends.llm_gateway import GatewayEndpoints, GatewayModel, endpoints
from honing.backends.reef import ReefChatRoute, chat_model
from honing.core import ChatMessage, ContractError, PolicyRef
from tests.fakes.chat_upstream import ChatUpstream

# Read when litellm is imported, which the unit tests below do: use the price
# map bundled with the pinned version instead of downloading the latest one.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
REEF = os.environ.get("HONING_REEF_BIN")
needs_reef = pytest.mark.skipif(not REEF, reason="HONING_REEF_BIN names no Reef executable")
TOKEN = "reef-test"
UPSTREAM_MODEL = "stand-in"
TAGS = {"episode": "ep-1", "task": "t1", "harness": "h1", "policy": "p1"}
ROUTE = {"url": "http://127.0.0.1:8977", "scenario": "reflection", "tags": TAGS}
MODEL = GatewayModel(base_url="https://example.invalid/api/v1", model="vendor/model", api_key_env="PROVIDER_KEY")


def policy(model: GatewayModel = MODEL) -> PolicyRef:
    snapshot = model.snapshot()
    return PolicyRef(snapshot.checkpoint, snapshot.architecture, "master")


@pytest.mark.unit
def test_a_route_names_its_scenario_in_the_url_and_sends_honings_tags_as_headers() -> None:
    route = ReefChatRoute(**ROUTE)

    assert route.base_url == "http://127.0.0.1:8977/reef/scenarios/reflection/v1"
    assert route.headers() == {
        "x-reef-tag-episode": "ep-1",
        "x-reef-tag-task": "t1",
        "x-reef-tag-harness": "h1",
        "x-reef-tag-policy": "p1",
    }
    assert route.token_env == "REEF_TOKEN"
    assert ReefChatRoute("https://reef.example/", "a b").base_url == "https://reef.example/reef/scenarios/a%20b/v1"
    assert ReefChatRoute("http://reef", "s").headers() == {}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"url": "reef:8977"}, "HTTP URL"),
        ({"scenario": ""}, "scenario"),
        ({"scenario": "a/b"}, "scenario"),
        ({"scenario": ".."}, "scenario"),
        ({"scenario": " s"}, "scenario"),
        ({"tags": ["task"]}, "tags must be an object"),
        ({"tags": {"seat": "blue"}}, "unknown tags"),
        ({"tags": {"task": ""}}, "task tag"),
        ({"token_env": ""}, "token_env"),
    ],
)
def test_a_route_refuses_what_reef_or_the_record_format_would_not_take(fields: dict[str, Any], message: str) -> None:
    with pytest.raises(ContractError, match=message):
        ReefChatRoute(**{**ROUTE, **fields})


@pytest.mark.unit
def test_without_reef_the_chat_model_is_the_plain_client() -> None:
    pytest.importorskip("litellm", reason="the chat client needs the 'litellm' extra")
    from honing.harness.models import LiteLLMChatModel

    fields = {"model": "hosted_vllm/honed", "api_base": "http://gpu-node:30000/v1", "timeout_seconds": 5.0}

    assert chat_model(None, **fields) == LiteLLMChatModel(**fields)  # type: ignore[arg-type]
    assert chat_model(None, **fields).extra_headers is None


@pytest.mark.unit
def test_through_reef_the_chat_model_calls_the_route_with_reefs_token_and_the_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("litellm", reason="the chat client needs the 'litellm' extra")
    monkeypatch.setenv("REEF_TOKEN", TOKEN)

    chat = chat_model(None, model="hosted_vllm/honed", reef=ROUTE, timeout_seconds=5.0)

    assert chat.api_base == "http://127.0.0.1:8977/reef/scenarios/reflection/v1"
    assert chat.api_key_env == "REEF_TOKEN"
    assert chat.extra_headers == ReefChatRoute(**ROUTE).headers()
    assert chat.timeout_seconds == 5.0
    assert chat_model(None, model="hosted_vllm/honed", reef=ReefChatRoute(**ROUTE)) == chat_model(
        None, model="hosted_vllm/honed", reef=ROUTE
    )


@pytest.mark.unit
@pytest.mark.parametrize("field", ["api_base", "api_key_env", "extra_headers"])
def test_through_reef_the_chat_model_names_no_server_or_key_of_its_own(field: str) -> None:
    pytest.importorskip("litellm", reason="the chat client needs the 'litellm' extra")
    with pytest.raises(ContractError, match=field):
        chat_model(None, model="hosted_vllm/honed", reef=ROUTE, **{field: "x"})


@pytest.mark.unit
def test_without_reef_a_gateway_policy_calls_its_model() -> None:
    gateway = endpoints(None, base_url=MODEL.base_url, model=MODEL.model, api_key_env=MODEL.api_key_env)

    assert gateway.resolve(policy()) == MODEL.endpoint()
    assert gateway.api_key_env(policy()) == "PROVIDER_KEY"


@pytest.mark.unit
def test_through_reef_a_gateway_policy_calls_the_route_with_reefs_token_and_keeps_its_identity() -> None:
    route = {"url": "http://127.0.0.1:8977", "scenario": "officeqa"}
    gateway = endpoints(None, reef=route, base_url=MODEL.base_url, model=MODEL.model, api_key_env="PROVIDER_KEY")

    endpoint = gateway.resolve(policy())

    assert endpoint.base_url == "http://127.0.0.1:8977/reef/scenarios/officeqa/v1"
    assert (endpoint.served_model, endpoint.runtime) == ("vendor/model", "openai_chat")
    assert gateway.api_key_env(policy()) == "REEF_TOKEN"
    assert gateway.model(policy()) == MODEL  # the same snapshot and checksum
    with pytest.raises(ContractError, match="checkpoint"):
        gateway.resolve(policy(GatewayModel(base_url="https://other.invalid/v1", model="m")))


@pytest.mark.unit
def test_a_gateways_route_takes_one_model_and_no_tags() -> None:
    other = GatewayModel(base_url="https://other.invalid/v1", model="m")
    with pytest.raises(ContractError, match="one model"):
        GatewayEndpoints(MODEL, other, reef=ReefChatRoute("http://reef", "s"))
    with pytest.raises(ContractError, match="no Reef tags"):
        GatewayEndpoints(MODEL, reef=ReefChatRoute(**ROUTE))


class Reef:
    """``reef serve`` in front of ``upstream`` with no config file, as Reef's quickstart runs it."""

    def __init__(self, directory: Path, upstream: ChatUpstream) -> None:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        log = directory / "serve.log"
        self.process = subprocess.Popen(
            [
                str(REEF),
                "serve",
                "--reef.port",
                str(self.port),
                "--inference.upstream-url",
                upstream.url.removesuffix("/v1"),
                "--inference.upstream-model",
                UPSTREAM_MODEL,
            ],
            cwd=directory,
            env={**os.environ, "REEF_TOKEN": TOKEN},
            stdout=log.open("ab"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.wait(lambda: self.call("GET", "/healthz", token=False) is not None, f"Reef did not start, see {log}")

    def stop(self) -> None:
        os.killpg(self.process.pid, signal.SIGTERM)
        try:
            self.process.wait(30)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)

    def call(self, method: str, path: str, body: object = None, *, token: bool = True) -> Any:
        headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {TOKEN}"} if token else {})}
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(f"{self.url}{path}", data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read() or b"{}")
        except OSError:
            return None

    def scenario(self, name: str) -> None:
        """Creates ``name`` and waits until Reef serves it, as an operator does before the first call."""
        self.call("POST", "/reef/scenarios", {"name": name})
        status = f"/reef/scenarios/{name}/inference/status"
        self.wait(lambda: (self.call("GET", status) or {}).get("state") == "ready", f"{name} is not served")

    def records(self, scenario: str) -> list[dict[str, Any]]:
        """The scenario's inference records with their payloads, oldest first."""
        rows = self.call("GET", f"/reef/scenarios/{scenario}/records?request_type=inference")["records"]
        return [self.call("GET", f"/reef/scenarios/{scenario}/records/{row['agent_record_id']}") for row in rows]

    @staticmethod
    def wait(condition: Any, message: str) -> None:
        deadline = time.monotonic() + 120
        while not condition():
            if time.monotonic() > deadline:
                raise AssertionError(message)
            time.sleep(0.25)


@pytest.fixture(scope="module")
def upstream() -> Iterator[ChatUpstream]:
    with ChatUpstream() as server:
        yield server


@pytest.fixture(scope="module")
def reef(tmp_path_factory: pytest.TempPathFactory, upstream: ChatUpstream) -> Iterator[Reef]:
    server = Reef(tmp_path_factory.mktemp("reef"), upstream)
    try:
        yield server
    finally:
        server.stop()


@needs_reef
@pytest.mark.integration
def test_a_chat_model_call_through_reef_reaches_the_model_and_is_recorded_with_its_tags(
    reef: Reef, upstream: ChatUpstream, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("litellm", reason="the chat client needs the 'litellm' extra")
    monkeypatch.setenv("REEF_TOKEN", TOKEN)
    reef.scenario("reflection")
    chat = chat_model(None, model="hosted_vllm/honed", reef={**ROUTE, "url": reef.url}, max_attempts=1)
    messages = [ChatMessage(role="user", content="Propose one instruction.")]

    reply = chat.complete(messages, max_tokens=16, temperature=0.0)

    # Reef asked the stand-in for its upstream model, with no key of Honing's.
    assert reply.text == f"pong from {UPSTREAM_MODEL}"
    assert (upstream.requests[-1]["model"], upstream.requests[-1]["authorization"]) == (UPSTREAM_MODEL, None)
    (record,) = reef.records("reflection")
    assert record["payload"]["metadata"]["tags"] == TAGS
    assert record["payload"]["messages"] == [{"role": "user", "content": "Propose one instruction."}]
    assert record["payload"]["response"]["choices"][0]["message"]["content"] == reply.text


@needs_reef
@pytest.mark.integration
def test_a_gateway_policys_call_through_reef_reaches_the_model_and_is_recorded(
    reef: Reef, upstream: ChatUpstream
) -> None:
    reef.scenario("officeqa")
    model = GatewayModel(base_url=upstream.url, model="honed")
    gateway = endpoints(None, reef={"url": reef.url, "scenario": "officeqa"}, base_url=upstream.url, model="honed")
    endpoint = gateway.resolve(policy(model))
    assert gateway.api_key_env(policy(model)) == "REEF_TOKEN"

    # What an agent's OpenAI client sends: the endpoint's URL and model, and the key the map names. No other header.
    body = {"model": endpoint.served_model, "messages": [{"role": "user", "content": "What is the figure?"}]}
    request = urllib.request.Request(
        f"{endpoint.base_url}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        answer = json.loads(response.read())

    assert answer["choices"][0]["message"]["content"] == f"pong from {UPSTREAM_MODEL}"
    (record,) = reef.records("officeqa")
    assert record["payload"]["messages"] == body["messages"]
    assert "tags" not in record["payload"].get("metadata", {})
