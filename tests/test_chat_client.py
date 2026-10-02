"""LiteLLMChatModel against a local stub server reached through ``api_base``; no request leaves the machine."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

import pytest

from honing.core import ChatMessage, ContractError, protocols, validate_method_state
from honing.harness.models import ChatModelError, LiteLLMChatModel

# Read when litellm is imported: use the price map bundled with the pinned
# version instead of downloading the latest one from GitHub.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
litellm = pytest.importorskip("litellm", reason="the chat client needs the 'litellm' extra")

KEY_ENV = "HONING_TEST_CHAT_KEY"
KEY = "sk-test-0123456789abcdef"
REQUEST_MARK = "request-body-mark-7f3a"
RESPONSE_MARK = "response-body-mark-9c1e"
PRICED_MODEL = "openai/gpt-4o-mini"
LOCAL_MODEL = "hosted_vllm/local-model"
MESSAGES = [ChatMessage(role="system", content="Be brief."), ChatMessage(role="user", content=f"Hi {REQUEST_MARK}")]


class Stub(BaseHTTPRequestHandler):
    """An OpenAI-compatible chat endpoint that answers with scripted statuses, in order, then repeats the last."""

    statuses: ClassVar[list[int]] = [200]
    delays_seconds: ClassVar[list[float]] = []
    requests: ClassVar[list[tuple[str, dict[str, str], dict[str, object]]]] = []

    def log_message(self, *_: object) -> None:
        pass

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append((self.path, dict(self.headers), body))
        if self.delays_seconds:
            time.sleep(type(self).delays_seconds.pop(0))
        status = type(self).statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        if status == 200:
            payload: dict[str, object] = {
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "created": 1,
                "model": body["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
            }
        else:
            # A provider that echoes the request and the key back, as some do in error bodies.
            # litellm picks the exception class partly from the message text, so a
            # context-window error needs the wording OpenAI uses.
            echoed = f"{RESPONSE_MARK} {json.dumps(body)} {self.headers.get('Authorization')}"
            if status == 413:
                status, echoed = 400, f"This model's maximum context length is 8192 tokens. {echoed}"
            payload = {"error": {"message": echoed, "type": "error", "code": str(status)}}
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    monkeypatch.setattr(Stub, "statuses", [200])
    monkeypatch.setattr(Stub, "delays_seconds", [])
    monkeypatch.setattr(Stub, "requests", [])
    monkeypatch.setenv(KEY_ENV, KEY)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()


def client(api_base: str, model: str = PRICED_MODEL, **settings: object) -> tuple[LiteLLMChatModel, list[float]]:
    sleeps: list[float] = []
    chat = LiteLLMChatModel(model, api_base=api_base, api_key_env=KEY_ENV, sleep=sleeps.append, **settings)  # type: ignore[arg-type]
    return chat, sleeps


def complete(chat: LiteLLMChatModel) -> object:
    return chat.complete(MESSAGES, max_tokens=64, temperature=0.5)


@pytest.mark.unit
def test_the_client_implements_the_chat_model_protocol(stub: str) -> None:
    assert isinstance(client(stub)[0], protocols.ChatModel)


@pytest.mark.unit
def test_a_reply_carries_the_text_and_the_token_counts(stub: str) -> None:
    chat, sleeps = client(stub, top_p=0.9, seed=7)

    reply = chat.complete(MESSAGES, max_tokens=64, temperature=0.5)

    assert (reply.text, reply.input_tokens, reply.output_tokens) == ("hello", 11, 3)
    [(path, _, body)] = Stub.requests
    assert path == "/v1/chat/completions"
    assert body == {
        "model": "gpt-4o-mini",
        "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": f"Hi {REQUEST_MARK}"}],
        "max_tokens": 64,
        "temperature": 0.5,
        "top_p": 0.9,
        "seed": 7,
    }
    assert sleeps == []


@pytest.mark.unit
def test_unset_sampling_settings_are_not_sent(stub: str) -> None:
    complete(client(stub)[0])

    [(_, _, body)] = Stub.requests
    assert "top_p" not in body and "seed" not in body


@pytest.mark.unit
def test_a_server_error_is_retried_then_succeeds(stub: str) -> None:
    Stub.statuses[:] = [503, 200]
    chat, sleeps = client(stub, retry_backoff_seconds=0.25)

    reply = complete(chat)

    assert getattr(reply, "text") == "hello"
    assert len(Stub.requests) == 2
    assert sleeps == [0.25]


@pytest.mark.unit
@pytest.mark.parametrize("status", [429, 500, 502, 503])
def test_rate_limits_and_server_errors_are_retried(stub: str, status: int) -> None:
    Stub.statuses[:] = [status, 200]
    chat, _ = client(stub)

    complete(chat)

    assert len(Stub.requests) == 2


@pytest.mark.unit
def test_a_timeout_is_retried(stub: str) -> None:
    Stub.delays_seconds[:] = [1.0]
    chat, sleeps = client(stub, timeout_seconds=0.2)

    complete(chat)

    assert len(Stub.requests) == 2
    assert len(sleeps) == 1


@pytest.mark.unit
def test_retries_stop_after_max_attempts(stub: str) -> None:
    Stub.statuses[:] = [503]
    chat, sleeps = client(stub, max_attempts=3, retry_backoff_seconds=1.0)

    with pytest.raises(ChatModelError) as raised:
        complete(chat)

    assert (raised.value.attempts, raised.value.status_code) == (3, 503)
    assert len(Stub.requests) == 3
    assert sleeps == [1.0, 2.0]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status", "reported_status"),
    # 413 is the stub's code for a 400 that says the context window is exceeded.
    [(400, 400), (401, 401), (403, 403), (404, 404), (422, 422), (413, 400)],
)
def test_a_client_error_is_raised_without_a_retry(stub: str, status: int, reported_status: int) -> None:
    # Only the status is asserted: litellm picks its exception class partly from
    # the message text, and not consistently (403 can arrive as ``APIError``).
    Stub.statuses[:] = [status]
    chat, sleeps = client(stub)

    with pytest.raises(ChatModelError) as raised:
        complete(chat)

    assert (raised.value.status_code, raised.value.attempts) == (reported_status, 1)
    assert len(Stub.requests) == 1
    assert sleeps == []


@pytest.mark.unit
def test_a_missing_key_variable_fails_at_construction(stub: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(KEY_ENV)

    with pytest.raises(ContractError, match=f"{KEY_ENV} is unset or empty"):
        client(stub)

    assert Stub.requests == []


@pytest.mark.unit
def test_a_key_variable_removed_after_construction_fails_before_any_request(
    stub: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat, _ = client(stub)
    monkeypatch.setenv(KEY_ENV, "")

    with pytest.raises(ContractError, match=f"{KEY_ENV} is unset or empty"):
        complete(chat)

    assert Stub.requests == []


@pytest.mark.unit
def test_the_key_is_read_on_each_call_and_never_set_globally(stub: str, monkeypatch: pytest.MonkeyPatch) -> None:
    chat, _ = client(stub)
    complete(chat)
    monkeypatch.setenv(KEY_ENV, "sk-rotated")
    complete(chat)

    assert [headers["Authorization"] for _, headers, _ in Stub.requests] == [f"Bearer {KEY}", "Bearer sk-rotated"]
    assert litellm.api_key is None
    assert litellm.openai_key is None
    assert litellm.num_retries is None


@pytest.mark.unit
def test_cost_comes_from_the_bundled_price_map_without_a_remote_connection(
    stub: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    connected: list[object] = []
    real_connect = socket.socket.connect

    def loopback_only(self: socket.socket, address: object) -> None:
        connected.append(address)
        real_connect(self, address)  # type: ignore[arg-type]

    monkeypatch.setattr(socket.socket, "connect", loopback_only)
    price = litellm.model_cost["gpt-4o-mini"]

    priced = complete(client(stub, PRICED_MODEL)[0])
    local = complete(client(stub, LOCAL_MODEL)[0])

    expected = 11 * price["input_cost_per_token"] + 3 * price["output_cost_per_token"]
    assert getattr(priced, "cost_usd") == pytest.approx(expected)
    assert expected > 0
    assert getattr(local, "cost_usd") is None
    assert connected and all(isinstance(address, tuple) and address[0] == "127.0.0.1" for address in connected)
    source = litellm.litellm_core_utils.get_model_cost_map.get_model_cost_map_source_info()
    assert (source["source"], source["is_env_forced"]) == ("local", True)


# Runs in a fresh interpreter, so litellm is imported by the client's first call.
# Every non-loopback lookup or connection is recorded and refused.
FRESH_PROCESS_CALL = """
import json, os, socket, sys

attempts = []
LOOPBACK = {"127.0.0.1", "::1", "localhost"}
real_connect, real_getaddrinfo = socket.socket.connect, socket.getaddrinfo

def connect(self, address):
    host = str(address[0] if isinstance(address, tuple) else address)
    attempts.append(host)
    if host not in LOOPBACK:
        raise OSError("refused by the test: " + host)
    return real_connect(self, address)

def getaddrinfo(host, *args, **kwargs):
    attempts.append(str(host))
    if str(host) not in LOOPBACK:
        raise socket.gaierror("refused by the test: " + str(host))
    return real_getaddrinfo(host, *args, **kwargs)

socket.socket.connect, socket.getaddrinfo = connect, getaddrinfo

from honing.core import ChatMessage
from honing.harness.models import LiteLLMChatModel

assert "litellm" not in sys.modules
chat = LiteLLMChatModel(sys.argv[1], api_base=sys.argv[2], api_key_env=sys.argv[3])
reply = chat.complete([ChatMessage(role="user", content="hi")], max_tokens=8, temperature=0.0)

from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map_source_info

print(json.dumps({
    "variable": os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP"),
    "remote": sorted({host for host in attempts if host not in LOOPBACK}),
    "source": get_model_cost_map_source_info()["source"],
    "cost_usd": reply.cost_usd,
}))
"""


@pytest.mark.integration  # starts a subprocess
def test_the_client_uses_the_bundled_price_map_when_the_variable_is_unset(stub: str) -> None:
    environment = {name: value for name, value in os.environ.items() if name != "LITELLM_LOCAL_MODEL_COST_MAP"}
    command = [sys.executable, "-c", FRESH_PROCESS_CALL, PRICED_MODEL, stub, KEY_ENV]

    result = subprocess.run(command, env=environment, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr[-2000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["variable"] == "True"
    assert report["remote"] == []
    assert report["source"] == "local"
    assert report["cost_usd"] > 0
    assert len(Stub.requests) == 1


@pytest.mark.unit
@pytest.mark.parametrize("status", [200, 400, 401, 503])
def test_no_body_or_key_reaches_an_error_or_what_litellm_logs(
    stub: str, status: int, caplog: pytest.LogCaptureFixture, capfd: pytest.CaptureFixture[str]
) -> None:
    Stub.statuses[:] = [status]
    chat, _ = client(stub, max_attempts=2)

    try:
        complete(chat)
    except ChatModelError as error:
        shown = [str(error), repr(error), "".join(traceback.format_exception(error))]
    else:
        assert status == 200
        shown = []
    out, err = capfd.readouterr()
    shown += [caplog.text, out, err]

    for text in shown:
        for secret in (KEY, REQUEST_MARK, RESPONSE_MARK):
            assert secret not in text


@pytest.mark.unit
def test_settings_are_json_native_and_hold_no_key(stub: str) -> None:
    chat, _ = client(stub, top_p=0.9, seed=3)

    settings = dict(chat.settings())

    validate_method_state({"schema_version": 1, "chat": settings})
    assert json.loads(json.dumps(settings)) == settings
    assert KEY not in json.dumps(settings)
    assert settings["model"] == PRICED_MODEL and settings["api_key_env"] == KEY_ENV


@pytest.mark.unit
@pytest.mark.parametrize(
    "settings",
    [
        {"model": ""},
        {"timeout_seconds": 0.0},
        {"timeout_seconds": float("inf")},
        {"max_attempts": 0},
        {"retry_backoff_seconds": -1.0},
        {"top_p": 0.0},
        {"top_p": 1.5},
    ],
)
def test_invalid_settings_are_refused(stub: str, settings: dict[str, object]) -> None:
    arguments: dict[str, object] = {"model": PRICED_MODEL, "api_base": stub, **settings}
    with pytest.raises(ContractError):
        LiteLLMChatModel(**arguments)  # type: ignore[arg-type]


@pytest.mark.integration
def test_a_real_model_answers() -> None:
    """One real call, run by hand: ``HONING_LITELLM_SMOKE_MODEL`` names the model, and optionally
    ``HONING_LITELLM_SMOKE_API_BASE`` the server and ``HONING_LITELLM_SMOKE_KEY_ENV`` the key's variable."""
    model = os.environ.get("HONING_LITELLM_SMOKE_MODEL")
    if not model:
        pytest.skip("set HONING_LITELLM_SMOKE_MODEL to call a real model")
    chat = LiteLLMChatModel(
        model,
        api_base=os.environ.get("HONING_LITELLM_SMOKE_API_BASE"),
        api_key_env=os.environ.get("HONING_LITELLM_SMOKE_KEY_ENV"),
    )

    reply = chat.complete([ChatMessage(role="user", content="Reply with the word ok.")], max_tokens=16, temperature=0.0)

    assert reply.text.strip()
    assert reply.input_tokens > 0
    assert reply.cost_usd is None or reply.cost_usd > 0
