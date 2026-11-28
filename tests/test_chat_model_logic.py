"""LiteLLMChatModel's own decisions against a fake litellm (tests/fakes/litellm_stub.py): no extra, no socket.

``tests/test_chat_client.py`` runs the client against the real litellm and a
loopback server, in the CI job that installs the extra. These tests pin the
client's own logic, which that job and this one share: which failures are
retried and how long it waits, what an error may say, when a missing provider
key ends a call, and when a reply has a known cost.
"""

from __future__ import annotations

import math
import os
import sys

import pytest

from honing.core import ChatMessage, ContractError
from honing.harness.models import ChatModelError, LiteLLMChatModel, completion_cost_usd
from tests.fakes.litellm_stub import APIConnectionError, APIError, FakeLiteLLM, Timeout, install, response

KEY_ENV = "HONING_TEST_CHAT_KEY"
KEY = "sk-test-0123456789abcdef"
BODY = "response-body-mark-9c1e"
MESSAGES = [ChatMessage(role="system", content="Be brief."), ChatMessage(role="user", content="Hi")]


def client(**settings: object) -> tuple[LiteLLMChatModel, list[float]]:
    sleeps: list[float] = []
    chat = LiteLLMChatModel(**{"model": "openai/m", "sleep": sleeps.append, **settings})  # type: ignore[arg-type]
    return chat, sleeps


@pytest.fixture
def keyed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_ENV, KEY)


@pytest.mark.unit
def test_a_call_sends_one_attempt_with_the_settings_and_returns_the_text_and_tokens(
    monkeypatch: pytest.MonkeyPatch, keyed: None
) -> None:
    fake = install(monkeypatch)
    chat, sleeps = client(api_base="http://gpu:30000/v1", api_key_env=KEY_ENV, top_p=0.5, seed=7, timeout_seconds=9)

    reply = chat.complete(MESSAGES, max_tokens=32, temperature=0.2)

    assert (reply.text, reply.input_tokens, reply.output_tokens) == ("hello", 7, 3)
    assert fake.calls == [
        {
            "model": "openai/m",
            "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}],
            "max_tokens": 32,
            "temperature": 0.2,
            "top_p": 0.5,
            "seed": 7,
            "api_base": "http://gpu:30000/v1",
            "api_key": KEY,
            "timeout": 9,
            "num_retries": 0,
            "max_retries": 0,
        }
    ]
    assert sleeps == []


@pytest.mark.unit
@pytest.mark.parametrize(
    "error",
    [Timeout("slow"), APIConnectionError("reset"), APIError(408), APIError(429), APIError(500), APIError(503)],
    ids=["timeout", "connection", "408", "429", "500", "503"],
)
def test_timeouts_dropped_connections_rate_limits_and_server_errors_are_retried(
    monkeypatch: pytest.MonkeyPatch, keyed: None, error: Exception
) -> None:
    fake = install(monkeypatch, FakeLiteLLM([error, "hello"]))
    chat, sleeps = client(api_key_env=KEY_ENV, retry_backoff_seconds=0.25)

    assert chat.complete(MESSAGES, max_tokens=8, temperature=0.0).text == "hello"
    assert len(fake.calls) == 2
    assert sleeps == [0.25]


@pytest.mark.unit
def test_each_retry_waits_twice_as_long_and_the_last_failure_names_every_attempt(
    monkeypatch: pytest.MonkeyPatch, keyed: None
) -> None:
    fake = install(monkeypatch, FakeLiteLLM([APIError(503, BODY)]))
    chat, sleeps = client(api_key_env=KEY_ENV, max_attempts=4, retry_backoff_seconds=0.5)

    with pytest.raises(ChatModelError) as raised:
        chat.complete(MESSAGES, max_tokens=8, temperature=0.0)

    assert len(fake.calls) == 4
    assert sleeps == [0.5, 1.0, 2.0]
    assert (raised.value.error_type, raised.value.status_code, raised.value.attempts) == ("APIError", 503, 4)
    assert str(raised.value) == "chat completion for 'openai/m' failed after 4 attempts: APIError (status 503)"


@pytest.mark.unit
@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_a_request_the_provider_refuses_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, keyed: None, status: int
) -> None:
    fake = install(monkeypatch, FakeLiteLLM([APIError(status, BODY), "hello"]))
    chat, sleeps = client(api_key_env=KEY_ENV)

    with pytest.raises(ChatModelError) as raised:
        chat.complete(MESSAGES, max_tokens=8, temperature=0.0)

    assert (raised.value.status_code, raised.value.attempts) == (status, 1)
    assert str(raised.value).endswith(f"after 1 attempt: APIError (status {status})")
    assert len(fake.calls) == 1 and sleeps == []
    assert fake.validated == []  # a key was passed, so no provider key can be missing


@pytest.mark.unit
def test_an_error_without_a_status_is_not_retried_and_says_so(monkeypatch: pytest.MonkeyPatch, keyed: None) -> None:
    install(monkeypatch, FakeLiteLLM([RuntimeError(BODY), "hello"]))
    chat, _ = client(api_key_env=KEY_ENV)

    with pytest.raises(ChatModelError, match=r"failed after 1 attempt: RuntimeError \(no status\)$"):
        chat.complete(MESSAGES, max_tokens=8, temperature=0.0)


@pytest.mark.unit
def test_the_error_holds_no_body_no_key_and_does_not_chain_litellm_s(
    monkeypatch: pytest.MonkeyPatch, keyed: None
) -> None:
    install(monkeypatch, FakeLiteLLM([APIError(400, f"{BODY} {KEY}")]))
    chat, _ = client(api_key_env=KEY_ENV)

    with pytest.raises(ChatModelError) as raised:
        chat.complete(MESSAGES, max_tokens=8, temperature=0.0)

    assert BODY not in str(raised.value) and KEY not in str(raised.value)
    assert raised.value.__cause__ is None and raised.value.__suppress_context__


@pytest.mark.unit
@pytest.mark.parametrize("status", [None, 401, 500])
def test_an_unset_provider_key_ends_the_call_on_its_first_attempt_with_a_hint(
    monkeypatch: pytest.MonkeyPatch, status: int | None
) -> None:
    fake = install(monkeypatch, FakeLiteLLM([APIError(status, BODY), "hello"], missing_keys=["OPENAI_API_KEY"]))
    chat, sleeps = client(api_base="http://gpu:30000/v1")

    with pytest.raises(ChatModelError) as raised:
        chat.complete(MESSAGES, max_tokens=8, temperature=0.0)

    assert (raised.value.error_type, raised.value.status_code, raised.value.attempts) == ("MissingAPIKey", status, 1)
    assert "OPENAI_API_KEY is unset; set api_key_env, or use a hosted_vllm/ model" in str(raised.value)
    assert len(fake.calls) == 1 and sleeps == []
    assert fake.validated == [{"model": "openai/m", "api_base": "http://gpu:30000/v1"}]


@pytest.mark.unit
@pytest.mark.parametrize(
    "error", [Timeout("slow"), APIConnectionError("reset"), APIError(429)], ids=["timeout", "connection", "429"]
)
def test_a_transient_failure_without_a_key_is_retried_without_asking_for_keys(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    fake = install(monkeypatch, FakeLiteLLM([error, "hello"], missing_keys=["OPENAI_API_KEY"]))
    chat, sleeps = client()

    assert chat.complete(MESSAGES, max_tokens=8, temperature=0.0).text == "hello"
    assert fake.validated == [] and len(sleeps) == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    "report",
    [RuntimeError("cannot tell"), "not a mapping", {"missing_keys": "OPENAI_API_KEY"}, {"missing_keys": []}],
    ids=["raises", "not-a-mapping", "not-a-list", "none-missing"],
)
def test_a_key_check_that_names_no_key_leaves_the_error_as_it_was(
    monkeypatch: pytest.MonkeyPatch, report: object
) -> None:
    fake = install(monkeypatch, FakeLiteLLM([APIError(500), "hello"]))

    def validate_environment(model: str, api_base: str | None = None) -> object:
        if isinstance(report, Exception):
            raise report
        return report

    monkeypatch.setattr(fake, "validate_environment", validate_environment)
    chat, sleeps = client()

    # Without a missing key, a 500 is a server error like any other, and is retried.
    assert chat.complete(MESSAGES, max_tokens=8, temperature=0.0).text == "hello"
    assert len(sleeps) == 1


@pytest.mark.unit
def test_the_key_is_read_again_on_every_call(monkeypatch: pytest.MonkeyPatch, keyed: None) -> None:
    fake = install(monkeypatch)
    chat, _ = client(api_key_env=KEY_ENV)
    chat.complete(MESSAGES, max_tokens=8, temperature=0.0)
    monkeypatch.setenv(KEY_ENV, "sk-rotated")
    chat.complete(MESSAGES, max_tokens=8, temperature=0.0)
    monkeypatch.delenv(KEY_ENV)

    with pytest.raises(ContractError, match=f"{KEY_ENV} is unset or empty"):
        chat.complete(MESSAGES, max_tokens=8, temperature=0.0)
    assert [call["api_key"] for call in fake.calls] == [KEY, "sk-rotated"]


@pytest.mark.unit
@pytest.mark.parametrize("value", [None, ""])
def test_an_unset_or_empty_key_variable_fails_at_construction(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    fake = install(monkeypatch)
    if value is None:
        monkeypatch.delenv(KEY_ENV, raising=False)
    else:
        monkeypatch.setenv(KEY_ENV, value)

    with pytest.raises(ContractError, match=f"environment variable {KEY_ENV} is unset or empty"):
        client(api_key_env=KEY_ENV)
    assert fake.calls == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"model": ""}, "model must be a litellm model string"),
        ({"timeout_seconds": 0}, "timeout_seconds must be > 0"),
        ({"timeout_seconds": math.inf}, "timeout_seconds must be > 0"),
        ({"max_attempts": 0}, "max_attempts must be >= 1"),
        ({"retry_backoff_seconds": -1.0}, "retry_backoff_seconds must be >= 0"),
        ({"retry_backoff_seconds": math.nan}, "retry_backoff_seconds must be >= 0"),
        ({"top_p": 0.0}, r"top_p must be in \(0, 1\]"),
        ({"top_p": 1.5}, r"top_p must be in \(0, 1\]"),
    ],
)
def test_settings_that_cannot_work_are_refused_at_construction(
    monkeypatch: pytest.MonkeyPatch, settings: dict[str, object], message: str
) -> None:
    install(monkeypatch)

    with pytest.raises(ContractError, match=message):
        client(**settings)


@pytest.mark.unit
def test_without_the_extra_construction_says_how_to_install_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "litellm", None)

    with pytest.raises(ImportError, match=r"pip install 'llm-honing\[litellm\]'"):
        client()


@pytest.mark.unit
@pytest.mark.parametrize(
    "reply",
    [response(None), response(["a", "list"]), type("Empty", (), {"choices": []})()],
    ids=["no-content", "not-text", "no-choices"],
)
def test_a_reply_without_text_is_an_error_and_is_not_retried(monkeypatch: pytest.MonkeyPatch, reply: object) -> None:
    fake = install(monkeypatch)
    monkeypatch.setattr(fake, "completion", lambda **_: reply)
    chat, sleeps = client()

    with pytest.raises(ChatModelError) as raised:
        chat.complete(MESSAGES, max_tokens=8, temperature=0.0)

    assert (raised.value.error_type, raised.value.status_code, raised.value.attempts) == ("NoTextInReply", None, 1)
    assert sleeps == []


@pytest.mark.unit
def test_a_reply_without_usage_counts_no_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = install(monkeypatch)
    monkeypatch.setattr(fake, "completion", lambda **_: response("hi", prompt_tokens=None, completion_tokens=None))
    chat, _ = client()

    reply = chat.complete(MESSAGES, max_tokens=8, temperature=0.0)

    assert (reply.text, reply.input_tokens, reply.output_tokens) == ("hi", 0, 0)


@pytest.mark.unit
def test_a_priced_model_s_reply_costs_what_the_price_map_says(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, FakeLiteLLM(prices={"openai/m": (0.5, 2.0)}))
    chat, _ = client()

    assert chat.complete(MESSAGES, max_tokens=8, temperature=0.0).cost_usd == pytest.approx(7 * 0.5 + 3 * 2.0)


@pytest.mark.unit
def test_a_server_named_by_api_base_has_no_known_cost_even_for_a_priced_model(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, FakeLiteLLM(prices={"openai/m": (0.5, 2.0)}))
    chat, _ = client(api_base="http://127.0.0.1:8000/v1")

    assert chat.complete(MESSAGES, max_tokens=8, temperature=0.0).cost_usd is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "cost",
    [0.0, -1.0, math.nan, math.inf, "0.1", ValueError("not mapped")],
    ids=["zero", "negative", "nan", "inf", "not-a-number", "unmapped"],
)
def test_a_reply_whose_price_is_zero_unknown_or_not_a_number_has_no_known_cost(
    monkeypatch: pytest.MonkeyPatch, cost: object
) -> None:
    install(monkeypatch, FakeLiteLLM(reply_cost=cost))
    chat, _ = client()

    reply = chat.complete(MESSAGES, max_tokens=8, temperature=0.0)

    assert reply.text == "hello" and reply.cost_usd is None


@pytest.mark.unit
def test_the_bundled_price_map_is_chosen_unless_the_process_chose_already(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch)
    chat, _ = client()
    # Recorded as set, so the variable is removed again when the test ends.
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "chosen")
    monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP")

    chat.complete(MESSAGES, max_tokens=8, temperature=0.0)
    assert os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] == "True"
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "False")
    chat.complete(MESSAGES, max_tokens=8, temperature=0.0)
    assert os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] == "False"


@pytest.mark.unit
def test_the_settings_a_method_records_name_the_key_variable_but_hold_no_key(
    monkeypatch: pytest.MonkeyPatch, keyed: None
) -> None:
    install(monkeypatch)
    chat, _ = client(api_base="http://gpu/v1", api_key_env=KEY_ENV, top_p=0.9, seed=3)

    assert chat.settings() == {
        "client": "litellm",
        "model": "openai/m",
        "api_base": "http://gpu/v1",
        "api_key_env": KEY_ENV,
        "timeout_seconds": 60.0,
        "max_attempts": 3,
        "top_p": 0.9,
        "seed": 3,
    }
    assert KEY not in repr(chat) and "sleep" not in repr(chat)


# ---------------------------------------------------------------- completion_cost_usd


@pytest.mark.unit
def test_tokens_of_a_priced_model_cost_their_price_and_zero_tokens_cost_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, FakeLiteLLM(prices={"openai/m": (0.5, 2.0)}))

    assert completion_cost_usd("openai/m", 1000, 10) == pytest.approx(1000 * 0.5 + 10 * 2.0)
    assert completion_cost_usd("openai/m", 0, 0) == 0.0


@pytest.mark.unit
@pytest.mark.parametrize(
    "prices", [{}, {"openai/m": (0.0, 0.0)}, {"openai/m": (math.inf, 0.0)}], ids=["unmapped", "free", "infinite"]
)
def test_tokens_of_an_unmapped_free_or_unpriceable_model_have_no_known_cost(
    monkeypatch: pytest.MonkeyPatch, prices: dict[str, tuple[float, float]]
) -> None:
    install(monkeypatch, FakeLiteLLM(prices=prices))

    assert completion_cost_usd("openai/m", 1000, 10) is None


@pytest.mark.unit
def test_tokens_sent_to_a_named_server_have_no_known_cost_and_import_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "litellm", None)  # importing it would fail

    assert completion_cost_usd("openai/m", 1000, 10, api_base="http://127.0.0.1:8000/v1") is None


@pytest.mark.unit
def test_a_price_the_map_cannot_add_up_has_no_known_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = install(monkeypatch, FakeLiteLLM(prices={"openai/m": (0.5, 2.0)}))
    costs = iter([(0.5, 2.0), (-1.0, 0.0)])
    monkeypatch.setattr(fake, "cost_per_token", lambda **_: next(costs))

    assert completion_cost_usd("openai/m", 1000, 10) is None


@pytest.mark.unit
@pytest.mark.parametrize(("input_tokens", "output_tokens"), [(-1, 0), (0, -1)])
def test_negative_token_counts_are_refused(input_tokens: int, output_tokens: int) -> None:
    with pytest.raises(ContractError, match="token counts must be >= 0"):
        completion_cost_usd("openai/m", input_tokens, output_tokens)
