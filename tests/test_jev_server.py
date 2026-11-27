"""The decision server and endpoint map, with a fake model: no GPU, no torch."""

from __future__ import annotations

import http.client
import json
import math
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import ClassVar

import pytest

from honing.backends.jev import DecisionModel, JevEndpoints
from honing.backends.jev.server import MAX_BODY_BYTES, MAX_BRANCHES, DecisionServer, RequestError, parse_decide
from honing.core import ContractError, PolicyEndpoint, PolicyRef
from tests.fakes.model import FakeModelSource, FakeWorld


class FakeDecisionModel:
    """Picks the first option of every question, slowly, and records how many calls overlap."""

    def __init__(self, delay: float = 0.0, fail: bool = False) -> None:
        self.delay = delay
        self.fail = fail
        self.active = 0
        self.peak = 0
        self._count = threading.Lock()

    @property
    def metadata(self) -> Mapping[str, object]:
        return {"revision": "fake", "dtype": "none"}

    def predict(self, parts: Mapping[str, object], questions: Mapping[str, object]) -> Mapping[str, object]:
        with self._count:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(self.delay)
            if self.fail:
                raise RuntimeError("model fell over")
            answers = {key: {"choice": next(iter(q["criteria"]))} for key, q in questions.items()}  # type: ignore[index]
            return {"answers": answers, "stateText": "\n".join(parts.get("essential", []))}  # type: ignore[arg-type]
        finally:
            with self._count:
                self.active -= 1


@pytest.fixture
def serve() -> Iterator[object]:
    servers: list[tuple[DecisionServer, threading.Thread]] = []

    def start(model: DecisionModel) -> str:
        server = DecisionServer(("127.0.0.1", 0), model)
        thread = threading.Thread(target=server.serve, daemon=True)
        thread.start()
        servers.append((server, thread))
        return server.url

    yield start
    for server, thread in servers:
        server.stop()
        thread.join(timeout=10)
        server.server_close()


def call(url: str, path: str, body: object | None = None, timeout: float = 10) -> tuple[int, dict[str, object]]:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


QUESTION = {"root": {"type": "choice", "instructions": "Next?", "criteria": {"power": "Power plant", "wait": "Wait"}}}


@pytest.mark.unit
def test_health_reports_what_is_loaded(serve) -> None:  # type: ignore[no-untyped-def]
    url = serve(FakeDecisionModel())
    assert call(url, "/v1/health") == (
        200,
        {"ready": True, "metadata": {"revision": "fake", "dtype": "none"}, "maxBranches": 16},
    )


class CheckedModel(FakeDecisionModel):
    """A fake whose ``healthy`` answers what the test sets, or raises it."""

    def __init__(self, healthy: bool | Exception = True, delay: float = 0.0, concurrent: bool = False) -> None:
        super().__init__(delay=delay)
        self.state = healthy
        self.concurrent = concurrent

    def healthy(self) -> bool:
        if isinstance(self.state, Exception):
            raise self.state
        return self.state


@pytest.mark.unit
def test_a_healthy_model_answers_health_as_before(serve) -> None:  # type: ignore[no-untyped-def]
    url = serve(CheckedModel())
    assert call(url, "/v1/health") == (
        200,
        {"ready": True, "metadata": {"revision": "fake", "dtype": "none"}, "maxBranches": 16},
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("state", "error"),
    [
        (False, "the model reports that its engine is down"),
        (ConnectionRefusedError("engine gone"), "ConnectionRefusedError: engine gone"),
    ],
)
def test_a_model_that_is_down_or_whose_check_fails_is_a_503(serve, state, error) -> None:  # type: ignore[no-untyped-def]
    url = serve(CheckedModel(state))
    assert call(url, "/v1/health") == (
        503,
        {"ready": False, "error": error, "metadata": {"revision": "fake", "dtype": "none"}, "maxBranches": 16},
    )


@pytest.mark.unit
def test_a_healthy_attribute_that_is_not_callable_is_ignored(serve) -> None:  # type: ignore[no-untyped-def]
    model = FakeDecisionModel()
    model.healthy = True  # type: ignore[attr-defined]  # a flag, not a check
    assert call(serve(model), "/v1/health")[0] == 200


@pytest.mark.unit
@pytest.mark.parametrize("concurrent", [False, True])
def test_health_answers_while_a_long_decision_runs(serve, concurrent: bool) -> None:  # type: ignore[no-untyped-def]
    # Model calls run one at a time on the serving thread; the check must not wait its turn behind them.
    model = CheckedModel(delay=2.0, concurrent=concurrent)
    url = serve(model)
    deciding = threading.Thread(target=call, args=(url, "/v1/decide", {"parts": {}, "questions": QUESTION}))
    deciding.start()
    deadline = time.monotonic() + 2
    while model.active == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert model.active == 1
    assert call(url, "/v1/health", timeout=1)[0] == 200
    assert model.active == 1  # the decision was still running when health answered
    deciding.join()


@pytest.mark.unit
def test_a_decision_is_answered_and_its_id_echoed(serve) -> None:  # type: ignore[no-untyped-def]
    url = serve(FakeDecisionModel())
    status, body = call(url, "/v1/decide", {"id": 7, "parts": {"essential": ["tick 30"]}, "questions": QUESTION})
    assert status == 200
    assert body == {"answers": {"root": {"choice": "power"}}, "stateText": "tick 30", "id": 7}


@pytest.mark.unit
def test_a_decision_after_an_idle_spell_is_still_answered(serve) -> None:  # type: ignore[no-untyped-def]
    """The serving loop wakes once a second while idle and must keep going, not stop, when nothing arrived."""
    url = serve(FakeDecisionModel())
    time.sleep(1.5)
    status, body = call(url, "/v1/decide", {"id": 8, "parts": {"essential": ["tick 31"]}, "questions": QUESTION}, timeout=5)
    assert status == 200 and body["id"] == 8


@pytest.mark.unit
@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"parts": {}}, "questions must be a non-empty object"),
        ({"parts": [], "questions": QUESTION}, "parts must be an object"),
        ({"parts": {}, "questions": QUESTION, "temperature": 2}, "unknown fields: temperature"),
        ({"parts": {}, "questions": QUESTION, "branches": []}, "branches must be an object of at most 16"),
        (
            {"parts": {}, "questions": QUESTION, "branches": {str(i): {} for i in range(17)}},
            "branches must be an object of at most 16",
        ),
        ({"parts": {}, "questions": QUESTION, "branches": {"a": []}}, "branch a must be a JSON object"),
        ({"parts": {}, "questions": QUESTION, "branches": {"a": {"parts": {}}}}, "questions in branch a must be"),
        (
            {
                "parts": {},
                "questions": QUESTION,
                "branches": {"a": {"parts": {}, "questions": QUESTION, "branches": {}}},
            },
            "unknown fields in branch a: branches",
        ),
    ],
)
def test_a_malformed_request_is_refused_without_calling_the_model(serve, body, message) -> None:  # type: ignore[no-untyped-def]
    model = FakeDecisionModel()
    url = serve(model)
    status, reply = call(url, "/v1/decide", body)
    assert status == 400 and message in str(reply["error"])
    assert model.peak == 0


@pytest.mark.unit
def test_a_model_failure_is_a_500_that_names_it(serve) -> None:  # type: ignore[no-untyped-def]
    url = serve(FakeDecisionModel(fail=True))
    status, reply = call(url, "/v1/decide", {"parts": {}, "questions": QUESTION})
    assert status == 500 and reply["error"] == "RuntimeError: model fell over"


@pytest.mark.unit
def test_an_unknown_route_is_a_404(serve) -> None:  # type: ignore[no-untyped-def]
    url = serve(FakeDecisionModel())
    assert call(url, "/v1/other")[0] == 404


@pytest.mark.unit
def test_concurrent_games_share_one_model_one_request_at_a_time(serve) -> None:  # type: ignore[no-untyped-def]
    model = FakeDecisionModel(delay=0.05)
    url = serve(model)
    with ThreadPoolExecutor(max_workers=4) as pool:
        replies = list(
            pool.map(lambda i: call(url, "/v1/decide", {"id": i, "parts": {}, "questions": QUESTION}), range(8))
        )
    assert [status for status, _ in replies] == [200] * 8
    assert sorted(body["id"] for _, body in replies) == list(range(8))  # type: ignore[type-var]
    assert model.peak == 1


@pytest.mark.unit
def test_parse_refuses_a_body_that_is_not_json() -> None:
    with pytest.raises(RequestError, match="not JSON"):
        parse_decide(b"{")


def _policy() -> PolicyRef:
    snapshot = FakeModelSource(FakeWorld()).materialize(__import__("pathlib").Path("unused"))
    return PolicyRef(snapshot.checkpoint, snapshot.architecture, "master")


@pytest.mark.unit
def test_endpoints_resolve_a_served_checkpoint_and_refuse_others() -> None:
    policy = _policy()
    endpoint = PolicyEndpoint("http://127.0.0.1:1", "tok", "laya", runtime="jev_decision")
    endpoints = JevEndpoints({policy.checkpoint.checksum: endpoint})
    assert endpoints.resolve(policy) is endpoint
    with pytest.raises(ContractError, match="no decision server"):
        JevEndpoints({}).resolve(policy)


@pytest.mark.unit
def test_endpoints_refuse_a_language_model_server() -> None:
    with pytest.raises(ContractError, match="not a Jev server"):
        JevEndpoints({"c": PolicyEndpoint("http://h", "tok", "m")})


@pytest.mark.unit
def test_the_fake_satisfies_the_decision_model_protocol() -> None:
    assert isinstance(FakeDecisionModel(), DecisionModel)


class ThreadRecordingModel:
    """Records the thread each call runs on."""

    metadata: ClassVar[dict[str, object]] = {"family": "fake"}

    def __init__(self) -> None:
        self.threads: set[int] = set()

    def predict(self, parts: object, questions: object) -> dict[str, object]:
        self.threads.add(threading.get_ident())
        return {"answers": {}}


@pytest.mark.unit
def test_every_model_call_runs_on_the_serving_thread() -> None:
    # A CUDA kernel can be chosen per thread, so the server never runs the model on a request's thread.
    model = ThreadRecordingModel()
    server = DecisionServer(("127.0.0.1", 0), model)  # type: ignore[arg-type]
    serving: list[int] = []

    def serve() -> None:
        serving.append(threading.get_ident())
        server.serve()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    with ThreadPoolExecutor(max_workers=4) as pool:
        replies = list(
            pool.map(lambda i: call(server.url, "/v1/decide", {"id": i, "parts": {}, "questions": QUESTION}), range(8))
        )
    server.stop()
    thread.join(timeout=10)
    server.server_close()

    assert [status for status, _ in replies] == [200] * 8
    assert model.threads == set(serving)
    assert not thread.is_alive()


BRANCHED = {
    "id": "d",
    "parts": {"essential": ["tick 30"]},
    "questions": QUESTION,
    "branches": {
        "x": {"id": "d:x", "parts": {"essential": ["after x"]}, "questions": QUESTION},
        "y": {"parts": {"essential": ["after y"]}, "questions": QUESTION},
    },
}


@pytest.mark.unit
def test_branches_are_answered_in_one_call_each_as_if_alone(serve) -> None:  # type: ignore[no-untyped-def]
    url = serve(FakeDecisionModel())
    status, body = call(url, "/v1/decide", BRANCHED)
    assert status == 200
    assert body == {
        "answers": {"root": {"choice": "power"}},
        "stateText": "tick 30",
        "id": "d",
        "branches": {
            "x": {"answers": {"root": {"choice": "power"}}, "stateText": "after x", "id": "d:x"},
            "y": {"answers": {"root": {"choice": "power"}}, "stateText": "after y"},
        },
    }


class SplitModel(FakeDecisionModel):
    """Offers predict_many, or prepare and run, and records the threads and batches they ran with."""

    def __init__(self, split: bool) -> None:
        super().__init__()
        self.batches: list[int] = []
        self.prepared_on: set[int] = set()
        self.ran_on: set[int] = set()
        if split:
            self.prepare = self._prepare
            self.run = self._run

    def predict_many(self, requests: list[tuple[Mapping[str, object], Mapping[str, object]]]) -> list[object]:
        self.batches.append(len(requests))
        self.ran_on.add(threading.get_ident())
        return [self.predict(parts, questions) for parts, questions in requests]

    def _prepare(self, requests: list[tuple[Mapping[str, object], Mapping[str, object]]]) -> object:
        self.prepared_on.add(threading.get_ident())
        return list(requests)

    def _run(self, prepared: list[tuple[Mapping[str, object], Mapping[str, object]]]) -> list[object]:
        return self.predict_many(prepared)


@pytest.mark.unit
@pytest.mark.parametrize("split", [False, True])
def test_a_model_s_own_batch_and_split_calls_are_used(split: bool) -> None:
    model = SplitModel(split)
    server = DecisionServer(("127.0.0.1", 0), model)
    serving: list[int] = []

    def serve() -> None:
        serving.append(threading.get_ident())
        server.serve()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    status, body = call(server.url, "/v1/decide", BRANCHED)
    single = call(server.url, "/v1/decide", {"parts": {}, "questions": QUESTION})
    server.stop()
    thread.join(timeout=10)
    server.server_close()
    assert status == 200 and set(body["branches"]) == {"x", "y"}  # type: ignore[arg-type]
    assert single[0] == 200
    assert model.ran_on == set(serving)
    if split:
        # prepare runs on the request's thread, so one call is encoded while another uses the GPU.
        assert model.batches == [3, 1] and serving[0] not in model.prepared_on
    else:
        assert model.batches == [3] and not model.prepared_on


def post_raw(
    url: str, body: bytes, length: str | None = None, path: str = "/v1/decide"
) -> tuple[int, dict[str, object]]:
    """POST ``body`` as it is, with ``length`` as the Content-Length header ("omit" sends none) when given.

    The response must be strict JSON with a matching Content-Type and Content-Length.
    """
    host, port = url.removeprefix("http://").split(":")
    connection = http.client.HTTPConnection(host, int(port), timeout=10)
    try:
        connection.putrequest("POST", path, skip_accept_encoding=True)
        if length != "omit":
            connection.putheader("Content-Length", str(len(body)) if length is None else length)
        connection.endheaders(body)
        response = connection.getresponse()
        data = response.read()
        assert response.getheader("Content-Type") == "application/json"
        assert response.getheader("Content-Length") == str(len(data))
        return response.status, json.loads(data, parse_constant=lambda name: pytest.fail(f"response has {name}"))
    finally:
        connection.close()


# Found by tests/test_jev_server_fuzz.py: each of these used to escape parse_decide or _send, so the server
# dropped the connection instead of answering, the last one after running the model.
@pytest.mark.unit
@pytest.mark.parametrize(
    ("body", "length", "message"),
    [
        (b"\xff", None, "not JSON"),
        (b"[" * 100_000, None, "not JSON"),
        (b'{"id": 1e999, "parts": {}, "questions": {"q": {}}}', None, "out of range"),
        (b'{"id": NaN, "parts": {}, "questions": {"q": {}}}', None, "NaN, which is not JSON"),
        (b"{}", "two", "body must be 1 to"),
        (b"", "omit", "body must be 1 to"),
    ],
)
def test_a_body_the_parser_cannot_carry_is_a_400(serve, body, length, message) -> None:  # type: ignore[no-untyped-def]
    model = FakeDecisionModel()
    url = serve(model)
    status, reply = post_raw(url, body, length)
    assert status == 400 and message in str(reply["error"])
    assert model.peak == 0


# Found by mutation testing with ``mutmut``: each test below kills mutants that the tests above let survive.
@pytest.mark.unit
def test_exactly_the_branch_limit_is_answered_and_one_more_refused(serve) -> None:  # type: ignore[no-untyped-def]
    url = serve(FakeDecisionModel())
    request = {"parts": {}, "questions": QUESTION}
    status, body = call(url, "/v1/decide", {**request, "branches": {str(i): request for i in range(MAX_BRANCHES)}})
    assert status == 200 and len(body["branches"]) == MAX_BRANCHES  # type: ignore[arg-type]
    status, body = call(url, "/v1/decide", {**request, "branches": {str(i): request for i in range(MAX_BRANCHES + 1)}})
    assert status == 400 and "at most 16" in str(body["error"])


@pytest.mark.unit
def test_a_body_at_the_size_limit_is_read_and_one_byte_more_refused(serve) -> None:  # type: ignore[no-untyped-def]
    url = serve(FakeDecisionModel())
    request = json.dumps({"parts": {}, "questions": QUESTION}).encode()
    at_limit = request + b" " * (MAX_BODY_BYTES - len(request))
    assert post_raw(url, at_limit)[0] == 200
    status, body = post_raw(url, b"{}", str(MAX_BODY_BYTES + 1))
    assert status == 400 and "body must be 1 to" in str(body["error"])


@pytest.mark.unit
def test_a_body_that_is_not_an_object_and_an_unknown_field_are_named(serve) -> None:  # type: ignore[no-untyped-def]
    url = serve(FakeDecisionModel())
    assert call(url, "/v1/decide", [1])[1]["error"] == "body must be a JSON object"
    reply = call(url, "/v1/decide", {"parts": {}, "questions": QUESTION, "b": 1, "a": 2})[1]
    assert reply["error"] == "unknown fields: a, b"
    assert call(url, "/v1/other")[1]["error"] == "no route /v1/other"
    assert post_raw(url, b"{}", path="/v1/other") == (404, {"error": "no route /v1/other"})


class NaNModel(FakeDecisionModel):
    def predict(self, parts: Mapping[str, object], questions: Mapping[str, object]) -> Mapping[str, object]:
        return {"answers": {key: {"choice": "power", "probabilities": {"power": math.nan}} for key in questions}}


@pytest.mark.unit
def test_an_answer_strict_json_cannot_write_is_a_500(serve) -> None:  # type: ignore[no-untyped-def]
    status, body = post_raw(serve(NaNModel()), json.dumps({"parts": {}, "questions": QUESTION}).encode())
    assert status == 500 and "the model's answer is not JSON" in str(body["error"])


class ConcurrentModel(FakeDecisionModel):
    """An engine that batches concurrent calls itself, as SGLang does."""

    concurrent = True

    def __init__(self) -> None:
        super().__init__(delay=0.1)
        self.threads: set[int] = set()
        self.seen: list[tuple[object, object]] = []

    def predict(self, parts: Mapping[str, object], questions: Mapping[str, object]) -> Mapping[str, object]:
        self.threads.add(threading.get_ident())
        self.seen.append((dict(parts), dict(questions)))
        return super().predict(parts, questions)


@pytest.mark.unit
def test_a_concurrent_model_is_called_on_the_request_threads_at_once(serve) -> None:  # type: ignore[no-untyped-def]
    model = ConcurrentModel()
    url = serve(model)
    request = {"parts": {"essential": ["tick 1"]}, "questions": QUESTION}
    with ThreadPoolExecutor(max_workers=4) as pool:
        replies = list(pool.map(lambda _: call(url, "/v1/decide", request), range(4)))
    assert [status for status, _ in replies] == [200] * 4
    assert model.peak > 1 and len(model.threads) > 1
    assert model.seen == [(request["parts"], request["questions"])] * 4


@pytest.mark.unit
def test_one_branch_is_answered_through_predict_many(serve) -> None:  # type: ignore[no-untyped-def]
    model = SplitModel(split=False)
    url = serve(model)
    request = {"parts": {}, "questions": QUESTION}
    assert call(url, "/v1/decide", {**request, "branches": {"x": request}})[0] == 200
    assert model.batches == [2]
