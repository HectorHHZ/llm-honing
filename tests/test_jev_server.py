"""The decision server and endpoint map, with a fake model: no GPU, no torch."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import ClassVar

import pytest

from honing.backends.jev import DecisionModel, JevEndpoints
from honing.backends.jev.server import DecisionServer, RequestError, parse_decide
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


def call(url: str, path: str, body: object | None = None) -> tuple[int, dict[str, object]]:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


QUESTION = {"root": {"type": "choice", "instructions": "Next?", "criteria": {"power": "Power plant", "wait": "Wait"}}}


@pytest.mark.unit
def test_health_reports_what_is_loaded(serve) -> None:  # type: ignore[no-untyped-def]
    url = serve(FakeDecisionModel())
    assert call(url, "/v1/health") == (200, {"ready": True, "metadata": {"revision": "fake", "dtype": "none"}})


@pytest.mark.unit
def test_a_decision_is_answered_and_its_id_echoed(serve) -> None:  # type: ignore[no-untyped-def]
    url = serve(FakeDecisionModel())
    status, body = call(url, "/v1/decide", {"id": 7, "parts": {"essential": ["tick 30"]}, "questions": QUESTION})
    assert status == 200
    assert body == {"answers": {"root": {"choice": "power"}}, "stateText": "tick 30", "id": 7}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"parts": {}}, "questions must be a non-empty object"),
        ({"parts": [], "questions": QUESTION}, "parts must be an object"),
        ({"parts": {}, "questions": QUESTION, "temperature": 2}, "unknown fields: temperature"),
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
