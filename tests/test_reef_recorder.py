"""The decision server's Reef recorder (#223): its records, and that recording never touches a decision.

Tests that talk to Reef use the fake in ``tests/fakes/reef.py``, or a real Reef when
``HONING_REEF_URL`` (and ``REEF_TOKEN``) name one. Tests that take Reef down always use the fake.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("reef_client")

from honing.backends.jev.server import DecisionServer
from honing.backends.reef.recorder import ReefRecorder, backoff_seconds, decision_records, request_tags
from honing.backends.reef.records import RecordedDecision, decision_record_id
from tests.fakes.reef import FakeReef
from tests.fakes.tank import ForecastModel

REPO = Path(__file__).resolve().parents[1]
REAL_REEF = os.environ.get("HONING_REEF_URL")
TOKEN = os.environ.get("REEF_TOKEN")
CRITERIA = {"power": "Power plant", "wait": "Wait"}
QUESTION = {"root": {"type": "choice", "instructions": "Next?", "criteria": CRITERIA}}
TAGS = {"episode": "ep-1", "task": "t-1", "harness": "h-1", "policy": "p-1"}


class ProbabilityModel:
    """Satisfies DecisionModel: the first option with probability 0.75, and the state text, as a Jev model answers."""

    metadata: Mapping[str, object] = {"family": "fake", "model": "fake"}

    def __init__(self) -> None:
        self.threads: set[int] = set()

    def predict(self, parts: Mapping[str, Any], questions: Mapping[str, Any]) -> Mapping[str, object]:
        self.threads.add(threading.get_ident())
        answers = {}
        for key, question in questions.items():
            options = list(question["criteria"])
            probabilities = {o: (0.75 if i == 0 else 0.25 / (len(options) - 1)) for i, o in enumerate(options)}
            answers[key] = {"type": "choice", "choice": options[0], "probabilities": probabilities}
        return {"answers": answers, "stateText": "\n".join(parts.get("essential", []))}


class Store:
    """Reads back what a Reef, fake or real, stored for one scenario."""

    def __init__(self, url: str, fake: FakeReef | None) -> None:
        self.url, self.fake = url, fake

    def payload(self, scenario: str, record_id: str) -> dict[str, Any] | None:
        if self.fake is not None:
            return self.fake.payloads(scenario).get(record_id)
        request = urllib.request.Request(
            f"{self.url}/reef/scenarios/{scenario}/records/{record_id}",
            headers={"Authorization": f"Bearer {TOKEN}"} if TOKEN else {},
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return json.loads(response.read())["payload"]  # type: ignore[no-any-return]
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            raise


@pytest.fixture
def reef() -> Iterator[Store]:
    if REAL_REEF:
        yield Store(REAL_REEF.rstrip("/"), None)
        return
    fake = FakeReef()
    yield Store(fake.url, fake)
    fake.close()


@pytest.fixture
def fake_reef() -> Iterator[FakeReef]:
    fake = FakeReef()
    yield fake
    fake.close()


@pytest.fixture
def scenario() -> str:
    # A real Reef keeps every scenario, so each test writes to a new one.
    return f"honing-test-{uuid.uuid4().hex[:12]}"


@pytest.fixture
def serve() -> Iterator[Any]:
    started: list[tuple[DecisionServer, threading.Thread]] = []

    def start(model: Any, recorder: ReefRecorder | None) -> DecisionServer:
        server = DecisionServer(("127.0.0.1", 0), model, recorder)
        thread = threading.Thread(target=server.serve, daemon=True)
        thread.start()
        started.append((server, thread))
        return server

    yield start
    for server, thread in started:
        server.stop()
        thread.join(timeout=30)
        server.server_close()


def decide(url: str, body: Mapping[str, Any], tags: Mapping[str, str] | None = None) -> tuple[int, dict[str, Any]]:
    headers = {"Content-Type": "application/json", **{f"x-reef-tag-{k}": v for k, v in (tags or {}).items()}}
    request = urllib.request.Request(url + "/v1/decide", data=json.dumps(body).encode(), headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def health(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url + "/v1/health", timeout=10) as response:
        return json.loads(response.read())  # type: ignore[no-any-return]


def wait_for(condition: Any, seconds: float = 20.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("timed out")
        time.sleep(0.02)


def recorder(url: str, scenario: str, **kwargs: Any) -> ReefRecorder:
    kwargs.setdefault("backoff_seconds", (0.05, 0.2))
    return ReefRecorder(url, scenario, token=TOKEN, **kwargs)


def body(request_id: object = "d1", **branches: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"id": request_id, "parts": {"essential": [f"state {request_id}"]}, "questions": QUESTION}
    if branches:
        out["branches"] = dict(branches)
    return out


def reply(*choices: str, branches: Mapping[str, Any] | None = None) -> dict[str, Any]:
    answers = {"root": {"choice": choices[0], "probabilities": {"power": 0.75, "wait": 0.25}}}
    return {"answers": answers, **({"branches": branches} if branches else {})}


# The mapping from a request and its reply to records.


@pytest.mark.unit
def test_a_request_and_each_branch_are_one_record_each_with_the_branch_named() -> None:
    sent = body("m", fire={"id": "m:weapon:fire", "parts": {"essential": ["f"]}, "questions": QUESTION})
    records, skipped = decision_records(sent, reply("power", branches={"fire": reply("wait")}), TAGS)
    assert skipped == []
    assert [(r.request_id, r.branch) for r in records] == [("m", None), ("m:weapon:fire", "fire")]
    assert records[1].parts == {"essential": ["f"]} and dict(records[1].tags) == TAGS
    assert len({r.record_id for r in records}) == 2


@pytest.mark.unit
@pytest.mark.parametrize(
    ("request_id", "recorded"), [(7, "7"), (0, "0"), ("blue-1:motion", "blue-1:motion"), (True, None), (1.0, None)]
)
def test_an_integer_id_is_recorded_as_its_string_and_another_non_string_id_is_skipped(
    request_id: object, recorded: str | None
) -> None:
    records, skipped = decision_records(body(request_id), reply("power"), TAGS)
    if recorded is None:
        assert records == [] and [s.reason for s in skipped] == ["no_id"]
    else:
        assert [r.request_id for r in records] == [recorded] and skipped == []


@pytest.mark.unit
def test_a_branch_without_an_id_is_skipped_and_the_rest_recorded() -> None:
    sent = body("m", a={"parts": {}, "questions": QUESTION}, b={"id": "m:b", "parts": {}, "questions": QUESTION})
    records, skipped = decision_records(sent, reply("power", branches={"a": reply("wait"), "b": reply("wait")}), TAGS)
    assert [r.request_id for r in records] == ["m", "m:b"]
    assert [s.reason for s in skipped] == ["no_id"]


@pytest.mark.unit
def test_a_request_without_an_episode_tag_makes_no_record() -> None:
    sent = body("m", a={"id": "m:a", "parts": {}, "questions": QUESTION})
    records, skipped = decision_records(sent, reply("power", branches={"a": reply("wait")}), {"task": "t"})
    assert records == [] and [s.reason for s in skipped] == ["no_episode", "no_episode"]


@pytest.mark.unit
def test_an_answer_without_probabilities_is_invalid_not_recorded() -> None:
    records, skipped = decision_records(body(), {"answers": {"root": {"choice": "power"}}}, TAGS)
    assert records == [] and [s.reason for s in skipped] == ["invalid"]


@pytest.mark.unit
def test_request_tags_keep_honing_s_tags_only() -> None:
    tags, ignored = request_tags(
        [("X-Reef-Tag-Episode", " e "), ("x-reef-tag-task", ""), ("x-reef-tag-seat", "blue"), ("Content-Type", "a")]
    )
    assert tags == {"episode": "e"} and ignored == ["seat"]


@pytest.mark.unit
def test_the_backoff_doubles_to_its_cap_and_survives_an_outage_of_any_length() -> None:
    assert 0.25 <= backoff_seconds(1, 0.5, 30.0) <= 0.5
    assert 1.0 <= backoff_seconds(3, 0.5, 30.0) <= 2.0
    # An outage of hours is thousands of attempts, and 2.0 ** 1024 overflows.
    for failures in (64, 1025, 10**6):
        assert 15.0 <= backoff_seconds(failures, 0.5, 30.0) <= 30.0


# The server and Reef.


@pytest.mark.integration
def test_every_answered_request_and_branch_is_one_reef_record_of_what_was_asked_and_answered(
    serve: Any, reef: Store, scenario: str
) -> None:
    server = serve(ProbabilityModel(), recorder(reef.url, scenario))
    sent = body("m", x={"id": "m:x", "parts": {"essential": ["x"]}, "questions": QUESTION})
    status, answer = decide(server.url, sent, TAGS)
    assert status == 200
    wait_for(lambda: health(server.url)["reef"]["sent"] == 2)
    for request_id, branch, parts, answers in (
        ("m", None, sent["parts"], answer["answers"]),
        ("m:x", "x", {"essential": ["x"]}, answer["branches"]["x"]["answers"]),
    ):
        stored = reef.payload(scenario, decision_record_id("ep-1", request_id))
        assert stored is not None
        assert RecordedDecision.from_payload(stored) == RecordedDecision(
            request_id=request_id, parts=parts, questions=QUESTION, answers=answers, tags=TAGS, branch=branch
        )
    counts = health(server.url)["reef"]
    assert {k: counts[k] for k in ("requests", "sent", "refused", "pending_records", "pending_bytes")} == {
        "requests": 1,
        "sent": 2,
        "refused": 0,
        "pending_records": 0,
        "pending_bytes": 0,
    }


@pytest.mark.integration
def test_a_request_with_no_episode_tag_is_answered_and_counted_not_recorded(
    serve: Any, fake_reef: FakeReef, scenario: str
) -> None:
    server = serve(ProbabilityModel(), recorder(fake_reef.url, scenario))
    assert decide(server.url, body("a"), {"task": "t"})[0] == 200
    assert decide(server.url, body(None), TAGS)[0] == 200
    wait_for(lambda: health(server.url)["reef"]["skipped_no_id"] == 1)
    counts = health(server.url)["reef"]
    assert counts["skipped_no_episode"] == 1 and counts["sent"] == 0
    assert fake_reef.posts == 0


@pytest.mark.integration
def test_the_server_answers_while_reef_is_down_and_its_records_arrive_once_reef_is_back(
    serve: Any, fake_reef: FakeReef, scenario: str
) -> None:
    fake_reef.down = True
    server = serve(ProbabilityModel(), recorder(fake_reef.url, scenario))
    for i in range(20):
        started = time.monotonic()
        assert decide(server.url, body(f"d{i}"), TAGS)[0] == 200
        assert time.monotonic() - started < 1.0
    wait_for(lambda: fake_reef.posts >= 3)  # it keeps trying
    waiting = health(server.url)["reef"]
    assert waiting["pending_requests"] + waiting["pending_records"] == 20 and waiting["sent"] == 0
    fake_reef.down = False
    wait_for(lambda: health(server.url)["reef"]["sent"] == 20)
    assert set(fake_reef.payloads(scenario)) == {decision_record_id("ep-1", f"d{i}") for i in range(20)}


@pytest.mark.integration
@pytest.mark.parametrize(
    ("status", "count", "message"),
    [
        (401, "rejected_token", "rejected the token"),
        (403, "rejected_token", "rejected the token"),
        (404, "no_route", "no such route, check --reef-url"),
    ],
)
def test_a_wrong_token_or_url_is_logged_and_counted_apart_and_its_records_wait(
    serve: Any,
    fake_reef: FakeReef,
    scenario: str,
    caplog: pytest.LogCaptureFixture,
    status: int,
    count: str,
    message: str,
) -> None:
    from http import HTTPStatus

    fake_reef.answer = HTTPStatus(status)
    server = serve(ProbabilityModel(), recorder(fake_reef.url, scenario))
    with caplog.at_level(logging.WARNING, logger="honing.backends.reef.recorder"):
        for i in range(3):
            assert decide(server.url, body(f"d{i}"), TAGS)[0] == 200
        wait_for(lambda: fake_reef.posts >= 3)
    counts = health(server.url)["reef"]
    assert counts[count] >= 3 and counts["sent"] == 0 and counts["refused"] == 0
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1 and message in errors[0]  # once, then a summary every 30 s
    assert not any("not taking records" in m for m in caplog.messages)  # not reported as an outage
    fake_reef.answer = None
    wait_for(lambda: health(server.url)["reef"]["sent"] == 3)


@pytest.mark.integration
def test_while_reef_is_down_the_recorder_backs_off_however_busy_the_server_is(fake_reef: FakeReef) -> None:
    fake_reef.down = True
    sender = recorder(fake_reef.url, "s", backoff_seconds=(0.4, 0.4), flush_seconds=0.0)
    started = time.monotonic()
    while time.monotonic() - started < 1.2:
        sender.submit(
            json.dumps(body("d")).encode(), json.dumps(reply("power")).encode(), [("x-reef-tag-episode", "e")]
        )
        time.sleep(0.01)
    sender.close()
    # One try, then one every 0.2 to 0.4 s: at most 7 in 1.2 s, where a retry per submit would be about a hundred.
    assert 2 <= fake_reef.posts <= 7


@pytest.mark.integration
def test_records_wait_while_nothing_listens_and_arrive_when_reef_starts() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    sender = recorder(f"http://127.0.0.1:{port}", "s")
    sender.submit(json.dumps(body("d")).encode(), json.dumps(reply("power")).encode(), [("x-reef-tag-episode", "e")])
    time.sleep(0.5)
    assert sender.counts()["pending_records"] == 1
    fake = FakeReef(port)
    try:
        wait_for(lambda: sender.counts()["sent"] == 1)
        assert set(fake.payloads("s")) == {decision_record_id("e", "d")}
    finally:
        sender.close()
        fake.close()


@pytest.mark.integration
def test_a_resent_batch_adds_nothing_and_a_changed_resend_is_refused(reef: Store, scenario: str) -> None:
    from reef_client import ReefClient, ReefClientError

    client = ReefClient(reef.url, token=TOKEN)
    records = [RecordedDecision("r1", {"essential": ["s"]}, QUESTION, reply("power")["answers"], TAGS).to_record()]
    first, _ = client.post("/reef/records/batch", scenario, {"records": records})
    again, _ = client.post("/reef/records/batch", scenario, {"records": records})
    assert first == again
    changed = json.loads(json.dumps(records))
    changed[0]["payload"]["honing"]["answers"]["root"]["choice"] = "wait"
    with pytest.raises(ReefClientError) as refused:
        client.post("/reef/records/batch", scenario, {"records": changed})
    assert refused.value.status == 409
    stored = reef.payload(scenario, records[0]["agent_record_id"])
    assert stored is not None and stored["honing"]["answers"]["root"]["choice"] == "power"


@pytest.mark.integration
def test_a_record_reef_refuses_is_logged_and_counted_and_the_rest_of_its_batch_stored(
    reef: Store, scenario: str, caplog: pytest.LogCaptureFixture
) -> None:
    from reef_client import ReefClient

    # Store request d3 with other answers first, so the recorder's d3 is a changed resend.
    other = RecordedDecision("d3", {"essential": ["state d3"]}, QUESTION, reply("wait")["answers"], TAGS)
    ReefClient(reef.url, token=TOKEN).post("/reef/records/batch", scenario, {"records": [other.to_record()]})
    sender = recorder(reef.url, scenario)
    headers = [(f"x-reef-tag-{k}", v) for k, v in TAGS.items()]
    with caplog.at_level(logging.WARNING, logger="honing.backends.reef.recorder"):
        for i in range(6):
            answers = {"root": {"choice": "power", "probabilities": {"power": 0.75, "wait": 0.25}}}
            if i == 3:
                answers = {"root": {"choice": "power", "probabilities": {"power": 0.5, "wait": 0.5}}}
            sender.submit(json.dumps(body(f"d{i}")).encode(), json.dumps({"answers": answers}).encode(), headers)
        wait_for(lambda: sender.counts()["sent"] + sender.counts()["refused"] == 6)
        sender.close()
    assert sender.counts()["sent"] == 5 and sender.counts()["refused"] == 1
    assert sender.counts()["pending_bytes"] == 0
    refused_id = decision_record_id("ep-1", "d3")
    assert any(refused_id in message and "409" in message for message in caplog.messages)
    for i in (0, 1, 2, 4, 5):
        assert reef.payload(scenario, decision_record_id("ep-1", f"d{i}")) is not None


@pytest.mark.integration
def test_a_record_larger_than_any_batch_is_invalid_and_the_records_around_it_are_sent(
    fake_reef: FakeReef, caplog: pytest.LogCaptureFixture
) -> None:
    sender = recorder(fake_reef.url, "s", max_pending_bytes=64 * 1024 * 1024)
    headers = [("x-reef-tag-episode", "e")]
    big = body("big")
    big["parts"] = {"essential": ["x" * (10 * 1024 * 1024)]}
    with caplog.at_level(logging.WARNING, logger="honing.backends.reef.recorder"):
        for sent in (body("a"), big, body("b")):
            sender.submit(json.dumps(sent).encode(), json.dumps(reply("power")).encode(), headers)
        wait_for(lambda: sender.counts()["sent"] == 2)
        sender.close()
    assert sender.counts()["invalid"] == 1 and sender.counts()["pending_bytes"] == 0
    assert set(fake_reef.payloads("s")) == {decision_record_id("e", "a"), decision_record_id("e", "b")}
    assert any("big" in message or "over" in message for message in caplog.messages)


@pytest.mark.integration
def test_a_fault_while_sending_is_logged_and_the_batch_sent_again(
    fake_reef: FakeReef, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    sender = recorder(fake_reef.url, "s")
    faults = iter([AttributeError("a receipt that is not an object")])
    deliver = sender._deliver

    def flaky(entries: Any) -> int:
        fault = next(faults, None)
        if fault is not None:
            raise fault
        return deliver(entries)

    monkeypatch.setattr(sender, "_deliver", flaky)
    with caplog.at_level(logging.WARNING, logger="honing.backends.reef.recorder"):
        tags = [("x-reef-tag-episode", "e")]
        sender.submit(json.dumps(body("a")).encode(), json.dumps(reply("power")).encode(), tags)
        wait_for(lambda: sender.counts()["sent"] == 1)
        sender.close()
    assert any("a receipt that is not an object" in message for message in caplog.messages)


@pytest.mark.integration
def test_past_the_memory_limit_a_decision_is_answered_and_its_record_dropped_loudly(
    serve: Any, fake_reef: FakeReef, scenario: str, caplog: pytest.LogCaptureFixture
) -> None:
    fake_reef.down = True
    # Room for about two requests and their replies.
    server = serve(ProbabilityModel(), recorder(fake_reef.url, scenario, max_pending_bytes=900))
    with caplog.at_level(logging.WARNING, logger="honing.backends.reef.recorder"):
        statuses = [decide(server.url, body(f"d{i}"), TAGS)[0] for i in range(5)]
    assert statuses == [200] * 5
    counts = health(server.url)["reef"]
    assert counts["requests"] == 5 and counts["dropped_requests"] >= 1
    assert counts["pending_bytes"] <= 900
    assert any("not recorded" in message for message in caplog.messages)
    fake_reef.down = False
    wait_for(lambda: health(server.url)["reef"]["sent"] == 5 - counts["dropped_requests"])


@pytest.mark.integration
def test_closing_while_reef_is_down_stops_in_time_and_logs_what_is_unsent(
    fake_reef: FakeReef, caplog: pytest.LogCaptureFixture
) -> None:
    fake_reef.down = True
    sender = recorder(fake_reef.url, "s", flush_seconds=0.5, timeout_seconds=1.0)
    sender.submit(json.dumps(body("d")).encode(), json.dumps(reply("power")).encode(), [("x-reef-tag-episode", "e")])
    started = time.monotonic()
    with caplog.at_level(logging.WARNING, logger="honing.backends.reef.recorder"):
        sender.close()
    # Within the flush time and one backoff step, not the join's safety margin (flush + timeout + 1 s).
    assert time.monotonic() - started < 1.5
    assert sender.counts()["unsent"] == 1
    assert any("not sent" in message for message in caplog.messages)
    posts = fake_reef.posts
    time.sleep(0.5)
    assert fake_reef.posts == posts  # the recorder's thread has stopped


@pytest.mark.integration
def test_the_model_runs_on_the_serving_thread_and_recording_never_does(serve: Any, fake_reef: FakeReef) -> None:
    submitted: list[int] = []

    closed: list[bool] = []

    class Spy(ReefRecorder):
        def submit(self, *args: Any) -> None:
            submitted.append(threading.get_ident())
            super().submit(*args)

        def close(self) -> None:
            closed.append(True)
            super().close()

    model = ProbabilityModel()
    server = DecisionServer(("127.0.0.1", 0), model, Spy(fake_reef.url, "s", backoff_seconds=(0.05, 0.2)))  # type: ignore[arg-type]
    serving: list[int] = []
    thread = threading.Thread(target=lambda: (serving.append(threading.get_ident()), server.serve()), daemon=True)
    thread.start()
    for i in range(4):
        assert decide(server.url, body(f"d{i}"), TAGS)[0] == 200
    server.stop()
    thread.join(timeout=30)
    server.server_close()
    assert model.threads == set(serving)
    assert len(submitted) == 4 and not set(submitted) & set(serving)
    assert closed == [True]  # stopping the server closed, and so flushed, the recorder
    assert len(fake_reef.payloads("s")) == 4


@pytest.mark.functional
def test_the_server_imports_no_reef_client_unless_asked_to_record() -> None:
    code = "import sys, honing.backends.jev.server as s; s.main(['--help'])"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    assert "--reef-url" in result.stdout
    check = "import sys, honing.backends.jev.server; print('reef_client' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", check], capture_output=True, text=True, cwd=REPO).stdout == "False\n"


@pytest.mark.unit
def test_reef_url_and_scenario_go_together() -> None:
    from honing.backends.jev.server import main

    with pytest.raises(SystemExit):
        main(["--family", "djev", "--reef-url", "http://127.0.0.1:1"])


# A tank round, end to end: the Node match process against the decision server, recorded in Reef.

NODE = shutil.which("node")


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(NODE is None, reason="needs node")
@pytest.mark.parametrize("sample", [False, True], ids=["evaluation", "rollout"])
def test_every_request_of_a_tank_round_is_one_reef_record_matching_its_decision_log_row(
    serve: Any, reef: Store, scenario: str, tmp_path: Path, sample: bool
) -> None:
    import asyncio

    from honing.backends.tank import TankEpisodeRunner, load_suite, seed_harness
    from honing.core import PolicyEndpoint, PolicyRef
    from tests.fakes.model import FakeModelSource, FakeWorld

    integration = REPO / "integrations/tank-war"
    server = serve(ForecastModel(0.9), recorder(reef.url, scenario))

    class One:
        def resolve(self, policy: Any) -> PolicyEndpoint:
            return PolicyEndpoint(server.url, "fake", "forecast", runtime="jev_decision")

    suite = load_suite(integration / "tasks/suite.jsonl")
    runner = TankEpisodeRunner(suite=suite, endpoints=One(), integration_dir=integration, runs_dir=tmp_path)
    if sample:
        runner = runner.sampling()
    task = suite.tasks("val", limit=1)[0]
    snapshot = FakeModelSource(FakeWorld()).materialize(tmp_path / "model")
    policy = PolicyRef(snapshot.checkpoint, snapshot.architecture, "master")
    outcome = asyncio.run(runner.run(seed_harness(integration), task, policy))
    assert outcome.exit_code == 0
    run = Path(outcome.workspace_location)  # type: ignore[arg-type]
    tags = json.loads((run / "manifest.json").read_text())["reef_tags"]
    assert set(tags) == {"episode", "task", "harness", "policy"} and tags["task"] == task.task_id
    rows = [json.loads(line) for line in (run / "decisions.jsonl").read_text().splitlines()]
    assert rows
    counts: dict[str, Any] = {}

    def settled() -> bool:
        counts.update(health(server.url)["reef"])
        return counts["pending_requests"] == 0 and counts["pending_records"] == 0

    wait_for(settled, 60)
    assert counts["refused"] == 0 and counts["invalid"] == 0 and counts["dropped_requests"] == 0
    for row in rows:
        stored = reef.payload(scenario, decision_record_id(tags["episode"], row["request_id"]))
        assert stored is not None, row["request_id"]
        recorded = RecordedDecision.from_payload(stored)
        assert replace_branch(recorded) == RecordedDecision.from_row(row, tags)
    # Every request the server answered is a record: the rows' requests, and the weapon branches the game did not use.
    assert counts["sent"] > len(rows)


def replace_branch(record: RecordedDecision) -> RecordedDecision:
    """``record`` as ``RecordedDecision.from_row`` builds it: a decisions.jsonl row does not name its branch."""
    import dataclasses

    return dataclasses.replace(record, branch=None)
