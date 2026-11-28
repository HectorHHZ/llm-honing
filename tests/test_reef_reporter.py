"""The grader's Reef reports (#225): one report per graded episode, referencing its decision records in play order.

Tests that talk to Reef use the fake in ``tests/fakes/reef.py``, or a real Reef
when ``HONING_REEF_BIN`` names the ``reef`` executable of an environment with Reef
bc8c150. The module then starts one, with a recipe whose report type is
``integrations/reef/report.py``'s (``tests/fakes/reef_recipe.py``), so Reef parses
every report at admission. A test that needs the report type skips on the fake;
tests that take Reef down always use the fake.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("reef_client")

from reef_client import ReefClient, ReefClientError

from honing.backends.reef.records import EpisodeReport, RecordedDecision, batches, decision_record_id
from honing.backends.reef.reporter import ReefReporter, ReefReportingGrader, Skipped, episode_report
from honing.core import ContractError, EpisodeOutcome, Grade, Task
from tests.fakes.reef import FakeReef

REPO = Path(__file__).resolve().parents[1]
REEF_BIN = os.environ.get("HONING_REEF_BIN")
TOKEN = "reef-test"
FIXTURE = REPO / "tests/fixtures/decision_traces/s3114-blue-destroyed"
NODE = shutil.which("node")
TASK = Task(task_id="val-3114-blue", prompt="")


class RealReef:
    """One ``reef serve`` with the test recipe, on a free port, with its state in ``directory``."""

    def __init__(self, directory: Path) -> None:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.log = directory / "serve.log"
        self.process = subprocess.Popen(
            [
                str(REEF_BIN),
                "serve",
                "--reef.port",
                str(self.port),
                "--recipe.implementation",
                "tests.fakes.reef_recipe:EpisodeReportRecipe",
                "--inference.upstream-url",
                "http://127.0.0.1:9/v1",
                "--inference.upstream-model",
                "none",
            ],
            cwd=directory,
            env={**os.environ, "REEF_TOKEN": TOKEN, "PYTHONPATH": str(REPO)},
            stdout=self.log.open("ab"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        wait_for(self.healthy, 120, f"Reef did not start; see {self.log}")

    def healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.url}/healthz", timeout=2) as response:
                return response.status == 200
        except OSError:
            return False

    def stop(self) -> None:
        os.killpg(self.process.pid, signal.SIGTERM)
        try:
            self.process.wait(30)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)


class Store:
    """Writes records to, and reads reports back from, a Reef, fake or real."""

    def __init__(self, url: str, fake: FakeReef | None) -> None:
        self.url, self.fake = url, fake
        self.client = ReefClient(url, token=TOKEN)

    def import_records(self, scenario: str, records: list[dict[str, Any]]) -> None:
        for batch in batches(records):
            self.client.post("/reef/records/batch", scenario, {"records": batch})

    def reports(self, scenario: str) -> dict[str, dict[str, Any]]:
        """The scenario's stored reports by id, with their payloads."""
        if self.fake is not None:
            return self.fake.reports(scenario)
        listed: list[dict[str, Any]] = []
        after = 0
        while True:
            page = self.client.get(
                f"/reef/scenarios/{scenario}/records?request_type=report&limit=100&after_sequence={after}"
            )
            listed += page["records"]
            if not page["records"] or page.get("next_after_sequence") is None:
                break
            after = page["next_after_sequence"]
        path = f"/reef/scenarios/{scenario}/records"
        return {
            row["agent_record_id"]: self.client.get(f"{path}/{row['agent_record_id']}")["payload"] for row in listed
        }


@pytest.fixture(scope="module")
def real_reef(tmp_path_factory: pytest.TempPathFactory) -> Iterator[RealReef | None]:
    if not REEF_BIN:
        yield None
        return
    server = RealReef(tmp_path_factory.mktemp("reef"))
    yield server
    server.stop()


@pytest.fixture
def reef(real_reef: RealReef | None) -> Iterator[Store]:
    if real_reef is not None:
        yield Store(real_reef.url, None)
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


def wait_for(condition: Callable[[], bool], seconds: float = 20.0, message: str = "timed out") -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(message)
        time.sleep(0.02)


def reporter(url: str, scenario: str, **kwargs: Any) -> ReefReporter:
    kwargs.setdefault("backoff_seconds", (0.05, 0.2))
    kwargs.setdefault("reference_wait_seconds", 0.0)
    return ReefReporter(url, scenario, token=TOKEN, **kwargs)


def settled(sender: ReefReporter) -> Callable[[], bool]:
    return lambda: sender.counts()["pending"] == 0 and sender.counts()["episodes"] > 0


def run_dir(tmp_path: Path, tags: dict[str, str] | None) -> Path:
    """The fixture round as a run directory, with ``tags`` as its manifest's ``reef_tags``."""
    run = tmp_path / f"run-{uuid.uuid4().hex[:8]}"
    shutil.copytree(FIXTURE, run)
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["reef_tags"] = tags
    (run / "manifest.json").write_text(json.dumps(manifest))
    return run


def rows(run: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (run / "decisions.jsonl").read_text().splitlines() if line.strip()]


def outcome(run: Path) -> EpisodeOutcome:
    return EpisodeOutcome(
        task_id=TASK.task_id, spec_id="h-1", policy_checksum="p-1", exit_code=0, workspace_location=str(run)
    )


def tags(episode: str | None = None) -> dict[str, str]:
    return {"episode": episode or uuid.uuid4().hex, "task": TASK.task_id, "harness": "h-1", "policy": "p-1"}


def record_round(store: Store, scenario: str, run: Path, episode_tags: dict[str, str]) -> None:
    """What the decision server records for the round: one record per row."""
    store.import_records(scenario, [RecordedDecision.from_row(row, episode_tags).to_record() for row in rows(run)])


class Graded:
    """A grader that returns ``grade`` and counts its calls, or raises ``error``."""

    def __init__(self, grade: Grade | None = None, error: Exception | None = None) -> None:
        self.result, self.error, self.calls = grade, error, 0

    async def grade(self, task: Task, outcome: EpisodeOutcome) -> Grade:
        self.calls += 1
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


GRADE = Grade(score=0.0, feedback="The agent's tank was destroyed.")


# Making the report.


@pytest.mark.unit
def test_a_report_references_every_row_in_play_order_with_its_chosen_and_the_grade(tmp_path: Path) -> None:
    episode_tags = tags()
    run = run_dir(tmp_path, episode_tags)
    report = episode_report(TASK, outcome(run), GRADE)
    assert isinstance(report, EpisodeReport)
    played = rows(run)
    assert len(played) > 1
    assert report.references == tuple(decision_record_id(episode_tags["episode"], row["request_id"]) for row in played)
    assert [dict(chosen) for _, chosen in report.decisions] == [row["chosen"] for row in played]
    assert (report.episode, report.score, report.feedback, report.task) == (
        episode_tags["episode"],
        GRADE.score,
        GRADE.feedback,
        TASK.task_id,
    )
    payload = report.payload()
    assert set(payload["metadata"]["honing"]) == {"format", "version", "episode", "chosen"}
    assert EpisodeReport.from_payload(payload) == report


@pytest.mark.unit
@pytest.mark.parametrize("value", [None, {}, {"task": TASK.task_id}], ids=["null", "empty", "no-episode"])
def test_a_run_without_an_episode_tag_makes_no_report(tmp_path: Path, value: dict[str, str] | None) -> None:
    made = episode_report(TASK, outcome(run_dir(tmp_path, value)), GRADE)
    assert isinstance(made, Skipped) and made.reason == "no_tags"


@pytest.mark.unit
def test_a_run_with_no_manifest_or_no_directory_makes_no_report(tmp_path: Path) -> None:
    run = run_dir(tmp_path, tags())
    (run / "manifest.json").unlink()
    assert episode_report(TASK, outcome(run), GRADE) == Skipped("no_tags", f"{run} has no manifest.json")
    nowhere = EpisodeOutcome(task_id=TASK.task_id, spec_id="h-1", policy_checksum="p-1", exit_code=0)
    made = episode_report(TASK, nowhere, GRADE)
    assert isinstance(made, Skipped) and made.reason == "no_tags"


@pytest.mark.unit
@pytest.mark.parametrize("text", [None, "", "\n"], ids=["missing", "empty", "blank"])
def test_a_run_with_no_decision_rows_makes_no_report(tmp_path: Path, text: str | None) -> None:
    run = run_dir(tmp_path, tags())
    if text is None:
        (run / "decisions.jsonl").unlink()
    else:
        (run / "decisions.jsonl").write_text(text)
    made = episode_report(TASK, outcome(run), GRADE)
    assert isinstance(made, Skipped) and made.reason == "no_decisions"


@pytest.mark.unit
@pytest.mark.parametrize(
    "break_run",
    [
        lambda run: (run / "decisions.jsonl").write_text("{not json\n"),
        lambda run: (run / "decisions.jsonl").write_text(json.dumps({"chosen": {}}) + "\n"),
        lambda run: (run / "manifest.json").write_text("[1"),
        lambda run: (run / "decisions.jsonl").write_text((run / "decisions.jsonl").read_text() * 2),
    ],
    ids=["bad-json", "no-request-id", "bad-manifest", "repeated-request"],
)
def test_a_run_whose_files_are_broken_is_invalid_not_reported(tmp_path: Path, break_run: Any) -> None:
    run = run_dir(tmp_path, tags())
    break_run(run)
    made = episode_report(TASK, outcome(run), GRADE)
    assert isinstance(made, Skipped) and made.reason == "invalid"


@pytest.mark.unit
def test_a_run_tagged_with_another_task_is_invalid(tmp_path: Path) -> None:
    run = run_dir(tmp_path, {**tags(), "task": "val-9999-red"})
    made = episode_report(TASK, outcome(run), GRADE)
    assert isinstance(made, Skipped) and made.reason == "invalid"


# The grader wrapper.


@pytest.mark.functional
def test_the_wrapper_returns_the_grade_unchanged_and_its_errors_propagate_unreported(
    tmp_path: Path, fake_reef: FakeReef, scenario: str
) -> None:
    sender = reporter(fake_reef.url, scenario)
    graded = ReefReportingGrader(Graded(GRADE), sender)
    run = run_dir(tmp_path, None)
    assert asyncio.run(graded.grade(TASK, outcome(run))) is GRADE
    failing = ReefReportingGrader(Graded(error=ContractError("no outcome")), sender)
    with pytest.raises(ContractError, match="no outcome"):
        asyncio.run(failing.grade(TASK, outcome(run)))
    wait_for(settled(sender))
    sender.close()
    assert sender.counts()["episodes"] == 1 and sender.counts()["skipped_no_tags"] == 1
    assert fake_reef.posts == 0


@pytest.mark.unit
def test_without_a_reef_url_the_tank_grader_is_returned_as_before(tmp_path: Path) -> None:
    from honing.backends.tank import load_suite
    from honing.backends.tank.e2e import grader
    from honing.backends.tank.grader import TankGrader

    class Context:
        def __init__(self) -> None:
            self.root, self.work_dir = REPO, tmp_path
            self.shared = {"tank_suite": load_suite(REPO / "integrations/tank-war/tasks/suite.jsonl")}

    assert type(grader(Context())) is TankGrader
    with pytest.raises(ContractError, match="go together"):
        grader(Context(), reef_url="http://127.0.0.1:9")
    with pytest.raises(ContractError, match="go together"):
        grader(Context(), reef_scenario="s")


@pytest.mark.integration
def test_the_tank_grader_reports_with_the_reef_token_from_the_environment(
    tmp_path: Path, fake_reef: FakeReef, scenario: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from honing.backends.tank import load_suite
    from honing.backends.tank.e2e import grader

    class Context:
        def __init__(self) -> None:
            self.root, self.work_dir = REPO, tmp_path
            self.shared = {"tank_suite": load_suite(REPO / "integrations/tank-war/tasks/suite.jsonl")}

    monkeypatch.setenv("REEF_TOKEN", "token-from-env")
    graded = grader(Context(), reef_url=fake_reef.url, reef_scenario=scenario)
    assert isinstance(graded, ReefReportingGrader) and graded.reporter.scenario == scenario
    episode_tags = tags()
    run = run_dir(tmp_path, episode_tags)
    record_round(Store(fake_reef.url, fake_reef), scenario, run, episode_tags)
    imports = len(fake_reef.headers_seen)
    graded.reporter.submit(TASK, outcome(run), GRADE)
    wait_for(settled(graded.reporter))
    graded.close()
    assert graded.reporter.counts()["sent"] == 1
    assert [seen.get("authorization") for seen in fake_reef.headers_seen[imports:]] == ["Bearer token-from-env"]


@pytest.mark.unit
@pytest.mark.parametrize("fails", [False, True], ids=["finishes", "raises"])
def test_the_tank_campaign_closes_its_grader_when_the_campaign_ends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fails: bool
) -> None:
    from types import SimpleNamespace

    from honing.backends.tank import e2e

    class Closing:
        closed = 0

        def close(self) -> None:
            self.closed += 1

    closing = Closing()

    def run(scheduler: object) -> list[object]:
        assert closing.closed == 0, "closed before the campaign ended"
        if fails:
            raise RuntimeError("the campaign failed")
        return []

    work = tmp_path / "work"
    work.mkdir()
    built = SimpleNamespace(
        campaign=SimpleNamespace(run=run),
        scheduler=None,
        context=SimpleNamespace(work_dir=work, components={"grader": closing}),
    )
    monkeypatch.setattr(e2e, "build", lambda config, root, work_dir: built)
    monkeypatch.setattr(
        e2e, "summarize", lambda outcomes, run: {"steps": [], "head": "r0", "releases": "", "runs": "", "budget": None}
    )
    config = tmp_path / "config.json"
    config.write_text("{}")
    if fails:
        with pytest.raises(RuntimeError, match="the campaign failed"):
            e2e.main(["--config", str(config)])
    else:
        assert e2e.main(["--config", str(config)]) == 0
    assert closing.closed == 1


@pytest.mark.functional
def test_the_tank_grader_imports_no_reef_client_unless_asked_to_report() -> None:
    code = "import sys, honing.backends.tank.e2e; print('reef_client' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO)
    assert result.stdout.strip() == "False"


# Against Reef, fake or real.


@pytest.mark.integration
def test_a_graded_episode_is_one_report_and_a_resend_adds_nothing(tmp_path: Path, reef: Store, scenario: str) -> None:
    episode_tags = tags()
    run = run_dir(tmp_path, episode_tags)
    record_round(reef, scenario, run, episode_tags)
    sender = reporter(reef.url, scenario)
    graded = ReefReportingGrader(Graded(GRADE), sender)
    asyncio.run(graded.grade(TASK, outcome(run)))
    asyncio.run(graded.grade(TASK, outcome(run)))  # graded again, the same report: a retry
    wait_for(lambda: sender.counts()["pending"] == 0 and sender.counts()["episodes"] == 2)
    graded.close()
    assert sender.counts()["sent"] == 2 and sender.counts()["refused"] == 0
    stored = reef.reports(scenario)
    assert list(stored) == [EpisodeReport.from_rows(rows(run), episode=episode_tags["episode"], score=0).record_id]
    report = EpisodeReport.from_payload(next(iter(stored.values())))
    assert report.references == tuple(
        decision_record_id(episode_tags["episode"], row["request_id"]) for row in rows(run)
    )
    assert [dict(chosen) for _, chosen in report.decisions] == [row["chosen"] for row in rows(run)]
    assert (report.score, report.feedback, report.task) == (GRADE.score, GRADE.feedback, TASK.task_id)


@pytest.mark.integration
def test_a_changed_resend_of_a_report_is_refused_and_logged(
    tmp_path: Path, reef: Store, scenario: str, caplog: pytest.LogCaptureFixture
) -> None:
    episode_tags = tags()
    run = run_dir(tmp_path, episode_tags)
    record_round(reef, scenario, run, episode_tags)
    sender = reporter(reef.url, scenario)
    sender.submit(TASK, outcome(run), GRADE)
    wait_for(settled(sender))
    with caplog.at_level(logging.ERROR, logger="honing.backends.reef.reporter"):
        sender.submit(TASK, outcome(run), Grade(score=1.0, feedback="won"))
        wait_for(lambda: sender.counts()["pending"] == 0 and sender.counts()["episodes"] == 2)
    sender.close()
    assert sender.counts()["sent"] == 1 and sender.counts()["refused"] == 1
    assert episode_tags["episode"] in caplog.text and "409" in caplog.text
    assert EpisodeReport.from_payload(next(iter(reef.reports(scenario).values()))).score == GRADE.score


@pytest.mark.integration
def test_a_report_with_an_unknown_reference_is_refused_at_admission_logged_and_counted(
    tmp_path: Path, reef: Store, scenario: str, caplog: pytest.LogCaptureFixture
) -> None:
    episode_tags = tags()
    run = run_dir(tmp_path, episode_tags)
    played = rows(run)
    # The recorder lost the last decision: every other record is stored.
    reef.import_records(scenario, [RecordedDecision.from_row(row, episode_tags).to_record() for row in played[:-1]])
    report = EpisodeReport.from_rows(played, episode=episode_tags["episode"], score=0.0, task=TASK.task_id)
    with pytest.raises(ReefClientError) as refused:
        reef.client.report(scenario, {**report.payload(), "agent_record_id": report.record_id})
    assert refused.value.status == 400 and "must identify an existing inference" in refused.value.body
    sender = reporter(reef.url, scenario, reference_wait_seconds=0.3)
    with caplog.at_level(logging.ERROR, logger="honing.backends.reef.reporter"):
        sender.submit(TASK, outcome(run), GRADE)
        wait_for(settled(sender))
    sender.close()
    assert sender.counts()["unknown_reference"] == 1 and sender.counts()["sent"] == 0
    assert episode_tags["episode"] in caplog.text and TASK.task_id in caplog.text
    assert reef.reports(scenario) == {}


@pytest.mark.integration
@pytest.mark.parametrize(
    ("change", "typed"),
    [
        (lambda payload: {**payload, "score": "lost"}, False),
        (lambda payload: {**payload, "score": float("inf")}, False),
        (lambda payload: {**payload, "score": None}, True),
        (lambda payload: {**payload, "metadata": {"honing": payload["metadata"]["honing"]}}, True),
        (lambda payload: {**payload, "metadata": {**payload["metadata"], "task": 7}}, True),
    ],
    ids=["text-score", "infinite-score", "no-score", "no-task", "number-task"],
)
def test_a_malformed_report_is_refused_at_admission(
    tmp_path: Path, reef: Store, scenario: str, change: Callable[[dict[str, Any]], dict[str, Any]], typed: bool
) -> None:
    if typed and reef.fake is not None:
        pytest.skip("only a real Reef parses the report type; the fake admits what a record-only recipe admits")
    episode_tags = tags()
    run = run_dir(tmp_path, episode_tags)
    record_round(reef, scenario, run, episode_tags)
    report = EpisodeReport.from_rows(rows(run), episode=episode_tags["episode"], score=1.0, task=TASK.task_id)
    payload = change({**report.payload(), "agent_record_id": report.record_id})
    with pytest.raises(ReefClientError) as refused:
        reef.client.report(scenario, payload)
    assert refused.value.status == 400, refused.value.body
    assert reef.reports(scenario) == {}


@pytest.mark.integration
def test_a_report_waits_for_records_that_arrive_late(tmp_path: Path, reef: Store, scenario: str) -> None:
    episode_tags = tags()
    run = run_dir(tmp_path, episode_tags)
    sender = reporter(reef.url, scenario, reference_wait_seconds=30.0)
    sender.submit(TASK, outcome(run), GRADE)
    time.sleep(0.3)
    assert sender.counts()["pending"] == 1 and reef.reports(scenario) == {}
    record_round(reef, scenario, run, episode_tags)
    wait_for(settled(sender))
    sender.close()
    assert sender.counts()["sent"] == 1 and sender.counts()["unknown_reference"] == 0
    assert len(reef.reports(scenario)) == 1


# Reef down, or misconfigured: always the fake.


@pytest.mark.integration
def test_a_grade_never_waits_for_reef_and_its_report_arrives_once_reef_is_back(
    tmp_path: Path, fake_reef: FakeReef, scenario: str
) -> None:
    store = Store(fake_reef.url, fake_reef)
    episode_tags = tags()
    run = run_dir(tmp_path, episode_tags)
    record_round(store, scenario, run, episode_tags)
    fake_reef.down = True
    sender = reporter(fake_reef.url, scenario)
    graded = ReefReportingGrader(Graded(GRADE), sender)
    started = time.monotonic()
    assert asyncio.run(graded.grade(TASK, outcome(run))) is GRADE
    assert time.monotonic() - started < 0.5
    wait_for(lambda: fake_reef.posts >= 3)
    assert sender.counts()["pending"] == 1 and fake_reef.reports(scenario) == {}
    fake_reef.down = False
    wait_for(settled(sender))
    graded.close()
    assert sender.counts()["sent"] == 1 and len(fake_reef.reports(scenario)) == 1


@pytest.mark.integration
def test_while_reef_is_down_the_reporter_backs_off_however_many_episodes_are_graded(
    tmp_path: Path, fake_reef: FakeReef, scenario: str
) -> None:
    fake_reef.down = True
    sender = reporter(fake_reef.url, scenario, backoff_seconds=(0.2, 0.2))
    run = run_dir(tmp_path, tags())
    for _ in range(50):
        sender.submit(TASK, outcome(run), GRADE)
        time.sleep(0.01)
    posts = fake_reef.posts
    sender.close()
    # 0.5 s of grading with a backoff of 0.1 to 0.2 s: at most a handful of attempts, not one per episode.
    assert posts <= 8, posts


@pytest.mark.integration
def test_a_wrong_token_or_url_is_logged_and_counted_apart_and_the_report_waits(
    tmp_path: Path, fake_reef: FakeReef, scenario: str, caplog: pytest.LogCaptureFixture
) -> None:
    from http import HTTPStatus

    store = Store(fake_reef.url, fake_reef)
    episode_tags = tags()
    run = run_dir(tmp_path, episode_tags)
    record_round(store, scenario, run, episode_tags)
    for status, count in ((HTTPStatus.UNAUTHORIZED, "rejected_token"), (HTTPStatus.NOT_FOUND, "no_route")):
        fake_reef.answer = status
        sender = reporter(fake_reef.url, scenario)
        with caplog.at_level(logging.ERROR, logger="honing.backends.reef.reporter"):
            sender.submit(TASK, outcome(run), GRADE)
            wait_for(lambda sender=sender, count=count: sender.counts()[count] >= 2)  # type: ignore[misc]
        assert sender.counts()["pending"] == 1 and sender.counts()["refused"] == 0
        fake_reef.answer = None
        wait_for(settled(sender))
        sender.close()
        assert sender.counts()["sent"] == 1
    assert "REEF_TOKEN" in caplog.text and "reef_url" in caplog.text


@pytest.mark.integration
def test_closing_while_reef_is_down_stops_in_time_and_counts_what_is_unsent(
    tmp_path: Path, fake_reef: FakeReef, scenario: str, caplog: pytest.LogCaptureFixture
) -> None:
    fake_reef.down = True
    sender = reporter(fake_reef.url, scenario, flush_seconds=0.3)
    sender.submit(TASK, outcome(run_dir(tmp_path, tags())), GRADE)
    started = time.monotonic()
    with caplog.at_level(logging.WARNING, logger="honing.backends.reef.reporter"):
        sender.close()
    assert time.monotonic() - started < 2.0
    assert sender.counts()["unsent"] == 1 and "not sent" in caplog.text
    sender.submit(TASK, outcome(run_dir(tmp_path, tags())), GRADE)
    assert sender.counts()["dropped"] == 1


@pytest.mark.functional
def test_past_the_pending_limit_an_episode_is_graded_and_its_report_dropped_loudly(
    tmp_path: Path,
    fake_reef: FakeReef,
    scenario: str,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from honing.backends.reef import reporter as module

    made = module.episode_report

    def slowly(*args: Any) -> Any:
        # A report being made is in neither queue: the limit must still count it.
        time.sleep(0.1)
        return made(*args)

    monkeypatch.setattr(module, "episode_report", slowly)
    fake_reef.down = True
    sender = reporter(fake_reef.url, scenario, max_pending_reports=2, flush_seconds=0.0)
    graded = ReefReportingGrader(Graded(GRADE), sender)
    run = run_dir(tmp_path, tags())
    with caplog.at_level(logging.WARNING, logger="honing.backends.reef.reporter"):
        for _ in range(4):
            assert asyncio.run(graded.grade(TASK, outcome(run))) is GRADE
    graded.close()
    assert sender.counts()["dropped"] == 2 and "not reported" in caplog.text


# A real tank round: the decision server records, the grader reports (#225's "done when").


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(NODE is None, reason="needs node")
@pytest.mark.parametrize("sample", [False, True], ids=["evaluation", "rollout"])
def test_every_graded_episode_of_a_tank_round_is_one_report_referencing_its_decisions(
    reef: Store, scenario: str, tmp_path: Path, sample: bool
) -> None:
    import threading

    from honing.backends.jev.server import DecisionServer
    from honing.backends.reef.recorder import ReefRecorder
    from honing.backends.tank import TankEpisodeRunner, load_suite, seed_harness
    from honing.backends.tank.grader import TankGrader
    from honing.core import PolicyEndpoint, PolicyRef
    from tests.fakes.model import FakeModelSource, FakeWorld
    from tests.fakes.tank import ForecastModel

    integration = REPO / "integrations/tank-war"
    recorder = ReefRecorder(reef.url, scenario, token=TOKEN, backoff_seconds=(0.05, 0.2))
    server = DecisionServer(("127.0.0.1", 0), ForecastModel(0.9), recorder)
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    try:

        class One:
            def resolve(self, policy: Any) -> PolicyEndpoint:
                return PolicyEndpoint(server.url, "fake", "forecast", runtime="jev_decision")

        suite = load_suite(integration / "tasks/suite.jsonl")
        runner = TankEpisodeRunner(suite=suite, endpoints=One(), integration_dir=integration, runs_dir=tmp_path)
        if sample:
            runner = runner.sampling()
        sender = reporter(reef.url, scenario, reference_wait_seconds=60.0)
        graded = ReefReportingGrader(TankGrader(suite=suite), sender)
        snapshot = FakeModelSource(FakeWorld()).materialize(tmp_path / "model")
        policy = PolicyRef(snapshot.checkpoint, snapshot.architecture, "master")
        played: list[tuple[Task, EpisodeOutcome, Grade]] = []
        for task in suite.tasks("val", limit=2):
            episode = asyncio.run(runner.run(seed_harness(integration), task, policy))
            assert episode.exit_code == 0
            # Graded as soon as the round ends: its records may still be on the recorder's thread.
            played.append((task, episode, asyncio.run(graded.grade(task, episode))))
        wait_for(lambda: sender.counts()["pending"] == 0, 90)
        graded.close()
    finally:
        server.stop()
        thread.join(timeout=30)
        server.server_close()
    counts = sender.counts()
    assert counts["sent"] == len(played) == 2, counts
    assert counts["refused"] == counts["unknown_reference"] == counts["invalid"] == 0
    stored = reef.reports(scenario)
    assert len(stored) == len(played)
    for task, episode, grade in played:
        run = Path(episode.workspace_location)  # type: ignore[arg-type]
        episode_tags = json.loads((run / "manifest.json").read_text())["reef_tags"]
        expected = EpisodeReport.from_rows(
            rows(run), episode=episode_tags["episode"], score=grade.score, feedback=grade.feedback, task=task.task_id
        )
        report = EpisodeReport.from_payload(stored[expected.record_id])
        assert report == expected
        assert [dict(chosen) for _, chosen in report.decisions] == [row["chosen"] for row in rows(run)]
        assert report.references == tuple(
            decision_record_id(episode_tags["episode"], row["request_id"]) for row in rows(run)
        )
