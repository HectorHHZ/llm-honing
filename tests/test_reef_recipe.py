"""Honing's Reef recipe (#226) against a real Reef: records in, job data spooled for Honing, nothing published.

Reef runs as its own service in its own Python 3.12 environment, as in
production. Set ``HONING_REEF_BIN`` to the ``reef`` executable of an
environment with Reef bc8c150 (the CI ``reef`` job does). Without it the module
is skipped.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from honing.backends.reef import EpisodeReport, RecordedDecision, batches, decisions_from_job
from honing.backends.tank.decisions import decisions_from_log
from honing.core import DecisionSample

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/reef"))
from import_tank_runs import main as import_runs

REEF = os.environ.get("HONING_REEF_BIN")
pytestmark = pytest.mark.skipif(not REEF, reason="HONING_REEF_BIN names no Reef executable")

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = sorted((ROOT / "tests" / "fixtures" / "decision_traces").glob("*/decisions.jsonl"))
TOKEN = "reef-test"
SCENARIO = "tank"


class Reef:
    """One ``reef serve`` of ``integrations/reef/deployment.yaml``, with its state in ``directory``."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        self.process: subprocess.Popen[bytes] | None = None

    @property
    def spool(self) -> Path:
        return self.directory / ".reef" / "honing-spool"

    def start(self) -> None:
        env = {**os.environ, "REEF_TOKEN": TOKEN, "PYTHONPATH": str(ROOT)}
        self.process = subprocess.Popen(
            [str(REEF), "serve", "-c", str(ROOT / "integrations/reef/deployment.yaml"), "--reef.port", str(self.port)],
            cwd=self.directory,
            env=env,
            stdout=(self.directory / "serve.log").open("ab"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        wait(lambda: self.healthy(), 120, f"Reef did not start; see {self.directory / 'serve.log'}")

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
        self.process = None

    def healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/healthz", timeout=2) as response:
                return response.status == 200
        except OSError:
            return False

    def call(self, method: str, path: str, body: object = None, *, scenario: bool = True) -> Any:
        headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
        if scenario:
            headers["x-reef-scenario"] = SCENARIO
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=None if body is None else json.dumps(body).encode(),
            headers=headers,
            method=method,
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read() or b"{}")

    def jobs(self) -> dict[str, str]:
        return {job["id"]: job["state"] for job in self.call("GET", f"/reef/scenarios/{SCENARIO}/jobs")["jobs"]}

    def spooled(self) -> list[dict[str, Any]]:
        files = [json.loads(path.read_text()) for path in self.spool.glob("*.json")]
        return sorted(files, key=lambda job: job["submitted_at"])

    def consume(self, job: dict[str, Any]) -> None:
        (self.spool / f"{job['job_id']}.consumed").touch()


def wait(condition: Callable[[], bool], seconds: float, message: str) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(message)
        time.sleep(0.5)


@pytest.fixture
def reef(tmp_path: Path) -> Iterator[Reef]:
    server = Reef(tmp_path)
    server.start()
    try:
        yield server
    finally:
        server.stop()


def tags(log: Path) -> dict[str, str]:
    return {"episode": log.parent.name, "task": f"task-{log.parent.name}", "release": "r1", "harness": "h1"}


def score(log: Path) -> float:
    return -1.0 if "destroyed" in log.parent.name else 1.0


def push_episode(reef: Reef, log: Path, episode_tags: dict[str, str], report: dict[str, Any] | None = None) -> None:
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    records = [RecordedDecision.from_row(row, episode_tags).to_record() for row in rows]
    for batch in batches(records):
        reef.call("POST", "/reef/records/batch", {"records": batch})
    reef.call("POST", "/reef/records/batch", {"records": records[:3]})  # a resend is a retry
    if report is None:
        episode = EpisodeReport.from_rows(
            rows, episode=episode_tags["episode"], score=score(log), task=episode_tags["task"]
        )
        report = episode.to_record()
    reef.call("POST", "/reef/records/batch", {"records": [report]})


def samples_of(job: dict[str, Any]) -> list[DecisionSample]:
    """The decision samples one spooled job holds, read as Honing's data provider (#227) reads them."""
    return list(decisions_from_job(job, behavior=True))


def test_episodes_reach_the_spool_once_and_reef_publishes_nothing(reef: Reef) -> None:
    reef.call("POST", "/reef/scenarios", {"name": SCENARIO}, scenario=False)
    for log in FIXTURES:
        push_episode(reef, log, tags(log))

    # One job at a time: the second waits until Honing consumes the first.
    wait(lambda: len(reef.spooled()) == 1, 60, "no job data was spooled")
    first = reef.spooled()[0]
    reef.consume(first)
    wait(lambda: len(reef.spooled()) == 2, 60, "the second episode's job data was not spooled")
    wait(lambda: reef.jobs()[first["job_id"]] == "finished", 60, "a consumed job did not finish")

    by_episode = {job["samples"][0]["trajectory"]["extra"]["honing"]["episode"]: job for job in reef.spooled()}
    assert set(by_episode) == {log.parent.name for log in FIXTURES}
    for log in FIXTURES:
        job = by_episode[log.parent.name]
        assert samples_of(job) == decisions_from_log(
            log.parent,
            reward=score(log),
            group=tags(log)["task"],
            release_id="r1",
            harness_id="h1",
            behavior=True,
        )
        assert len(job["samples"][0]["source_agent_record_ids"]) == len(samples_of(job)) + 1  # and the report

    # Reef serves its root and publishes nothing: a finished job saved no checkpoint.
    artifacts = reef.call("GET", f"/reef/scenarios/{SCENARIO}/artifacts")
    assert len(next(value for value in artifacts.values() if isinstance(value, list))) == 1

    # After a restart the running job resumes, and consumed data never comes back.
    reef.stop()
    reef.start()
    second = reef.spooled()[1]
    wait(lambda: reef.jobs().get(second["job_id"]) == "running", 60, "the running job did not resume")
    reef.consume(second)
    wait(lambda: set(reef.jobs().values()) == {"finished"}, 60, "the resumed job did not finish")
    time.sleep(5)
    assert len(reef.spooled()) == 2
    assert reef.call("GET", f"/reef/scenarios/{SCENARIO}/next-job")["samples"] == []


def test_a_report_of_another_format_trains_nothing(reef: Reef) -> None:
    reef.call("POST", "/reef/scenarios", {"name": SCENARIO}, scenario=False)
    log = FIXTURES[0]
    episode_tags = {**tags(log), "episode": "plain"}
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    plain = EpisodeReport.from_rows(rows, episode="plain", score=1.0).to_record()
    plain["payload"]["metadata"].pop("honing")
    push_episode(reef, log, episode_tags, report=plain)
    push_episode(reef, FIXTURES[1], tags(FIXTURES[1]))

    wait(lambda: len(reef.spooled()) == 1, 60, "no job data was spooled")
    time.sleep(5)
    (job,) = reef.spooled()
    assert job["samples"][0]["trajectory"]["extra"]["honing"]["episode"] == FIXTURES[1].parent.name


def test_runs_imported_with_reef_client_train_as_decisions_from_log(reef: Reef, tmp_path: Path) -> None:
    """``scripts/reef/import_tank_runs.py``: runs played before recording reach the provider unchanged."""
    from honing.backends.reef import ReefDataProvider
    from honing.backends.reef.tank import run_tags
    from honing.core import DataContract

    reef.call("POST", "/reef/scenarios", {"name": SCENARIO}, scenario=False)
    runs = [log.parent for log in FIXTURES]
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("REEF_TOKEN", TOKEN)
        url = f"http://127.0.0.1:{reef.port}"
        argv = [*map(str, runs), "--out", str(tmp_path / "records.jsonl"), "--release", "r1", "--harness", "h1"]
        assert import_runs([*argv, "--url", url, "--scenario", SCENARIO]) == 0

    provider = ReefDataProvider(reef.spool, wait_s=60, poll_s=0.5)
    drawn = []
    for _ in runs:
        batch = provider.next_batch(DataContract("typed_decisions"))
        assert batch is not None
        drawn.append(batch)
    for batch in drawn:
        episode = batch.decisions[0].group
        (run,) = [run for run in runs if run_tags(run)["task"] == episode]
        assert list(batch.decisions) == [
            replace(decision, reward=batch.decisions[0].reward)
            for decision in decisions_from_log(
                run, reward=None, group=episode, release_id="r1", harness_id="h1", behavior=False
            )
        ]
    wait(lambda: set(reef.jobs().values()) == {"finished"}, 60, "consumed jobs did not finish")
