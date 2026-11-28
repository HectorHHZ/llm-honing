"""The tank dry run with Reef recording on (#228): each stage's "done when" of #137, against a real Reef.

``configs/e2e/tank_dry_run_reef.json`` is ``tank_dry_run.json`` with recording on: its decision servers
record in Reef, its grader posts each graded episode's report, and its sft move trains on the job data
Reef spools. Reef is ``integrations/reef/deployment.yaml`` with the test's own record-to-sample stage in
place of Honing's (``--pipeline.record2sample.module``), so the rewards the trainer gets are that stage's,
with no change to Honing.

Reef runs in its own Python 3.12 environment, as in production. Set ``HONING_REEF_BIN`` to the ``reef``
executable of an environment with Reef bc8c150 (the CI ``reef`` job does). Without it the test is skipped.
It also needs Node and the ``gepa`` and ``reef`` extras. A chat call through Reef is
``tests/test_reef_chat.py``'s, in the same CI job: a Reef that trains refuses an upstream chat server.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import threading
import urllib.error
import uuid
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from honing.backends.reef import EpisodeReport, RecordedDecision, ReefDataProvider, batches, decisions_from_job
from honing.backends.reef.tank import run_records
from honing.backends.tank.decisions import decisions_from_log
from honing.backends.tank.e2e import build
from honing.core import DataContract, DecisionSample
from tests.test_reef_recipe import SCENARIO, TOKEN, Reef, wait

REEF = os.environ.get("HONING_REEF_BIN")
ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/e2e/tank_dry_run_reef.json"
#: The test's record-to-sample stage: Honing's, with the score minus 0.5 as the reward.
STAGE = "tests.fakes.reef_recipe:CenteredEpisodeSamples"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(not REEF, reason="HONING_REEF_BIN names no Reef executable"),
    pytest.mark.skipif(shutil.which("node") is None, reason="the dry run plays real rounds with Node"),
]


@pytest.fixture
def reef(tmp_path: Path) -> Iterator[Reef]:
    (tmp_path / "reef").mkdir()
    server = Reef(tmp_path / "reef", "--pipeline.record2sample.module", STAGE)
    try:
        yield server
    finally:
        server.stop()


def tags(run: Path) -> dict[str, str]:
    return json.loads((run / "manifest.json").read_text())["reef_tags"]


def stored(reef: Reef, kind: str) -> list[str]:
    """The ids of the scenario's records of one request type, ``inference`` or ``report``."""
    ids: list[str] = []
    after = 0
    while True:
        path = f"/reef/scenarios/{SCENARIO}/records?request_type={kind}&limit=100&after_sequence={after}"
        page = reef.call("GET", path)
        ids += [row["agent_record_id"] for row in page["records"]]
        if not page["records"] or page.get("next_after_sequence") is None:
            return ids
        after = page["next_after_sequence"]


def payload(reef: Reef, record_id: str) -> dict[str, Any]:
    return reef.call("GET", f"/reef/scenarios/{SCENARIO}/records/{record_id}")["payload"]


def refused(reef: Reef, path: str, body: object) -> tuple[int, str]:
    """Reef's status and message for a write it refuses."""
    with pytest.raises(urllib.error.HTTPError) as error:
        reef.call("POST", path, body)
    return error.value.code, error.value.read().decode()


def sampled(reef: Reef, report: dict[str, Any]) -> bool:
    """Whether Reef made a sample of the episode of ``report``. A sample is named ``<scenario>.<set>:<index>``, and
    an episode's record set by its report (``store_unit``, reef/train/pipeline.py)."""
    prefix, after = f"{SCENARIO}.{report['agent_record_id']}:", 0
    while True:
        page = reef.call("GET", f"/reef/scenarios/{SCENARIO}/samples?after={after}&limit=1000")
        if any(sample["sample_id"].startswith(prefix) for sample in page["samples"]):
            return True
        if page["next"] is None:
            return False
        after = page["next"]


def from_logs(job: dict[str, Any], runs: dict[str, Path], *, behavior: bool) -> list[DecisionSample]:
    """What ``decisions_from_log`` builds from the rounds of one job, rewarded as the test's stage rewards them."""
    decisions: list[DecisionSample] = []
    for sample in job["samples"]:
        run = runs[sample["trajectory"]["extra"]["honing"]["episode"]]
        score = run_records(run, tags(run))[-1]["payload"]["score"]
        decisions += decisions_from_log(
            run,
            reward=score - 0.5,
            group=tags(run)["task"],
            release_id=None,
            harness_id=tags(run)["harness"],
            behavior=behavior,
            policy_checksum=tags(run)["policy"],
        )
    return decisions


def post(reef: Reef, records: list[dict[str, Any]]) -> None:
    for batch in batches(records):
        reef.call("POST", "/reef/records/batch", {"records": batch})


def test_the_tank_dry_run_records_every_stage_in_reef_and_trains_on_its_job_data(
    reef: Reef, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("gepa", reason="the dry run's harness move runs GEPA")
    pytest.importorskip("reef_client", reason="the recorder and the reporter need the 'reef' extra")
    reef.start()
    reef.call("POST", "/reef/scenarios", {"name": SCENARIO}, scenario=False)
    # Reef is down when the campaign starts, and comes back once decisions were answered without it.
    reef.stop()
    monkeypatch.setenv("REEF_TOKEN", TOKEN)
    monkeypatch.setenv("HONING_REEF_SPOOL", str(reef.spool))
    config = json.loads(CONFIG.read_text())
    for slot in ("endpoints", "grader"):
        config["components"][slot]["args"]["reef_url"] = f"http://127.0.0.1:{reef.port}"
    work = tmp_path / "out"
    run = build(config, ROOT, work)

    def answered() -> int:
        return sum(len(log.read_text().splitlines()) for log in (work / "runs").glob("*/decisions.jsonl"))

    answered_while_down: list[int] = []

    def bring_back() -> None:
        wait(lambda: answered() > 0, 300, "no decision was answered")
        answered_while_down.append(answered())
        reef.start()

    returning = threading.Thread(target=bring_back, daemon=True)
    returning.start()
    try:
        outcomes = run.campaign.run(run.scheduler)
        returning.join()
        runs = {tags(path.parent)["episode"]: path.parent for path in (work / "runs").glob("*/manifest.json")}
        # What each round should have left in Reef: one record per row, then the round's report.
        expected = {episode: run_records(path, tags(path)) for episode, path in runs.items()}
        decisions = {record["agent_record_id"]: record for records in expected.values() for record in records[:-1]}
        wait(
            lambda: set(decisions) <= set(stored(reef, "inference")) and len(stored(reef, "report")) == len(runs),
            300,
            "Reef did not get every decision and every report",
        )
    finally:
        run.context.components["endpoints"].close()
        run.context.components["grader"].close()
    assert [outcome.move.kind for outcome in outcomes] == ["harness", "train", "train"]

    # 1. Every question is one Reef record matching its decisions.jsonl row, those answered while Reef was down too.
    # Reef also holds the weapon branches the game did not take: recorded, never referenced. A row names no branch,
    # so the record of a branch the game kept differs from the row's in its branch name alone.
    assert answered_while_down[0] > 0
    inferences = stored(reef, "inference")
    assert len(inferences) == len(set(inferences))
    for record_id in inferences:
        record = RecordedDecision.from_payload(payload(reef, record_id))
        if record_id in decisions:
            assert replace(record, branch=None) == RecordedDecision.from_payload(decisions[record_id]["payload"])
        else:
            assert record.branch is not None and record.tags["episode"] in runs

    # 5. Every graded episode has one report referencing its decisions, and a malformed one is refused at admission.
    for records in expected.values():
        report = records[-1]
        assert EpisodeReport.from_payload(payload(reef, report["agent_record_id"])) == EpisodeReport.from_payload(
            report["payload"]
        )
    no_task = copy.deepcopy(report["payload"])
    del no_task["metadata"]["task"]
    status, message = refused(reef, "/reef/report", {**no_task, "agent_record_id": uuid.uuid4().hex})
    assert status == 400 and "metadata.task" in message

    # 2. A resend adds nothing, a changed resend is refused, and so is a report with an unknown reference.
    # The resend is what the recorder and the reporter sent for one round, read back from Reef.
    one = next(iter(expected.values()))
    first = [
        *(RecordedDecision.from_payload(payload(reef, r["agent_record_id"])).to_record() for r in one[:-1]),
        EpisodeReport.from_payload(payload(reef, one[-1]["agent_record_id"])).to_record(),
    ]
    post(reef, first)
    assert len(stored(reef, "inference")) == len(inferences) and len(stored(reef, "report")) == len(runs)
    changed = copy.deepcopy(first[0])
    changed["payload"]["metadata"]["tags"]["task"] = "another"
    assert refused(reef, "/reef/records/batch", {"records": [changed]})[0] == 409
    unknown = {**copy.deepcopy(report["payload"]), "references": [uuid.uuid4().hex]}
    unknown["metadata"]["honing"]["chosen"] = unknown["metadata"]["honing"]["chosen"][:1]
    status, message = refused(reef, "/reef/report", {**unknown, "agent_record_id": uuid.uuid4().hex})
    assert status == 400 and "must identify an existing inference" in message

    # 3 and 4. The sft move trained on Reef's first job: the test stage's samples of the rounds, as decisions_from_log
    # builds them, and its commit record names the job.
    trainer = run.context.components["trainer"]
    sft = trainer.batches[0]
    job_id = sft.batch_id.removeprefix(f"reef:{SCENARIO}/")
    consumed = json.loads((reef.spool / f"{job_id}.json").read_text())
    assert list(sft.decisions) == from_logs(consumed, runs, behavior=sft.contract.requires_logprobs)
    commits = [json.loads(line) for line in (work / "releases/commits.jsonl").read_text().splitlines()]
    assert commits[2]["candidate"]["batch_id"] == sft.batch_id
    assert commits[2]["op"] in ("publish", "reject") and commits[2]["decision"]["policy"] == "paired_interval"

    # After a restart the consumed job stays consumed: the next move draws the next job, and no record is in two jobs.
    wait(lambda: len(reef.spooled()) == 2, 120, "the next job was not spooled")
    reef.stop()
    reef.start()
    drawn = ReefDataProvider(reef.spool, wait_s=120, poll_s=0.5).next_batch(DataContract("typed_decisions"))
    assert drawn is not None and drawn.batch_id != sft.batch_id
    wait(lambda: len(reef.spooled()) == 3, 120, "the job after a restart was not spooled")
    jobs = reef.spooled()
    sources = [record for job in jobs for sample in job["samples"] for record in sample["source_agent_record_ids"]]
    assert len(sources) == len(set(sources))
    for job in jobs:
        assert list(decisions_from_job(job, behavior=True)) == from_logs(job, runs, behavior=True)
    wait(lambda: reef.jobs()[job_id] == "finished", 60, "the consumed job did not finish")

    # 2. An episode with an aborted call trains nothing. Decision records carry no response, so this one is given
    # what Reef stores for an engine's abort; the same round again, as another task, still makes its sample.
    played = next(iter(runs.values()))
    aborted = run_records(played, {**tags(played), "episode": uuid.uuid4().hex, "task": "aborted"})
    aborted[0]["payload"]["response"] = {"finish_reason": "abort"}
    post(reef, aborted)
    control = run_records(played, {**tags(played), "episode": uuid.uuid4().hex, "task": "control"})
    post(reef, control)
    wait(lambda: sampled(reef, control[-1]), 60, "the control episode made no sample")
    assert not sampled(reef, aborted[-1])
