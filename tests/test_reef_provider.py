"""ReefDataProvider (#227) on spooled job data. ``tests/test_reef_recipe.py`` reads what a real Reef spools."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from honing.backends.reef import (
    EPISODE_FORMAT,
    FORMAT_VERSION,
    JOB_FORMAT,
    JOB_VERSION,
    EpisodeReport,
    RecordedDecision,
    ReefDataProvider,
    decisions_from_job,
)
from honing.backends.tank.decisions import decisions_from_log
from honing.campaign import rules
from honing.core import CampaignRuleError, ContractError, DataContract, TrainJobSpec
from tests.fakes.harness import SEED_HARNESS
from tests.test_campaign_rules import head_release

FIXTURES = sorted((Path(__file__).parent / "fixtures" / "decision_traces").glob("*/decisions.jsonl"))
HARNESS = SEED_HARNESS.spec_id
POLICY = head_release().policy().checkpoint.checksum
DECISIONS = DataContract("typed_decisions")


def tags(log: Path) -> dict[str, str]:
    return {
        "episode": log.parent.name,
        "task": f"task-{log.parent.name}",
        "release": "r0000",
        "harness": HARNESS,
        "policy": POLICY,
    }


def score(log: Path) -> float:
    return 0.0 if "destroyed" in log.parent.name else 1.0


def trajectory(log: Path) -> dict[str, Any]:
    """One episode as ``DecisionEpisodeSamples`` (``integrations/reef/processor.py``) makes it."""
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    report = EpisodeReport.from_rows(rows, episode=log.parent.name, score=score(log), task=tags(log)["task"])
    sent = report.to_record()["payload"]
    references = zip(sent["references"], sent["metadata"]["honing"]["chosen"], strict=True)
    steps = []
    for index, (row, (record_id, chosen)) in enumerate(zip(rows, references, strict=True)):
        payload = RecordedDecision.from_row(row, tags(log)).to_record()["payload"]
        extra = {"honing": {**payload["honing"], "record_id": record_id, "chosen": chosen}}
        steps.append({"step_id": index + 1, "source": "agent", "message": "", "extra": extra})
    # Reef stores payloads with their keys sorted.
    honing = {"format": EPISODE_FORMAT, "version": FORMAT_VERSION, "episode": log.parent.name, "tags": tags(log)}
    return json.loads(
        json.dumps(
            {"steps": steps, "extra": {"reef": {"reward": score(log), "feedback": None}, "honing": honing}},
            sort_keys=True,
        )
    )


def spool_job(spool: Path, job_id: str, logs: list[Path], submitted_at: str) -> dict[str, Any]:
    job = {
        "format": JOB_FORMAT,
        "version": JOB_VERSION,
        "job_id": job_id,
        "scenario": "tank",
        "data_id": f"tank:data:{job_id}",
        "submitted_at": submitted_at,
        "parameters": {"objective": "honing"},
        "samples": [
            {"trajectory": trajectory(log), "group_id": tags(log)["task"], "source_agent_record_ids": []}
            for log in logs
        ],
    }
    spool.mkdir(parents=True, exist_ok=True)
    # As integrations/reef/backend.py does: a provider polling the spool never sees a half-written job.
    partial = spool / f"{job_id}.json.tmp"
    partial.write_text(json.dumps(job))
    os.replace(partial, spool / f"{job_id}.json")
    return job


def expected(logs: list[Path], *, behavior: bool) -> list[Any]:
    return [
        decision
        for log in logs
        for decision in decisions_from_log(
            log.parent,
            reward=score(log),
            group=tags(log)["task"],
            release_id="r0000",
            harness_id=HARNESS,
            behavior=behavior,
            policy_checksum=POLICY,
        )
    ]


@pytest.mark.parametrize("behavior", [False, True])
def test_a_job_gives_the_samples_decisions_from_log_builds(tmp_path: Path, behavior: bool) -> None:
    job = spool_job(tmp_path, "a", FIXTURES, "2026-11-25T00:00:00+00:00")
    assert list(decisions_from_job(job, behavior=behavior)) == expected(FIXTURES, behavior=behavior)

    contract = DataContract("typed_decisions", requires_logprobs=behavior, requires_reward=True)
    batch = ReefDataProvider(tmp_path).next_batch(contract)
    assert batch is not None and batch.batch_id == "reef:tank/a"
    assert list(batch.decisions) == expected(FIXTURES, behavior=behavior)


def test_jobs_are_drawn_oldest_first_and_each_once_across_restarts(tmp_path: Path) -> None:
    spool_job(tmp_path, "late", FIXTURES[:1], "2026-11-25T00:00:02+00:00")
    spool_job(tmp_path, "early", FIXTURES[1:], "2026-11-25T00:00:01+00:00")
    spool_job(tmp_path, "gone", FIXTURES[:1], "2026-11-25T00:00:00+00:00")
    (tmp_path / "gone.cancelled").touch()

    provider = ReefDataProvider(tmp_path)
    first = provider.next_batch(DECISIONS)
    assert first is not None and first.batch_id == "reef:tank/early"
    assert (tmp_path / "early.consumed").exists()

    restarted = ReefDataProvider(tmp_path)
    second = restarted.next_batch(DECISIONS)
    assert second is not None and second.batch_id == "reef:tank/late"
    assert restarted.next_batch(DECISIONS) is None
    assert ReefDataProvider(tmp_path).pending() == []


def test_other_contracts_and_broken_jobs_consume_nothing(tmp_path: Path) -> None:
    job = spool_job(tmp_path, "a", FIXTURES[:1], "2026-11-25T00:00:00+00:00")
    provider = ReefDataProvider(tmp_path)
    assert provider.next_batch(DataContract("token_sequences")) is None

    (tmp_path / "a.json").write_text(json.dumps({**job, "version": 2}))
    with pytest.raises(ContractError, match="version 1"):
        provider.next_batch(DECISIONS)
    assert not (tmp_path / "a.consumed").exists()


def test_a_move_waits_for_reef_to_spool_its_job(tmp_path: Path) -> None:
    assert ReefDataProvider(tmp_path, wait_s=0.2, poll_s=0.05).next_batch(DECISIONS) is None

    timer = threading.Timer(0.3, spool_job, (tmp_path, "a", FIXTURES[:1], "2026-11-25T00:00:00+00:00"))
    timer.start()
    started = time.monotonic()
    batch = ReefDataProvider(tmp_path, wait_s=10, poll_s=0.05).next_batch(DECISIONS)
    timer.join()
    assert batch is not None and time.monotonic() - started < 5


def test_a_batch_meets_the_campaign_rules_as_they_are(tmp_path: Path) -> None:
    spool_job(tmp_path, "a", FIXTURES, "2026-11-25T00:00:00+00:00")
    batch = ReefDataProvider(tmp_path).next_batch(DECISIONS)
    assert batch is not None
    rules.batch_meets_contract(TrainJobSpec(algorithm="sft", contract="typed_decisions"), batch)
    rules.batch_from_one_release(batch, head_release())

    spool_job(tmp_path, "b", FIXTURES[:1], "2026-11-25T00:00:01+00:00")
    job = json.loads((tmp_path / "b.json").read_text())
    job["samples"][0]["trajectory"]["extra"]["honing"]["tags"]["release"] = "r0007"
    (tmp_path / "b.json").write_text(json.dumps(job))
    foreign = ReefDataProvider(tmp_path).next_batch(DECISIONS)
    assert foreign is not None
    with pytest.raises(CampaignRuleError, match=r"^batch_from_one_release:"):
        rules.batch_from_one_release(foreign, head_release())


def test_a_batch_from_other_weights_is_refused_without_a_release_tag(tmp_path: Path) -> None:
    # A runner knows the weights it served, not their release (#241), so the policy tag keeps a batch to the head.
    for job_id, policy in (("a", POLICY), ("b", "other-weights")):
        spool_job(tmp_path, job_id, FIXTURES[:1], f"2026-11-25T00:00:0{job_id == 'b':d}+00:00")
        job = json.loads((tmp_path / f"{job_id}.json").read_text())
        episode_tags = job["samples"][0]["trajectory"]["extra"]["honing"]["tags"]
        del episode_tags["release"]
        episode_tags["policy"] = policy
        (tmp_path / f"{job_id}.json").write_text(json.dumps(job))
    same = ReefDataProvider(tmp_path).next_batch(DECISIONS)
    assert same is not None and same.source_releases == frozenset() and same.source_policies == {POLICY}
    rules.batch_from_one_release(same, head_release())
    other = ReefDataProvider(tmp_path).next_batch(DECISIONS)
    assert other is not None and other.source_policies == {"other-weights"}
    with pytest.raises(CampaignRuleError, match=r"^batch_from_one_release:.*other-weights"):
        rules.batch_from_one_release(other, head_release())


def test_the_gpu_config_builds_a_provider_over_the_spool_it_names(tmp_path: Path, monkeypatch: Any) -> None:
    from types import SimpleNamespace

    from honing.core import DataProvider
    from honing.e2e.components import build_component

    config = json.loads((Path(__file__).parents[1] / "configs/e2e/tank_direction_edit_reef.json").read_text())
    monkeypatch.setenv("HONING_REEF_SPOOL", str(tmp_path))
    provider = build_component(config["components"]["data"], SimpleNamespace(root=tmp_path), "data", DataProvider)
    assert isinstance(provider, ReefDataProvider) and provider.spool_dir == tmp_path

    monkeypatch.delenv("HONING_REEF_SPOOL")
    with pytest.raises(ContractError, match="unset environment variable"):
        build_component(config["components"]["data"], SimpleNamespace(root=tmp_path), "data", DataProvider)
