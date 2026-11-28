"""The Reef record format (#222): decisions and episode reports read back to what the decision log gives."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from honing.backends.reef import (
    MAX_BATCH_RECORDS,
    EpisodeReport,
    RecordedDecision,
    batches,
    decision_record_id,
    episode_report_id,
    tag_headers,
)
from honing.backends.tank.decisions import decisions_from_log
from honing.core import ContractError

FIXTURES = sorted((Path(__file__).parent / "fixtures" / "decision_traces").glob("*/decisions.jsonl"))
TAGS = {"episode": "s3114-a", "task": "tank-3114", "release": "r1", "harness": "h1", "policy": "c0ffee"}


def rows(log: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]


def through_json(value: Any) -> Any:
    """``value`` as Reef hands it back: its stores sort object keys."""
    return json.loads(json.dumps(value, sort_keys=True))


@pytest.mark.unit
@pytest.mark.parametrize("log", FIXTURES, ids=lambda log: log.parent.name)
@pytest.mark.parametrize("behavior", [True, False])
def test_records_and_report_give_the_decision_logs_samples(log: Path, behavior: bool) -> None:
    logged = rows(log)
    tags = {**TAGS, "episode": log.parent.name}
    stored = {
        record["agent_record_id"]: through_json(record["payload"])
        for record in (RecordedDecision.from_row(row, tags).to_record() for row in logged)
    }
    report = EpisodeReport.from_payload(
        through_json(EpisodeReport.from_rows(logged, episode=tags["episode"], score=1.0, task="tank-3114").payload())
    )
    kwargs: dict[str, Any] = {"reward": 1.0, "group": "tank-3114", "release_id": "r1", "harness_id": "h1"}

    samples = [
        RecordedDecision.from_payload(stored[record_id]).sample(chosen, behavior=behavior, **kwargs)
        for record_id, chosen in report.decisions
    ]

    assert len(stored) == len(logged)
    assert samples == decisions_from_log(log.parent, behavior=behavior, **kwargs)


@pytest.mark.unit
def test_a_record_reads_back_whole() -> None:
    row = rows(FIXTURES[0])[0]
    decision = RecordedDecision.from_row(row, TAGS)
    back = RecordedDecision.from_payload(through_json(decision.payload()))
    assert back == RecordedDecision(
        request_id=row["request_id"],
        parts=row["contextParts"],
        questions=row["questions"],
        answers=row["answers"],
        tags=TAGS,
    )
    assert back.record_id == decision.record_id == decision_record_id("s3114-a", row["request_id"])
    assert decision.to_record()["request_type"] == "inference"
    assert decision.payload()["metadata"] == {"tags": TAGS}


@pytest.mark.unit
def test_ids_are_stable_and_name_the_episode() -> None:
    # A changed derivation would orphan every stored record, so the value is pinned.
    assert decision_record_id("ep", "blue-3114-1:motion") == "482791861431155914b0c6f20fc3c8fb"
    assert decision_record_id("ep", "blue-3114-1:motion") != decision_record_id("ep2", "blue-3114-1:motion")
    assert episode_report_id("ep") != decision_record_id("ep", "ep")
    assert len(episode_report_id("ep")) == 32


@pytest.mark.unit
def test_a_report_reads_back_whole() -> None:
    logged = rows(FIXTURES[0])
    report = EpisodeReport.from_rows(logged, episode="ep", score=-0.5, feedback="lost", task="t")
    payload = report.payload()
    assert set(payload) == {"score", "feedback", "references", "metadata"}
    assert payload["references"] == [decision_record_id("ep", row["request_id"]) for row in logged]
    assert EpisodeReport.from_payload(through_json(payload)) == report
    assert report.to_record()["agent_record_id"] == episode_report_id("ep")


@pytest.mark.parametrize(
    ("tags", "message"),
    [
        ({"task": "t"}, "episode tag"),
        ({"episode": "ep", "seat": "blue"}, "unknown tags"),
        ({"episode": "ep", "task": " t"}, "task tag"),
    ],
)
@pytest.mark.unit
def test_tags_are_refused(tags: dict[str, str], message: str) -> None:
    with pytest.raises(ContractError, match=message):
        RecordedDecision.from_row(rows(FIXTURES[0])[0], tags)


@pytest.mark.unit
def test_tag_headers() -> None:
    assert tag_headers({"policy": "c", "episode": "e"}) == {"x-reef-tag-episode": "e", "x-reef-tag-policy": "c"}


@pytest.mark.unit
def test_a_choice_outside_the_options_is_refused() -> None:
    row = rows(FIXTURES[0])[0]
    decision = RecordedDecision.from_row(row, TAGS)
    key = next(iter(row["questions"]))
    for chosen, message in (({key: "nowhere"}, "not an option"), ({}, "the report chose")):
        with pytest.raises(ContractError, match=message):
            decision.sample(chosen, reward=None, group=None, release_id=None, harness_id=None, behavior=False)


@pytest.mark.unit
def test_a_record_without_a_probability_per_option_is_refused() -> None:
    row = rows(FIXTURES[0])[0]
    key = next(iter(row["questions"]))
    answers = {**row["answers"], key: {**row["answers"][key], "probabilities": {}}}
    with pytest.raises(ContractError, match="a probability per option"):
        RecordedDecision.from_row({**row, "answers": answers}, TAGS)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda p: p["honing"].update(version=2), "version 2"),
        (lambda p: p["honing"].update(format="other"), "not a honing-decision"),
        (lambda p: p["honing"].update(extra=1), "has keys"),
        (lambda p: p.pop("metadata"), "metadata.tags"),
    ],
)
@pytest.mark.unit
def test_another_format_is_refused(change: Any, message: str) -> None:
    payload = through_json(RecordedDecision.from_row(rows(FIXTURES[0])[0], TAGS).payload())
    change(payload)
    with pytest.raises(ContractError, match=message):
        RecordedDecision.from_payload(payload)


@pytest.mark.unit
def test_a_report_is_refused() -> None:
    with pytest.raises(ContractError, match="each record once"):
        EpisodeReport(episode="ep", score=1.0, decisions=(("a", {"q": "x"}), ("a", {"q": "y"})))
    with pytest.raises(ContractError, match="finite number"):
        EpisodeReport(episode="ep", score=float("nan"))
    payload = through_json(EpisodeReport(episode="ep", score=1.0, decisions=(("a", {"q": "x"}),)).payload())
    payload["references"].append("b")
    with pytest.raises(ContractError, match="one choice per reference"):
        EpisodeReport.from_payload(payload)


@pytest.mark.unit
def test_batches_keep_order_under_both_limits() -> None:
    records = [
        {"agent_record_id": str(i), "request_type": "inference", "payload": {"x": "y" * 100}} for i in range(25)
    ]
    by_count = list(batches(records, max_records=10))
    assert [len(batch) for batch in by_count] == [10, 10, 5]
    by_bytes = list(batches(records, max_bytes=600))
    assert all(len(json.dumps({"records": batch}).encode()) <= 600 for batch in by_bytes)
    assert [r for batch in by_bytes for r in batch] == records == [r for batch in by_count for r in batch]
    assert MAX_BATCH_RECORDS == 1000
    with pytest.raises(ContractError, match="over 100"):
        list(batches(records, max_bytes=100))
