"""Meta-Harness's archive: JSON state that renders into upstream's on-disk layout for the proposer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from honing.core import ContractError, HarnessFile, HarnessSpec
from honing.core.state import validate_method_state
from honing.harness.methods.meta_harness.archive import (
    ARCHIVE_SCHEMA_VERSION,
    Archive,
    Candidate,
    Trace,
    read_candidate,
    write_archive,
    write_candidate,
)
from tests.fakes.harness import INSTRUCTIONS_PATH, SEED_HARNESS

TASK_IDS = ("sum", "sort", "parse")
POLICY = "ck-a"


def _files(text: str) -> tuple[tuple[str, HarnessFile], ...]:
    return ((INSTRUCTIONS_PATH, HarnessFile("instructions", text)),)


def _scored(tmp_path: Path) -> Archive:
    transcript = tmp_path / "episode.jsonl"
    transcript.write_text('{"turn": 1}\n')
    archive = Archive.seed(SEED_HARNESS, TASK_IDS)
    archive = archive.score(
        "seed",
        (0.5, 0.5, 0.5),
        POLICY,
        {"sum": Trace(str(transcript), "answered without checking"), "sort": Trace(str(tmp_path / "gone"), None)},
    )
    archive = archive.add(Candidate("verify", "seed", _files("Verify."), "checking helps", "added a rule"))
    return archive.score("verify", (1.0, 0.5, 0.0), POLICY, {})


@pytest.mark.unit
def test_the_state_round_trips_and_is_json_native(tmp_path: Path) -> None:
    archive = _scored(tmp_path)
    state = archive.to_state()

    validate_method_state(state)
    assert state["schema_version"] == ARCHIVE_SCHEMA_VERSION
    assert Archive.from_state(json.loads(json.dumps(state))) == archive


@pytest.mark.unit
def test_state_under_another_schema_is_refused() -> None:
    state = {**Archive.seed(SEED_HARNESS, TASK_IDS).to_state(), "schema_version": ARCHIVE_SCHEMA_VERSION + 1}
    with pytest.raises(ContractError, match="schema"):
        Archive.from_state(state)


@pytest.mark.unit
def test_names_must_be_unique_snake_case_and_parents_must_come_first() -> None:
    archive = Archive.seed(SEED_HARNESS, TASK_IDS)
    with pytest.raises(ContractError, match="used twice"):
        archive.add(Candidate("seed", "seed", _files("x")))
    with pytest.raises(ContractError, match="earlier candidate"):
        archive.add(Candidate("child", "nobody", _files("x")))
    with pytest.raises(ContractError, match=r"\[a-z0-9_\]"):
        Candidate("Bad-Name", "seed", _files("x"))


@pytest.mark.unit
def test_adopting_the_head_serves_it_once_as_a_new_root() -> None:
    archive = Archive.seed(SEED_HARNESS, TASK_IDS)
    assert archive.adopt_head(SEED_HARNESS) is archive

    head = SEED_HARNESS.with_file(INSTRUCTIONS_PATH, HarnessFile("instructions", "Edited elsewhere."))
    adopted = archive.adopt_head(head)

    root = adopted.served_candidate
    assert root.name == "head_" + head.spec_id[2:10] and root.parent is None
    assert root.spec(head.agent) == head
    assert adopted.adopt_head(head) is adopted
    # Coming back to a known harness serves it again rather than adding a duplicate.
    assert adopted.adopt_head(SEED_HARNESS).served == "seed"
    assert adopted.iteration == 1


@pytest.mark.unit
def test_the_written_archive_has_upstreams_layout(tmp_path: Path) -> None:
    directory = tmp_path / "archive"
    (directory / "leftover").mkdir(parents=True)

    write_archive(_scored(tmp_path), directory, agent="claude_code", policy=POLICY)

    assert not (directory / "leftover").exists()
    assert (directory / "candidates/verify/CLAUDE.md").read_text() == "Verify."
    assert json.loads((directory / "candidates/seed/harness.json").read_text()) == {
        "agent": "claude_code",
        "files": {"CLAUDE.md": "instructions"},
    }
    rows = [json.loads(line) for line in (directory / "evolution_summary.jsonl").read_text().splitlines()]
    assert [(row["name"], row["parent"], row["served"], row["stale"]) for row in rows] == [
        ("seed", None, True, False),
        ("verify", "seed", False, False),
    ]
    assert rows[1]["per_task"] == {"sum": 1.0, "sort": 0.5, "parse": 0.0}
    assert rows[1]["hypothesis"] == "checking helps" and rows[1]["changes"] == "added a rule"
    # Strictly better replaces; a tie keeps the earlier candidate, as upstream's update_frontier does.
    assert json.loads((directory / "frontier_val.json").read_text()) == {
        "sum": {"best_candidate": "verify", "score": 1.0},
        "sort": {"best_candidate": "seed", "score": 0.5},
        "parse": {"best_candidate": "seed", "score": 0.5},
        "_best": {"candidate": "seed", "mean": 0.5},
    }
    assert (directory / "traces/seed/sum/transcript.jsonl").read_text() == '{"turn": 1}\n'
    assert (directory / "traces/seed/sum/feedback.txt").read_text() == "answered without checking"
    assert (directory / "traces/seed/sort/transcript.unavailable").read_text() == f"{tmp_path / 'gone'}\n"
    assert not (directory / "traces/seed/sort/feedback.txt").exists()


@pytest.mark.unit
def test_scores_under_other_weights_are_stale_and_leave_the_frontier(tmp_path: Path) -> None:
    directory = tmp_path / "archive"

    write_archive(_scored(tmp_path), directory, agent="claude_code", policy="ck-b")

    rows = [json.loads(line) for line in (directory / "evolution_summary.jsonl").read_text().splitlines()]
    assert all(row["stale"] for row in rows)
    assert json.loads((directory / "frontier_val.json").read_text()) == {}


@pytest.mark.unit
def test_a_task_id_is_quoted_into_one_folder(tmp_path: Path) -> None:
    archive = Archive.seed(SEED_HARNESS, ("a/b",)).score("seed", (1.0,), POLICY, {"a/b": Trace(None, "ok")})

    write_archive(archive, tmp_path / "archive", agent="claude_code", policy=POLICY)

    assert (tmp_path / "archive/traces/seed/a%2Fb/feedback.txt").read_text() == "ok"


@pytest.mark.unit
def test_a_candidate_folder_reads_back_as_its_spec(tmp_path: Path) -> None:
    spec = HarnessSpec(
        agent="claude_code",
        files=(
            ("CLAUDE.md", HarnessFile("instructions", "Solve.")),
            (".claude/skills/check/SKILL.md", HarnessFile("skill", "Check.")),
        ),
    )
    write_candidate(spec, tmp_path / "c")

    assert read_candidate(tmp_path / "c").spec_id == spec.spec_id


@pytest.mark.unit
def test_a_folder_that_disagrees_with_its_manifest_is_refused(tmp_path: Path) -> None:
    write_candidate(SEED_HARNESS, tmp_path / "extra")
    (tmp_path / "extra" / "notes.md").write_text("unlisted")
    write_candidate(SEED_HARNESS, tmp_path / "missing")
    (tmp_path / "missing" / "CLAUDE.md").unlink()
    write_candidate(SEED_HARNESS, tmp_path / "kind")
    (tmp_path / "kind" / "harness.json").write_text('{"agent": "claude_code", "files": {"CLAUDE.md": "prompt"}}')

    for name in ("extra", "missing", "kind"):
        with pytest.raises(ValueError):
            read_candidate(tmp_path / name)
