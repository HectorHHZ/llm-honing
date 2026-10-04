"""OfficeQA as a task suite: loading questions, grading with OfficeQA's scorer, and what the feedback reveals."""

from __future__ import annotations

import asyncio
import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

from honing.backends.officeqa import OfficeQAGrader, load_suite, reward
from honing.backends.officeqa.fetch import neighbors
from honing.core import ContractError, EpisodeOutcome, Grade, protocols

ROWS = [
    {
        "uid": "UID0002",
        "question": "What were VA expenditures in FY 1934?",
        "answer": "507",
        "source_docs": "https://example.invalid/page=18",
        "source_files": "treasury_bulletin_1944_01.txt",
        "difficulty": "easy",
    },
    {
        "uid": "UID0008",
        "question": "By what percent did receipts grow?",
        "answer": "73",
        "source_docs": "a\nb",
        "source_files": "treasury_bulletin_2012_06.txt\ntreasury_bulletin_2022_06.txt",
        "difficulty": "easy",
    },
]


@pytest.fixture
def csv_path(tmp_path: Path) -> Path:
    path = tmp_path / "officeqa_full.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ROWS[0]))
        writer.writeheader()
        writer.writerows(ROWS)
    return path


def _outcome(tmp_path: Path, final_output: str, exit_code: int = 0, status: str = "finished") -> EpisodeOutcome:
    (tmp_path / "result.json").write_text(json.dumps({"final_output": final_output, "status": status}))
    return EpisodeOutcome(
        task_id="UID0002",
        spec_id="h-1",
        policy_checksum="c",
        exit_code=exit_code,
        final_output=final_output,
        workspace_location=str(tmp_path),
    )


def _grade(grader: OfficeQAGrader, outcome: EpisodeOutcome) -> Grade:
    task = next(t for t in grader._suite.tasks() if t.task_id == outcome.task_id)
    return asyncio.run(grader.grade(task, outcome))


@pytest.mark.unit
def test_a_suite_keeps_the_named_questions_in_order(csv_path: Path) -> None:
    suite = load_suite(csv_path, ["UID0008", "UID0002"])

    assert [t.task_id for t in suite.tasks()] == ["UID0008", "UID0002"]
    assert suite.question_for(suite.tasks()[0]).source_files == (
        "treasury_bulletin_2012_06.txt",
        "treasury_bulletin_2022_06.txt",
    )
    prompt = suite.tasks()[1].prompt
    assert prompt.startswith("What were VA expenditures in FY 1934?")
    assert "`corpus/`" in prompt and "<FINAL_ANSWER></FINAL_ANSWER>" in prompt
    assert "507" not in prompt


@pytest.mark.unit
def test_an_unknown_question_is_refused(csv_path: Path) -> None:
    with pytest.raises(ContractError, match="no questions \\['UID9999'\\]"):
        load_suite(csv_path, ["UID0002", "UID9999"])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("final_output", "score", "feedback"),
    [
        ("Table 3 shows it. <FINAL_ANSWER>507</FINAL_ANSWER>", 1.0, "Correct: answered '507'"),
        ("<FINAL_ANSWER>$507 million</FINAL_ANSWER>", 1.0, "Correct"),
        (
            "<FINAL_ANSWER>508</FINAL_ANSWER>",
            0.0,
            "Incorrect: answered '508'. The figure is in treasury_bulletin_1944",
        ),
        ("The answer is 507.", 0.0, "without <FINAL_ANSWER> tags"),
    ],
)
def test_answers_are_scored_by_officeqa_s_own_scorer(
    tmp_path: Path, csv_path: Path, final_output: str, score: float, feedback: str
) -> None:
    grader = OfficeQAGrader(load_suite(csv_path, ["UID0002"]))

    grade = _grade(grader, _outcome(tmp_path, final_output))

    assert grade.score == score
    assert feedback in (grade.feedback or "")


@pytest.mark.unit
def test_a_wrong_answer_s_feedback_hides_the_reference_unless_asked(tmp_path: Path, csv_path: Path) -> None:
    suite = load_suite(csv_path, ["UID0002"])
    outcome = _outcome(tmp_path, "<FINAL_ANSWER>508</FINAL_ANSWER>")

    hidden = _grade(OfficeQAGrader(suite), outcome)
    shown = _grade(OfficeQAGrader(suite, reveal_answer=True), outcome)

    assert "507" not in (hidden.feedback or "")
    assert "reference answer is '507'" in (shown.feedback or "")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("exit_code", "status", "feedback"),
    [
        (-9, "running", "stopped at the time limit"),
        (2, "error", "did not finish (error)"),
        (1, "failed", "did not finish (failed)"),
    ],
)
def test_an_agent_that_did_not_finish_scores_zero_with_the_reason(
    tmp_path: Path, csv_path: Path, exit_code: int, status: str, feedback: str
) -> None:
    grader = OfficeQAGrader(load_suite(csv_path, ["UID0002"]))

    grade = _grade(grader, _outcome(tmp_path, "<FINAL_ANSWER>507</FINAL_ANSWER>", exit_code, status))

    assert grade.score == 0.0
    assert feedback in (grade.feedback or "")


@pytest.mark.unit
def test_a_copy_of_a_task_under_another_id_grades_as_the_original(tmp_path: Path, csv_path: Path) -> None:
    grader = OfficeQAGrader(load_suite(csv_path, ["UID0002"]))
    copy = replace(grader._suite.tasks()[0], task_id="UID0002-3")

    grade = asyncio.run(grader.grade(copy, _outcome(tmp_path, "<FINAL_ANSWER>507</FINAL_ANSWER>")))

    assert grade.score == 1.0


@pytest.mark.unit
def test_the_grader_satisfies_the_task_grader_protocol(csv_path: Path) -> None:
    assert isinstance(OfficeQAGrader(load_suite(csv_path, ["UID0002"])), protocols.TaskGrader)


@pytest.mark.unit
def test_the_vendored_scorer_is_upstream_s() -> None:
    assert reward.score_answer("103,375", "<FINAL_ANSWER>103375</FINAL_ANSWER>") == 1.0
    assert reward.score_answer("507", "<FINAL_ANSWER>507.6</FINAL_ANSWER>", tolerance=0.0) == 0.0
    assert reward.score_answer("507", "<FINAL_ANSWER>507.6</FINAL_ANSWER>", tolerance=0.01) == 1.0


@pytest.mark.unit
def test_neighbors_span_months_across_a_year_boundary() -> None:
    assert neighbors("treasury_bulletin_1944_01.txt", 2) == [
        "treasury_bulletin_1943_11.txt",
        "treasury_bulletin_1943_12.txt",
        "treasury_bulletin_1944_01.txt",
        "treasury_bulletin_1944_02.txt",
        "treasury_bulletin_1944_03.txt",
    ]
    assert neighbors("other.txt", 2) == ["other.txt"]
