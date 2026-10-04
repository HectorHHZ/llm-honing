"""OfficeQA questions (databricks/officeqa, CC-BY-SA 4.0) as Honing tasks.

A task's prompt is the question plus what the grader depends on: where the
documents are, and that the answer goes in ``<FINAL_ANSWER>`` tags. How to
search the documents is left to the harness.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from honing.core import ContractError, Task, require

CORPUS_DIR = "corpus"

_PROMPT = """{question}

The documents are U.S. Treasury Bulletins, one plain-text file per issue, in the directory `{corpus}/` of your \
working directory (files are named treasury_bulletin_YYYY_MM.txt). Answer from those documents.

Give your final answer inside <FINAL_ANSWER></FINAL_ANSWER> tags, the value only, in the unit the question asks for."""


@dataclass(frozen=True)
class OfficeQAQuestion:
    """One OfficeQA question and what grades it.

    Attributes:
        uid: The question's id, which is also its task id.
        question: The question text.
        answer: The reference answer.
        source_files: The bulletin files that hold the answer.
        difficulty: ``easy`` or ``hard``.
    """

    uid: str
    question: str
    answer: str
    source_files: tuple[str, ...]
    difficulty: str

    def task(self) -> Task:
        """Returns the question as a task; the answer stays here."""
        return Task(task_id=self.uid, prompt=_PROMPT.format(question=self.question.strip(), corpus=CORPUS_DIR))


@dataclass(frozen=True)
class OfficeQASuite:
    """The questions a run uses, in order.

    Attributes:
        questions: The questions, each named once.
    """

    questions: tuple[OfficeQAQuestion, ...]

    def __post_init__(self) -> None:
        require(len(self.questions) > 0, "an OfficeQA suite needs at least one question")
        uids = [q.uid for q in self.questions]
        require(len(set(uids)) == len(uids), "an OfficeQA suite names each question once")

    def tasks(self) -> tuple[Task, ...]:
        """Returns the questions as tasks, in order."""
        return tuple(q.task() for q in self.questions)

    def question_for(self, task: Task) -> OfficeQAQuestion:
        """Returns a task's question, matched by id or, for a copy under another id, by prompt.

        The conformance check runs copies of one task under new ids to test concurrency.

        Args:
            task: A task from :meth:`tasks`, or a copy of one.

        Raises:
            ContractError: The task is not one of the suite's.
        """
        for q in self.questions:
            if q.uid == task.task_id or q.task().prompt == task.prompt:
                return q
        raise ContractError(f"task {task.task_id!r} is not in the OfficeQA suite")


def read_questions(csv_path: Path) -> dict[str, OfficeQAQuestion]:
    """Reads every question in an OfficeQA CSV.

    Args:
        csv_path: ``officeqa_full.csv`` or ``officeqa_pro.csv``.

    Returns:
        The questions by uid.

    Raises:
        ContractError: A required column is missing.
    """
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    missing = {"uid", "question", "answer", "source_files"} - set(rows[0] if rows else {})
    if missing:
        raise ContractError(f"{csv_path} lacks the columns {sorted(missing)}")
    return {
        row["uid"]: OfficeQAQuestion(
            uid=row["uid"],
            question=row["question"],
            answer=row["answer"],
            source_files=tuple(line.strip() for line in row["source_files"].splitlines() if line.strip()),
            difficulty=row.get("difficulty", ""),
        )
        for row in rows
    }


def load_suite(csv_path: Path, uids: Sequence[str]) -> OfficeQASuite:
    """Loads the named questions from an OfficeQA CSV.

    Args:
        csv_path: The questions file.
        uids: The questions to use, in order.

    Returns:
        The suite.

    Raises:
        ContractError: A uid is not in the file.
    """
    questions = read_questions(csv_path)
    unknown = [uid for uid in uids if uid not in questions]
    if unknown:
        raise ContractError(f"{csv_path} has no questions {unknown}")
    return OfficeQASuite(tuple(questions[uid] for uid in uids))
