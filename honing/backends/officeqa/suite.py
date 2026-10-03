"""OfficeQA questions as Honing tasks.

OfficeQA (databricks/officeqa, CC-BY-SA 4.0) asks numeric and short-answer
questions about U.S. Treasury Bulletins. A row has ``uid``, ``question``,
``answer``, ``source_docs`` (page URLs), ``source_files`` (the bulletin text
files, one per line), and ``difficulty``. The task's prompt is the question plus
the protocol the grader depends on: where the documents are, and that the
answer goes inside ``<FINAL_ANSWER>`` tags. How to search them is the harness's
business, not the task's.

The answer and source files stay in the suite; a task carries only its id and prompt.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from honing.core import ContractError, Task, require

CORPUS_DIR = "corpus"

PROMPT = """{question}

The documents are U.S. Treasury Bulletins, one plain-text file per issue, in the directory `{corpus}/` of your \
working directory (files are named treasury_bulletin_YYYY_MM.txt). Answer from those documents.

Give your final answer inside <FINAL_ANSWER></FINAL_ANSWER> tags, the value only, in the unit the question asks for."""


@dataclass(frozen=True)
class OfficeQAQuestion:
    uid: str
    question: str
    answer: str
    source_files: tuple[str, ...]
    difficulty: str

    def task(self, corpus: str = CORPUS_DIR) -> Task:
        return Task(task_id=self.uid, prompt=PROMPT.format(question=self.question.strip(), corpus=corpus))


@dataclass(frozen=True)
class OfficeQASuite:
    """The questions a run uses, in the order given."""

    questions: tuple[OfficeQAQuestion, ...]

    def __post_init__(self) -> None:
        require(len(self.questions) > 0, "an OfficeQA suite needs at least one question")
        uids = [q.uid for q in self.questions]
        require(len(set(uids)) == len(uids), "an OfficeQA suite names each question once")

    def tasks(self) -> tuple[Task, ...]:
        return tuple(q.task() for q in self.questions)

    def question(self, task_id: str) -> OfficeQAQuestion:
        for q in self.questions:
            if q.uid == task_id:
                return q
        raise ContractError(f"task {task_id!r} is not in the OfficeQA suite")

    def question_for(self, task: Task) -> OfficeQAQuestion:
        """The task's question, by id, or by prompt for a copy of a task under another id.

        Copies are how a runner's concurrency is checked (``honing.e2e.conformance``):
        the same task several times at once, each under its own id.
        """
        for q in self.questions:
            if q.uid == task.task_id or q.task().prompt == task.prompt:
                return q
        raise ContractError(f"task {task.task_id!r} is not in the OfficeQA suite")


def read_questions(csv_path: Path) -> dict[str, OfficeQAQuestion]:
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
    """The named questions, refusing an unknown uid."""
    questions = read_questions(csv_path)
    unknown = [uid for uid in uids if uid not in questions]
    if unknown:
        raise ContractError(f"{csv_path} has no questions {unknown}")
    return OfficeQASuite(tuple(questions[uid] for uid in uids))
