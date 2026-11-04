"""Grade OfficeQA episodes with OfficeQA's own scorer, which :mod:`integrations.officeqa.fetch` downloads."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from collections.abc import Callable
from pathlib import Path

from integrations.officeqa.fetch import SCORER_SHA256
from integrations.officeqa.suite import OfficeQASuite

from honing.core import ContractError, EpisodeOutcome, Grade, Task, require

Scorer = Callable[[str, str, float], float]
"""``score(reference, prediction, tolerance)``: 1.0 when the prediction matches, else 0.0."""

_TIMED_OUT = -9
_FINAL_ANSWER = re.compile(r"<FINAL_ANSWER>\s*(.*?)\s*</FINAL_ANSWER>", re.DOTALL | re.IGNORECASE)


def load_scorer(path: Path) -> Scorer:
    """Loads ``score_answer`` from OfficeQA's ``reward.py``, refusing any other file.

    Args:
        path: The ``reward.py`` the fetcher wrote.

    Returns:
        The scorer.

    Raises:
        ContractError: The file is missing or is not the pinned ``reward.py``.
    """
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != SCORER_SHA256:
        raise ContractError(f"{path} is not OfficeQA's pinned reward.py; run python -m integrations.officeqa.fetch")
    spec = importlib.util.spec_from_file_location("officeqa_reward", path)
    if spec is None or spec.loader is None:
        raise ContractError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    score: Scorer = module.score_answer
    return score


class OfficeQAGrader:
    """Satisfies :class:`~honing.core.TaskGrader` for OfficeQA tasks.

    A score is 1 when the last ``<FINAL_ANSWER>`` matches the reference within
    ``tolerance``, else 0. An agent that did not finish scores 0 with the reason
    rather than raising, since the harness may be why. Feedback names the
    agent's answer, the reference answer, and the source bulletin, which gives a
    reflective method the most to work with. A method can then write answers
    into a skill and pass the next gate on the same questions by memory; set
    ``reveal_answer`` to False, or gate on questions the method never sees, when
    that matters.

    Args:
        suite: The questions being graded.
        scorer: OfficeQA's scorer, from :func:`load_scorer`.
        tolerance: The relative error accepted; 0 is the benchmark's setting.
        reveal_answer: Put the reference answer in the feedback of every zero score.

    Raises:
        ContractError: ``tolerance`` is outside ``[0, 1)``.
    """

    def __init__(
        self, suite: OfficeQASuite, scorer: Scorer, tolerance: float = 0.0, reveal_answer: bool = True
    ) -> None:
        require(0.0 <= tolerance < 1.0, f"tolerance must be in [0, 1), got {tolerance}")
        self._suite = suite
        self._scorer = scorer
        self._tolerance = tolerance
        self._reveal = reveal_answer

    async def grade(self, task: Task, outcome: EpisodeOutcome) -> Grade:
        """Scores one episode's final answer.

        Args:
            task: The task the episode ran.
            outcome: The episode's outcome.

        Returns:
            The score and feedback.

        Raises:
            ContractError: The task is not in the suite.
        """
        question = self._suite.question(task.task_id)
        where = ", ".join(question.source_files) or "an unlisted bulletin"
        label = f" The reference answer is {question.answer!r}." if self._reveal else ""
        if outcome.exit_code == _TIMED_OUT:
            return Grade(0.0, f"The agent was stopped at the time limit before answering.{label} It is in {where}.")
        if outcome.exit_code != 0:
            return Grade(0.0, f"The agent did not finish ({_status(outcome)}).{label} It is in {where}.")
        answers = _FINAL_ANSWER.findall(outcome.final_output)
        if not answers:
            return Grade(0.0, f"The agent answered without <FINAL_ANSWER> tags.{label} It is in {where}.")
        if self._scorer(question.answer, outcome.final_output, self._tolerance) >= 1.0:
            return Grade(1.0, f"Correct: answered {answers[-1]!r}, from {where}.")
        return Grade(0.0, f"Incorrect: answered {answers[-1]!r}.{label} It is in {where}.")


def _status(outcome: EpisodeOutcome) -> str:
    result = None if outcome.workspace_location is None else Path(outcome.workspace_location) / "result.json"
    if result is None or not result.exists():
        return "no result"
    parsed = json.loads(result.read_text())
    return str(parsed.get("error") or parsed.get("status") or "unknown")
