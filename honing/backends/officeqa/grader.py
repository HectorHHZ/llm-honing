"""Grade an OfficeQA episode with OfficeQA's own scorer.

The score is ``reward.score_answer`` from databricks/officeqa (vendored
unchanged in :mod:`honing.backends.officeqa.reward`): 1 when the text inside the
last ``<FINAL_ANSWER>`` tags matches the reference within ``tolerance``, else 0.
A tolerance of 0 is the benchmark's own setting.

The feedback is what a reflective method reads. It says what the agent
answered and which bulletin holds the figure, and by default not the reference
answer: a method that saw answers could write them into a skill, and the next
gate on the same questions would reward the leak, not the harness.
"""

from __future__ import annotations

import json
from pathlib import Path

from honing.backends.officeqa import reward
from honing.backends.officeqa.suite import OfficeQASuite
from honing.core import EpisodeOutcome, Grade, Task, require

TIMED_OUT = -9


def _status(outcome: EpisodeOutcome) -> str:
    if outcome.workspace_location is None:
        return "unknown"
    result = Path(outcome.workspace_location) / "result.json"
    if not result.exists():
        return "no result"
    parsed = json.loads(result.read_text())
    return str(parsed.get("error") or parsed.get("status") or "unknown")


class OfficeQAGrader:
    """Satisfies :class:`~honing.core.TaskGrader` for OfficeQA tasks.

    An agent that did not finish (error, iteration or budget limit, timeout)
    scores 0 with feedback saying why, rather than raising: the harness may be
    the reason it did not finish, and a gate must be able to compare that.
    """

    def __init__(self, suite: OfficeQASuite, tolerance: float = 0.0, reveal_answer: bool = False) -> None:
        require(0.0 <= tolerance < 1.0, f"tolerance must be in [0, 1), got {tolerance}")
        self._suite = suite
        self._tolerance = tolerance
        self._reveal = reveal_answer

    async def grade(self, task: Task, outcome: EpisodeOutcome) -> Grade:
        question = self._suite.question_for(task)
        where = ", ".join(question.source_files) or "an unlisted bulletin"
        if outcome.exit_code == TIMED_OUT:
            return Grade(0.0, f"The agent was stopped at the time limit before answering. The figure is in {where}.")
        if outcome.exit_code != 0:
            return Grade(0.0, f"The agent did not finish ({_status(outcome)}). The figure is in {where}.")
        answer = reward.extract_final_answer(outcome.final_output)
        if "<final_answer>" not in outcome.final_output.lower():
            return Grade(0.0, f"The agent answered without <FINAL_ANSWER> tags. The figure is in {where}.")
        score = reward.score_answer(question.answer, outcome.final_output, self._tolerance)
        if score >= 1.0:
            return Grade(1.0, f"Correct: answered {answer!r}, from {where}.")
        expected = f" The reference answer is {question.answer!r}." if self._reveal else ""
        return Grade(0.0, f"Incorrect: answered {answer!r}. The figure is in {where}.{expected}")
