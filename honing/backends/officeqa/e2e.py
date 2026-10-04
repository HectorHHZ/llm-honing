"""Factories that put OfficeQA into a pipeline config (see :mod:`honing.e2e.components`).

``data_dir`` is what :mod:`honing.backends.officeqa.fetch` wrote, and may use ``{root}`` and environment variables.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from honing.backends.officeqa.fetch import QUESTIONS
from honing.backends.officeqa.grader import OfficeQAGrader
from honing.backends.officeqa.suite import OfficeQASuite, load_suite
from honing.core import Task


def tasks(context: object, data_dir: str, uids: Sequence[str]) -> tuple[Task, ...]:
    """Builds the named questions as tasks.

    Args:
        context: The pipeline context.
        data_dir: The fetched data directory.
        uids: The questions, in order.

    Returns:
        The tasks.
    """
    return _suite(context, data_dir, uids).tasks()


def grader(
    context: object, data_dir: str, uids: Sequence[str], tolerance: float = 0.0, reveal_answer: bool = False
) -> OfficeQAGrader:
    """Builds the grader for the named questions.

    Args:
        context: The pipeline context.
        data_dir: The fetched data directory.
        uids: The questions, as given to :func:`tasks`.
        tolerance: The relative error accepted.
        reveal_answer: Put reference answers in feedback.

    Returns:
        The grader.
    """
    return OfficeQAGrader(_suite(context, data_dir, uids), tolerance=tolerance, reveal_answer=reveal_answer)


def _suite(context: object, data_dir: str, uids: Sequence[str]) -> OfficeQASuite:
    path = Path(os.path.expandvars(data_dir.format(root=str(getattr(context, "root", "."))))).expanduser()
    shared = getattr(context, "shared", None)
    key = f"officeqa:{path}:{','.join(uids)}"
    cached = shared.get(key) if isinstance(shared, dict) else None
    if isinstance(cached, OfficeQASuite):
        return cached
    suite = load_suite(path / QUESTIONS, uids)
    if isinstance(shared, dict):
        shared[key] = suite
    return suite
