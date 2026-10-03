"""Factories that put OfficeQA into a pipeline config (see ``honing.e2e.components``).

``data_dir`` is what ``python -m honing.backends.officeqa.fetch`` wrote: the
questions CSV and ``corpus/``. Paths may use ``{root}`` and environment variables.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

from honing.backends.officeqa.fetch import QUESTIONS
from honing.backends.officeqa.grader import OfficeQAGrader
from honing.backends.officeqa.suite import OfficeQASuite, load_suite
from honing.backends.openhands.runner import AGENT, AGENT_SETTINGS
from honing.core import HarnessFile, HarnessSpec, Task

SEED_INSTRUCTIONS = """You answer questions about the documents in your working directory.
Look figures up in the documents rather than answering from memory, and report them in the unit the question asks for.
"""
SEED_AGENT_SETTINGS = {"tools": ["terminal", "file_editor"], "max_iterations": 25}


def _suite(context: object, data_dir: str, uids: Sequence[str]) -> OfficeQASuite:
    root = str(getattr(context, "root", "."))
    path = Path(os.path.expandvars(data_dir.format(root=root))) / QUESTIONS
    shared = getattr(context, "shared", None)
    key = f"officeqa:{path}:{','.join(uids)}"
    cached = shared.get(key) if isinstance(shared, dict) else None
    if isinstance(cached, OfficeQASuite):
        return cached
    suite = load_suite(path, uids)
    if isinstance(shared, dict):
        shared[key] = suite
    return suite


def tasks(context: object, data_dir: str, uids: Sequence[str]) -> tuple[Task, ...]:
    return _suite(context, data_dir, uids).tasks()


def grader(
    context: object, data_dir: str, uids: Sequence[str], tolerance: float = 0.0, reveal_answer: bool = False
) -> OfficeQAGrader:
    return OfficeQAGrader(_suite(context, data_dir, uids), tolerance=tolerance, reveal_answer=reveal_answer)


def seed_harness(context: object) -> HarnessSpec:
    """A plain OpenHands harness for OfficeQA: short instructions and the default tools, no skills."""
    return HarnessSpec(
        agent=AGENT,
        files=(
            ("AGENTS.md", HarnessFile("instructions", SEED_INSTRUCTIONS)),
            (AGENT_SETTINGS, HarnessFile("settings", json.dumps(SEED_AGENT_SETTINGS, indent=2) + "\n")),
        ),
    )
