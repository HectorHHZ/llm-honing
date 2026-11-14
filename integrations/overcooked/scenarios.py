"""A small layout-disjoint smoke suite, not a generalization benchmark."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from honing.core import Task
from integrations.overcooked.engine import ROOT, Scenario


@dataclass(frozen=True)
class Suite:
    records: tuple[dict, ...]

    def __post_init__(self) -> None:
        ids: set[str] = set()
        layouts: dict[str, str] = {}
        for row in self.records:
            task_id, split = row["task_id"], row["split"]
            if not isinstance(task_id, str) or not task_id or task_id in ids:
                raise ValueError("task IDs must be unique nonempty strings")
            if split not in {"train", "val", "heldout"} or not isinstance(row["prompt"], str) or not row["prompt"]:
                raise ValueError("each task needs a known split and nonempty prompt")
            scenario = Scenario(**row["overcooked"])
            if scenario.layout in layouts and layouts[scenario.layout] != split:
                raise ValueError("kitchen layouts must not overlap between splits")
            layouts[scenario.layout] = split
            ids.add(task_id)

    def scenario(self, task_id: str) -> Scenario:
        for row in self.records:
            if row["task_id"] == task_id:
                return Scenario(**row["overcooked"])
        raise ValueError(f"unknown Overcooked task: {task_id}")

    def tasks(self, split: str, limit: int | None = None) -> tuple[Task, ...]:
        if split not in {"train", "val", "heldout"} or (limit is not None and (type(limit) is not int or limit < 1)):
            raise ValueError("invalid split or limit")
        rows = [row for row in self.records if row["split"] == split]
        return tuple(Task(row["task_id"], row["prompt"]) for row in rows[:limit])


def load_suite(path: Path = ROOT / "tasks/suite.jsonl") -> Suite:
    return Suite(tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()))
