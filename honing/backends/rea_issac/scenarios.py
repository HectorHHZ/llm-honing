"""Seeded boss-defeat tasks whose game budgets are fixed outside the harness."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from honing.core import ContractError, Task, require

Split = Literal["train", "val", "heldout"]
SPLITS = ("train", "val", "heldout")


@dataclass(frozen=True)
class ReaIssacScenario:
    seed: int = 7
    max_ticks: int = 6000
    decision_frames: int = 3

    def __post_init__(self) -> None:
        require(type(self.seed) is int and 0 <= self.seed < 2**32, "seed must be uint32")
        require(type(self.max_ticks) is int and 1 <= self.max_ticks <= 108000, "max_ticks must be in 1..108000")
        require(
            type(self.decision_frames) is int and 1 <= self.decision_frames <= 30, "decision_frames must be in 1..30"
        )

    def record(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class ReaIssacSuite:
    records: tuple[dict[str, object], ...]

    def __post_init__(self) -> None:
        ids: set[str] = set()
        seeds: dict[int, str] = {}
        for record in self.records:
            require(isinstance(record, dict), "suite records must be objects")
            task_id, split = record.get("task_id"), record.get("split")
            require(isinstance(task_id, str) and bool(task_id) and task_id not in ids, "task IDs must be unique text")
            require(split in SPLITS, "unknown split")
            require(isinstance(record.get("prompt"), str) and bool(record["prompt"]), "task prompt is required")
            scenario = self.scenario(str(task_id))
            require(scenario.seed not in seeds or seeds[scenario.seed] == split, "splits must not share seeds")
            seeds[scenario.seed] = str(split)
            ids.add(str(task_id))

    def scenario(self, task_id: str) -> ReaIssacScenario:
        for record in self.records:
            if record.get("task_id") == task_id:
                config = record.get("rea_issac")
                require(isinstance(config, dict), "rea_issac must be a scenario object")
                assert isinstance(config, dict)
                return ReaIssacScenario(**config)
        raise ContractError(f"task {task_id!r} is not in the REA-Issac suite")

    def tasks(self, split: Split, *, limit: int | None = None) -> tuple[Task, ...]:
        require(split in SPLITS, "unknown split")
        require(limit is None or (type(limit) is int and limit > 0), "limit must be positive")
        chosen = [r for r in self.records if r["split"] == split]
        return tuple(Task(str(r["task_id"]), str(r["prompt"])) for r in chosen[:limit])


def build_suite() -> list[dict[str, object]]:
    return [
        {
            "task_id": f"rea-issac-{split}-s{seed}",
            "split": split,
            "prompt": "Defeat the Warden before the tick budget expires. Survive, clear rooms and collect upgrades.",
            "rea_issac": ReaIssacScenario(seed).record(),
        }
        for split, seeds in (("train", (1001, 1002)), ("val", (2001, 2002)), ("heldout", (3001, 3002)))
        for seed in seeds
    ]


def load_suite(path: Path) -> ReaIssacSuite:
    try:
        return ReaIssacSuite(tuple(json.loads(line) for line in path.read_text().splitlines() if line.strip()))
    except (KeyError, TypeError, ValueError) as error:
        raise ContractError(f"invalid REA-Issac suite: {error}") from error
