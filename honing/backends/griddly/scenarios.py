"""Fixed native levels, seeds and budgets; global observations define a separate task."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from honing.backends.griddly.catalog import PROTOCOL
from honing.core import ContractError, Task, require

Split = Literal["train", "val", "heldout"]
SPLITS = ("train", "val", "heldout")


@dataclass(frozen=True)
class GriddlyScenario:
    seed: int = 1
    level_id: int = 0
    max_steps: int = 100
    observation: str = PROTOCOL

    def __post_init__(self) -> None:
        require(type(self.seed) is int and 0 <= self.seed < 2**32, "seed must be uint32")
        require(type(self.level_id) is int and 0 <= self.level_id <= 4, "Spiders level_id must be in 0..4")
        require(type(self.max_steps) is int and 1 <= self.max_steps <= 10000, "max_steps must be in 1..10000")
        require(self.observation == PROTOCOL, f"observation must be {PROTOCOL}")

    def record(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GriddlySuite:
    records: tuple[dict[str, object], ...]

    def __post_init__(self) -> None:
        ids: set[str] = set()
        levels: dict[int, str] = {}
        seeds: dict[int, str] = {}
        for record in self.records:
            task_id, split = record.get("task_id"), record.get("split")
            require(isinstance(task_id, str) and bool(task_id) and task_id not in ids, "task IDs must be unique")
            require(split in SPLITS, "unknown task split")
            require(isinstance(record.get("prompt"), str) and bool(record["prompt"]), "a task needs a prompt")
            scenario = self.scenario(str(task_id))
            for key, seen in ((scenario.level_id, levels), (scenario.seed, seeds)):
                require(key not in seen or seen[key] == split, "splits must not share levels or seeds")
                seen[key] = str(split)
            ids.add(str(task_id))

    def scenario(self, task_id: str) -> GriddlyScenario:
        for record in self.records:
            if record.get("task_id") == task_id:
                config = record.get("griddly")
                require(isinstance(config, dict), "griddly must be a scenario object")
                assert isinstance(config, dict)
                return GriddlyScenario(**config)
        raise ContractError(f"unknown Griddly task {task_id!r}")

    def split_of(self, task_id: str) -> str:
        self.scenario(task_id)
        return str(next(r["split"] for r in self.records if r["task_id"] == task_id))

    def tasks(self, split: Split, *, limit: int | None = None) -> tuple[Task, ...]:
        require(split in SPLITS, "unknown task split")
        require(limit is None or (type(limit) is int and limit > 0), "limit must be positive")
        rows = [r for r in self.records if r["split"] == split]
        return tuple(Task(str(r["task_id"]), str(r["prompt"])) for r in rows[:limit])


def build_suite() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for split, levels, start in (("train", (0, 1), 1001), ("val", (2, 3), 2001), ("heldout", (4,), 3001)):
        for level in levels:
            for offset in range(4):
                scenario = GriddlyScenario(seed=start + level * 10 + offset, level_id=level)
                records.append(
                    {
                        "task_id": f"spiders-{split}-l{level}-s{scenario.seed}",
                        "split": split,
                        "prompt": "Reach the gem while avoiding spiders, using the global object observation.",
                        "griddly": scenario.record(),
                    }
                )
    return records


def load_suite(path: Path) -> GriddlySuite:
    try:
        return GriddlySuite(
            tuple(json.loads(row) for row in path.read_text(encoding="utf-8").splitlines() if row.strip())
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ContractError(f"invalid Griddly suite {path}: {error}") from error
