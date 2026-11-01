"""Seeded full-game tasks with explicit achievement goals and disjoint splits."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from honing.backends.craftax.catalog import ACHIEVEMENTS, DEFAULT_GOALS
from honing.core import ContractError, Task, require

Split = Literal["train", "val", "heldout"]
SPLITS = ("train", "val", "heldout")


@dataclass(frozen=True)
class CraftaxScenario:
    """The world seed, action budget and goals belong to the task, not the harness."""

    seed: int = 1
    max_steps: int = 300
    goals: tuple[str, ...] = DEFAULT_GOALS

    def __post_init__(self) -> None:
        require(type(self.seed) is int and 0 <= self.seed < 2**32, "seed must be a uint32 integer")
        require(type(self.max_steps) is int and self.max_steps > 0, "max_steps must be a positive integer")
        require(isinstance(self.goals, tuple) and bool(self.goals), "goals must be a nonempty tuple")
        require(all(isinstance(g, str) for g in self.goals), "goals must contain achievement names")
        require(len(set(self.goals)) == len(self.goals), "goals must be unique")
        require(set(self.goals) <= ACHIEVEMENTS, f"unknown goals: {sorted(set(self.goals) - ACHIEVEMENTS)}")

    def record(self) -> dict[str, object]:
        return {**asdict(self), "goals": list(self.goals)}


@dataclass(frozen=True)
class CraftaxSuite:
    """Task records whose seeds cannot leak between train, validation and heldout."""

    records: tuple[dict[str, object], ...]

    def __post_init__(self) -> None:
        ids: set[str] = set()
        seeds: dict[int, str] = {}
        for record in self.records:
            require(isinstance(record, dict), "each suite record must be an object")
            task_id, split = record.get("task_id"), record.get("split")
            require(isinstance(task_id, str) and bool(task_id) and task_id not in ids, "task IDs must be unique text")
            require(isinstance(record.get("prompt"), str) and bool(record["prompt"]), "a task needs a prompt")
            require(split in SPLITS, f"unknown split {split!r}")
            scenario = self.scenario(str(task_id))
            require(scenario.seed not in seeds or seeds[scenario.seed] == split, "splits must not share seeds")
            seeds[scenario.seed] = str(split)
            ids.add(str(task_id))

    def scenario(self, task_id: str) -> CraftaxScenario:
        for record in self.records:
            if record.get("task_id") == task_id:
                config = record.get("craftax")
                require(isinstance(config, dict), "craftax must be a scenario object")
                assert isinstance(config, dict)
                return CraftaxScenario(**{**config, "goals": tuple(config.get("goals", DEFAULT_GOALS))})
        raise ContractError(f"task {task_id!r} is not in the Craftax suite")

    def tasks(self, split: Split, *, limit: int | None = None) -> tuple[Task, ...]:
        require(split in SPLITS, f"unknown split {split!r}")
        require(limit is None or (type(limit) is int and limit > 0), "limit must be positive")
        chosen = [r for r in self.records if r["split"] == split]
        return tuple(Task(str(r["task_id"]), str(r["prompt"])) for r in chosen[:limit])


def build_suite() -> list[dict[str, object]]:
    """Introductory crafting tasks in the complete game; the full action set remains available."""
    records: list[dict[str, object]] = []
    for split, seeds in (("train", (1001, 1002)), ("val", (2001, 2002)), ("heldout", (3001, 3002))):
        for seed in seeds:
            scenario = CraftaxScenario(seed=seed)
            records.append(
                {
                    "task_id": f"craftax-{split}-s{seed}",
                    "split": split,
                    "prompt": "Collect wood, place a crafting table, and make a wooden pickaxe.",
                    "craftax": scenario.record(),
                }
            )
    return records


def load_suite(path: Path) -> CraftaxSuite:
    try:
        return CraftaxSuite(tuple(json.loads(line) for line in path.read_text().splitlines() if line.strip()))
    except (KeyError, TypeError, ValueError) as error:
        raise ContractError(f"invalid Craftax suite {path}: {error}") from error
