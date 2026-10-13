"""Tank duel scenarios and the task suite.

Each suite line is one task: ``task_id``, ``prompt``, ``split``, and a ``tank``
block with the scenario. A task is one Classic round on the seeded maze, with
the policy in one seat and the planner opponent in the other. Every seed is
played from both seats, because the two spawns are not equally placed on a
given maze. Splits: ``train`` for a method's minibatches and for rollouts,
``val`` for the gate, ``heldout`` for reports the gate never sees.

Build the suite: ``python -m honing.backends.tank.scenarios <suite.jsonl>``
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, get_args

from honing.core import ContractError, Task, require, require_choice

Seat = Literal["blue", "red"]
Split = Literal["train", "val", "heldout"]
Opponent = Literal["planner"]
SEATS: tuple[Seat, ...] = ("blue", "red")
# Seeds per split; a seed is played once from each seat.
SPLIT_SEEDS: dict[Split, range] = {
    "train": range(1001, 1017),
    "val": range(2001, 2009),
    "heldout": range(3001, 3009),
}


@dataclass(frozen=True)
class TankScenario:
    """One round: the maze seed, the policy's seat, the opponent, and the clock.

    ``interval_ms`` is how often, in game time, the policy is asked, and
    ``opponent_interval_ms`` the opponent. They belong to the task, as RA2's
    cadence does, so no harness can buy more decisions than its opponent.
    """

    seed: int
    seat: Seat
    opponent: Opponent = "planner"
    seconds: float = 30.0
    interval_ms: int = 200
    opponent_interval_ms: int = 300

    def __post_init__(self) -> None:
        require(self.seed >= 0, "a scenario seed is non-negative")
        require_choice(self.seat, get_args(Seat), "TankScenario.seat")
        require_choice(self.opponent, get_args(Opponent), "TankScenario.opponent")
        require(self.seconds > 0, "a scenario needs a positive time limit")
        require(self.interval_ms > 0 and self.opponent_interval_ms > 0, "decision intervals must be positive")


def _prompt(scenario: TankScenario) -> str:
    return (
        f"Play the {scenario.seat} tank against the {scenario.opponent} opponent on the Classic maze with seed "
        f"{scenario.seed}. Win the round by hitting the enemy within {scenario.seconds:g} seconds; every shell, "
        "including your own, is lethal."
    )


def _task_id(split: Split, scenario: TankScenario) -> str:
    return f"tank-{split}-s{scenario.seed}-{scenario.seat}"


def build_suite(seconds: float = 30.0) -> list[dict[str, object]]:
    """Every task, one record per split, seed, and seat, in a stable order (both seats of a seed together)."""
    records: list[dict[str, object]] = []
    for split, seeds in SPLIT_SEEDS.items():
        for seed in seeds:
            for seat in SEATS:
                scenario = TankScenario(seed=seed, seat=seat, seconds=seconds)
                records.append(
                    {
                        "task_id": _task_id(split, scenario),
                        "prompt": _prompt(scenario),
                        "split": split,
                        "tank": asdict(scenario),
                    }
                )
    return records


@dataclass(frozen=True)
class TankSuite:
    """A loaded task suite: tasks by split, and the scenario behind each task."""

    records: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        ids = [str(record["task_id"]) for record in self.records]
        require(len(set(ids)) == len(ids), "a task suite names each task once")

    def tasks(self, split: Split, *, limit: int | None = None) -> tuple[Task, ...]:
        require_choice(split, get_args(Split), "split")
        chosen = [r for r in self.records if r["split"] == split]
        return tuple(Task(task_id=str(r["task_id"]), prompt=str(r["prompt"])) for r in chosen[:limit])

    def scenario(self, task_id: str) -> TankScenario:
        for record in self.records:
            if record["task_id"] == task_id:
                return TankScenario(**record["tank"])  # type: ignore[arg-type]
        raise ContractError(f"task {task_id!r} is not in the tank suite")

    def split_of(self, task_id: str) -> Split:
        for record in self.records:
            if record["task_id"] == task_id:
                return record["split"]  # type: ignore[return-value]
        raise ContractError(f"task {task_id!r} is not in the tank suite")


def load_suite(path: Path) -> TankSuite:
    """Read a suite file, refusing a line without a known split or a valid scenario."""
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        try:
            require_choice(record.get("split"), get_args(Split), "split")
            TankScenario(**record["tank"])
        except (ContractError, KeyError, TypeError) as error:
            raise ContractError(f"{path}:{number}: {error}") from None
        records.append(record)
    return TankSuite(tuple(records))


def write_suite(records: Iterable[Mapping[str, object]], path: Path) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    (suite_path,) = (argv or sys.argv[1:])[:1]
    write_suite(build_suite(), Path(suite_path))


if __name__ == "__main__":
    main()
