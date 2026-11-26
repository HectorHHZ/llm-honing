"""Tank duel scenarios and the task suite.

Each suite line is one task: ``task_id``, ``prompt``, ``split``, and a ``tank``
block with the scenario. A task is one Tank War round on the seeded maze, with
the policy in one seat and the planner opponent in the other. Every seed is
played from both seats, because the two spawns are not equally placed on a
given maze. Splits: ``train`` for a method's minibatches and for rollouts,
``val`` for validation, ``heldout`` for reports no campaign sees.

``val`` and ``heldout`` also have large seed pools beyond the suite file
(:data:`SEED_POOLS`). The suite file's ``val`` tasks are a fixed set a method
such as GEPA may select on. Every gate draws its rounds from the ``val`` pool
instead, and no two gates of a campaign play the same seed
(:class:`~honing.backends.tank.gate.FreshSeedEvaluator`). No campaign code
path can ask for ``heldout``: :meth:`TankSuite.tasks` and :func:`pool_tasks`
refuse it, and experiment scripts read it through
:meth:`TankSuite.heldout_tasks`.

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
#: Seeds beyond the suite file, disjoint from its seeds and from each other. A
#: pool round runs on the default clock, as the suite file's rounds do.
SEED_POOLS: dict[Split, range] = {"val": range(100_000, 200_000), "heldout": range(200_000, 300_000)}


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
        f"Play the {scenario.seat} tank against the {scenario.opponent} opponent on the tank maze with seed "
        f"{scenario.seed}. Win the round by hitting the enemy within {scenario.seconds:g} seconds; every shell, "
        "including your own, is lethal."
    )


def _task_id(split: Split, scenario: TankScenario) -> str:
    return f"tank-{split}-s{scenario.seed}-{scenario.seat}"


def _record(split: Split, scenario: TankScenario) -> dict[str, object]:
    return {
        "task_id": _task_id(split, scenario),
        "prompt": _prompt(scenario),
        "split": split,
        "tank": asdict(scenario),
    }


def build_suite(seconds: float = 30.0) -> list[dict[str, object]]:
    """Every task, one record per split, seed, and seat, in a stable order (both seats of a seed together)."""
    return [
        _record(split, TankScenario(seed=seed, seat=seat, seconds=seconds))
        for split, seeds in SPLIT_SEEDS.items()
        for seed in seeds
        for seat in SEATS
    ]


def _refuse_heldout(split: str) -> None:
    require(split != "heldout", "heldout tasks are read only by experiment scripts, through TankSuite.heldout_tasks")


def _pool_tasks(split: Split, seeds: Iterable[int]) -> tuple[Task, ...]:
    tasks = []
    for seed in seeds:
        require(seed in SEED_POOLS[split], f"seed {seed} is not in the {split} pool")
        for seat in SEATS:
            record = _record(split, TankScenario(seed=seed, seat=seat))
            tasks.append(Task(task_id=str(record["task_id"]), prompt=str(record["prompt"])))
    return tuple(tasks)


def _pool_scenario(split: Split, task_id: str) -> TankScenario | None:
    """The scenario of the split's pool task named ``task_id``, or None when no pool task has that id."""
    seed = task_id.removeprefix(f"tank-{split}-s").rpartition("-")[0]
    if not (seed.isdigit() and int(seed) in SEED_POOLS[split]):
        return None
    for seat in SEATS:
        scenario = TankScenario(int(seed), seat)
        if _task_id(split, scenario) == task_id:
            return scenario
    return None


def pool_tasks(split: Split, seeds: Iterable[int]) -> tuple[Task, ...]:
    """Both seats of each seed from the split's pool, seed by seed. ``heldout`` is refused."""
    require_choice(split, tuple(SEED_POOLS), "pool")
    _refuse_heldout(split)
    return _pool_tasks(split, seeds)


@dataclass(frozen=True)
class TankSuite:
    """A loaded task suite: tasks by split, and the scenario behind each task, pool tasks included."""

    records: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        ids = [str(record["task_id"]) for record in self.records]
        require(len(set(ids)) == len(ids), "a task suite names each task once")

    def tasks(self, split: Split, *, limit: int | None = None) -> tuple[Task, ...]:
        """The split's tasks in the suite file, in file order. ``heldout`` is refused."""
        require_choice(split, get_args(Split), "split")
        _refuse_heldout(split)
        return self._file_tasks(split, limit)

    def heldout_tasks(self, seeds: Iterable[int] | None = None) -> tuple[Task, ...]:
        """Heldout tasks, for experiment scripts. No campaign code path calls this.

        Without ``seeds``, the suite file's heldout tasks in file order. With
        them, both seats of each seed from the heldout pool.
        """
        return self._file_tasks("heldout", None) if seeds is None else _pool_tasks("heldout", seeds)

    def scenario(self, task_id: str) -> TankScenario:
        return TankScenario(**self._record(task_id)["tank"])  # type: ignore[arg-type]

    def split_of(self, task_id: str) -> Split:
        return self._record(task_id)["split"]  # type: ignore[return-value]

    def pool_overlap(self, split: Split) -> dict[str, int]:
        """The suite file's tasks a task of the split's pool could play, each with its seed.

        A line reaches the pool if its seed is in it, or if its id is a pool
        task's, since :meth:`scenario` reads the suite file before the pools.
        """
        require_choice(split, tuple(SEED_POOLS), "pool")
        overlap = {}
        for record in self.records:
            task_id, seed = str(record["task_id"]), TankScenario(**record["tank"]).seed  # type: ignore[arg-type]
            if seed in SEED_POOLS[split] or _pool_scenario(split, task_id) is not None:
                overlap[task_id] = seed
        return overlap

    def _file_tasks(self, split: Split, limit: int | None) -> tuple[Task, ...]:
        chosen = [r for r in self.records if r["split"] == split]
        return tuple(Task(task_id=str(r["task_id"]), prompt=str(r["prompt"])) for r in chosen[:limit])

    def _record(self, task_id: str) -> Mapping[str, object]:
        """The task's suite line, or for a pool task the line it would have."""
        for record in self.records:
            if record["task_id"] == task_id:
                return record
        for split in SEED_POOLS:
            scenario = _pool_scenario(split, task_id)
            if scenario is not None:
                return _record(split, scenario)
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
