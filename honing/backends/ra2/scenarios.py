"""RA2 scenarios and the task suite built from the league protocol.

Each suite line is one task: ``task_id``, ``prompt``, ``split``, and an ``ra2``
block with the scenario. Splits follow the protocol: ``train`` for learning,
``val`` for the gate (the development matrix), and ``heldout`` for the
qualification and sealed cases, which the gate never uses.

Build the suite: ``python -m honing.backends.ra2.scenarios <protocol.json> <suite.jsonl>``
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, get_args

from honing.core import ContractError, Task, require, require_choice

Seat = Literal["red", "blue"]
Split = Literal["train", "val", "heldout"]
SEATS: tuple[Seat, ...] = ("red", "blue")


@dataclass(frozen=True)
class RA2Scenario:
    """One match: which map, which seat the policy plays, the seed, and how long it may run."""

    map: str
    seat: Seat
    seed: int
    ticks: int
    opponent: str
    cadence: int = 3

    def __post_init__(self) -> None:
        require(self.map.endswith(".map"), f"a scenario map is a .map file, got {self.map!r}")
        require_choice(self.seat, get_args(Seat), "RA2Scenario.seat")
        require(self.ticks > 0, "a scenario needs a positive tick limit")
        require(self.cadence > 0 and self.cadence % 3 == 0, "cadence must be a positive multiple of 3")


def _prompt(scenario: RA2Scenario) -> str:
    return (
        f"Play {scenario.seat} as Russians against the Supalosa bot on {scenario.map.removesuffix('.map')} "
        f"with seed {scenario.seed}. Win by destroying the enemy base within {scenario.ticks} ticks."
    )


def _task_id(split: Split, scenario: RA2Scenario) -> str:
    return f"ra2-{split}-{scenario.map.removesuffix('.map')}-s{scenario.seed}-{scenario.seat}"


def build_suite(protocol: Mapping[str, object]) -> list[dict[str, object]]:
    """Every task the protocol defines, one record per map, seed, and seat, in a stable order."""
    opponent = f"supalosa@{protocol['baselineRevision']}"
    cadence = int(protocol["cadenceTicks"])  # type: ignore[call-overload]
    groups: list[tuple[Split, str]] = [
        ("train", "train"),
        ("val", "dev"),
        ("heldout", "layaQualification"),
        ("heldout", "sealedTest"),
    ]
    records: list[dict[str, object]] = []
    for split, key in groups:
        group = protocol[key]
        if not isinstance(group, Mapping):
            raise ContractError(f"protocol section {key!r} is missing")
        for map_name in group["maps"]:
            for seed in group["seeds"]:
                for seat in group.get("seats", SEATS):
                    scenario = RA2Scenario(map_name, seat, seed, int(group["maxTicks"]), opponent, cadence)
                    records.append(
                        {
                            "task_id": _task_id(split, scenario),
                            "prompt": _prompt(scenario),
                            "split": split,
                            "source": key,
                            "ra2": asdict(scenario),
                        }
                    )
    return records


@dataclass(frozen=True)
class RA2Suite:
    """A loaded task suite: tasks by split, and the scenario behind each task."""

    records: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        ids = [str(record["task_id"]) for record in self.records]
        require(len(set(ids)) == len(ids), "a task suite names each task once")

    def tasks(self, split: Split, *, limit: int | None = None) -> tuple[Task, ...]:
        require_choice(split, get_args(Split), "split")
        chosen = [r for r in self.records if r["split"] == split]
        return tuple(Task(task_id=str(r["task_id"]), prompt=str(r["prompt"])) for r in chosen[:limit])

    def scenario(self, task_id: str) -> RA2Scenario:
        for record in self.records:
            if record["task_id"] == task_id:
                return RA2Scenario(**record["ra2"])  # type: ignore[arg-type]
        raise ContractError(f"task {task_id!r} is not in the RA2 suite")

    def split_of(self, task_id: str) -> Split:
        for record in self.records:
            if record["task_id"] == task_id:
                return record["split"]  # type: ignore[return-value]
        raise ContractError(f"task {task_id!r} is not in the RA2 suite")


def load_suite(path: Path) -> RA2Suite:
    """Read a suite file, refusing a line without a known split or a valid scenario."""
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        try:
            require_choice(record.get("split"), get_args(Split), "split")
            RA2Scenario(**record["ra2"])
        except (ContractError, KeyError, TypeError) as error:
            raise ContractError(f"{path}:{number}: {error}") from None
        records.append(record)
    return RA2Suite(tuple(records))


def write_suite(records: Iterable[Mapping[str, object]], path: Path) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    protocol_path, suite_path = (argv or sys.argv[1:])[:2]
    write_suite(build_suite(json.loads(Path(protocol_path).read_text())), Path(suite_path))


if __name__ == "__main__":
    main()
