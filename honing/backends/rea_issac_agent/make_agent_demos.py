"""Collect public-state REA-Issac combat demonstrations from the fixed rule baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from honing.backends.rea_issac.catalog import DIRECTIONS
from honing.backends.rea_issac.harness import parse_harness, seed_harness
from honing.backends.rea_issac.match import history_entry
from honing.backends.rea_issac.model import make_request
from honing.backends.rea_issac.scenarios import ReaIssacScenario
from integrations.rea_issac.engine import Engine
from integrations.rea_issac.policy import Baseline

ROOT = Path(__file__).resolve().parents[3]
INTEGRATION = Path(__file__).resolve().parent
REVERSE_DIRECTIONS = {vector: name for name, vector in DIRECTIONS.items()}
TRAIN_SEEDS = (7, 11, 1001, 1002, 2002)
VALIDATION_SEEDS = (2001,)
HELDOUT_SEEDS = (3001, 3002)


def verify_frozen() -> dict[str, Any]:
    frozen = json.loads((INTEGRATION / "agent_frozen.json").read_text())
    for name, expected in frozen["files"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"frozen game or observation file changed: {name}")
    return frozen


def choose_evenly(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(rows) <= limit:
        return rows
    return [rows[index * len(rows) // limit] for index in range(limit)]


def choose_across_rooms(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Spread a fixed per-seed budget over every observed combat room."""
    by_room: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_room.setdefault(row["room"], []).append(row)
    room_rows = list(by_room.values())
    quotas = [0] * len(room_rows)
    remaining = min(limit, len(rows))
    while remaining:
        for index, group in enumerate(room_rows):
            if remaining and quotas[index] < len(group):
                quotas[index] += 1
                remaining -= 1
    groups = [choose_evenly(group, quota) for group, quota in zip(room_rows, quotas, strict=True)]
    selected: list[dict[str, Any]] = []
    for index in range(max(map(len, groups), default=0)):
        selected.extend(group[index] for group in groups if index < len(group))
    return selected


def collect_seed(
    seed: int, *, max_ticks: int, limit: int, max_state_chars: int, selection: str = "chronological"
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scenario = ReaIssacScenario(seed, max_ticks, 3)
    harness = parse_harness(seed_harness(INTEGRATION))
    engine, teacher = Engine(seed, max_ticks), Baseline()
    history: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    while engine.status == "running":
        observation = engine.observe()
        action = teacher.act(observation)
        if observation["room"]["enemies"]:
            request = make_request(observation, scenario, harness, history)
            selected = {
                "move": REVERSE_DIRECTIONS[action.get("move_x", 0), action.get("move_y", 0)],
                "shoot": REVERSE_DIRECTIONS[action.get("shoot_x", 0), action.get("shoot_y", 0)],
                "bomb": "YES" if action.get("bomb", False) else "NO",
            }
            candidates.append(
                {
                    "seed": seed,
                    "tick": engine.tick,
                    "room": observation["room"]["id"],
                    "request": request,
                    "chosen": selected,
                    "group": f"rea-issac:{seed}:{observation['room']['id']}",
                }
            )
        next_observation = engine.step(action, scenario.decision_frames)
        history.append(history_entry(action, next_observation))
    eligible = [row for row in candidates if len(row["request"]["parts"]["essential"][0]) <= max_state_chars]
    if selection not in ("chronological", "balanced_rooms"):
        raise ValueError(f"unsupported demonstration selection: {selection}")
    chosen = choose_across_rooms(eligible, limit) if selection == "balanced_rooms" else choose_evenly(eligible, limit)
    return chosen, {
        "seed": seed,
        "combat_decisions_available": len(candidates),
        "combat_decisions_within_char_limit": len(eligible),
        "combat_decisions_selected": min(len(eligible), limit),
        "result": engine.result(),
    }


def build_dataset(
    output: Path,
    *,
    max_ticks: int = 3000,
    per_seed: int = 40,
    max_state_chars: int = 7000,
    selection: str = "chronological",
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    frozen = verify_frozen()
    output.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "purpose": "nanojev-rea-issac-agent-sft-demonstrations",
        "teacher": "integrations.rea_issac.policy.Baseline",
        "teacher_is_reference_action": False,
        "frozen": frozen,
        "train_seeds": list(TRAIN_SEEDS),
        "validation_seeds": list(VALIDATION_SEEDS),
        "heldout_seeds": list(HELDOUT_SEEDS),
        "max_ticks": max_ticks,
        "decision_frames": 3,
        "per_seed": per_seed,
        "max_state_chars": max_state_chars,
        "episodes": [],
    }
    if selection != "chronological":
        summary["selection"] = selection
    for split, seeds in (("train", TRAIN_SEEDS), ("validation", VALIDATION_SEEDS)):
        for seed in seeds:
            selected, episode = collect_seed(
                seed, max_ticks=max_ticks, limit=per_seed, max_state_chars=max_state_chars, selection=selection
            )
            records.extend({"split": split, **row} for row in selected)
            summary["episodes"].append({"split": split, **episode})
    with (output / "decisions.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    summary["decisions"] = len(records)
    summary["decisions_sha256"] = hashlib.sha256((output / "decisions.jsonl").read_bytes()).hexdigest()
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-ticks", type=int, default=3000)
    parser.add_argument("--per-seed", type=int, default=40)
    parser.add_argument("--max-state-chars", type=int, default=7000)
    parser.add_argument("--selection", choices=("chronological", "balanced_rooms"), default="chronological")
    args = parser.parse_args()
    if args.per_seed < 1 or args.max_state_chars < 1:
        parser.error("--per-seed and --max-state-chars must be positive")
    print(
        json.dumps(
            build_dataset(
                args.out,
                max_ticks=args.max_ticks,
                per_seed=args.per_seed,
                max_state_chars=args.max_state_chars,
                selection=args.selection,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
