"""Single-component REA-Issac fault candidates for issue #184."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from honing.backends.rea_issac.catalog import DIRECTIONS
from honing.backends.rea_issac.harness import QUESTIONS_PATH, parse_harness, seed_harness
from honing.backends.rea_issac_agent.make_agent_demos import verify_frozen
from honing.backends.rea_issac_agent.train_agent import load_batch
from honing.core import HarnessFile, HarnessSpec

HARNESS_REVERSE_MOVE = "harness-reverse-move-v1"
CONTROLLER_SWAP_MOVE_AXES = "controller-swap-move-axes-v1"
WEIGHTS_REVERSE_MOVE = "weights-reverse-move-sft-v1"

REVERSED_MOVE_INSTRUCTION = (
    "Movement controls are inverted: choose the compass direction opposite to where you want the player to move. "
    "For example, choose W to travel east, E to travel west, N to travel south, and S to travel north. "
    "Apply the same reversal to diagonal movement. Choose NONE only when staying still is safest."
)

OPPOSITE = {
    name: next(other for other, candidate in DIRECTIONS.items() if candidate == (-vector[0], -vector[1]))
    for name, vector in DIRECTIONS.items()
}
PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]


def verify_clean_baseline() -> dict[str, Any]:
    """Refuse a changed game, agent controller, harness, or selected checkpoint pin."""
    frozen = json.loads((PACKAGE / "fault_freeze.json").read_text())
    if frozen.get("format") != "rea-issac-agent-fault-freeze-v1":
        raise ValueError("unknown REA-Issac fault freeze")
    verify_frozen()
    for name, expected in frozen["files"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"frozen REA-Issac agent file changed: {name}")
    if seed_harness(PACKAGE).spec_id != frozen["harness_spec_id"]:
        raise ValueError("frozen REA-Issac agent harness changed")
    return frozen


def reverse_movement_harness(clean: HarnessSpec) -> HarnessSpec:
    """Change only the model-facing movement instruction, never the observed state or option IDs."""
    frozen = verify_clean_baseline()
    if clean.spec_id != frozen["harness_spec_id"]:
        raise ValueError("harness fault must start from the frozen clean agent")
    original = clean.get(QUESTIONS_PATH)
    if original is None or original.kind != "instructions":
        raise ValueError("clean REA-Issac harness has no movement instructions")
    questions = json.loads(original.content)
    if set(questions) != {"move", "shoot", "bomb"}:
        raise ValueError("clean REA-Issac questions changed")
    questions["move"] = REVERSED_MOVE_INSTRUCTION
    faulty = clean.with_file(QUESTIONS_PATH, HarnessFile("instructions", json.dumps(questions, indent=2) + "\n"))
    parse_harness(faulty)
    return faulty


def swap_move_axes(action: dict[str, Any]) -> dict[str, Any]:
    """Actuator fault after the clean agent controller has chosen an action."""
    changed = action.copy()
    changed["move_x"], changed["move_y"] = action["move_y"], action["move_x"]
    return changed


def build_reversed_move_dataset(source: Path, output: Path) -> dict[str, Any]:
    """Train only the move question, with a deliberately opposite label for every nonzero move."""
    verify_clean_baseline()
    _, summary = load_batch(source)
    if output.exists():
        raise FileExistsError(output)
    rows = [json.loads(line) for line in (source / "decisions.jsonl").read_text().splitlines() if line]
    transformed: list[dict[str, Any]] = []
    per_seed: dict[int, int] = {}
    for row in rows:
        correct = row["chosen"]["move"]
        if correct == "NONE":
            continue
        changed = copy.deepcopy(row)
        changed["chosen"] = {"move": OPPOSITE[correct]}
        changed["request"]["questions"] = {"move": changed["request"]["questions"]["move"]}
        changed["source_release"] = WEIGHTS_REVERSE_MOVE
        transformed.append(changed)
        per_seed[row["seed"]] = per_seed.get(row["seed"], 0) + 1
    if not transformed or not any(row["split"] == "validation" for row in transformed):
        raise ValueError("reverse-movement fault dataset has no train or validation movements")
    result = {**summary}
    result.update(
        purpose="rea-issac-agent-reversed-movement-sft",
        fault_family=WEIGHTS_REVERSE_MOVE,
        source_decisions_sha256=summary["decisions_sha256"],
        excluded_zero_move=len(rows) - len(transformed),
        decisions=len(transformed),
        episodes=[
            {**episode, "combat_decisions_selected": per_seed.get(episode["seed"], 0)}
            for episode in summary["episodes"]
        ],
    )
    output.mkdir(parents=True)
    data = output / "decisions.jsonl"
    with data.open("x") as stream:
        for row in transformed:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    result["decisions_sha256"] = hashlib.sha256(data.read_bytes()).hexdigest()
    (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the issue #184 reversed-movement weights fault dataset")
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_reversed_move_dataset(args.source_dataset, args.out), sort_keys=True))


if __name__ == "__main__":
    main()
