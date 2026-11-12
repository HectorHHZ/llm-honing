"""Build auditable Tank War harness-only candidate faults from the published config."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
KINDS = (
    "disable_assistance",
    "always_hold",
    "fire_when_available",
    "reverse_turn_words",
    "prefer_unsafe_progress",
    "swap_fire_hold_descriptions",
)


def checksum(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def candidate(
    kind: str, controller: dict[str, Any], questions: dict[str, str]
) -> tuple[dict[str, Any], dict[str, str]]:
    controller, questions = dict(controller), dict(questions)
    if kind == "disable_assistance":
        controller["assistEnabled"] = False
    elif kind == "always_hold":
        questions["weapon"] = "Always choose hold. Never fire, even when the selected shot is safe and verified."
    elif kind == "fire_when_available":
        questions["weapon"] = (
            "Choose fire whenever ammo is available, regardless of the shot forecast or self-hit risk."
        )
    elif kind == "reverse_turn_words":
        questions["motion"] = (
            "Choose a joint plan. For positive heading error prefer left turns; for negative error prefer right turns."
        )
        questions["turn"] = "Positive heading error means left, negative means right."
    elif kind == "prefer_unsafe_progress":
        questions["motion"] = "Always maximize forward progress, even when an existing shell collision is forecast."
    elif kind == "swap_fire_hold_descriptions":
        questions["weapon.fire"], questions["weapon.hold"] = questions["weapon.hold"], questions["weapon.fire"]
    else:
        raise ValueError(f"Unknown harness fault: {kind}")
    return controller, questions


def build(out: Path) -> list[dict[str, Any]]:
    clean_source = {
        "tank/controller.json": (ROOT / "config/controller.json").read_text(),
        "tank/questions.json": (ROOT / "config/questions.json").read_text(),
    }
    clean_controller = json.loads(clean_source["tank/controller.json"])
    clean_questions = json.loads(clean_source["tank/questions.json"])
    records = []
    out.mkdir(parents=True, exist_ok=False)
    for kind in KINDS:
        fault_dir = out / kind
        (fault_dir / "tank").mkdir(parents=True)
        controller, questions = candidate(kind, clean_controller, clean_questions)
        changed = {
            "tank/controller.json": (
                json.dumps(controller, indent=2) + "\n"
                if controller != clean_controller
                else clean_source["tank/controller.json"]
            ),
            "tank/questions.json": (
                json.dumps(questions, indent=2) + "\n"
                if questions != clean_questions
                else clean_source["tank/questions.json"]
            ),
        }
        altered = [path for path in changed if clean_source[path] != changed[path]]
        if len(altered) != 1:
            raise AssertionError(f"Fault {kind} changed {altered}")
        for path, content in changed.items():
            (fault_dir / path).write_text(content)
        index = [
            {"path": "tank/controller.json", "kind": "settings"},
            {"path": "tank/questions.json", "kind": "instructions"},
        ]
        (fault_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n")
        record = {
            "layer": "harness",
            "family": kind,
            "instance": f"{kind}-v1",
            "changed_paths": altered,
            "clean_sha256": {path: checksum(clean_source[path]) for path in clean_source},
            "fault_sha256": {path: checksum(changed[path]) for path in changed},
            "validation_status": "pending",
        }
        (fault_dir / "fault.json").write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)
    (out / "catalog.json").write_text(json.dumps(records, indent=2) + "\n")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"candidates": len(build(args.out)), "out": str(args.out)}))


if __name__ == "__main__":
    main()
