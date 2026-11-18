"""Freeze unseen Tank War whole-game cases across newly validated fault instances."""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

LAYERS = ("harness", "weights", "controller")
EXPOSED_INSTANCES = {
    "point_blank_pursuit-v1",
    "reverse_strength_2p5-v1",
    "swap_selected_turn-v1",
}
SOURCE_FILES = (
    "manifest.json",
    "result.json",
    "states.jsonl",
    "decisions.jsonl",
    "execution.jsonl",
    "events.jsonl",
    "model_calls.jsonl",
)
FORBIDDEN = ("inject", "fault", "clean", "gold", "manifest", "source_run", "seed")
VERSION = "tank183-v4-multifamily-blind-v1"


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(item: dict[str, Any]) -> str:
    identity = ":".join(str(item[key]) for key in ("layer", "fault_instance", "seed", "seat"))
    return hashlib.sha256(f"{VERSION}:{identity}".encode()).hexdigest()


def select_faults(labels: list[dict[str, Any]], eligible_seeds: set[int], per_layer: int = 10) -> list[dict[str, Any]]:
    chosen = []
    for layer in LAYERS:
        by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in labels:
            if (
                item["layer"] == layer
                and item["fault_instance"] not in EXPOSED_INSTANCES
                and item["seed"] in eligible_seeds
                and item["fault_loss"]
                and item["cause_label"] == layer
                and item["verified_local_effect"]
            ):
                by_instance[item["fault_instance"]].append(item)
        for group in by_instance.values():
            group.sort(key=rank)
        used_within_instance: set[tuple[str, int]] = set()
        instance_names = sorted(by_instance)
        while sum(item["layer"] == layer for item in chosen) < per_layer:
            advanced = False
            for instance in instance_names:
                while by_instance[instance]:
                    item = by_instance[instance].pop(0)
                    key = instance, item["seed"]
                    if key in used_within_instance:
                        continue
                    used_within_instance.add(key)
                    chosen.append(item)
                    advanced = True
                    break
                if sum(item["layer"] == layer for item in chosen) == per_layer:
                    break
            if not advanced:
                break
    return chosen


def select_controls(controls: list[dict[str, Any]], eligible_seeds: set[int], count: int = 10) -> list[dict[str, Any]]:
    candidates = [
        item
        for item in controls
        if item["seed"] in eligible_seeds and item["outcome"] == -1 and item["cause_label"] is None
    ]
    candidates.sort(
        key=lambda item: hashlib.sha256(f"{VERSION}:control:{item['seed']}:{item['seat']}".encode()).hexdigest()
    )
    chosen = []
    used = set()
    for item in candidates:
        if item["seed"] in used:
            continue
        chosen.append(item)
        used.add(item["seed"])
        if len(chosen) == count:
            break
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--seed-plan", type=Path, required=True)
    parser.add_argument("--exposure-register", type=Path, required=True)
    parser.add_argument("--pre-registration", type=Path, required=True)
    parser.add_argument("--bare-prompt", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    registration = read(args.pre_registration)
    exposure = read(args.exposure_register)
    if registration["sampling"]["excluded_seed_positions_source_sha256"] != sha(args.exposure_register):
        raise ValueError("seed-exposure register differs from preregistration")
    if registration["prompt_sha256"]["bare"] != sha(args.bare_prompt) or registration["schema_sha256"] != sha(
        args.schema
    ):
        raise ValueError("prompt or schema differs from preregistration")
    plan = read(args.seed_plan)
    if exposure["final_first_80_seed_plan_sha256"] != sha(args.seed_plan):
        raise ValueError("seed plan differs from exposure register")
    eligible_seeds = {plan["groups"]["final_test"][index] for index in exposure["unexposed_final_positions"]}
    if len(eligible_seeds) != len(exposure["unexposed_final_positions"]):
        raise ValueError("eligible seed positions duplicate a reset")
    labels = rows(args.dataset / "labels.jsonl")
    controls = rows(args.controls)
    selected_faults = select_faults(labels, eligible_seeds, registration["sampling"]["target_faults_per_layer"])
    selected_controls = select_controls(
        controls, eligible_seeds, registration["sampling"]["target_no_injection_controls"]
    )
    public_case = runpy.run_path(str(Path(__file__).with_name("prepare-critical-blind.py")))["public_case"]
    inputs = args.out / "inputs"
    inputs.mkdir(parents=True)
    cases = []
    for item in [*selected_faults, *selected_controls]:
        layer = item.get("layer", "no_injection")
        source = Path(item["fault_run"] if layer in LAYERS else item["source_run"])
        instance = item.get("fault_instance", "control")
        episode_id = hashlib.sha256(f"{VERSION}:{instance}:{item['seed']}:{item['seat']}".encode()).hexdigest()[:20]
        if layer in LAYERS and any(
            sha(source / name) != item["source_hashes"]["fault"][name] for name in SOURCE_FILES
        ):
            raise ValueError(f"fault source log changed: {episode_id}")
        public = public_case({"source_run": str(source), "episode_id": episode_id})
        serialized = json.dumps(public, separators=(",", ":"))
        if (
            len(serialized) > 500_000
            or any(term in serialized.lower() for term in FORBIDDEN)
            or str(item["seed"]) in serialized
            or str(source) in serialized
        ):
            raise ValueError(f"blind input leaks construction data: {episode_id}")
        input_path = inputs / f"{episode_id}.json"
        input_path.write_text(serialized)
        cases.append(
            {
                "episode_id": episode_id,
                "seed": item["seed"],
                "seat": item["seat"],
                "fault_instance": instance if layer in LAYERS else None,
                "technical_label": layer,
                "technical_reference_index": (
                    item["technical_reference"]["decision_index"] if layer in LAYERS else None
                ),
                "source_run": str(source),
                "source_hashes": {name: sha(source / name) for name in SOURCE_FILES},
                "input_sha256": sha(input_path),
                "decisions": len(public["timeline"]),
            }
        )
    if len({item["episode_id"] for item in cases}) != len(cases):
        raise ValueError("opaque episode ID collision")
    (args.out / "protocol.json").write_bytes(args.pre_registration.read_bytes())
    (args.out / "prompt.txt").write_bytes(args.bare_prompt.read_bytes())
    (args.out / "schema.json").write_bytes(args.schema.read_bytes())
    summary = {
        "status": "frozen_before_model_answers",
        "version": VERSION,
        "pre_registration_sha256": sha(args.pre_registration),
        "dataset_summary_sha256": sha(args.dataset / "summary.json"),
        "dataset_labels_sha256": sha(args.dataset / "labels.jsonl"),
        "controls_sha256": sha(args.controls),
        "seed_exposure_sha256": sha(args.exposure_register),
        "protocol_sha256": sha(args.out / "protocol.json"),
        "prompt_sha256": sha(args.out / "prompt.txt"),
        "schema_sha256": sha(args.out / "schema.json"),
        "preparation_script_sha256": sha(Path(__file__)),
        "public_builder_sha256": sha(Path(__file__).with_name("prepare-critical-blind.py")),
        "counts": dict(Counter(item["technical_label"] for item in cases)),
        "instance_counts": dict(Counter(item["fault_instance"] for item in cases if item["fault_instance"])),
        "independent_seed_blocks": len({item["seed"] for item in cases}),
        "cases": cases,
    }
    (args.out / "sample-freeze.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                "cases": len(cases),
                "counts": summary["counts"],
                "instance_counts": summary["instance_counts"],
                "independent_seed_blocks": summary["independent_seed_blocks"],
                "freeze_sha256": sha(args.out / "sample-freeze.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
