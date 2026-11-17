"""Freeze an unseen, balanced Tank War attribution set from screened games."""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path
from typing import Any

LAYERS = ("harness", "weights", "controller")
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


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(layer: str, seed: int, seat: str) -> str:
    return hashlib.sha256(f"tank183-sustained-blind-v1:{layer}:{seed}:{seat}".encode()).hexdigest()


def select(labels: list[dict[str, Any]], controls: list[dict[str, Any]], exposed: set[int]) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    used = set(exposed)
    for layer in (*LAYERS, "no_injection"):
        candidates = (
            [
                item
                for item in labels
                if item["cause_label"] == layer
                and item["fault_loss"]
                and item["verified_local_effect"]
                and item["technical_reference"]["verified"]
            ]
            if layer != "no_injection"
            else [item for item in controls if item["outcome"] == -1 and item["cause_label"] is None]
        )
        candidates.sort(key=lambda item: rank(layer, item["seed"], item["seat"]))
        selected = 0
        for item in candidates:
            if item["seed"] in used:
                continue
            chosen.append({**item, "test_layer": layer})
            used.add(item["seed"])
            selected += 1
            if selected == 10:
                break
        if selected != 10:
            raise ValueError(f"only {selected}/10 independent {layer} games remain after exclusions")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prior-rapid-freeze", type=Path, required=True)
    parser.add_argument("--prior-blind-freeze", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    if not out.is_dir() or (out / "inputs").exists() or (out / "sample-freeze.json").exists():
        raise ValueError("output must contain only the preregistered protocol, prompt, and schema")
    for name in ("protocol.json", "prompt.txt", "schema.json"):
        if not (out / name).is_file():
            raise FileNotFoundError(out / name)
    labels_path = args.dataset / "labels.jsonl"
    controls_path = args.dataset / "controls.jsonl"
    prior_paths = (args.prior_rapid_freeze, args.prior_blind_freeze)
    exposed = {case["seed"] for path in prior_paths for case in read(path)["cases"] if "seed" in case}
    chosen = select(rows(labels_path), rows(controls_path), exposed)
    public_case = runpy.run_path(str(Path(__file__).with_name("prepare-critical-blind.py")))["public_case"]
    (out / "inputs").mkdir()
    frozen = []
    for item in chosen:
        layer = item["test_layer"]
        source = Path(item["fault_run"] if layer != "no_injection" else item["source_run"])
        episode_id = hashlib.sha256(f"tank183-v2:{layer}:{item['seed']}:{item['seat']}".encode()).hexdigest()[:20]
        public = public_case({"source_run": str(source), "episode_id": episode_id})
        serialized = json.dumps(public, separators=(",", ":"))
        if (
            len(serialized) > 500_000
            or any(term in serialized.lower() for term in FORBIDDEN)
            or str(item["seed"]) in serialized
            or str(source) in serialized
        ):
            raise ValueError(f"blind input leaks construction or is oversized: {episode_id}")
        input_path = out / "inputs" / f"{episode_id}.json"
        input_path.write_text(serialized)
        frozen.append(
            {
                "episode_id": episode_id,
                "seed": item["seed"],
                "seat": item["seat"],
                "source_run": str(source),
                "technical_label": layer,
                "technical_reference_index": (
                    item["technical_reference"]["decision_index"] if layer != "no_injection" else None
                ),
                "source_hashes": {name: digest(source / name) for name in SOURCE_FILES},
                "input_sha256": digest(input_path),
                "decisions": len(public["timeline"]),
            }
        )
    freeze = {
        "status": "frozen_before_model_answers",
        "version": "tank183-sustained-blind-v1",
        "selection": (
            "ten per class and ten no-injection losses; SHA-256 rank; distinct seed; "
            "prior rapid/blind seeds excluded"
        ),
        "dataset_hashes": {"labels": digest(labels_path), "controls": digest(controls_path)},
        "excluded_freezes": {str(path): digest(path) for path in prior_paths},
        "protocol_sha256": digest(out / "protocol.json"),
        "prompt_sha256": digest(out / "prompt.txt"),
        "schema_sha256": digest(out / "schema.json"),
        "preparation_script_sha256": digest(Path(__file__)),
        "public_builder_sha256": digest(Path(__file__).with_name("prepare-critical-blind.py")),
        "cases": frozen,
    }
    (out / "sample-freeze.json").write_text(json.dumps(freeze, indent=2) + "\n")
    print(
        json.dumps(
            {
                "cases": len(frozen),
                "by_layer": {
                    layer: sum(x["technical_label"] == layer for x in frozen) for layer in (*LAYERS, "no_injection")
                },
                "freeze_sha256": digest(out / "sample-freeze.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
