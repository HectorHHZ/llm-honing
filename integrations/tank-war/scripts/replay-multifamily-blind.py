"""Replay frozen Tank War cases with each fault instance's own local files."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import runpy
from pathlib import Path
from typing import Any

ONE = runpy.run_path(str(Path(__file__).with_name("replay-sustained-blind.py")))["one"]


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_paths(case: dict[str, Any], mapping: dict[str, Any]) -> tuple[Path, Path]:
    layer = case["technical_label"]
    if layer not in ("harness", "controller"):
        return Path("."), Path(".")
    instance = case["fault_instance"]
    item = mapping.get(instance)
    if not item or item["layer"] != layer:
        raise ValueError(f"missing {layer} replay files for {instance}")
    path = Path(item["path"])
    if layer == "harness":
        if (
            sha(path / "tank/controller.json") != item["controller_sha256"]
            or sha(path / "tank/questions.json") != item["questions_sha256"]
        ):
            raise ValueError(f"harness candidate changed for {instance}")
        return path, Path(".")
    if sha(path) != item["fault_sha256"]:
        raise ValueError(f"controller candidate changed for {instance}")
    return Path("."), path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--node", type=Path, required=True)
    parser.add_argument("--candidate-map", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    freeze = read(args.out / "sample-freeze.json")
    if freeze["status"] != "frozen_before_model_answers" or (args.out / "replays/summary.json").exists():
        raise ValueError("blind freeze changed or replay gate already finished")
    mapping = read(args.candidate_map)
    if not isinstance(mapping, dict):
        raise ValueError("candidate map must be a keyed object")
    (args.out / "replays").mkdir(exist_ok=True)

    def replay(case: dict[str, Any]) -> dict[str, Any]:
        harness, controller = candidate_paths(case, mapping)
        return ONE(case, args.out, args.repo, args.node, harness, controller)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        attempts = list(pool.map(replay, freeze["cases"]))
    summary = {
        "cases": len(attempts),
        "valid": sum(item["valid"] for item in attempts),
        "invalid": [item["episode_id"] for item in attempts if not item["valid"]],
        "seconds_sum": sum(item["elapsed_seconds"] for item in attempts),
        "sample_freeze_sha256": sha(args.out / "sample-freeze.json"),
        "candidate_map_sha256": sha(args.candidate_map),
    }
    (args.out / "replays/summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    if summary["invalid"]:
        raise ValueError("one or more full replay checks failed")


if __name__ == "__main__":
    main()
