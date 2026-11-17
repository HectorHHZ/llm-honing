"""Check full deterministic replays of frozen sustained-fault diagnosis games."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one(case: dict[str, Any], out: Path, repo: Path, node: Path, harness: Path, controller: Path) -> dict[str, Any]:
    source = Path(case["source_run"])
    target = out / "replays" / case["episode_id"]
    target.mkdir(parents=True, exist_ok=True)
    if (target / "attempt.json").exists():
        prior = read(target / "attempt.json")
        if not prior["valid"]:
            raise ValueError(f"invalid prior replay: {case['episode_id']}")
        return prior
    if (target / "run").exists():
        raise ValueError(f"incomplete prior replay: {case['episode_id']}")
    for name, expected in case["source_hashes"].items():
        if digest(source / name) != expected:
            raise ValueError(f"source recording changed: {case['episode_id']}/{name}")
    manifest = read(source / "manifest.json")
    for name, expected in manifest["sources"].items():
        if digest(repo / "integrations/tank-war" / name) != expected:
            raise ValueError(f"recorded game source differs: {case['episode_id']}/{name}")
    command = [
        str(node),
        str(repo / "integrations/tank-war/scripts/match.mjs"),
        f"--seed={case['seed']}",
        f"--seat={case['seat']}",
        f"--replay-dir={source}",
        f"--out={target / 'run'}",
        f"--seconds={manifest['seconds']}",
        f"--interval-ms={manifest['interval_ms']}",
        f"--opponent-interval-ms={manifest['opponent_interval_ms']}",
    ]
    if case["technical_label"] == "harness":
        command.append(f"--harness-dir={harness}")
    elif case["technical_label"] == "controller":
        command.append(f"--controller-fault-file={controller}")
    started = time.monotonic()
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=1800, check=False)
        attempt: dict[str, Any] = {
            "returncode": result.returncode,
            "stdout": result.stdout[-500:],
            "stderr": result.stderr[-2000:],
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        attempt = {"error": repr(error)}
    attempt.update(
        episode_id=case["episode_id"], command=command, elapsed_seconds=round(time.monotonic() - started, 2)
    )
    if attempt.get("returncode") == 0:
        replay = target / "run"
        attempt["result_equal"] = read(source / "result.json") == read(replay / "result.json")
        attempt["states_equal"] = [x["state_hash"] for x in rows(source / "states.jsonl")] == [
            x["state_hash"] for x in rows(replay / "states.jsonl")
        ]
        attempt["execution_equal"] = rows(source / "execution.jsonl") == rows(replay / "execution.jsonl")
        attempt["events_equal"] = rows(source / "events.jsonl") == rows(replay / "events.jsonl")
    attempt["valid"] = all(
        attempt.get(key) for key in ("result_equal", "states_equal", "execution_equal", "events_equal")
    )
    (target / "attempt.json").write_text(json.dumps(attempt, indent=2) + "\n")
    print(json.dumps({"episode_id": case["episode_id"], "valid": attempt["valid"]}), flush=True)
    return attempt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--node", type=Path, required=True)
    parser.add_argument("--harness-dir", type=Path, required=True)
    parser.add_argument("--controller-fault-file", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    freeze = read(args.out / "sample-freeze.json")
    if freeze["status"] != "frozen_before_model_answers" or (args.out / "replays" / "summary.json").exists():
        raise ValueError("freeze changed or replay gate already completed")
    (args.out / "replays").mkdir(exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        attempts = list(
            pool.map(
                lambda case: one(case, args.out, args.repo, args.node, args.harness_dir, args.controller_fault_file),
                freeze["cases"],
            )
        )
    summary = {
        "cases": len(attempts),
        "valid": sum(item["valid"] for item in attempts),
        "invalid": [item["episode_id"] for item in attempts if not item["valid"]],
        "seconds_sum": sum(item["elapsed_seconds"] for item in attempts),
    }
    (args.out / "replays" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
