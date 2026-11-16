"""Full deterministic replay of each frozen #183 blind case before diagnosis."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import time
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(
    case: dict, out: Path, repo: Path, old_repo: Path, harness_repo: Path, old_harness: Path, node: Path
) -> dict:
    source = Path(case["source_run"])
    target = out / "replays" / case["episode_id"]
    target.mkdir(parents=True, exist_ok=True)
    if (target / "attempt.json").exists():
        prior = read(target / "attempt.json")
        if not prior.get("valid"):
            raise ValueError(f"prior invalid replay must be investigated: {case['episode_id']}")
        return prior
    if (target / "run").exists():
        raise ValueError(f"incomplete prior replay must be investigated: {case['episode_id']}")
    manifest = read(source / "manifest.json")
    is_old = "formal-faults-final-v2" in source.parts
    run_repo = (
        old_repo
        if is_old
        else harness_repo if any(part.startswith("harness-selected-fire-") for part in source.parts) else repo
    )
    command = [
        str(node),
        str(run_repo / "integrations/tank-war/scripts/match.mjs"),
        f"--seed={case['seed']}",
        f"--seat={case['seat']}",
        f"--replay-dir={source}",
        f"--out={target / 'run'}",
        "--seconds=30",
        "--interval-ms=200",
        "--opponent-interval-ms=300",
    ]
    if is_old:
        command.append(f"--harness-dir={old_harness}")
    elif manifest.get("critical_fault_sha256"):
        configs = list(source.parents[2].glob("configs/*.json"))
        matches = [path for path in configs if sha(path) == manifest["critical_fault_sha256"]]
        if not matches:
            raise ValueError(f"critical fault config not found: {source}")
        command.append(f"--critical-fault-file={sorted(matches)[0]}")
    started = time.monotonic()
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=300, check=False)
        attempt = {
            "episode_id": case["episode_id"],
            "returncode": result.returncode,
            "stderr": result.stderr[-2000:],
            "stdout": result.stdout[-500:],
            "seconds": round(time.monotonic() - started, 2),
            "command": command,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        attempt = {"episode_id": case["episode_id"], "error": repr(error), "command": command}
    if attempt.get("returncode") == 0:
        original_states = [row["state_hash"] for row in rows(source / "states.jsonl")]
        replay_states = [row["state_hash"] for row in rows(target / "run" / "states.jsonl")]
        attempt["result_equal"] = read(source / "result.json") == read(target / "run" / "result.json")
        attempt["state_hashes_equal"] = original_states == replay_states
        attempt["execution_equal"] = rows(source / "execution.jsonl") == rows(target / "run" / "execution.jsonl")
    attempt["valid"] = bool(
        attempt.get("returncode") == 0
        and attempt.get("result_equal")
        and attempt.get("state_hashes_equal")
        and attempt.get("execution_equal")
    )
    (target / "attempt.json").write_text(json.dumps(attempt, indent=2) + "\n")
    print(
        json.dumps({"episode_id": case["episode_id"], "valid": attempt["valid"], "seconds": attempt.get("seconds")}),
        flush=True,
    )
    return attempt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--old-repo", type=Path, required=True)
    parser.add_argument("--harness-repo", type=Path, required=True)
    parser.add_argument("--old-harness", type=Path, required=True)
    parser.add_argument("--node", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    freeze = read(args.out / "sample-freeze.json")
    if freeze["status"] != "frozen_before_model_answers":
        raise ValueError("sample is not frozen")
    (args.out / "replays").mkdir(exist_ok=True)
    if (args.out / "replays" / "summary.json").exists():
        raise FileExistsError("replay gate already completed")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        attempts = list(
            pool.map(
                lambda case: replay(
                    case, args.out, args.repo, args.old_repo, args.harness_repo, args.old_harness, args.node
                ),
                freeze["cases"],
            )
        )
    summary = {
        "cases": len(attempts),
        "valid": sum(item["valid"] for item in attempts),
        "invalid": [item["episode_id"] for item in attempts if not item["valid"]],
        "seconds_sum": sum(item.get("seconds", 0) for item in attempts),
    }
    (args.out / "replays" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
