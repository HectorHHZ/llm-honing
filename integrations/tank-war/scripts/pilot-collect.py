"""Collect fixed Tank War seed-group rounds while preserving every attempt."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect(args: argparse.Namespace) -> dict[str, Any]:
    plan = _json(args.seed_plan)
    seeds = plan["groups"][args.seed_group][args.start_index : args.start_index + args.seed_count]
    if len(seeds) != args.seed_count or len(set(seeds)) != len(seeds):
        raise ValueError(f"{args.seed_group} seed slice is missing or duplicated")
    with urllib.request.urlopen(args.url.rstrip("/") + "/v1/health", timeout=30) as response:
        health = json.load(response)
    model = health["metadata"]
    if not health["ready"] or model["weightsSha256"] != args.expected_weights_sha256:
        raise ValueError("model service does not have the expected checkpoint")
    config = {
        "arm": args.arm,
        "seed_plan": str(args.seed_plan),
        "seed_plan_version": plan["version"],
        "seeds": seeds,
        "seats": ["blue", "red"],
        "url": args.url,
        "model": model,
        "harness_dir": str(args.harness_dir) if args.harness_dir else None,
        "controller_fault_file": str(args.controller_fault_file) if args.controller_fault_file else None,
        "expected_code_commit": args.expected_code_commit,
        "node": str(args.node),
        "timeout_seconds": args.timeout_seconds,
    }
    if args.seed_group != "pilot":
        config["seed_group"] = args.seed_group
    args.out.mkdir(parents=True, exist_ok=True)
    config_path = args.out / "config.json"
    if config_path.exists() and _json(config_path) != config:
        raise ValueError("pilot output already belongs to a different configuration")
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    attempts_path = args.out / "attempts.jsonl"
    script = Path(__file__).with_name("match.mjs")
    successful = failed = 0
    for seed in seeds:
        for seat in ("blue", "red"):
            run = args.out / f"s{seed}-{seat}"
            if run.exists():
                if (run / "result.json").exists() and not (run / "error.json").exists():
                    successful += 1
                    continue
                raise ValueError(f"preserve incomplete prior attempt: {run}")
            command = [
                os.fspath(args.node),
                os.fspath(script),
                f"--seed={seed}",
                f"--seat={seat}",
                f"--decision-server={args.url}",
                f"--out={run}",
                "--seconds=30",
                "--interval-ms=200",
                "--opponent-interval-ms=300",
            ]
            if args.harness_dir:
                command.append(f"--harness-dir={args.harness_dir}")
            if args.controller_fault_file:
                command.append(f"--controller-fault-file={args.controller_fault_file}")
            started = time.monotonic()
            try:
                result = subprocess.run(
                    command, capture_output=True, text=True, timeout=args.timeout_seconds, check=False
                )
                attempt = {
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            except (OSError, subprocess.TimeoutExpired) as error:
                attempt = {"error": repr(error)}
            attempt.update(
                {
                    "arm": args.arm,
                    "seed": seed,
                    "seat": seat,
                    "run_dir": str(run),
                    "command": command,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                }
            )
            with attempts_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(attempt) + "\n")
            if attempt.get("returncode") != 0:
                failed += 1
                continue
            manifest = _json(run / "manifest.json")
            if (
                manifest["code_commit"] != args.expected_code_commit
                or manifest["model"]["weights_sha256"] != args.expected_weights_sha256
                or manifest["seed"] != seed
                or manifest["seat"] != seat
            ):
                raise ValueError(f"frozen identity differs in {run}")
            successful += 1
    summary = {"arm": args.arm, "attempted": len(seeds) * 2, "successful": successful, "failed": failed}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed-plan", type=Path, required=True)
    parser.add_argument("--seed-group", choices=("pilot", "development", "final_test"), default="pilot")
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--expected-weights-sha256", required=True)
    parser.add_argument("--expected-code-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--node", type=Path, default=Path("node"))
    parser.add_argument("--harness-dir", type=Path)
    parser.add_argument("--controller-fault-file", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    if args.seed_count < 1 or args.start_index < 0 or args.timeout_seconds < 1:
        parser.error("seed count, start index, and timeout must be positive")
    if args.harness_dir and args.controller_fault_file:
        parser.error("change only one fault layer")
    print(json.dumps(collect(args)))


if __name__ == "__main__":
    main()
