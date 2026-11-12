"""Recollect one frozen seed's two seats after a replay provenance failure."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import subprocess
from pathlib import Path

signature = runpy.run_path(str(Path(__file__).with_name("reproducibility-gate.py")))["signature"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-plan", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--node", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.seed_plan.read_text(encoding="utf-8"))
    if args.seed not in plan["groups"]["identity"]:
        parser.error("seed is not in the frozen identity plan")
    args.out.mkdir(parents=True, exist_ok=False)
    log = args.out / "gate.jsonl"
    script = Path(__file__).with_name("match.mjs")
    for seat in ("blue", "red"):
        runs = []
        for repetition in (1, 2):
            run = args.out / f"seed-{args.seed}-{seat}-repeat-{repetition}"
            command = [
                os.fspath(args.node),
                os.fspath(script),
                f"--seed={args.seed}",
                f"--seat={seat}",
                f"--decision-server={args.url}",
                f"--out={run}",
                "--seconds=30",
                "--interval-ms=200",
                "--opponent-interval-ms=300",
            ]
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
                attempt = {
                    "seed": args.seed,
                    "seat": seat,
                    "repetition": repetition,
                    "returncode": result.returncode,
                    "stderr": result.stderr[-4000:],
                }
            except (OSError, subprocess.TimeoutExpired) as error:
                attempt = {"seed": args.seed, "seat": seat, "repetition": repetition, "error": repr(error)}
            with (args.out / "attempts.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(attempt) + "\n")
            if attempt.get("returncode") != 0:
                raise RuntimeError(f"repair run failed: {attempt}")
            runs.append(run)
        first, second = (signature(run) for run in runs)
        row = {
            "seed": args.seed,
            "seat": seat,
            "passed": first == second,
            "decisions": len(first["state_hashes"]),
            "run_dirs": [str(run) for run in runs],
        }
        with log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        if not row["passed"]:
            raise RuntimeError(f"repair replay differs for {seat}")
    print(json.dumps({"seed": args.seed, "seat_pairs": 2, "passed": True}))


if __name__ == "__main__":
    main()
