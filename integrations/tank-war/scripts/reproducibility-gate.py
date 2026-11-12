"""Run two independent clean matches per frozen seed and seat, comparing every decision."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def probabilities(reply: dict[str, Any]) -> dict[str, Any]:
    return {
        "answers": {
            question: {"choice": answer["choice"], "probabilities": dict(answer["probabilities"])}
            for question, answer in reply["answers"].items()
        },
        "branches": {key: probabilities(value) for key, value in reply.get("branches", {}).items()},
    }


def signature(run: Path) -> dict[str, Any]:
    calls = records(run / "model_calls.jsonl")
    states = records(run / "states.jsonl")
    if len(calls) != len(states):
        raise ValueError("Model-call and state counts differ")
    return {
        "state_hashes": [state["state_hash"] for state in states],
        "requests": [call["request"] for call in calls],
        "probabilities": [probabilities(call["reply"]) for call in calls],
        "result": json.loads((run / "result.json").read_text()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-plan", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--node", type=Path, default=Path("node"))
    args = parser.parse_args()
    seeds = json.loads(args.seed_plan.read_text())["groups"]["identity"]
    if len(seeds) != 64 or len(set(seeds)) != 64:
        parser.error("Identity plan must contain 64 unique seeds")
    args.out.mkdir(parents=True, exist_ok=False)
    log = args.out / "gate.jsonl"
    script = Path(__file__).with_name("match.mjs")
    for seed in seeds:
        for seat in ("blue", "red"):
            runs = []
            for repetition in (1, 2):
                run = args.out / f"seed-{seed}-{seat}-repeat-{repetition}"
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
                try:
                    result = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
                except (OSError, subprocess.TimeoutExpired) as error:
                    with log.open("a") as output:
                        output.write(
                            json.dumps(
                                {
                                    "seed": seed,
                                    "seat": seat,
                                    "passed": False,
                                    "repetition": repetition,
                                    "error": repr(error),
                                }
                            )
                            + "\n"
                        )
                    raise SystemExit(1) from error
                runs.append(run)
                if result.returncode:
                    row = {
                        "seed": seed,
                        "seat": seat,
                        "passed": False,
                        "repetition": repetition,
                        "returncode": result.returncode,
                        "stderr": result.stderr[-4000:],
                    }
                    with log.open("a") as output:
                        output.write(json.dumps(row) + "\n")
                    raise SystemExit(1)
            first, second = (signature(run) for run in runs)
            row = {
                "seed": seed,
                "seat": seat,
                "passed": first == second,
                "decisions": len(first["state_hashes"]),
                "run_dirs": [str(run) for run in runs],
            }
            with log.open("a") as output:
                output.write(json.dumps(row) + "\n")
            if not row["passed"]:
                raise SystemExit(1)
    print(json.dumps({"seed_blocks": len(seeds), "seat_pairs": len(seeds) * 2, "passed": True}))


if __name__ == "__main__":
    main()
