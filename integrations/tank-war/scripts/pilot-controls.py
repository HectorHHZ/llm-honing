"""Replay and index no-injection Tank War pilot controls for private review."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import subprocess
from pathlib import Path
from typing import Any


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-plan", type=Path, required=True)
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--node", type=Path, default=Path("node"))
    args = parser.parse_args()
    plan = _json(args.seed_plan)
    seeds = plan["groups"]["pilot"][args.start_index : args.start_index + args.seed_count]
    if len(seeds) != args.seed_count or args.seed_count < 1:
        parser.error("pilot seed slice is incomplete")
    args.out.mkdir(parents=True, exist_ok=False)
    signature = runpy.run_path(str(Path(__file__).with_name("reproducibility-gate.py")))["signature"]
    attempts = args.out / "attempts.jsonl"
    selected = args.out / "controls.jsonl"
    count = 0
    for seed in seeds:
        for seat in ("blue", "red"):
            source = args.clean_root / f"s{seed}-{seat}"
            manifest = _json(source / "manifest.json")
            result = _json(source / "result.json")
            if manifest["seed"] != seed or manifest["seat"] != seat:
                raise ValueError(f"control identity differs: {source}")
            sample_id = hashlib.sha256(f"{manifest['code_commit']}:no-injection:{seed}:{seat}".encode()).hexdigest()[
                :16
            ]
            directory = args.out / sample_id
            directory.mkdir()
            replay = directory / "replayed"
            command = [
                os.fspath(args.node),
                os.fspath(Path(__file__).with_name("match.mjs")),
                f"--seed={seed}",
                f"--seat={seat}",
                f"--replay-dir={source}",
                f"--out={replay}",
                f"--seconds={manifest['seconds']}",
                f"--interval-ms={manifest['interval_ms']}",
                f"--opponent-interval-ms={manifest['opponent_interval_ms']}",
            ]
            try:
                completed = subprocess.run(
                    command, cwd=args.repo, capture_output=True, text=True, timeout=300, check=False
                )
                attempt = {
                    "command": command,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }
            except (OSError, subprocess.TimeoutExpired) as error:
                attempt = {"command": command, "error": repr(error)}
            with attempts.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(attempt) + "\n")
            if attempt.get("returncode") != 0:
                continue
            if signature(source) != signature(replay) or any(
                (source / name).read_bytes() != (replay / name).read_bytes()
                for name in ("execution.jsonl", "events.jsonl")
            ):
                raise ValueError(f"control replay differs: {source}")
            labels = directory / "planner-reference-all-states.jsonl"
            command = [
                os.fspath(args.node),
                os.fspath(Path(__file__).with_name("label-recorded-states.mjs")),
                f"--run={source}",
                f"--out={labels}",
            ]
            completed = subprocess.run(
                command, cwd=args.repo, capture_output=True, text=True, timeout=300, check=False
            )
            with attempts.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "command": command,
                            "returncode": completed.returncode,
                            "stdout": completed.stdout,
                            "stderr": completed.stderr,
                        }
                    )
                    + "\n"
                )
            if completed.returncode:
                continue
            index = result["decisions"][seat] // 2
            reference = next((row for row in _rows(labels) if row["decision_index"] == index), None)
            if reference is None:
                raise ValueError(f"missing control reference at decision {index}")
            (directory / "reference-action.json").write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")
            packet = {
                "sample_id": sample_id,
                "split": "pilot",
                "condition": "no_injection",
                "seed": seed,
                "seat": seat,
                "outcome": result["outcome"][seat],
                "decision_index": index,
                "window": [max(0, index - 5), min(index + 5, result["decisions"][seat] - 1)],
                "source_run": str(source),
                "replay": str(replay),
                "reference_action": str(directory / "reference-action.json"),
            }
            (directory / "private-control-packet.json").write_text(
                json.dumps(packet, indent=2) + "\n", encoding="utf-8"
            )
            with selected.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(packet) + "\n")
            count += 1
    summary = {"expected_controls": len(seeds) * 2, "verified_controls": count}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
