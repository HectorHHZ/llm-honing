"""Replay completed clean rounds from saved answers and verify every decision.

This can run while the live collectors continue. A second invocation skips
complete verified replays and refuses incomplete or failed prior attempts.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import subprocess
from pathlib import Path
from typing import Any

signature = runpy.run_path(str(Path(__file__).with_name("reproducibility-gate.py")))["signature"]


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _same_recording(source: Path, replay: Path) -> bool:
    if signature(source) != signature(replay):
        return False
    for name in ("execution.jsonl", "events.jsonl"):
        if (source / name).read_bytes() != (replay / name).read_bytes():
            return False
    return True


def verify(source: Path, replay: Path, node: Path, attempt_log: Path) -> dict[str, Any]:
    if replay.exists():
        if not (replay / "result.json").exists() or (replay / "error.json").exists():
            raise ValueError(f"incomplete prior replay: {replay}")
    else:
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        command = [
            os.fspath(node),
            os.fspath(Path(__file__).with_name("match.mjs")),
            f"--seed={manifest['seed']}",
            f"--seat={manifest['seat']}",
            f"--replay-dir={source}",
            f"--out={replay}",
            f"--seconds={manifest['seconds']}",
            f"--interval-ms={manifest['interval_ms']}",
            f"--opponent-interval-ms={manifest['opponent_interval_ms']}",
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
            attempt = {
                "source": str(source),
                "replay": str(replay),
                "returncode": result.returncode,
                "stderr": result.stderr,
            }
        except (OSError, subprocess.TimeoutExpired) as error:
            attempt = {"source": str(source), "replay": str(replay), "error": repr(error)}
        with attempt_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(attempt) + "\n")
        if attempt.get("returncode") != 0:
            raise RuntimeError(f"recorded replay failed: {source}")
    if not _same_recording(source, replay):
        raise ValueError(f"recorded replay diverges: {source}")
    return {
        "source": str(source),
        "replay": str(replay),
        "passed": True,
        "decisions": len(signature(source)["state_hashes"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collector", type=Path, action="append", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--node", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    journal = args.out / "verified.jsonl"
    prior = {row["source"]: row for row in _rows(journal)}
    new_count = 0
    for collector in args.collector:
        for row in _rows(collector / "gate.jsonl"):
            if row.get("passed") is not True:
                raise ValueError(f"failed independent replay pair in {collector}")
            for source_name in row["run_dirs"]:
                source = Path(source_name)
                manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
                if manifest["code_commit"] != args.code_commit:
                    continue
                replay = args.out / collector.name / source.name
                if source_name in prior:
                    if not _same_recording(source, replay):
                        raise ValueError(f"prior verified replay now differs: {source}")
                    continue
                replay.parent.mkdir(parents=True, exist_ok=True)
                verified = verify(source, replay, args.node, args.out / "attempts.jsonl")
                with journal.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(verified) + "\n")
                new_count += 1
    print(json.dumps({"new_replays": new_count, "total_verified": len(prior) + new_count}))


if __name__ == "__main__":
    main()
