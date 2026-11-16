"""Convert sealed episode-only Codex answers to the existing aggregate scorer input."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    freeze = read(args.run / "freeze.json")
    if freeze["status"] != "frozen_before_episode_only_answers":
        raise ValueError("sensitivity sample not frozen")
    if any(read(args.run / mode / "summary.json")["status"] != "complete" for mode in ("read", "probe")):
        raise ValueError("diagnosis mode incomplete")
    records = []
    for case in freeze["cases"]:
        record = {"episode_id": case["episode_id"]}
        for mode in ("read", "probe"):
            target = args.run / mode / case["episode_id"]
            attempt, answer = read(target / "attempt.json"), read(target / "answer.json")
            if (
                attempt.get("exit_code") != 0
                or attempt.get("tool_call")
                or attempt.get("output_validation_error")
                or attempt.get("usage") is None
                or attempt.get("input_sha256") != case["input_sha256"]
                or attempt.get("prompt_sha256") != freeze["prompt_sha256"]
                or attempt.get("schema_sha256") != freeze["schema_sha256"]
            ):
                raise ValueError(f"invalid answer record: {mode}/{case['episode_id']}")
            record[mode] = {"valid": True, "answer": answer}
        records.append(record)
    result = {
        "source_run_freeze_sha256": sha(args.run / "freeze.json"),
        "records": records,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"records": len(records), "score_input_sha256": sha(args.out)}, indent=2))


if __name__ == "__main__":
    main()
