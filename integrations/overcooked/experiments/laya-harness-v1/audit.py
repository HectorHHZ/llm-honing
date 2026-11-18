"""Portable archive checks and score derivation; standard library only, no model calls."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    episodes = json.loads((root / "episodes.json").read_text())
    file_count, calls = 0, 0
    for row in episodes:
        for relative, expected in row["files"].items():
            path = root / relative
            assert path.resolve().is_relative_to(root), "Archive path escapes the experiment directory"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative
            file_count += 1
        directory = root / "data/episodes" / row["key"]
        frames = [
            json.loads(line)["frame"]
            for line in gzip.decompress((directory / "steps.jsonl.gz").read_bytes()).splitlines()
        ]
        decisions = [
            json.loads(line) for line in gzip.decompress((directory / "decisions.jsonl.gz").read_bytes()).splitlines()
        ]
        result = json.loads((directory / "result.json").read_text())
        assert len(frames) == 401 and len(decisions) == 400
        assert frames[-1]["score"] == row["score"] == result["score"]
        assert frames[-1]["deliveries"] == row["deliveries"] == result["deliveries"]
        assert dict(Counter(d["chosen"] for d in decisions)) == row["actions"]
        assert result["complete"] and result["steps"] == 400
        for frame in frames:
            content = {key: value for key, value in frame.items() if key != "frame_hash"}
            encoded = json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            assert hashlib.sha256(encoded).hexdigest() == frame["frame_hash"]
        actual_calls = sum(d["response"] is not None for d in decisions)
        assert actual_calls == row["model_calls"]
        calls += actual_calls
    summary = json.loads((root / "summary.json").read_text())
    assert summary["episodes"] == len(episodes) and summary["real_model_decisions"] == calls
    print(
        json.dumps(
            {
                "verified": True,
                "episodes": len(episodes),
                "files": file_count,
                "game_model_calls": calls,
                "selected": summary["selection"]["selected"],
            }
        )
    )


if __name__ == "__main__":
    main()
