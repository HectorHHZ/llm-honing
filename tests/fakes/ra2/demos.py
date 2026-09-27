"""Labelled RA2 training matches in #49's directory format, small enough for unit tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROTOCOL = {"train": {"maps": ["mp03t4.map", "mp06t2.map"], "seeds": [100, 101, 102]}}


def write_match(
    root: Path, seed: int, *, rows: int = 3, kind: str = "teacher-demonstration", map_name: str = "mp03t4.map"
) -> Path:
    directory = root / f"demo-s{seed}"
    (directory / "source").mkdir(parents=True)
    teacher = f"// teacher {seed}\n".encode()
    (directory / "source" / "teacher.mjs").write_bytes(teacher)
    manifest = {
        "split": "train",
        "seed": seed,
        "seat": "red",
        "map": map_name,
        "kind": kind,
        "sourceSha256": {"teacher.mjs": hashlib.sha256(teacher).hexdigest()},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "result.json").write_text(json.dumps({"notModelEvaluation": True, "kind": kind}))
    with (directory / "samples.jsonl").open("w") as handle:
        for index in range(rows):
            handle.write(
                json.dumps(
                    {
                        "split": "train",
                        "seed": seed,
                        "seat": "red",
                        "map": map_name,
                        "stage": 1,
                        "parts": {"essential": [f"Tick {3 * (index + 1)}. Seed {seed}."], "logs": []},
                        "questions": {
                            "g1": {
                                "type": "choice",
                                "instructions": "Choose the next intent.",
                                "criteria": {
                                    "A": "Take no action for this group this turn.",
                                    "B": "Build a power plant",
                                },
                            }
                        },
                        "answers": {"g1": {"choice": "B" if index % 2 else "A"}},
                    }
                )
                + "\n"
            )
    return directory
