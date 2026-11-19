"""Dataset-only checks for the optional Space Invaders SFT entry point."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
def test_training_dry_run_needs_no_checkpoint(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "integrations/space-invaders/train_rollout_pair_sft.py"
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"

    def row(seed: int, label: str) -> dict[str, object]:
        return {
            "seed": seed,
            "label": label,
            "delta": 20 if label == "A" else -20,
            "parts": {"essential": [f"seed {seed}, {label}"], "details": [], "memory": [], "logs": []},
            "question": {"control": {"instructions": "Choose a lane", "criteria": {"A": "left", "B": "right"}}},
        }

    train.write_text("".join(json.dumps(row(1, label)) + "\n" for label in ("A", "B")))
    validation.write_text("".join(json.dumps(row(2, label)) + "\n" for label in ("A", "B")))
    work = tmp_path / "work"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--train-records",
            str(train),
            "--validation-records",
            str(validation),
            "--work-dir",
            str(work),
            "--train-per-label",
            "1",
            "--validation-per-label",
            "1",
            "--validation-seeds",
            "auto",
            "--reserved-game-seeds",
            "3",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    audit = json.loads((work / "dataset-audit.json").read_text())
    assert audit["trainSeeds"] == [1] and audit["validationSeeds"] == [2]
    assert audit["crossSplitInputOverlap"] == 0
    assert audit["baseCheckpoint"] is None
    assert '"trainSeeds"' in result.stdout
    assert not (work / "train-result.json").exists()
