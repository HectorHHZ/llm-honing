"""Dataset-only checks for the optional Space Invaders SFT entry point."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_gpu_runner_rejects_wrong_base_before_touching_hold(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "integrations/space-invaders/run_counterfactual_sft_gpu4.sh"
    pilot = tmp_path / "pilot"
    inputs = pilot / "runs/counterfactual-sft-testtag"
    inputs.mkdir(parents=True)
    for name in ("train.jsonl", "validation.jsonl", "train_rollout_pair_sft.py"):
        (inputs / name).write_text("{}\n")
    base = tmp_path / "base"
    base.mkdir()
    (base / "best.safetensors").write_bytes(b"different weights")
    environment = {
        **os.environ,
        "SFT_PILOT_DIR": str(pilot),
        "SFT_CODE_ROOT": str(tmp_path),
        "SFT_RUNTIME": sys.executable,
        "SFT_SOURCE_DIR": str(tmp_path),
        "SFT_CACHE": str(tmp_path),
        "SFT_HOLD_SCRIPT": str(tmp_path / "hold.py"),
        "SFT_HOLD_UNIT": "never-touch-this-hold",
        "SFT_GPU_UUID": "GPU-test",
        "SFT_PINS": str(tmp_path / "pins.json"),
        "SFT_BASE_CHECKPOINT": str(base),
        "SFT_BASE_SHA256": "0" * 64,
        "SFT_WORK_ROOT": str(tmp_path),
        "SFT_TAG": "testtag",
        "SFT_RESERVED_GAME_SEEDS": "3",
    }
    result = subprocess.run(["bash", str(script)], env=environment, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Base checkpoint weights differ from the frozen incumbent" in result.stderr


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
