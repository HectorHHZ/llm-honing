"""Fit NanoJev's decision heads to fixed-game public-state combat demonstrations."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from honing.backends.jev.nanojev import nanojev_snapshot
from honing.backends.jev.nanojev_trainer import NanoJevTrainer, NanoJevTrainerSettings, release_path
from honing.backends.rea_issac_agent.make_agent_demos import (
    HELDOUT_SEEDS,
    INTEGRATION,
    TRAIN_SEEDS,
    VALIDATION_SEEDS,
    verify_frozen,
)
from honing.core import DataContract, DecisionQuestion, DecisionSample, TrainingBatch, TrainJobSpec, UpdatePolicy

ROOT = Path(__file__).resolve().parents[3]
PINS = ROOT / "integrations/ra2/pins.json"


def decision_sample(row: dict[str, Any], harness_id: str) -> DecisionSample:
    request = row["request"]
    parts = tuple((name, tuple(lines)) for name, lines in request["parts"].items())
    questions = tuple(
        (
            name,
            DecisionQuestion("choice", question["instructions"], tuple(question["criteria"].items())),
        )
        for name, question in request["questions"].items()
    )
    return DecisionSample(
        parts=parts,
        questions=questions,
        chosen=tuple(row["chosen"].items()),
        group=row["group"],
        source_release="rea-issac-scripted-baseline-v1",
        source_harness=harness_id,
    )


def load_batch(dataset: Path) -> tuple[TrainingBatch, dict[str, Any]]:
    frozen = verify_frozen()
    data = dataset / "decisions.jsonl"
    summary = json.loads((dataset / "summary.json").read_text())
    checksum = hashlib.sha256(data.read_bytes()).hexdigest()
    if checksum != summary["decisions_sha256"] or frozen != summary["frozen"]:
        raise ValueError("demonstration data or frozen game files differ from the dataset manifest")
    if (
        tuple(summary["train_seeds"]) != TRAIN_SEEDS
        or tuple(summary["validation_seeds"]) != VALIDATION_SEEDS
        or tuple(summary["heldout_seeds"]) != HELDOUT_SEEDS
    ):
        raise ValueError("demonstration seed split changed")
    rows = [json.loads(line) for line in data.read_text().splitlines() if line]
    if len(rows) != summary["decisions"]:
        raise ValueError("demonstration count differs from the manifest")
    for row in rows:
        expected = "validation" if row["seed"] in VALIDATION_SEEDS else "train"
        if row["seed"] not in (*TRAIN_SEEDS, *VALIDATION_SEEDS) or row["split"] != expected:
            raise ValueError("demonstration seed leaked across splits")
        if not row["group"].startswith(f"rea-issac:{row['seed']}:"):
            raise ValueError("demonstration group differs from its seed")
    harness_id = hashlib.sha256(
        (INTEGRATION / "config/controller.json").read_bytes() + (INTEGRATION / "config/questions.json").read_bytes()
    ).hexdigest()
    batch = TrainingBatch(
        f"rea-issac-agent-{checksum[:16]}",
        DataContract("typed_decisions"),
        decisions=tuple(decision_sample(row, harness_id) for row in rows),
    )
    return batch, summary


def train(
    dataset: Path,
    output: Path,
    *,
    source_dir: Path,
    gpu: str = "2",
    base_checkpoint: Path | None = None,
    max_length: int = 4096,
    epochs: int = 1,
) -> dict[str, Any]:
    if max_length < 1 or epochs < 1:
        raise ValueError("max_length and epochs must be positive")
    if output.exists():
        raise FileExistsError(output)
    batch, summary = load_batch(dataset)
    output.mkdir(parents=True)
    (output / "dataset_manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    pins = json.loads(PINS.read_text())
    base = nanojev_snapshot(base_checkpoint or release_path(pins))
    trainer = NanoJevTrainer(
        NanoJevTrainerSettings(
            pins_path=PINS,
            source_dir=source_dir,
            work_dir=output / "training",
            gpu=gpu,
            batch_questions=1,
            max_length=max_length,
            min_validation=30,
            validation_groups=r"^rea-issac:2001:",
            deterministic=True,
        )
    )
    (output / "training").mkdir()
    job = TrainJobSpec(
        "sft",
        steps=epochs,
        contract="typed_decisions",
        update_policy=UpdatePolicy(kind="param_subset", trainable_patterns=(r"^(norm|scalar|set_)",)),
    )
    result = trainer.train(job, base, batch)
    checkpoint = Path(result.snapshot.checkpoint.location)
    report = {
        "purpose": "rea-issac-nanojev-agent-head-sft",
        "dataset_sha256": summary["decisions_sha256"],
        "training_seeds": TRAIN_SEEDS,
        "validation_seeds": VALIDATION_SEEDS,
        "heldout_seeds": HELDOUT_SEEDS,
        "base_checksum": base.checkpoint.checksum,
        "trained_checksum": result.snapshot.checkpoint.checksum,
        "checkpoint_manifest": str(checkpoint.parent / "checkpoint.json"),
        "metrics": dict(result.metrics),
        "update_policy": "decision heads only",
        "steps": epochs,
        "batch_questions": 1,
        "max_length": max_length,
        "base_checkpoint": str(base_checkpoint) if base_checkpoint else None,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--nanojev-source", type=Path, required=True)
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    print(
        json.dumps(
            train(
                args.dataset,
                args.out,
                source_dir=args.nanojev_source,
                gpu=args.gpu,
                base_checkpoint=args.base_checkpoint,
                max_length=args.max_length,
                epochs=args.epochs,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
