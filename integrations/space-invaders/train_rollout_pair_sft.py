"""Train NanoJev on labeled A/B game decisions."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def model_input(row: dict[str, Any]) -> str:
    return json.dumps({"parts": row["parts"], "question": row["question"]}, sort_keys=True)


def examples(
    path: Path,
    per_label: int,
    seed: int,
    forbidden: set[str] | None = None,
    min_delta: float = 5.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    stats: dict[str, Any] = {"rows": 0, "ties": 0, "weak": 0, "duplicate": 0, "cross_split": 0}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as source:
        for line in source:
            row = json.loads(line)
            stats["rows"] += 1
            if row["label"] is None:
                stats["ties"] += 1
                continue
            if abs(row["delta"]) < min_delta:
                stats["weak"] += 1
                continue
            key = model_input(row)
            if key in (forbidden or set()):
                stats["cross_split"] += 1
                continue
            if key in seen:
                stats["duplicate"] += 1
                continue
            seen.add(key)
            groups[row["label"]].append(row)
    rng = random.Random(seed)
    selected = []
    for label in ("A", "B"):
        if len(groups[label]) < per_label:
            raise ValueError(f"Only {len(groups[label])} {label} examples; need {per_label}")
        selected.extend(rng.sample(groups[label], per_label))
    rng.shuffle(selected)
    stats["available"] = {label: len(groups[label]) for label in ("A", "B")}
    stats["selected"] = len(selected)
    return selected, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-records", type=Path, required=True)
    parser.add_argument("--validation-records", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--pins", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--base-checkpoint", type=Path, help="Start from an existing NanoJev checkpoint")
    parser.add_argument("--replay-records", type=Path, help="Older code-teacher examples mixed into training only")
    parser.add_argument("--replay-per-label", type=int, default=0)
    parser.add_argument("--replay-min-delta", type=float, default=0)
    parser.add_argument("--gpu", default="4")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-per-label", type=int, default=150)
    parser.add_argument("--validation-per-label", type=int, default=50)
    parser.add_argument("--min-delta", type=float, default=5.0)
    parser.add_argument("--validation-seeds", default="13,14,15,16,17,18")
    parser.add_argument("--reserved-game-seeds", default="55,56,57,58,59,60")
    parser.add_argument("--batch-name", default="spaceinvaders-rollout-pair-v1")
    parser.add_argument(
        "--teacher-description",
        default="paired 48-decision ALE rollouts; first A/B action then code-first",
    )
    parser.add_argument(
        "--teacher-value-description",
        default="discounted reward minus 100 per lost life",
    )
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (
        min(args.epochs, args.train_per_label, args.validation_per_label) <= 0
        or args.min_delta < 0
        or args.replay_per_label < 0
        or args.replay_min_delta < 0
    ):
        parser.error("Epoch and example counts must be positive and min-delta nonnegative")
    if bool(args.replay_records) != bool(args.replay_per_label):
        parser.error("--replay-records and positive --replay-per-label must be provided together")
    game_seeds = [int(seed) for seed in args.reserved_game_seeds.split(",")]
    expected_validation_seeds = (
        None if args.validation_seeds == "auto" else {int(seed) for seed in args.validation_seeds.split(",")}
    )
    train, train_stats = examples(args.train_records, args.train_per_label, 57, min_delta=args.min_delta)
    train_inputs = {model_input(row) for row in train}
    replay_stats = None
    if args.replay_records is not None:
        replay, replay_stats = examples(
            args.replay_records,
            args.replay_per_label,
            59,
            train_inputs,
            args.replay_min_delta,
        )
        train.extend(replay)
        train_inputs.update(model_input(row) for row in replay)
    validation, validation_stats = examples(
        args.validation_records,
        args.validation_per_label,
        58,
        train_inputs,
        args.min_delta,
    )
    validation_inputs = {model_input(row) for row in validation}
    train_seeds = sorted({row["seed"] for row in train})
    validation_seeds = sorted({row["seed"] for row in validation})
    if expected_validation_seeds is not None and set(validation_seeds) != expected_validation_seeds:
        raise ValueError("Validation examples must cover exactly the declared validation seeds")
    expected_validation_seeds = set(validation_seeds)
    if set(train_seeds) & set(validation_seeds) or train_inputs & validation_inputs:
        raise ValueError("Training and validation inputs must be disjoint")
    if not game_seeds or set(game_seeds) & (set(train_seeds) | set(validation_seeds)):
        raise ValueError("Reserved game seeds must be nonempty and disjoint from training and validation")
    audit = {
        "trainSource": str(args.train_records),
        "trainSha256": hashlib.sha256(args.train_records.read_bytes()).hexdigest(),
        "replaySource": str(args.replay_records) if args.replay_records else None,
        "replaySha256": hashlib.sha256(args.replay_records.read_bytes()).hexdigest() if args.replay_records else None,
        "baseCheckpoint": str(args.base_checkpoint) if args.base_checkpoint else None,
        "validationSource": str(args.validation_records),
        "validationSha256": hashlib.sha256(args.validation_records.read_bytes()).hexdigest(),
        "teacher": args.teacher_description + "; " + args.teacher_value_description + "; "
        f"minimum absolute difference {args.min_delta:g}",
        "trainSeeds": train_seeds,
        "validationSeeds": validation_seeds,
        "reservedGameSeeds": game_seeds,
        "train": train_stats,
        "replay": replay_stats,
        "validation": validation_stats,
        "crossSplitInputOverlap": len(train_inputs & validation_inputs),
        "updatePolicy": "full" if args.full else "param_subset",
        "epochs": args.epochs,
    }
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "dataset-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)
    if args.dry_run:
        return
    if args.pins is None or args.source_dir is None:
        parser.error("Training needs --pins and --source-dir")

    from honing.backends.jev.nanojev import nanojev_snapshot
    from honing.backends.jev.nanojev_trainer import NanoJevTrainer, NanoJevTrainerSettings, release_path
    from honing.core import (
        DecisionQuestion,
        DecisionSample,
        TrainingBatch,
        TrainJobSpec,
        UpdatePolicy,
        requirements_for,
    )

    decisions = []
    for row in [*train, *validation]:
        control = row["question"]["control"]
        question = DecisionQuestion("choice", control["instructions"], tuple(control["criteria"].items()))
        decisions.append(
            DecisionSample(
                parts=tuple((name, tuple(row["parts"][name])) for name in ("essential", "details", "memory", "logs")),
                questions=(("control", question),),
                chosen=(("control", row["label"]),),
                group=f"spaceinvaders-rollout:{row['seed']}",
            )
        )
    update_policy = (
        UpdatePolicy("full")
        if args.full
        else UpdatePolicy("param_subset", trainable_patterns=(r"^(norm|scalar|set_)",))
    )
    job = TrainJobSpec("sft", update_policy=update_policy, steps=args.epochs, contract="typed_decisions")
    batch = TrainingBatch(
        f"{args.batch_name}-{update_policy.kind}-{args.epochs}ep",
        requirements_for(job).data_contract(),
        decisions=tuple(decisions),
    )
    pins = json.loads(args.pins.read_text())
    base = nanojev_snapshot(args.base_checkpoint or release_path(pins))
    settings = NanoJevTrainerSettings(
        pins_path=args.pins,
        source_dir=args.source_dir,
        work_dir=args.work_dir / "training",
        gpu=args.gpu,
        validation_groups=r"^spaceinvaders-rollout:(" + "|".join(map(str, sorted(expected_validation_seeds))) + r")$",
        min_validation=2 * args.validation_per_label,
        seed=57,
    )
    result = NanoJevTrainer(settings).train(job, base, batch)
    checkpoint = Path(result.snapshot.checkpoint.location)
    report = {
        "checkpoint": str(checkpoint),
        "manifest": str(checkpoint.parent / "checkpoint.json"),
        "weightsSha256": result.snapshot.checkpoint.checksum,
        "metrics": dict(result.metrics),
    }
    (args.work_dir / "train-result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
