"""Resumable Laya DAgger generations with pure-model development evaluation.

This development loop does not consume the sealed test and cannot declare the full
cross-model objective complete. Every generation changes weights and archives evidence.
"""

import argparse
import fcntl
import hashlib
import json
import os
import sys
from pathlib import Path

from campaign import run, save

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def phase(directory, jobs, concurrency):
    directory.mkdir(parents=True, exist_ok=True)
    plan = {"version": 1, "jobs": jobs}
    file = directory / "plan.json"
    if file.exists() and json.loads(file.read_text()) != plan:
        raise ValueError("Cannot change a persisted phase plan")
    if not file.exists():
        save(file, plan)
    run(directory, concurrency)


def match_jobs(directory, checkpoint, cases, gpus, split, ticks):
    jobs = []
    for i, (map_name, seed, seat) in enumerate(cases):
        name = f"{Path(map_name).stem}-{seed}-{seat}"
        out = directory / name
        gpu = gpus[i % len(gpus)]
        jobs.append(
            {
                "id": name,
                "kind": "learned-policy-match",
                "split": split,
                "gpu": gpu,
                "cwd": str(ROOT),
                "result": str(out / "result.json"),
                "command": [
                    "env",
                    f"LAYA_CHECKPOINT_MANIFEST={checkpoint}",
                    "node",
                    "integration/laya/run.mjs",
                    "--model=typed-decisions",
                    "--candidate-mode=intent",
                    "--decision-mode=staged",
                    "--decision-interval=3",
                    "--cadence=3",
                    f"--ticks={ticks}",
                    f"--map={map_name}",
                    f"--seed={seed}",
                    f"--seat={seat}",
                    f"--gpu={gpu}",
                    f"--out={out}",
                ],
            }
        )
    return jobs


def verification_jobs(matches, split):
    return [
        {
            "id": "verify-" + p.name,
            "cwd": str(ROOT),
            "result": str(p / "evidence.json"),
            "command": [sys.executable, "league/verify_artifacts.py", "--match", str(p), "--split", split],
        }
        for p in matches
    ]


def optimize(args):
    args.root.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((ROOT / "league/protocol.json").read_text())
    conf = {
        "version": 1,
        "initialCheckpoint": str(args.initial_checkpoint),
        "initialManifestSha256": digest(args.initial_checkpoint),
        "data": [str(p) for p in args.data],
        "dataHashes": {str(p): digest(p / "samples.jsonl") for p in args.data},
        "generations": args.generations,
        "gpus": args.gpus,
        "epochs": args.epochs,
        "validationSeed": args.validation_seed,
        "protocolSha256": digest(ROOT / "league/protocol.json"),
    }
    path = args.root / "config.json"
    if path.exists() and json.loads(path.read_text()) != conf:
        raise ValueError("Resume arguments or initial data changed")
    if not path.exists():
        save(path, conf)
    data = list(args.data)
    checkpoint = args.initial_checkpoint
    summaries = []
    for generation in range(1, args.generations + 1):
        directory = args.root / f"generation-{generation:03d}"
        directory.mkdir(exist_ok=True)
        save(args.root / "progress.json", {"generation": generation, "stage": "training", "complete": False})
        trained = directory / "trained"
        train_job = {
            "id": "train",
            "gpu": args.gpus[0],
            "cwd": str(ROOT),
            "result": str(trained / "result.json"),
            "command": [
                os.environ.get("LAYA_PYTHON", sys.executable),
                "league/train_laya.py",
                "--matches",
                *map(str, data),
                "--out",
                str(trained),
                "--gpu",
                args.gpus[0],
                "--epochs",
                str(args.epochs),
                "--validation-seed",
                str(args.validation_seed),
                "--parent-manifest",
                str(checkpoint),
            ],
        }
        phase(directory / "training", [train_job], 1)
        checkpoint = trained / "checkpoint.json"
        save(
            args.root / "progress.json",
            {"generation": generation, "stage": "development-evaluation", "complete": False},
        )
        # Reuse the same development cases for paired comparisons; never sealed test cases.
        cases = [(m, protocol["dev"]["seeds"][0], seat) for m in protocol["dev"]["maps"] for seat in ("red", "blue")]
        jobs = match_jobs(directory / "dev", checkpoint, cases, args.gpus, "dev", protocol["dev"]["maxTicks"])
        phase(directory / "evaluation", jobs, min(3, len(args.gpus)))
        matches = [Path(j["result"]).parent for j in jobs]
        phase(directory / "verification", verification_jobs(matches, "dev"), 3)
        records = [json.loads((p / "evidence.json").read_text()) for p in matches]
        wins = sum(r["ended"] and not r["truncated"] and r["outcome"].get(r["seat"]) == 1 for r in records)
        summary = {
            "generation": generation,
            "checkpoint": str(checkpoint),
            "wins": wins,
            "games": len(records),
            "records": records,
            "sealedTestConsumed": False,
            "stableWinClaim": False,
            "crossModelGoalComplete": False,
        }
        save(directory / "summary.json", summary)
        summaries.append({k: summary[k] for k in ("generation", "checkpoint", "wins", "games")})
        best = max(summaries, key=lambda row: (row["wins"] / row["games"], row["generation"]))
        save(
            args.root / "development-ranking.json",
            {
                "generations": summaries,
                "bestObserved": best,
                "caveat": (
                    "Adaptive development cases, not a held-out success claim. Ties favor the latest checkpoint."
                ),
            },
        )
        if generation == args.generations:
            break
        save(
            args.root / "progress.json",
            {"generation": generation, "stage": "training-rollout-and-labels", "complete": False},
        )
        seed = next(s for s in protocol["train"]["seeds"] if s != args.validation_seed and s >= 106 + generation)
        cases = [(m, seed, seat) for m in protocol["train"]["maps"] for seat in ("red", "blue")]
        jobs = match_jobs(directory / "rollouts", checkpoint, cases, args.gpus, "train", protocol["train"]["maxTicks"])
        phase(directory / "collection", jobs, min(3, len(args.gpus)))
        matches = [Path(j["result"]).parent for j in jobs]
        phase(directory / "rollout-verification", verification_jobs(matches, "train"), 3)
        labels = directory / "labels"
        jobs = [
            {
                "id": p.name,
                "cwd": str(ROOT),
                "result": str(labels / p.name / "result.json"),
                "command": ["node", "league/label_rollout.mjs", f"--match={p}", f"--out={labels / p.name}"],
            }
            for p in matches
        ]
        phase(directory / "annotation", jobs, 3)
        data.extend(Path(j["result"]).parent for j in jobs)
    save(
        args.root / "result.json",
        {
            "iterationBatchComplete": True,
            "goalComplete": False,
            "generations": summaries,
            "next": "Inspect development evidence; continue learning and cross-model integration.",
        },
    )
    save(
        args.root / "progress.json",
        {"generation": args.generations, "stage": "iteration-batch-finished", "complete": True, "goalComplete": False},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--gpus", nargs="+", default=["1", "5"])
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--validation-seed", type=int, default=105)
    args = parser.parse_args()
    if not 1 <= args.generations <= 8 or len(set(args.gpus)) != len(args.gpus):
        raise ValueError("Use 1..8 generations and unique GPU indices")
    args.root, args.initial_checkpoint = args.root.resolve(), args.initial_checkpoint.resolve()
    args.data = [p.resolve() for p in args.data]
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        optimize(args)


if __name__ == "__main__":
    main()
