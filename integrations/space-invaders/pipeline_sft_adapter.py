"""Stage an audited A/B dataset, train on UCSD GPU 4, and return local evidence.

This adapter uses the existing verified GPU-hold training script. It never
trains locally and removes only its unique remote input directory.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import subprocess
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent


def rows(path: Path) -> list[dict[str, object]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as source:
        return [json.loads(line) for line in source]


def call(*command: str) -> None:
    subprocess.run(command, check=True, text=True)


def staging_sources(
    train_path: Path, validation_path: Path, replay_path: Path, bundle_path: Path
) -> list[tuple[Path, str]]:
    sources = [
        (train_path, "train.jsonl"),
        (validation_path, "validation.jsonl"),
        (HERE / "train_rollout_pair_sft.py", "train_rollout_pair_sft.py"),
        (HERE / "run_counterfactual_sft_gpu4.sh", "runner.sh"),
        (bundle_path, "honing-bundle.tar.gz"),
    ]
    if replay_path.is_file():
        sources.append((replay_path, "replay.jsonl.gz"))
    return sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path(os.environ.get("HONING_PIPELINE_RUN", ".")))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--development-games-spent", type=int, default=0)
    args = parser.parse_args()
    run = args.run.resolve()
    data_dir = args.data_dir.resolve()
    config = json.loads((run / "config.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["stage"] != "registered" or config["remote"]["host"] != "ds-serv2.ucsd.edu":
        raise ValueError("SFT needs a registered UCSD pipeline round")
    train_path, validation_path = data_dir / "train.jsonl", data_dir / "validation.jsonl"
    replay_path = data_dir / "replay.jsonl.gz"
    for path in (train_path, validation_path):
        if not path.is_file():
            raise ValueError(f"Missing SFT input {path}")
    train, validation = rows(train_path), rows(validation_path)
    if len(train) < 200 or len(validation) < 60:
        raise ValueError("SFT requires at least 200 training and 60 internal-validation examples")
    data_seeds = {row["seed"] for row in train + validation}
    if any(type(seed) is not int for seed in data_seeds):
        raise ValueError("Every SFT example needs an integer seed")
    development_seeds = set(manifest["seeds"]["development"])
    if not data_seeds <= development_seeds:
        raise ValueError("SFT data may use development seeds only")
    if {row["seed"] for row in train} & {row["seed"] for row in validation}:
        raise ValueError("SFT training and internal-validation seeds overlap")
    replay_seeds = {row["seed"] for row in rows(replay_path) if "seed" in row} if replay_path.is_file() else set()
    if replay_seeds & (set(manifest["seeds"]["validation"]) | set(manifest["seeds"]["gate"])):
        raise ValueError("Replay examples include validation or final Gate seeds")
    budget = json.loads((run / "proposal.json").read_text())["budget"]["development_games"]
    if not 0 <= args.development_games_spent <= budget:
        raise ValueError("Development game count exceeds the preregistered budget")
    remote = config["remote"]
    from pipeline_bundle import make_bundle

    version = json.loads((run / "pipeline-version.json").read_text())["source_sha256"]
    tag = "pipeline" + uuid.uuid4().hex[:12]
    if not re.fullmatch(r"[a-z0-9]+", tag):
        raise AssertionError("Unsafe generated tag")
    input_dir = f"{remote['pilot_dir']}/runs/counterfactual-sft-{tag}"
    work_dir = f"{remote['work_root']}/counterfactual-sft-{tag}"
    host = remote["host"]
    call("ssh", host, "mkdir", "-p", input_dir)
    evidence = run / "sft-evidence"
    evidence.mkdir(exist_ok=True)
    bundle_path = evidence / "honing-bundle.tar.gz"
    bundle_sha = make_bundle(HERE.parents[1], version, bundle_path)
    completed = False
    try:
        for source, name in staging_sources(train_path, validation_path, replay_path, bundle_path):
            call("scp", str(source), f"{host}:{input_dir}/{name}")
        remote_sha = subprocess.run(
            ["ssh", host, "sha256sum", f"{input_dir}/honing-bundle.tar.gz"], text=True, check=True, capture_output=True
        ).stdout.split()[0]
        if remote_sha != bundle_sha:
            raise ValueError("UCSD Honing source bundle differs from the frozen local bundle")
        call("ssh", host, "tar", "-xf", f"{input_dir}/honing-bundle.tar.gz", "-C", input_dir)
        gate_seeds = ",".join(map(str, manifest["seeds"]["gate"]))
        call(
            "ssh",
            host,
            "env",
            f"SFT_TAG={tag}",
            f"SFT_RESERVED_GAME_SEEDS={gate_seeds}",
            f"SFT_CODE_ROOT={input_dir}",
            f"SFT_PILOT_DIR={remote['pilot_dir']}",
            f"SFT_RUNTIME={remote['runtime']}",
            f"SFT_SOURCE_DIR={remote['source_dir']}",
            f"SFT_CACHE={remote['cache']}",
            f"SFT_HOLD_SCRIPT={remote['hold_script']}",
            f"SFT_HOLD_UNIT={remote['hold_unit']}",
            f"SFT_GPU_UUID={remote['gpu_uuid']}",
            f"SFT_PINS={remote['pins']}",
            f"SFT_BASE_CHECKPOINT={config['incumbent']['checkpoint']}",
            f"SFT_BASE_SHA256={manifest['weights_sha256']}",
            f"SFT_WORK_ROOT={remote['work_root']}",
            f"SFT_REPLAY_PER_LABEL={100 if replay_path.is_file() else 0}",
            "bash",
            f"{input_dir}/runner.sh",
        )
        for remote_path, local_name in (
            (f"{work_dir}/train-result.json", "train-result.json"),
            (f"{work_dir}/dataset-audit.json", "dataset-audit.json"),
            (f"{input_dir}/train.log", "train.log"),
        ):
            call("scp", f"{host}:{remote_path}", str(evidence / local_name))
        result = json.loads((evidence / "train-result.json").read_text())
        audit = json.loads((evidence / "dataset-audit.json").read_text())
        if result["weightsSha256"] == manifest["weights_sha256"]:
            raise ValueError("Training returned unchanged incumbent weights")
        method_result = {
            "method": "sft",
            "training_host": host,
            "gpu_uuid": remote["gpu_uuid"],
            "weights_sha256": result["weightsSha256"],
            "checkpoint": result["checkpoint"],
            "checkpoint_manifest": result["manifest"],
            "training_metrics": result["metrics"],
            "training_log": str(evidence / "train.log"),
            "dataset_audit": str(evidence / "dataset-audit.json"),
            "equivalent_metrics_explanation": (
                "The local train log and dataset audit retain train/validation counts, "
                "initial/final validation loss and held-out accuracy; wandb was not configured."
            ),
            "data_seeds": sorted(data_seeds),
            "development_games_spent": args.development_games_spent,
            "dataset_counts": audit.get("counts", {}),
            "honing_bundle_sha256": bundle_sha,
        }
        (run / "method-result.json").write_text(json.dumps(method_result, indent=2) + "\n")
        completed = True
    finally:
        if not (evidence / "train.log").exists():
            subprocess.run(
                ["scp", f"{host}:{input_dir}/train.log", str(evidence / "train.log")], text=True, capture_output=True
            )
        call("ssh", host, "rm", "-rf", input_dir)
        if not completed:
            call("ssh", host, "rm", "-rf", work_dir)
        active = subprocess.run(
            ["ssh", host, "systemctl", "--user", "is-active", remote["hold_unit"]], text=True, capture_output=True
        )
        if active.returncode != 0 or active.stdout.strip() != "active":
            raise RuntimeError("UCSD GPU 4 hold was not restored after SFT")


if __name__ == "__main__":
    main()
