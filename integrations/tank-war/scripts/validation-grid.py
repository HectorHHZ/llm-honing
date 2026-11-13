"""Run #183's fixed 128-seed paired screen across explicitly shared GPUs."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

from exp0_protocol import validate_fault

SEATS = ("blue", "red")
ARMS = ("clean", "harness", "controller", "weights")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_servers(records: list[dict[str, Any]]) -> None:
    if len(records) != 7 or {row["gpu"] for row in records} != set(range(7)):
        raise ValueError("expected all seven GPU servers")
    for row in records:
        with urllib.request.urlopen(row["url"] + "/v1/health", timeout=30) as response:
            health = json.load(response)
        meta = health["metadata"]
        if (
            not health["ready"]
            or meta["weightsSha256"] != row["expected_weights_sha256"]
            or meta["maxLen"] != 8192
            or meta["gpu"] != str(row["gpu"])
        ):
            raise ValueError(f"GPU {row['gpu']} is not serving the pinned checkpoint")


def _run_one(args: argparse.Namespace, row: dict[str, Any], arm: str, seed: int, seat: str, log: Path) -> str:
    root = args.out / arm
    root.mkdir(exist_ok=True)
    run = root / f"s{seed}-{seat}"
    if run.exists():
        if (run / "error.json").exists() or not (run / "result.json").exists():
            raise ValueError(f"incomplete prior run preserved at {run}")
        manifest = _json(run / "manifest.json")
        if (
            manifest["code_commit"] != args.code_commit
            or manifest["model"]["weights_sha256"] != row["expected_weights_sha256"]
            or manifest["decision_server"] != row["url"]
            or manifest["seed"] != seed
            or manifest["seat"] != seat
        ):
            raise ValueError(f"existing run has different frozen identity: {run}")
        return "existing"
    command = [
        os.fspath(args.node),
        os.fspath(Path(__file__).with_name("match.mjs")),
        f"--seed={seed}",
        f"--seat={seat}",
        f"--decision-server={row['url']}",
        f"--out={run}",
        "--seconds=30",
        "--interval-ms=200",
        "--opponent-interval-ms=300",
    ]
    if arm == "harness":
        command.append(f"--harness-dir={args.harness_dir}")
    elif arm == "controller":
        command.append(f"--controller-fault-file={args.controller_fault_file}")
    started = time.monotonic()
    try:
        result = subprocess.run(command, cwd=args.repo, capture_output=True, text=True, timeout=900, check=False)
        attempt = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        attempt = {"error": repr(error)}
    attempt.update(
        gpu=row["gpu"],
        arm=arm,
        seed=seed,
        seat=seat,
        run_dir=str(run),
        command=command,
        elapsed_seconds=round(time.monotonic() - started, 3),
    )
    with log.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(attempt) + "\n")
    if attempt.get("returncode") != 0:
        raise RuntimeError(f"match failed, preserved at {run}; see {log}")
    manifest = _json(run / "manifest.json")
    if (
        manifest["code_commit"] != args.code_commit
        or manifest["model"]["weights_sha256"] != row["expected_weights_sha256"]
        or manifest["decision_server"] != row["url"]
        or manifest["seed"] != seed
        or manifest["seat"] != seat
    ):
        raise ValueError(f"frozen identity differs in {run}")
    return "new"


def _worker(
    args: argparse.Namespace, row: dict[str, Any], seeds: list[int], clean_gpus: list[int], weight_gpus: list[int]
) -> dict[str, Any]:
    gpu = row["gpu"]
    own_clean = gpu in clean_gpus
    group = clean_gpus if own_clean else weight_gpus
    index = group.index(gpu)
    log = args.out / f"attempts-gpu{gpu}.jsonl"
    counts = {"new": 0, "existing": 0}
    for seed in seeds[index :: len(group)]:
        for seat in SEATS:
            for arm in (("clean", "harness", "controller") if own_clean else ("weights",)):
                counts[_run_one(args, row, arm, seed, seat, log)] += 1
        (args.out / f"progress-gpu{gpu}.json").write_text(
            json.dumps(
                {
                    "gpu": gpu,
                    "seed": seed,
                    "completed_seed_blocks": sum(counts.values()) // (6 if own_clean else 2),
                    **counts,
                }
            )
            + "\n"
        )
    return {"gpu": gpu, **counts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-plan", type=Path, required=True)
    parser.add_argument("--gate-log", type=Path, required=True)
    parser.add_argument("--servers", type=Path, required=True)
    parser.add_argument("--harness-dir", type=Path, required=True)
    parser.add_argument("--controller-fault-file", type=Path, required=True)
    parser.add_argument("--weights-fault-file", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--node", type=Path, default=Path("node"))
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    plan = _json(args.seed_plan)
    seeds = plan["groups"]["fault_validation"]
    if len(seeds) != 128 or len(set(seeds)) != 128:
        parser.error("fault validation must use exactly 128 distinct seed blocks")
    from importlib.machinery import SourceFileLoader

    gate = SourceFileLoader("screen_fault", str(Path(__file__).with_name("screen-fault.py"))).load_module()
    gate.require_gate(args.gate_log, plan["groups"]["identity"])
    records = json.loads(args.servers.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        parser.error("server manifest must be a list")
    _check_servers(records)
    clean_gpus = [row["gpu"] for row in records if row["kind"] == "clean"]
    weight_gpus = [row["gpu"] for row in records if row["kind"] == "weights"]
    if len(clean_gpus) != 5 or len(weight_gpus) != 2:
        parser.error("expected five clean servers and two edited-weights servers")
    if _json(args.weights_fault_file)["layer"] != "weights":
        parser.error("weights fault manifest is not a weights edit")
    if _json(args.controller_fault_file)["layer"] != "controller":
        parser.error("controller fault manifest is not a controller edit")
    if _json(args.harness_dir / "fault.json")["layer"] != "harness":
        parser.error("harness fault manifest is not a harness edit")
    config = {
        "seed_plan_sha256": _sha(args.seed_plan),
        "gate_log_sha256": _sha(args.gate_log),
        "seeds": seeds,
        "seats": SEATS,
        "servers": [
            {key: row[key] for key in ("gpu", "gpu_uuid", "url", "kind", "expected_weights_sha256")} for row in records
        ],
        "harness_fault_sha256": _sha(args.harness_dir / "fault.json"),
        "controller_fault_sha256": _sha(args.controller_fault_file),
        "weights_fault_sha256": _sha(args.weights_fault_file),
        "match_sha256": _sha(Path(__file__).with_name("match.mjs")),
        "code_commit": args.code_commit,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "validation-config.json"
    if path.exists() and _json(path) != config:
        raise ValueError("validation output belongs to another frozen configuration")
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
        futures = [pool.submit(_worker, args, row, seeds, clean_gpus, weight_gpus) for row in records]
        results.extend(future.result() for future in concurrent.futures.as_completed(futures))
    (args.out / "worker-summary.json").write_text(json.dumps(results, indent=2) + "\n")
    screens = {}
    for arm in ARMS[1:]:
        result = validate_fault(args.out / "clean", args.out / arm, seeds, arm)
        result["candidate"] = str(
            args.harness_dir / "fault.json"
            if arm == "harness"
            else args.controller_fault_file if arm == "controller" else args.weights_fault_file
        )
        (args.out / f"screen-{arm}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        screens[arm] = {key: result[key] for key in ("mean_fault_minus_clean", "ci95", "retained")}
    print(json.dumps({"screens": screens, "workers": results}))


if __name__ == "__main__":
    main()
