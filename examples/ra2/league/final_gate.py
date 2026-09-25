"""Explicit, resumable Laya final evaluation after a complete development gate.

Never called by optimize.py. Reservation consumes the held-out cases before any
match starts. Errors leave the reservation consumed and require inspection.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path

from gates import evaluate_gate, tick_budget_matches
from optimize import match_jobs, verification_jobs
from sealed import checksum, reserve
from storage import digest
from verify_artifacts import verify

from campaign import run, save

ROOT = Path(__file__).resolve().parents[1]


def development_gate(records, protocol, checkpoint):
    spec = protocol["dev"]
    expected = {(m, seed, seat) for m in spec["maps"] for seed in spec["seeds"] for seat in ("red", "blue")}
    actual = [(r["map"], r["seed"], r["seat"]) for r in records]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Need the complete unique development matrix before consuming sealed cases")
    for record in records:
        if not tick_budget_matches(record, spec):
            raise ValueError("Development tick budget mismatch")
        if (
            record.get("split") != "dev"
            or record.get("checkpointSha256") != checkpoint
            or record.get("controller") != "model"
            or record.get("normalView") is not True
            or record.get("auditPassed") is not True
            or record.get("replayVerified") is not True
            or record.get("fallbackCount") != 0
            or record.get("error") is not None
        ):
            raise ValueError("Development provenance failed")
    winners = [r for r in records if r["ended"] and not r["truncated"] and r["outcome"].get(r["seat"]) == 1]
    thresholds = protocol["sealedTest"]
    if len(winners) / len(records) < thresholds["minimumWinRate"]:
        raise ValueError("Development win rate is insufficient")
    for key, values, threshold in [
        ("map", spec["maps"], thresholds["minimumPerMapWinRate"]),
        ("seat", ("red", "blue"), thresholds["minimumPerSeatWinRate"]),
    ]:
        for value in values:
            wins = sum(r[key] == value for r in winners)
            count = sum(r[key] == value for r in records)
            if wins / count < threshold:
                raise ValueError("Development map/seat win rate is insufficient")
    return {"games": len(records), "wins": len(winners), "passed": True}


def phase(root, jobs, concurrency, gpu_pool):
    root.mkdir(parents=True, exist_ok=True)
    plan = {"version": 1, "jobs": jobs}
    path = root / "plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("Cannot change a persisted final evaluation plan")
    if not path.exists():
        save(path, plan)
    run(root, concurrency, gpu_pool)


def finalize(args):
    protocol = json.loads((ROOT / "league/protocol.json").read_text())
    checkpoint = json.loads(args.checkpoint.read_text())
    if checkpoint.get("kind") != "trained-laya":
        raise ValueError("This final driver currently supports verified trained Laya checkpoints only")
    weight_hash = digest(Path(checkpoint["path"]) / "model.safetensors")
    if weight_hash != checkpoint["weightsSha256"]:
        raise ValueError("Checkpoint weight mismatch")
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / "final-controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = []
        for match in args.development_matches:
            record = verify(match, "dev", args.replay_python)
            manifest = json.loads((match / "manifest.json").read_text())
            for relative, value in manifest["implementationSha256"].items():
                if relative.startswith("integration/") and digest(ROOT / relative) != value:
                    raise ValueError("Development policy source differs from final evaluation source")
            records.append(record)
        preflight = development_gate(records, protocol, weight_hash)
        sources = {
            str(p.relative_to(ROOT)): digest(p)
            for p in sorted((ROOT / "integration").rglob("*"))
            if p.is_file() and p.suffix in {".py", ".mjs"}
        }
        sources.update({str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / "league").glob("*.py"))})
        sources["pins.json"] = digest(ROOT / "pins.json")
        reservation = reserve(
            ROOT / ".cache/league/sealed-consumption.json", protocol, weight_hash, checksum(sources), args.root
        )
        save(args.root / "reservation.json", reservation)
        save(args.root / "development-preflight.json", preflight)
        save(args.root / "source-fingerprint.json", sources)
        spec = protocol["sealedTest"]
        cases = [(m, seed, seat) for m in spec["maps"] for seed in spec["seeds"] for seat in spec["seats"]]
        jobs = match_jobs(args.root / "matches", args.checkpoint, cases, args.gpus, "sealedTest", spec["maxTicks"])
        # Verify/compress each completed match without retaining 96 full decision logs.
        checks = verification_jobs([Path(job["result"]).parent for job in jobs], "sealedTest")
        for check, match in zip(checks, jobs, strict=True):
            check["dependsOn"] = [match["id"]]
            check["command"] += ["--replay-python", args.replay_python]
        phase(args.root / "campaign", [*checks, *jobs], min(3, len(args.gpus)), args.gpus)
        evidence = [json.loads(Path(check["result"]).read_text()) for check in checks]
        for job in jobs:
            manifest = json.loads((Path(job["result"]).parent / "manifest.json").read_text())
            for relative, value in manifest["implementationSha256"].items():
                if relative in sources and sources[relative] != value:
                    raise ValueError("Final match source differs from the consumed reservation")
        report = {
            **evaluate_gate(evidence, protocol),
            "checkpoint": str(args.checkpoint),
            "checkpointSha256": weight_hash,
            "reservation": reservation["id"],
            "records": evidence,
            "fullCrossModelGoalComplete": False,
        }
        save(args.root / "result.json", report)
        print(json.dumps({key: value for key, value in report.items() if key != "records"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--development-matches", type=Path, nargs="+", required=True)
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--replay-python", default=os.environ.get("RA2_REPLAY_PYTHON"))
    args = parser.parse_args()
    if not args.replay_python:
        parser.error("Set RA2_REPLAY_PYTHON or --replay-python")
    args.root = args.root.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.development_matches = [p.resolve() for p in args.development_matches]
    finalize(args)


if __name__ == "__main__":
    main()
