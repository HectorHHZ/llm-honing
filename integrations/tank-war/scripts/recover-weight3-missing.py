"""Finish missing frozen weight-3.0 games without changing old attempts."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

EXPECTED_SHA = "b2538b1649843791b5e5996b8364af88051e6f3aa1c7828770873ce90cc574f7"
EXPECTED_COMMIT = "31912a1a3378f49e542577fa383d42891a6f2dc3"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-plan", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--node", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--deadline-unix", type=float, required=True)
    args = parser.parse_args()
    with urllib.request.urlopen(args.url + "/v1/health", timeout=10) as response:
        health = json.load(response)
    if not health["ready"] or health["metadata"]["weightsSha256"] != EXPECTED_SHA:
        raise ValueError("edited checkpoint identity differs")
    seeds = read(args.seed_plan)["groups"]["final_test"][:80]
    pending = []
    for seed in seeds:
        for seat in ("blue", "red"):
            prior = args.original / f"s{seed}-{seat}"
            if (prior / "result.json").exists() and not (prior / "error.json").exists():
                continue
            pending.append((seed, seat, str(prior) if prior.exists() else None))
    args.out.mkdir(parents=True, exist_ok=True)
    freeze = args.out / "freeze.json"
    planned = {
        "checkpoint_sha256": EXPECTED_SHA,
        "code_commit": EXPECTED_COMMIT,
        "source_seed_plan": str(args.seed_plan),
        "pending": pending,
        "original_root": str(args.original),
        "url": args.url,
    }
    if freeze.exists() and read(freeze) != planned:
        raise ValueError("recovery plan changed")
    if not freeze.exists():
        freeze.write_text(json.dumps(planned, indent=2) + "\n")
    journal = args.out / "attempts.jsonl"
    lock = threading.Lock()

    def collect(item: tuple) -> dict:
        seed, seat, prior = item
        run = args.out / f"s{seed}-{seat}"
        if (run / "result.json").exists() and not (run / "error.json").exists():
            return {"seed": seed, "seat": seat, "reused": True}
        if run.exists():
            raise ValueError(f"incomplete recovery attempt remains: {run}")
        command = [
            str(args.node),
            str(args.repo / "integrations/tank-war/scripts/match.mjs"),
            f"--seed={seed}",
            f"--seat={seat}",
            f"--decision-server={args.url}",
            f"--out={run}",
            "--seconds=30",
            "--interval-ms=200",
            "--opponent-interval-ms=300",
        ]
        started = time.monotonic()
        try:
            process = subprocess.run(command, cwd=args.repo, text=True, capture_output=True, timeout=900, check=False)
            row = {"returncode": process.returncode, "stderr": process.stderr[-2000:]}
        except (OSError, subprocess.TimeoutExpired) as error:
            row = {"error": repr(error)}
        row.update(
            seed=seed,
            seat=seat,
            prior=prior,
            run=str(run),
            command=command,
            seconds=round(time.monotonic() - started, 2),
        )
        if row.get("returncode") == 0:
            manifest = read(run / "manifest.json")
            if manifest["code_commit"] != EXPECTED_COMMIT or manifest["model"]["weights_sha256"] != EXPECTED_SHA:
                row["validation_error"] = "source or checkpoint identity differs"
        with lock, journal.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(
            json.dumps({"seed": seed, "seat": seat, "returncode": row.get("returncode"), "seconds": row["seconds"]}),
            flush=True,
        )
        return row

    entries = iter(pending)
    active: set[concurrent.futures.Future] = set()
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        while active or time.time() < args.deadline_unix:
            while len(active) < args.workers and time.time() < args.deadline_unix:
                try:
                    item = next(entries)
                except StopIteration:
                    break
                active.add(pool.submit(collect, item))
            if not active:
                break
            finished, active = concurrent.futures.wait(
                active, timeout=5, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in finished:
                done += 1
                try:
                    future.result()
                except Exception as error:
                    with lock, journal.open("a") as stream:
                        stream.write(json.dumps({"error": repr(error)}) + "\n")
    print(json.dumps({"recovered": done, "original_completed": 160 - len(pending), "planned_missing": len(pending)}))


if __name__ == "__main__":
    main()
