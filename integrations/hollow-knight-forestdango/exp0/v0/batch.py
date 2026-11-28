"""Run a list of collect.py jobs, at most P at a time, each on its own display (:350-:379); keep every launch on record.

  python3 -B batch.py JOBS.json --root ROOT [--parallel 2] [--log batch_log.jsonl]

JOBS.json: [{"name": <attempt dir>, "task", "seed", "condition", "instance"?, "form"?, "display", "label"?}, ...].
Each job runs `python3 -B collect.py --out ROOT/<name> ...` with its output in ROOT/<name>.log; the log file
batch_log.jsonl gets one line when a job starts (pid, start time) and one when it ends (exit code, seconds). Only the
processes started here are waited for; a stop request (SIGTERM to this process) is passed to its own children by pid.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", type=Path)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--log", type=Path, default=None)
    a = ap.parse_args()
    jobs = json.loads(a.jobs.read_text())
    log = a.log or (a.root / "batch_log.jsonl")
    a.root.mkdir(parents=True, exist_ok=True)
    running: dict[int, tuple[subprocess.Popen, dict, float]] = {}

    def record(row: dict) -> None:
        with log.open("a") as f:
            f.write(json.dumps({**row, "time": time.strftime("%Y-%m-%d %H:%M:%S %z")}) + "\n")

    def stop(*_):
        for proc, job, _ in running.values():
            if proc.poll() is None:
                proc.terminate()
                record({"event": "terminated", "name": job["name"], "pid": proc.pid})
        sys.exit(143)

    signal.signal(signal.SIGTERM, stop)
    queue = list(jobs)
    while queue or running:
        while queue and len(running) < a.parallel:
            job = queue.pop(0)
            cmd = [
                sys.executable,
                "-B",
                str(HERE / "collect.py"),
                "--out",
                str(a.root / job["name"]),
                "--task",
                job["task"],
                "--condition",
                job["condition"],
                "--display",
                str(job["display"]),
                "--label",
                job.get("label", ""),
            ]
            if job.get("seed") is not None:
                cmd += ["--seed", str(job["seed"])]
            if job.get("group"):
                cmd += ["--group", job["group"]]
            if job.get("endpoint"):
                cmd += ["--endpoint", job["endpoint"]]
            if job.get("plan"):
                cmd += ["--plan", job["plan"]]
            if job.get("max_decisions"):
                cmd += ["--max-decisions", str(job["max_decisions"])]
            if job.get("instance"):
                cmd += ["--fault-instance", job["instance"]]
            if job.get("form"):
                cmd += ["--fault-form", job["form"]]
            out = (a.root / f"{job['name']}.log").open("w")
            proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, cwd=str(HERE))
            running[proc.pid] = (proc, job, time.monotonic())
            record(
                {"event": "started", "name": job["name"], "pid": proc.pid, "display": job["display"], "cmd": cmd[2:]}
            )
        time.sleep(2)
        for pid, (proc, job, t0) in list(running.items()):
            if proc.poll() is not None:
                record(
                    {
                        "event": "finished",
                        "name": job["name"],
                        "pid": pid,
                        "returncode": proc.returncode,
                        "seconds": round(time.monotonic() - t0, 1),
                    }
                )
                del running[pid]


if __name__ == "__main__":
    main()
