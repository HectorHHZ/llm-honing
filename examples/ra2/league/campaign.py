"""Durable process supervision for immutable experiment job plans."""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def execute(root, job):
    spec = json.loads((root / "jobs" / f"{job}.json").read_text())
    execution = root / "status" / f"{job}.execution.json"
    command = json.loads(execution.read_text())["command"] if execution.exists() else spec["command"]
    started = time.time()
    with (root / "logs" / f"{job}.log").open("ab") as log:
        try:
            result = subprocess.run(command, cwd=spec["cwd"], stdout=log, stderr=subprocess.STDOUT, check=False)
            code, error = result.returncode, None
        except Exception as exc:
            code, error = 127, str(exc)
    save(
        root / "status" / f"{job}.exit.json",
        {"exitCode": code, "error": error, "started": started, "finished": time.time()},
    )


def gpu_command(command, gpu):
    out = list(command)
    changed = False
    for i, argument in enumerate(out):
        if argument.startswith("--gpu="):
            out[i] = "--gpu=" + gpu
            changed = True
        elif argument == "--gpu" and i + 1 < len(out):
            out[i + 1] = gpu
            changed = True
    if not changed:
        raise ValueError("GPU job must expose a --gpu argument for recorded resource binding")
    return out


def run(root, concurrency, gpu_pool=None):
    import fcntl

    with (root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = json.loads((root / "plan.json").read_text())["jobs"]
        for name in ["jobs", "logs", "status"]:
            (root / name).mkdir(exist_ok=True)
        names = [j["id"] for j in jobs]
        if len(names) != len(set(names)) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in names):
            raise ValueError("Job IDs must be unique safe file names")
        for j in jobs:
            if any(name not in names or name == j["id"] for name in j.get("dependsOn", [])):
                raise ValueError("Unknown/self job dependency")
            path = root / "jobs" / f"{j['id']}.json"
            if path.exists() and json.loads(path.read_text()) != j:
                raise ValueError("An existing job specification cannot change")
            if not path.exists():
                save(path, j)
        while True:
            running, pending, failed, done = [], [], [], []
            for j in jobs:
                name = j["id"]
                exit_path = root / "status" / f"{name}.exit.json"
                live_path = root / "status" / f"{name}.live.json"
                if exit_path.exists():
                    record = json.loads(exit_path.read_text())
                    (done if record["exitCode"] == 0 and Path(j["result"]).exists() else failed).append(name)
                elif live_path.exists():
                    record = json.loads(live_path.read_text())
                    if record["processStart"] is not None and identity(record["pid"]) == record["processStart"]:
                        running.append(name)
                    else:
                        failed.append(name)  # Never restart an uncertain or vanished job implicitly.
                else:
                    pending.append(j)
            save(
                root / "progress.json",
                {
                    "running": running,
                    "pending": [j["id"] for j in pending],
                    "done": done,
                    "failed": failed,
                    "updated": time.time(),
                },
            )
            if failed:
                raise RuntimeError(f"Jobs need inspection: {failed}")
            if not running and not pending:
                return
            reserved = set()
            for job in jobs:
                if job["id"] in running and "gpu" in job:
                    live = json.loads((root / "status" / f"{job['id']}.live.json").read_text())
                    reserved.add(str(live.get("assignedGpu", job["gpu"])))
            busy = set()
            if any("gpu" in j for j in pending):
                devices = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True
                )
                used = subprocess.check_output(
                    ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True
                )
                busy = {row.split(",")[0].strip() for row in devices.splitlines() if row.split(",")[1].strip() in used}
            slots = max(0, concurrency - len(running))
            ready = [j for j in pending if all(name in done for name in j.get("dependsOn", []))]
            if not ready and pending and not running:
                raise ValueError("Cyclic job dependency")
            for j in ready:
                if not slots:
                    break
                requested = str(j["gpu"]) if "gpu" in j else None
                gpu = requested
                command = j["command"]
                if requested is not None:
                    pool = [str(x) for x in (gpu_pool or [requested])]
                    if requested not in pool:
                        raise ValueError("Requested GPU outside allowed pool")
                    ordered = [requested] + [x for x in pool if x != requested]
                    gpu = next((x for x in ordered if x not in (reserved | busy)), None)
                    if gpu is None:
                        continue
                    reserved.add(gpu)
                    command = gpu_command(command, gpu)
                save(
                    root / "status" / f"{j['id']}.execution.json",
                    {
                        "requestedGpu": requested,
                        "assignedGpu": gpu,
                        "allowedGpuPool": gpu_pool,
                        "command": command,
                        "plannedCommand": j["command"],
                        "boundAt": time.time(),
                    },
                )
                slots -= 1
                process = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), "worker", "--root", str(root), "--job", j["id"]],
                    start_new_session=True,
                )
                save(
                    root / "status" / f"{j['id']}.live.json",
                    {
                        "pid": process.pid,
                        "assignedGpu": gpu,
                        "processStart": identity(process.pid),
                        "specSha256": hashlib.sha256(json.dumps(j, sort_keys=True).encode()).hexdigest(),
                    },
                )
            time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["run", "worker"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--gpu-pool", nargs="+")
    args = parser.parse_args()
    if args.mode == "worker":
        execute(args.root.resolve(), args.job)
    else:
        if not 1 <= args.concurrency <= 3:
            raise ValueError("Concurrency must be between one and three")
        run(args.root.resolve(), args.concurrency, args.gpu_pool)


if __name__ == "__main__":
    main()
