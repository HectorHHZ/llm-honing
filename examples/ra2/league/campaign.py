"""Durable process supervision for immutable experiment job plans."""

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
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


def device_uuid(gpu):
    devices = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    for row in devices.splitlines():
        index, uuid = (part.strip() for part in row.split(","))
        if index == str(gpu):
            return uuid
    raise ValueError(f"GPU index {gpu} not found")


def try_device_lease(uuid):
    """Reserve across controllers, including workers still loading on the CPU."""
    directory = Path(os.environ.get("LAYA_LOCK_DIR", Path(tempfile.gettempdir()) / f"laya-{os.getuid()}"))
    directory.mkdir(parents=True, exist_ok=True)
    lease = (directory / f"{uuid}.dispatch.lock").open("a")
    try:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Existing model workers may predate dispatch leases. Their own UUID
        # lock covers data preparation even when nvidia-smi has no process yet.
        with (directory / f"{uuid}.lock").open("a") as worker:
            fcntl.flock(worker, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lease.close()
        return None
    except BaseException:
        lease.close()
        raise
    return lease


def wait_device_lease(gpu):
    """Compatibility for already-running controllers without FD handoff."""
    uuid = device_uuid(gpu)
    while True:
        lease = try_device_lease(uuid)
        if lease is not None:
            used = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True
            )
            if uuid not in used:
                return lease
            lease.close()
        time.sleep(1)


def execute(root, job, lease_fd=None):
    spec = json.loads((root / "jobs" / f"{job}.json").read_text())
    execution = root / "status" / f"{job}.execution.json"
    binding = json.loads(execution.read_text()) if execution.exists() else {}
    command = binding.get("command", spec["command"])
    started = time.time()
    with (root / "logs" / f"{job}.log").open("ab") as log:
        try:
            if lease_fd is not None:
                if lease_fd < 3:
                    raise ValueError("Invalid inherited lease descriptor")
                lease = os.fdopen(lease_fd, "a")
            elif "gpu" in spec:
                lease = wait_device_lease(binding.get("assignedGpu", spec["gpu"]))
            else:
                lease = contextlib.nullcontext()
            with lease:
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
            uuids = {}
            if any("gpu" in j for j in pending):
                devices = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True
                )
                used = subprocess.check_output(
                    ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True
                )
                uuids = dict(tuple(part.strip() for part in row.split(",")) for row in devices.splitlines())
                busy = {index for index, uuid in uuids.items() if uuid in used}
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
                lease = None
                if requested is not None:
                    pool = [str(x) for x in (gpu_pool or [requested])]
                    if requested not in pool:
                        raise ValueError("Requested GPU outside allowed pool")
                    ordered = [requested] + [x for x in pool if x != requested]
                    gpu = None
                    for candidate in ordered:
                        if candidate in (reserved | busy) or candidate not in uuids:
                            continue
                        lease = try_device_lease(uuids[candidate])
                        if lease is not None:
                            gpu = candidate
                            break
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
                worker_command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "worker",
                    "--root",
                    str(root),
                    "--job",
                    j["id"],
                ]
                descriptors = ()
                if lease is not None:
                    descriptors = (lease.fileno(),)
                    worker_command += ["--lease-fd", str(lease.fileno())]
                try:
                    process = subprocess.Popen(worker_command, start_new_session=True, pass_fds=descriptors)
                finally:
                    # Close only this reference; explicitly unlocking would
                    # also release the shared lock inherited by the child.
                    if lease is not None:
                        lease.close()
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
    parser.add_argument("--lease-fd", type=int)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--gpu-pool", nargs="+")
    args = parser.parse_args()
    if args.mode == "worker":
        execute(args.root.resolve(), args.job, args.lease_fd)
    else:
        if not 1 <= args.concurrency <= 3:
            raise ValueError("Concurrency must be between one and three")
        run(args.root.resolve(), args.concurrency, args.gpu_pool)


if __name__ == "__main__":
    main()
