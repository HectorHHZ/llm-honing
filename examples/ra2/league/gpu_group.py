"""Run a two-worker duel only after reserving a complete allowed GPU pair.

Uses the same UUID dispatch and worker locks as campaign.py. Partial acquisition
is released immediately, so independent controllers cannot deadlock. Dispatch
FDs are inherited by the match process to keep leases if this wrapper exits.
"""

import argparse
import itertools
import json
import subprocess
import time
from pathlib import Path

from campaign import device_uuid, identity, save, try_device_lease


def try_group(pool, count, uuids, busy, acquire=try_device_lease):
    for group in itertools.combinations(pool, count):
        if any(uuids[gpu] in busy for gpu in group):
            continue
        leases = []
        try:
            for gpu in group:
                lease = acquire(uuids[gpu])
                if lease is None:
                    break
                leases.append(lease)
            if len(leases) == count:
                return list(group), leases
        finally:
            if len(leases) != count:
                for lease in leases:
                    lease.close()
    return None


def bind(command, group):
    for i in range(len(group)):
        if not any("{gpu" + str(i) + "}" in item for item in command):
            raise ValueError("Every reserved device must have an explicit command placeholder")
    return [replace_tokens(item, group) for item in command]


def replace_tokens(item, group):
    for i, gpu in enumerate(group):
        item = item.replace("{gpu" + str(i) + "}", gpu)
    if "{gpu" in item:
        raise ValueError("Unknown GPU placeholder")
    return item


def validate_pool(pool, count):
    if (
        count != 2
        or len(pool) != len(set(pool))
        or len(pool) < count
        or any(not gpu.isascii() or not gpu.isdecimal() or str(int(gpu)) != gpu for gpu in pool)
    ):
        raise ValueError("Duel requires two distinct GPU indices from an explicit eligible pool")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", nargs="+", required=True)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    validate_pool(args.pool, args.count)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.record.exists() or args.timeout <= 0:
        raise ValueError("Require command, fresh resource record and positive timeout")
    bind(command, args.pool[: args.count])
    uuids = {gpu: device_uuid(gpu) for gpu in args.pool}
    args.record.parent.mkdir(parents=True, exist_ok=True)
    with args.record.open("x") as stream:
        json.dump(
            {
                "state": "waiting",
                "pool": args.pool,
                "uuids": uuids,
                "commandTemplate": command,
                "started": time.time(),
            },
            stream,
        )
    waiting = time.monotonic()
    while True:
        busy = set(
            subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True
            ).splitlines()
        )
        acquired = try_group(args.pool, args.count, uuids, busy)
        if acquired is not None:
            break
        if time.monotonic() - waiting > args.timeout:
            save(args.record, {"state": "timed-out", "pool": args.pool, "commandTemplate": command})
            raise TimeoutError("No complete idle GPU pair became available")
        time.sleep(2)
    group, leases = acquired
    execution = {
        "state": "running",
        "assignedGpu": group,
        "assignedUuid": [uuids[g] for g in group],
        "command": bind(command, group),
        "started": time.time(),
    }
    try:
        process = subprocess.Popen(execution["command"], pass_fds=tuple(lease.fileno() for lease in leases))
        execution.update(pid=process.pid, identity=identity(process.pid))
        save(args.record, execution)
        code = process.wait()
        execution.update(state="complete" if code == 0 else "failed", exitCode=code, finished=time.time())
        save(args.record, execution)
    except BaseException as error:
        # Do not claim completion or retry an uncertain child. A live child and
        # its workers retain the FD/UUID guards until they actually exit.
        execution.update(state="interrupted-or-error", error=str(error), observed=time.time())
        save(args.record, execution)
        raise
    finally:
        for lease in leases:
            lease.close()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
