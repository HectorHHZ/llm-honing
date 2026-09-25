"""Consume held-out cases once, allowing only exact evaluation resumes.

Use one project ledger. This is experiment discipline, not tamper-resistant
security against an operator deleting or replacing the ledger.
"""

import fcntl
import hashlib
import json
import os
import re
import time
from pathlib import Path


def checksum(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def cases(protocol):
    test = protocol["sealedTest"]
    if test.get("consumeOnce") is not True:
        raise ValueError("Sealed protocol must require consumeOnce")
    if any(set(test["maps"]) & set(protocol[s]["maps"]) for s in ("train", "dev")):
        raise ValueError("Sealed maps overlap training/development")
    if any(set(test["seeds"]) & set(protocol[s]["seeds"]) for s in ("train", "dev")):
        raise ValueError("Sealed seeds overlap training/development")
    result = [(m, seed, seat) for m in test["maps"] for seed in test["seeds"] for seat in test["seats"]]
    if not result or len(result) != len(set(result)) or len(result) < test["minimumGames"]:
        raise ValueError("Sealed matrix must be complete and unique")
    if set(test["seats"]) != {"red", "blue"}:
        raise ValueError("Sealed matrix must include both seats")
    return sorted(result)


def atomic_save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def reserve(ledger, protocol, checkpoint_sha256, source_sha256, evaluation_root):
    for value in (checkpoint_sha256, source_sha256):
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("Expected verified SHA256 digests")
    matrix = cases(protocol)
    spec = {
        "protocolSha256": checksum(protocol),
        "checkpointSha256": checkpoint_sha256,
        "sourceSha256": source_sha256,
        "evaluationRoot": str(Path(evaluation_root).resolve()),
        "cases": matrix,
    }
    run_id = checksum(spec)
    ledger = Path(ledger).resolve()
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.with_suffix(ledger.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(ledger.read_text()) if ledger.exists() else {"version": 1, "reservations": []}
        for record in data["reservations"]:
            if record["id"] == run_id:
                return record
            if {tuple(c) for c in record["spec"]["cases"]} & set(matrix):
                raise ValueError("Sealed cases already consumed; only the exact reserved evaluation may resume")
        record = {"id": run_id, "spec": spec, "reservedAt": time.time(), "consumedBeforeLaunch": True}
        data["reservations"].append(record)
        atomic_save(ledger, data)
        return record
