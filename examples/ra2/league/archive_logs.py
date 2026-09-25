"""Archive verified logs with zstd; restore gzip before using frozen older readers."""

import argparse
import fcntl
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from storage import digest, zstd_content


def checksum(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def decoded_checksum(path):
    h = hashlib.sha256()
    with zstd_content(path) as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def archive(match, restore=False):
    if not shutil.which("zstd"):
        raise RuntimeError("Install the zstd CLI before archiving or restoring logs")
    with (match / "archive.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        evidence_path = match / "evidence.json"
        evidence_bytes = evidence_path.read_bytes()
        evidence = json.loads(evidence_bytes)
        if not (match / "result.json").is_file() or not (
            evidence.get("auditPassed") and evidence.get("replayVerified")
        ):
            raise ValueError("Only completed, audited and replay-verified logs may be archived")
        logical = match / "decisions.jsonl"
        if logical.exists():
            raise ValueError("Compress the completed plain log to gzip first")
        expected = evidence["artifacts"][logical.name]
        gz, zst = Path(str(logical) + ".gz"), Path(str(logical) + ".zst")
        if restore:
            source, target = zst, gz
        else:
            source, target = gz, zst
        if not source.is_file():
            raise FileNotFoundError(source)
        before = checksum(source)
        temporary = target.with_name(target.name + ".archive-tmp")
        if temporary.exists():
            raise ValueError("Inspect the existing interrupted archive attempt before retrying")
        if restore:
            if decoded_checksum(source) != expected:
                raise ValueError("Archive content differs from verified evidence")
            if not target.exists():
                with zstd_content(source) as src, temporary.open("xb") as raw:
                    with gzip.GzipFile(filename="", mode="wb", compresslevel=9, fileobj=raw, mtime=0) as dst:
                        shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
                    raw.flush()
                    os.fsync(raw.fileno())
            check = target if target.exists() else temporary
            with gzip.open(check, "rb") as stream:
                h = hashlib.sha256()
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    h.update(block)
            actual = h.hexdigest()
        else:
            if digest(logical) != expected:
                raise ValueError("Gzip content differs from verified evidence")
            if not target.exists():
                with gzip.open(source, "rb") as src, temporary.open("xb") as raw:
                    process = subprocess.Popen(
                        ["zstd", "-q", "-10", "--long=27", "-c"], stdin=subprocess.PIPE, stdout=raw
                    )
                    try:
                        shutil.copyfileobj(src, process.stdin, 8 * 1024 * 1024)
                        process.stdin.close()
                        if process.wait() != 0:
                            raise ValueError("zstd compression failed; gzip retained")
                        raw.flush()
                        os.fsync(raw.fileno())
                    finally:
                        if process.poll() is None:
                            process.kill()
                            process.wait()
            check = target if target.exists() else temporary
            actual = decoded_checksum(check)
        if actual != expected or checksum(source) != before or evidence_path.read_bytes() != evidence_bytes:
            raise ValueError("Archive verification failed; original retained")
        if not target.exists():
            temporary.replace(target)
        record = {
            "format": "gzip-jsonl" if restore else "zstd-jsonl",
            "path": str(target),
            "originalName": logical.name,
            "contentSha256": expected,
            "compressedSha256": checksum(target),
            "compressedBytes": target.stat().st_size,
            "losslessVerified": True,
            "previousPath": str(source),
            "previousCompressedSha256": before,
            "previousCompressedBytes": source.stat().st_size,
            "evidencePreservedSha256": hashlib.sha256(evidence_bytes).hexdigest(),
            "time": time.time(),
        }
        current = match / "log-storage.json"
        backup = match / f"log-storage.before-{before[:12]}.json"
        if current.exists() and not backup.exists():
            shutil.copy2(current, backup)
        # Keep a recovery record before removing the now-redundant gzip copy.
        save(match / f"log-archive-{record['format']}-{before[:12]}.json", record)
        save(current, record)
        if not restore:
            source.unlink()
        directory = os.open(match, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore-gzip", action="store_true")
    parser.add_argument("matches", type=Path, nargs="+")
    args = parser.parse_args()
    for match in args.matches:
        print(json.dumps(archive(match.resolve(), args.restore_gzip)), flush=True)
