"""Losslessly compress completed decision logs, verifying decoded content before unlink."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def open_content(path):
    return path.open("rb") if path.exists() else gzip.open(str(path) + ".gz", "rb")


def digest(path):
    h = hashlib.sha256()
    with open_content(path) as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def compress(match):
    source = match / "decisions.jsonl"
    if not (match / "result.json").is_file():
        raise ValueError("Only complete matches can be archived")
    target = Path(str(source) + ".gz")
    if not source.exists():
        if target.exists():
            return {"alreadyCompressed": True, "path": str(target)}
        raise FileNotFoundError(source)
    original_size = source.stat().st_size
    original_hash = digest(source)
    temporary = target if target.exists() else Path(str(target) + ".tmp")
    if not target.exists():
        with source.open("rb") as src, gzip.open(temporary, "wb", compresslevel=3) as dst:
            for block in iter(lambda: src.read(8 * 1024 * 1024), b""):
                dst.write(block)
    decoded = hashlib.sha256()
    with gzip.open(temporary, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            decoded.update(block)
    if decoded.hexdigest() != original_hash:
        raise ValueError("Compression verification failed; original retained")
    if source.stat().st_size != original_size or digest(source) != original_hash:
        raise ValueError("Original changed during compression; original retained")
    if temporary != target:
        temporary.replace(target)
    report = {
        "path": str(target),
        "originalName": source.name,
        "originalBytes": original_size,
        "compressedBytes": target.stat().st_size,
        "contentSha256": original_hash,
        "compressedSha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "losslessVerified": True,
    }
    (match / "log-storage.json").write_text(json.dumps(report, indent=2) + "\n")
    source.unlink()
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("matches", nargs="+", type=Path)
    for directory in p.parse_args().matches:
        print(json.dumps(compress(directory.resolve())), flush=True)
