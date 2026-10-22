#!/usr/bin/env python3
"""Fetch and checksum one immutable public v33 release; never load a model/game."""

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen

REPO = "jayzou3773/laya-ra2-v33"
REVISION = "51d2f5a0b36c0a9921a450f14a5afb7cb6d30685"
WEIGHTS_SHA256 = "2ed46ac94c366db2dc68751d0eabfe945ceaeaf430e9fea1ec9e6c60c0207925"
SOURCE_REVISION = "970dc8c5f63d7b886a68409493f37d569424f933"
MANIFEST_SHA256 = "e9a1699e87f675582de111ac146faf21425085204d5baa8c6e5a8fe6cd309939"


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts or "\\" in name:
        raise ValueError(f"Unsafe release filename: {name}")
    return str(path)


def fetch(destination, name, checksum=None):
    name = safe_name(name)
    target = destination / name
    if target.exists():
        if checksum and digest(target) != checksum:
            raise ValueError(f"Existing file differs from pinned release: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".download-part")
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"
    request = Request(url, headers={"User-Agent": "ra2-laya-v33-cpu-setup/1.0"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=90) as response, temporary.open("wb") as stream:
                for chunk in iter(lambda: response.read(4 * 1024 * 1024), b""):
                    stream.write(chunk)
            if checksum and digest(temporary) != checksum:
                raise ValueError(f"Checksum mismatch: {name}")
            temporary.rename(target)
            print(f"Downloaded and checked: {name}", flush=True)
            return
        except (OSError, ValueError):  # noqa: PERF203 - Network retries need per-attempt handling.
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    fetch(destination, "SHA256SUMS", MANIFEST_SHA256)
    entries = {}
    for line in (destination / "SHA256SUMS").read_text().splitlines():
        if not line.strip():
            continue
        checksum, name = line.split(maxsplit=1)
        name = safe_name(name.lstrip("*"))
        if len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
            raise ValueError("Invalid checksum manifest")
        if name in entries:
            raise ValueError(f"Duplicate checksum entry: {name}")
        entries[name] = checksum
    if entries.get("model.safetensors") != WEIGHTS_SHA256:
        raise ValueError("Weight hash does not match the selected v33 release")
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fetch, destination, name, checksum) for name, checksum in entries.items()]
        for future in futures:
            future.result()
    release = json.loads((destination / "release.json").read_text())
    if (
        release["repo"] != REPO
        or release["weightsSha256"] != WEIGHTS_SHA256
        or release["sourceRevision"] != SOURCE_REVISION
    ):
        raise ValueError("Unexpected release identity")
    receipt = {
        "repo": REPO,
        "revision": REVISION,
        "weightsSha256": WEIGHTS_SHA256,
        "sourceRevision": SOURCE_REVISION,
        "verifiedFiles": len(entries),
    }
    (destination / ".snapshot.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"event": "release_verified", "destination": str(destination), **receipt}), flush=True)


if __name__ == "__main__":
    main()
