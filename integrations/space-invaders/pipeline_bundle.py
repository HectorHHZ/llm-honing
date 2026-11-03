"""Package the exact frozen Honing Python sources for isolated UCSD jobs."""

from __future__ import annotations

import hashlib
import shutil
import tarfile
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_bundle(repository: Path, version: dict[str, str], output: Path) -> str:
    names = sorted(name for name in version if name.startswith("honing/") and name.endswith(".py"))
    if not names:
        raise ValueError("Frozen version has no Honing Python sources")
    with tarfile.open(output, "w:gz") as archive:
        for name in names:
            source = repository / name
            if not source.is_file() or sha(source) != version[name]:
                raise ValueError(f"Honing source differs from frozen version: {name}")
            archive.add(source, arcname=name, recursive=False)
    return sha(output)


def extract_bundle(archive_path: Path, destination: Path, expected_sha: str) -> None:
    if sha(archive_path) != expected_sha:
        raise ValueError("Transferred Honing source bundle has the wrong hash")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        if not members or any(
            not member.isfile() or not member.name.startswith("honing/") or ".." in Path(member.name).parts
            for member in members
        ):
            raise ValueError("Honing source bundle contains an unsafe path")
        for member in members:
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Honing source bundle has unreadable file {member.name}")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
