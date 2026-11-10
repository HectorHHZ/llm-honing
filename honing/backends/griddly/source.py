"""Identify both the Honing adapter and the game runtime for exact replay."""

import hashlib
from pathlib import Path

from integrations.griddly import engine


def source_files() -> dict[str, Path]:
    """Return stable snapshot names for the adapter and its game implementation."""
    adapter = Path(__file__).parent
    runtime = Path(engine.__file__).parent
    return {
        **{
            name: adapter / name
            for name in ("harness.py", "match.py", "model.py", "replay.py", "scenarios.py", "source.py")
        },
        **{f"integration/{name}": runtime / name for name in ("catalog.py", "engine.py")},
    }


def adapter_sha256() -> str:
    result = hashlib.sha256()
    for name, path in sorted(source_files().items()):
        result.update(name.encode("utf-8"))
        result.update(path.read_bytes())
    return result.hexdigest()
