"""Identify headless match and replay code without compatibility modules."""

import hashlib
from pathlib import Path

from integrations.rea_issac import engine


def source_files() -> dict[str, Path]:
    adapter = Path(__file__).parent
    runtime = Path(engine.__file__).parent
    return {
        **{
            f"adapter/{name}": adapter / name
            for name in ("catalog.py", "harness.py", "match.py", "model.py", "replay.py", "scenarios.py", "source.py")
        },
        **{name: runtime / name for name in ("engine.py", "policy.py", "source_facts.json")},
    }


def runtime_fingerprint() -> dict[str, str]:
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in source_files().items()}
