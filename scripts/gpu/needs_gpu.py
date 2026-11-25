"""Whether a pull request needs the GPU check.

Reads the changed paths, one per line, on stdin and prints the first one that matches a pattern in
scripts/gpu/paths.txt, or nothing. The `gpu` workflow sets the pull request's `gpu` status from it.
"""

from __future__ import annotations

import fnmatch
import sys
from collections.abc import Iterable
from pathlib import Path

PATTERNS = Path(__file__).with_name("paths.txt")


def patterns(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]


def first_gpu_path(changed: Iterable[str], globs: list[str]) -> str | None:
    return next((path for path in changed if any(fnmatch.fnmatchcase(path, glob) for glob in globs)), None)


def main() -> None:
    hit = first_gpu_path((line.strip() for line in sys.stdin if line.strip()), patterns(PATTERNS.read_text()))
    if hit:
        print(hit)


if __name__ == "__main__":
    main()
