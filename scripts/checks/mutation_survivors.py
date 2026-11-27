"""Fail when mutmut leaves a mutant alive that tests/mutation_equivalent.txt does not list.

Run after ``mutmut run``. A survivor is named by its function and the line the
mutation wrote, not by mutmut's number, which shifts when the code changes. A
listed survivor is a mutant equivalent to the original (an HTTP header's case,
a ``zip`` whose lengths always match); anything else is a test gap. Prints the
kill ratio over the mutants some test covers.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ALLOWED = ROOT / "tests" / "mutation_equivalent.txt"
KILLED, SURVIVED, NO_TESTS = 1, 0, 33


def mutated_line(name: str) -> str:
    diff = subprocess.run(["mutmut", "show", name], capture_output=True, text=True, check=True).stdout
    added = [line[1:].strip() for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")]
    return added[0] if added else ""


def main() -> int:
    codes: Counter[int] = Counter()
    survivors = []
    for meta in sorted((ROOT / "mutants").rglob("*.meta")):
        for name, code in json.loads(meta.read_text())["exit_code_by_key"].items():
            codes[code] += 1
            if code == SURVIVED:
                survivors.append(name)
    allowed = {line.strip() for line in ALLOWED.read_text().splitlines() if line.strip() and not line.startswith("#")}
    unexplained = []
    for name in survivors:
        key = f"{re.sub(r'__mutmut_[0-9]+$', '', name)}: {mutated_line(name)}"
        if key not in allowed:
            unexplained.append(key)
    covered = codes[KILLED] + codes[SURVIVED]
    print(f"killed {codes[KILLED]} of {covered} covered mutants ({codes[KILLED] / max(covered, 1):.1%}),")
    listed = codes[SURVIVED] - len(unexplained)
    print(f"{codes[SURVIVED]} survived ({listed} listed as equivalent), {codes[NO_TESTS]} not covered")
    for key in unexplained:
        print(f"SURVIVED, not listed: {key}")
    return 1 if unexplained else 0


if __name__ == "__main__":
    sys.exit(main())
