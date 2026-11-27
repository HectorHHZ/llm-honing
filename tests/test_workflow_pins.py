"""Every action a workflow uses is pinned to a commit, so a moved or hijacked tag cannot change what CI runs.

A tag such as ``v7`` moves with each release, and a compromised action repository can point it anywhere.
A full commit SHA cannot move. The version comment after it is what Dependabot reads and updates.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

USES = re.compile(r"^\s*(?:-\s*)?uses:\s*(\S+)(.*)$")
PINNED = re.compile(r"^[\w.-]+/[\w./-]+@[0-9a-f]{40}$")
VERSION = re.compile(r"^\s*#\s*v\d+\.\d+\.\d+\s*$")


def references(repo_root: Path) -> list[tuple[str, str, str]]:
    """(file:line, action, trailing comment) for every ``uses:`` in a workflow or a local action."""
    files = [
        *(repo_root / ".github" / "workflows").glob("*.y*ml"),
        *(repo_root / ".github").glob("actions/*/action.y*ml"),
    ]
    found = []
    for path in sorted(files):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            match = USES.match(line)
            if match:
                found.append((f"{path.relative_to(repo_root)}:{number}", match.group(1), match.group(2)))
    return found


@pytest.mark.unit
def test_every_action_is_pinned_to_a_commit_with_its_version_beside_it(repo_root: Path) -> None:
    refs = references(repo_root)
    assert refs, "no workflow uses an action, so this test checks nothing"
    loose = [
        f"{where}: {action}{comment}"
        for where, action, comment in refs
        if not action.startswith("./") and not (PINNED.match(action) and VERSION.match(comment))
    ]
    assert not loose, "pin these to a full commit SHA with a '# vX.Y.Z' comment:\n" + "\n".join(loose)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("line", "pinned"),
    [
        ("      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1", True),
        ("      - uses: actions/checkout@v7", False),
        ("      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1", False),  # no version to update
        ("      - uses: actions/checkout@3d3c42e # v7.0.1", False),  # a short SHA can be ambiguous
        ("        uses: actions/labeler@main # v7.0.0", False),
    ],
)
def test_the_pin_rule(line: str, pinned: bool) -> None:
    action, comment = USES.match(line).groups()  # type: ignore[union-attr]
    assert bool(PINNED.match(action) and VERSION.match(comment)) is pinned
