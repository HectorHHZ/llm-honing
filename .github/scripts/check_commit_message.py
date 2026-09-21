#!/usr/bin/env python3
"""Check a commit subject against the Conventional Commits rules in CONTRIBUTING.md.

Run as a ``commit-msg`` pre-commit hook (the message file is the argument) or
over an explicit subject with ``--subject``. CI uses the second form to check
a pull request title, since a squash merge derives the commit subject from it.

Merge, revert, and fixup subjects pass unchanged: git writes them, not the
author.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

TYPES = ("feat", "fix", "docs", "test", "refactor", "perf", "build", "ci", "chore")
SCOPES = ("compression", "train", "data", "harness", "campaign", "serve", "core", "design", "docs", "ci")
MAX_SUBJECT = 72

#: type(scope)!: summary — the scope and the breaking-change marker are optional.
SUBJECT = re.compile(r"^(?P<type>[a-z]+)" r"(?:\((?P<scope>[a-z0-9-]+)\))?" r"(?P<breaking>!)?" r": (?P<summary>.+)$")

#: Subjects git generates on the author's behalf.
GENERATED = ("Merge ", "Revert ", "fixup! ", "squash! ")


def check(subject: str, body: str = "") -> list[str]:
    """Return the rule violations in ``subject``; an empty list means it passes."""
    if subject.startswith(GENERATED):
        return []

    match = SUBJECT.match(subject)
    if match is None:
        return [
            f"subject does not match '<type>(<scope>): <summary>': {subject!r}",
            f"  valid types: {', '.join(TYPES)}",
        ]

    problems = []
    kind, scope, summary = match["type"], match["scope"], match["summary"]

    if kind not in TYPES:
        problems.append(f"unknown type {kind!r}; valid types: {', '.join(TYPES)}")
    if scope is not None and scope not in SCOPES:
        problems.append(f"unknown scope {scope!r}; valid scopes: {', '.join(SCOPES)}")
    if len(subject) > MAX_SUBJECT:
        problems.append(f"subject is {len(subject)} characters; the limit is {MAX_SUBJECT}")
    if summary.endswith("."):
        problems.append("subject must not end with a period")
    if summary[:1].isupper():
        problems.append(f"summary starts with a capital: {summary!r}")
    # A breaking change must be visible in `git log --oneline` *and* explained
    # in the body, so a reader who sees only one of the two still finds the other.
    if match["breaking"] and "BREAKING CHANGE:" not in body:
        problems.append("a '!' subject requires a 'BREAKING CHANGE:' footer in the body")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("message_file", nargs="?", type=Path, help="path to the commit message file")
    parser.add_argument("--subject", help="check this subject instead of a message file")
    args = parser.parse_args()

    if args.subject is not None:
        subject, body = args.subject, ""
    elif args.message_file is not None:
        # Comment lines are git's own template text, not the author's message.
        lines = [
            line for line in args.message_file.read_text(encoding="utf-8").splitlines() if not line.startswith("#")
        ]
        subject = lines[0].strip() if lines else ""
        body = "\n".join(lines[1:])
    else:
        parser.error("pass a message file or --subject")

    if not subject:
        print("commit message is empty", file=sys.stderr)
        return 1

    problems = check(subject, body)
    if problems:
        print(f"invalid commit subject: {subject}\n", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print("\nSee the commit message rules in CONTRIBUTING.md.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
