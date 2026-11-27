#!/usr/bin/env python3
"""Check a commit subject against the Conventional Commits rules in CONTRIBUTING.md.

Run as a ``commit-msg`` pre-commit hook (the message file is the argument),
over an explicit subject with ``--subject``, or over every commit of a pull
request with ``--range BASE..HEAD``. CI checks the title and the range: a pull
request lands with a merge commit, so each of its commits reaches main.

Merge, revert, and fixup subjects pass unchanged as a single subject: git writes
them, not the author. In a range, a merge commit fails (a branch is rebased onto
main, never merged with it) and so does a fixup or squash commit (fold it in
before the pull request merges).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

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


class Commit(NamedTuple):
    sha: str
    subject: str
    body: str
    parents: int


def range_problems(commits: list[Commit]) -> list[str]:
    """The problems of every commit a pull request brings to main, each line naming its commit."""
    problems = []
    for commit in commits:
        name = f"{commit.sha[:7]} {commit.subject!r}"
        if commit.parents > 1:
            problems.append(f"{name} is a merge commit: rebase the branch onto main instead of merging main into it")
        elif commit.subject.startswith(("fixup! ", "squash! ")):
            problems.append(f"{name} is a {commit.subject.split('!')[0]} commit: fold it in before merging")
        else:
            problems += [f"{name}: {problem}" for problem in check(commit.subject, commit.body)]
    return problems


def read_commits(revision_range: str) -> list[Commit]:
    """The commits in ``revision_range``, oldest first, as git log gives them."""
    out = subprocess.run(
        ["git", "log", "--reverse", "--format=%H%x00%P%x00%s%x00%b%x1e", revision_range],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    commits = []
    for record in out.split("\x1e"):
        if record.strip():
            sha, parents, subject, body = record.lstrip("\n").split("\x00")
            commits.append(Commit(sha, subject, body, len(parents.split())))
    return commits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("message_file", nargs="?", type=Path, help="path to the commit message file")
    parser.add_argument("--subject", help="check this subject instead of a message file")
    parser.add_argument("--range", dest="revision_range", help="check every commit in BASE..HEAD")
    args = parser.parse_args()

    if args.revision_range is not None:
        problems = range_problems(read_commits(args.revision_range))
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        if problems:
            print("\nSee the commit message rules in CONTRIBUTING.md.", file=sys.stderr)
        return 1 if problems else 0

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
