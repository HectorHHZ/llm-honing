"""Set each open pull request's commit status ``review``, the merge rule CONTRIBUTING.md states.

A lead reviewer's own pull request passes: a lead decides it is ready. Anyone else's passes when a lead reviewer
approves it. Either way nobody's latest review may request changes on the current diff. A review
counts while the pull request's diff is the one reviewed: rebasing onto main keeps it, a push that changes the
diff drops it. The ``review`` workflow runs this on every review, comment and push, and every 15 minutes.
Usage: ``review_gate.py [--dry-run]`` with GH_TOKEN and GITHUB_REPOSITORY set.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass

#: The maintainers' lead reviewers, by GitHub login: Xuan, Hector, Lucas and Jay (Xinkai). Their own pull requests
#: need no other review, and everyone else's need one of theirs.
LEADS = ("Xuan-1998", "HectorHHZ", "Brocoliman", "jayzou3773")


@dataclass(frozen=True)
class Event:
    """One timeline entry: ``kind`` is reviewed, commented or review_requested, ``state`` a review's state and
    ``commit`` the commit it reviewed."""

    kind: str
    who: str
    when: dt.datetime
    state: str = ""
    commit: str = ""


def names(people: list[str]) -> str:
    """One login and how many more, so a description stays within a commit status's 140 characters."""
    return people[0] + (f" and {len(people) - 1} more" if len(people) > 1 else "")


def decide(
    author: str, events: Iterable[Event], current: Callable[[str], bool] = lambda commit: True
) -> tuple[str, str]:
    """The ``review`` status as (state, description) for a pull request by ``author``.

    ``current(commit)`` says whether the diff a review saw at ``commit`` is still the pull request's diff.
    """
    events = sorted((e for e in events if not e.who.endswith("[bot]")), key=lambda e: e.when)
    latest = {e.who: e for e in events if e.kind == "reviewed" and e.who != author and e.state != "commented"}
    blockers = sorted(who for who, e in latest.items() if e.state == "changes_requested" and current(e.commit))
    if blockers:
        return "pending", f"Changes requested by {names(blockers)}"
    if author in LEADS:
        return "success", "A lead reviewer's own pull request"
    approvals = sorted(
        who for who, e in latest.items() if who in LEADS and e.state == "approved" and current(e.commit)
    )
    if approvals:
        return "success", f"Approved by {names(approvals)}"
    return "pending", "Needs a lead reviewer's approval of the current diff"


def gh(*args: str) -> object:
    out = subprocess.run(["gh", *args], capture_output=True, text=True, check=True).stdout
    return json.loads(out) if out.strip() else None


def changes(files: list[dict[str, object]]) -> tuple[tuple[str, ...], ...]:
    """A diff as compare's ``files`` gives it, without line numbers or context, so a merge from main that
    leaves the pull request's own changes alone gives the same value. A file without a patch (binary, or too
    large) is named by its content."""
    return tuple(
        sorted(
            (
                str(f["filename"]),
                str(f["status"]),
                *(
                    [line for line in str(f["patch"]).splitlines() if line[:1] in "+-"]
                    if f.get("patch")
                    else [str(f.get("sha"))]
                ),
            )
            for f in files
        )
    )


def diff_of(repo: str, base: str, commit: str, seen: dict[str, object]) -> object:
    """The pull request's changes at ``commit``: from its merge base with ``base`` to ``commit``."""
    if commit not in seen:
        files = gh("api", f"repos/{repo}/compare/{base}...{commit}")["files"]  # type: ignore[index]
        seen[commit] = changes(files) if len(files) < 300 else commit  # compare lists at most 300 files
    return seen[commit]


def timeline(repo: str, number: int) -> list[Event]:
    pages = gh("api", f"repos/{repo}/issues/{number}/timeline", "--paginate", "--slurp")
    events = []
    for entry in (e for page in pages for e in page):  # type: ignore[union-attr]
        kind, when = entry.get("event"), entry.get("submitted_at") or entry.get("created_at")
        who = (entry.get("user") or entry.get("actor") or {}).get("login")
        if kind in ("reviewed", "commented", "review_requested") and when and who:
            parsed = dt.datetime.fromisoformat(when.replace("Z", "+00:00"))
            events.append(Event(kind, who, parsed, (entry.get("state") or "").lower(), entry.get("commit_id") or ""))
    return events


def main() -> None:
    repo, dry = os.environ["GITHUB_REPOSITORY"], "--dry-run" in sys.argv
    pulls = gh("api", f"repos/{repo}/pulls?state=open&per_page=100", "--paginate", "--slurp")
    seen: dict[str, object] = {}
    for pull in (p for page in pulls for p in page):  # type: ignore[union-attr]
        head, base = pull["head"]["sha"], pull["base"]["ref"]

        def current(commit: str, head: str = head, base: str = base) -> bool:
            return commit == head or (
                bool(commit) and diff_of(repo, base, commit, seen) == diff_of(repo, base, head, seen)
            )

        state, description = decide(pull["user"]["login"], timeline(repo, pull["number"]), current)
        print(f"#{pull['number']} {state}: {description}")
        if not dry:
            gh(
                "api",
                f"repos/{repo}/statuses/{pull['head']['sha']}",
                "-f",
                f"state={state}",
                "-f",
                "context=review",
                "-f",
                f"description={description[:140]}",
            )


if __name__ == "__main__":
    main()
