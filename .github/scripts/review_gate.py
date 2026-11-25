"""Set each open pull request's commit status ``review``, the merge rule CONTRIBUTING.md states.

Success when two people other than the author approve and nobody's latest review requests changes, or when
the author asked for review (a review request or a comment) and five days passed with no review or comment
from anyone else. Pending otherwise. The ``review`` workflow runs this on every review, comment and push, and
hourly so that the five days can run out. Usage: ``review_gate.py [--dry-run]`` with GH_TOKEN and
GITHUB_REPOSITORY set.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass

APPROVALS = 2
SILENCE = dt.timedelta(days=5)


@dataclass(frozen=True)
class Event:
    """One timeline entry: ``kind`` is reviewed, commented or review_requested, ``state`` a review's state."""

    kind: str
    who: str
    when: dt.datetime
    state: str = ""


def names(people: list[str]) -> str:
    """One login and how many more, so a description stays within a commit status's 140 characters."""
    return people[0] + (f" and {len(people) - 1} more" if len(people) > 1 else "")


def decide(author: str, events: Iterable[Event], now: dt.datetime) -> tuple[str, str]:
    """The ``review`` status as (state, description) for a pull request by ``author``."""
    events = sorted((e for e in events if not e.who.endswith("[bot]")), key=lambda e: e.when)
    latest = {e.who: e.state for e in events if e.kind == "reviewed" and e.who != author and e.state != "commented"}
    approvers = sorted(who for who, state in latest.items() if state == "approved")
    blockers = sorted(who for who, state in latest.items() if state == "changes_requested")
    if len(approvers) >= APPROVALS and not blockers:
        return "success", f"Approved by {names(approvers)}"
    heard = [e.when for e in events if e.who != author and e.kind in ("reviewed", "commented")]
    asks = [e.when for e in events if (e.kind == "review_requested" or (e.kind == "commented" and e.who == author))]
    asks = [when for when in asks if not heard or when > max(heard)]
    if asks and now - min(asks) >= SILENCE:
        return "success", f"No reply from a reviewer in the 5 days since {min(asks):%Y-%m-%d}"
    have = f"{len(approvers)}/{APPROVALS} approvals"
    if blockers:
        have += f", changes requested by {names(blockers)}"
    if not asks:
        return "pending", f"{have}. The author asking again starts the 5 days"
    until, left = min(asks) + SILENCE, SILENCE - (now - min(asks))
    return (
        "pending",
        f"{have}, or silence until {until:%Y-%m-%d %H:%M} UTC ({left.days}d {left.seconds // 3600}h left)",
    )


def gh(*args: str) -> object:
    out = subprocess.run(["gh", *args], capture_output=True, text=True, check=True).stdout
    return json.loads(out) if out.strip() else None


def timeline(repo: str, number: int) -> list[Event]:
    pages = gh("api", f"repos/{repo}/issues/{number}/timeline", "--paginate", "--slurp")
    events = []
    for entry in (e for page in pages for e in page):  # type: ignore[union-attr]
        kind, when = entry.get("event"), entry.get("submitted_at") or entry.get("created_at")
        who = (entry.get("user") or entry.get("actor") or {}).get("login")
        if kind in ("reviewed", "commented", "review_requested") and when and who:
            parsed = dt.datetime.fromisoformat(when.replace("Z", "+00:00"))
            events.append(Event(kind, who, parsed, (entry.get("state") or "").lower()))
    return events


def main() -> None:
    repo, dry = os.environ["GITHUB_REPOSITORY"], "--dry-run" in sys.argv
    now = dt.datetime.now(dt.timezone.utc)
    pulls = gh("api", f"repos/{repo}/pulls?state=open&per_page=100", "--paginate", "--slurp")
    for pull in (p for page in pulls for p in page):  # type: ignore[union-attr]
        state, description = decide(pull["user"]["login"], timeline(repo, pull["number"]), now)
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
