"""The merge rule: the review status, and the ruleset that requires it with every CI job and the GPU check."""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".github" / "scripts"))

from review_gate import LEADS, Event, changes, decide

LEAD, OTHER_LEAD = LEADS[0], LEADS[1]

T0 = dt.datetime(2026, 11, 1, tzinfo=dt.timezone.utc)


def review(who: str, state: str, days: float, commit: str = "") -> Event:
    return Event("reviewed", who, T0 + dt.timedelta(days=days), state, commit)


@pytest.mark.unit
def test_a_lead_reviewers_own_pull_request_passes_without_a_review() -> None:
    for lead in LEADS:
        assert decide(lead, []) == ("success", "A lead reviewer's own pull request")
    assert decide(LEAD, [review(OTHER_LEAD, "approved", 1)])[0] == "success"


@pytest.mark.unit
def test_the_lead_reviewers_are_xuan_hector_lucas_and_jay() -> None:
    assert set(LEADS) == {"Xuan-1998", "HectorHHZ", "Brocoliman", "jayzou3773"}


@pytest.mark.unit
@pytest.mark.parametrize(
    "events",
    [
        [],
        [review("a", "approved", 1)],  # anyone but a lead reviewer
        [review("ci[bot]", "approved", 1)],
        [review(LEAD, "approved", 1), review(LEAD, "changes_requested", 2)],  # the lead's latest verdict counts
        [review(LEAD, "approved", 1), review(LEAD, "dismissed", 2)],  # an admin dismissed it
        [review(LEAD, "approved", 1), review("a", "changes_requested", 2)],
    ],
)
def test_anyone_elses_pull_request_needs_a_lead_reviewers_approval(events: list[Event]) -> None:
    assert decide("leo", events)[0] == "pending"


@pytest.mark.unit
def test_any_lead_reviewers_approval_passes_anyone_elses_pull_request() -> None:
    for lead in LEADS:
        assert decide("leo", [review(lead, "approved", 1)]) == ("success", f"Approved by {lead}")
    both = [review(LEAD, "approved", 1), review(OTHER_LEAD, "approved", 2)]
    assert decide("leo", both)[1] == f"Approved by {min(LEAD, OTHER_LEAD)} and 1 more"
    # One lead's approval does not outweigh another lead's request for changes.
    split = [review(LEAD, "approved", 1), review(OTHER_LEAD, "changes_requested", 2)]
    assert decide("leo", split) == ("pending", f"Changes requested by {OTHER_LEAD}")
    later = [review(LEAD, "changes_requested", 1), review(LEAD, "approved", 2)]
    assert decide("leo", later)[0] == "success"
    kept = [review(LEAD, "approved", 1), review(LEAD, "commented", 2)]  # a comment keeps the verdict
    assert decide("leo", kept)[0] == "success"


@pytest.mark.unit
def test_a_review_counts_while_the_diff_is_the_one_reviewed() -> None:
    approved = [review(LEAD, "approved", 1, "old")]
    assert decide("leo", approved, current=lambda commit: True)[0] == "success"  # rebased onto main
    assert decide("leo", approved, current=lambda commit: commit == "head")[0] == "pending"  # the diff changed
    blocked = [review("a", "changes_requested", 1, "old")]
    assert decide(LEAD, blocked, current=lambda commit: True) == ("pending", "Changes requested by a")
    assert decide(LEAD, blocked, current=lambda commit: commit == "head")[0] == "success"  # answered by a push


@pytest.mark.unit
def test_a_rebase_onto_main_leaves_the_diff_the_same() -> None:
    before = [{"filename": "a.py", "status": "modified", "patch": "@@ -10,3 +10,3 @@ def f\n x = 1\n-y = 2\n+y = 3"}]
    after = [{"filename": "a.py", "status": "modified", "patch": "@@ -40,3 +40,3 @@ def f\n x = 0\n-y = 2\n+y = 3"}]
    pushed = [{"filename": "a.py", "status": "modified", "patch": "@@ -40,3 +40,3 @@ def f\n x = 0\n-y = 2\n+y = 4"}]
    binary = [{"filename": "w.bin", "status": "modified", "sha": "abc"}]
    assert changes(before) == changes(after) != changes(pushed)
    assert changes(binary) != changes([{**binary[0], "sha": "abd"}])


@pytest.mark.unit
def test_every_description_fits_a_commit_status() -> None:
    longest = [f"{i}".rjust(39, "x") for i in range(5)]  # a GitHub login has at most 39 characters
    for author in (LEAD, "y" * 39):
        for events in ([review(who, "changes_requested", 1) for who in longest], [review(LEAD, "approved", 1)], []):
            assert len(decide(author, events)[1]) <= 140


def ci_jobs(workflow: str) -> set[str]:
    """The job names of a workflow: the keys indented two spaces under ``jobs:``."""
    body = workflow.split("\njobs:\n", 1)[1]
    keys = [line for line in body.splitlines() if line.startswith("  ") and line[2] != " " and line.endswith(":")]
    return {line[2:-1] for line in keys}


@pytest.mark.unit
def test_the_ruleset_requires_the_review_status_and_every_ci_job(repo_root: Path) -> None:
    ruleset = json.loads((repo_root / ".github" / "rulesets" / "main.json").read_text())
    rules = {rule["type"]: rule.get("parameters", {}) for rule in ruleset["rules"]}
    # The review status knows the lead reviewers. A native approval count would block a lead's own pull requests.
    assert rules["pull_request"]["required_approving_review_count"] == 0
    # The review status drops an approval when the diff changes. A native dismissal would drop it on every
    # rebase onto main, which the strict up-to-date rule makes before each merge.
    assert not rules["pull_request"]["dismiss_stale_reviews_on_push"] and ruleset["bypass_actors"] == []
    assert rules["required_status_checks"]["strict_required_status_checks_policy"]
    required = {check["context"] for check in rules["required_status_checks"]["required_status_checks"]}
    assert {"review", "gpu"} <= required
    for job in ci_jobs((repo_root / ".github" / "workflows" / "ci.yml").read_text()):
        matrix = any(name.startswith(f"{job} (") for name in required)  # such as "test (3.12)"
        assert job in required or matrix, f"ci.yml's {job} is not required"
    # Branches are rebased onto main to update them, and land on main with a merge commit.
    assert rules["pull_request"]["allowed_merge_methods"] == ["merge"]
