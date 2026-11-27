"""The merge rule: the review status, and the ruleset that requires it with every CI job."""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".github" / "scripts"))

from review_gate import Event, changes, decide

T0 = dt.datetime(2026, 11, 1, tzinfo=dt.timezone.utc)


def at(days: float) -> dt.datetime:
    return T0 + dt.timedelta(days=days)


def ask(days: float) -> Event:
    return Event("review_requested", "author", at(days))


def review(who: str, state: str, days: float) -> Event:
    return Event("reviewed", who, at(days), state)


@pytest.mark.unit
def test_one_approval_passes() -> None:
    assert decide("author", [ask(0), review("a", "approved", 1)], at(1)) == ("success", "Approved by a")
    events = [ask(0), review("a", "approved", 1), review("b", "approved", 2)]
    assert decide("author", events, at(2)) == ("success", "Approved by a and 1 more")


@pytest.mark.unit
@pytest.mark.parametrize(
    "events",
    [
        [ask(0)],
        [review("author", "approved", 2)],  # the author's own does not count
        [review("ci[bot]", "approved", 2)],  # nor a bot's
        [review("a", "approved", 1), review("a", "changes_requested", 2)],  # a reviewer's latest verdict counts
        [review("a", "approved", 1), review("c", "changes_requested", 3)],
        [review("a", "approved", 1), review("a", "dismissed", 3)],  # an admin dismissed it
    ],
)
def test_no_approval_or_a_change_request_do_not(events: list[Event]) -> None:
    assert decide("author", events, at(3))[0] == "pending"


@pytest.mark.unit
def test_an_approval_counts_while_the_diff_is_the_one_approved() -> None:
    def approve(who: str, days: float, commit: str) -> Event:
        return Event("reviewed", who, at(days), "approved", commit)

    events = [approve("a", 1, "old")]
    assert decide("author", events, at(2), current=lambda commit: True)[0] == "success"  # main merged in
    state, description = decide("author", events, at(2), current=lambda commit: commit == "head")
    assert state == "pending" and description.startswith("0/1 approvals")  # a push changed the diff


@pytest.mark.unit
def test_a_merge_from_main_leaves_the_diff_the_same() -> None:
    before = [{"filename": "a.py", "status": "modified", "patch": "@@ -10,3 +10,3 @@ def f\n x = 1\n-y = 2\n+y = 3"}]
    after = [{"filename": "a.py", "status": "modified", "patch": "@@ -40,3 +40,3 @@ def f\n x = 0\n-y = 2\n+y = 3"}]
    pushed = [{"filename": "a.py", "status": "modified", "patch": "@@ -40,3 +40,3 @@ def f\n x = 0\n-y = 2\n+y = 4"}]
    binary = [{"filename": "w.bin", "status": "modified", "sha": "abc"}]
    assert changes(before) == changes(after) != changes(pushed)
    assert changes(binary) != changes([{**binary[0], "sha": "abd"}])


@pytest.mark.unit
def test_a_later_approval_replaces_the_same_reviewers_change_request() -> None:
    events = [review("a", "changes_requested", 1), review("a", "approved", 2), review("b", "approved", 3)]
    assert decide("author", events, at(3))[0] == "success"


@pytest.mark.unit
def test_a_comment_only_review_keeps_the_reviewers_earlier_verdict() -> None:
    events = [review("a", "approved", 1), review("a", "commented", 2), review("b", "approved", 3)]
    assert decide("author", events, at(3))[0] == "success"


@pytest.mark.unit
def test_five_silent_days_after_the_ask_pass() -> None:
    assert decide("author", [ask(0)], at(5) - dt.timedelta(seconds=1))[0] == "pending"
    assert decide("author", [ask(0)], at(5)) == ("success", "No reply from a reviewer in the 5 days since 2026-11-01")


@pytest.mark.unit
def test_the_clock_starts_at_the_first_ask_after_the_reviewers_last_word() -> None:
    events = [ask(0), review("a", "changes_requested", 1), Event("commented", "author", at(2)), ask(4)]
    assert decide("author", events, at(6))[0] == "pending"
    assert decide("author", events, at(7))[0] == "success"


@pytest.mark.unit
@pytest.mark.parametrize("reply", [review("a", "commented", 3), Event("commented", "someone", at(3))])
def test_any_reply_stops_the_clock_until_the_author_asks_again(reply: Event) -> None:
    state, description = decide("author", [ask(0), reply], at(30))
    assert state == "pending" and "asking again" in description
    assert decide("author", [ask(0), reply, Event("commented", "author", at(31))], at(36))[0] == "success"


@pytest.mark.unit
def test_a_bot_comment_does_not_stop_the_clock() -> None:
    assert decide("author", [ask(0), Event("commented", "codecov[bot]", at(1))], at(5))[0] == "success"


@pytest.mark.unit
def test_without_an_ask_nothing_passes_on_silence() -> None:
    assert decide("author", [], at(100))[0] == "pending"


@pytest.mark.unit
def test_every_description_fits_a_commit_status() -> None:
    longest = [f"{i}".rjust(39, "x") for i in range(5)]  # a GitHub login has at most 39 characters
    cases = [
        [review(who, "approved", 1) for who in longest],
        [review(who, "changes_requested", 1) for who in longest],
        [*(review(who, "changes_requested", 1) for who in longest), Event("commented", "author", at(2))],
        [ask(0)],
    ]
    for events in cases:
        assert len(decide("author", events, at(3))[1]) <= 140


def ci_jobs(workflow: str) -> set[str]:
    """The job names of a workflow: the keys indented two spaces under ``jobs:``."""
    body = workflow.split("\njobs:\n", 1)[1]
    keys = [line for line in body.splitlines() if line.startswith("  ") and line[2] != " " and line.endswith(":")]
    return {line[2:-1] for line in keys}


@pytest.mark.unit
def test_the_ruleset_requires_the_review_status_and_every_ci_job(repo_root: Path) -> None:
    ruleset = json.loads((repo_root / ".github" / "rulesets" / "main.json").read_text())
    rules = {rule["type"]: rule.get("parameters", {}) for rule in ruleset["rules"]}
    # Approvals are counted by the review status, which also knows the five-day rule. A native count would block it.
    assert rules["pull_request"]["required_approving_review_count"] == 0
    # The review status drops an approval when the diff changes. A native dismissal would drop it on every
    # update from main, which the strict up-to-date rule makes before each merge.
    assert not rules["pull_request"]["dismiss_stale_reviews_on_push"] and ruleset["bypass_actors"] == []
    assert rules["required_status_checks"]["strict_required_status_checks_policy"]
    required = {check["context"] for check in rules["required_status_checks"]["required_status_checks"]}
    assert {"review", "gpu"} <= required
    for job in ci_jobs((repo_root / ".github" / "workflows" / "ci.yml").read_text()):
        matrix = any(name.startswith(f"{job} (") for name in required)  # such as "test (3.12)"
        assert job in required or matrix, f"ci.yml's {job} is not required"
