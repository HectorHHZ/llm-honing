"""Contract tests for the Conventional Commits check.

The rules these pin are the ones CONTRIBUTING.md states, so a change to either
side without the other fails here.
"""

from __future__ import annotations

import pytest

from tests.conftest import load_script

check_commit_message = load_script(".github/scripts/check_commit_message.py")
check = check_commit_message.check


@pytest.mark.unit
@pytest.mark.parametrize(
    "subject",
    [
        "feat(prune): add expert-drop analysis over router activation counts",
        "fix(train): reject a sparse-mask selection spanning pruned experts",
        "docs(design): record the co-adaptation invariant for harness releases",
        "chore: update the pinned pre-commit hooks",  # scope is optional
        "ci(ci): pin the checkout action by major version",
    ],
)
def test_a_conforming_subject_passes(subject: str) -> None:
    assert check(subject) == []


@pytest.mark.unit
def test_an_unknown_type_is_rejected() -> None:
    problems = check("feature(prune): add expert drop")
    assert any("unknown type" in problem for problem in problems)


@pytest.mark.unit
def test_an_unknown_scope_is_rejected() -> None:
    problems = check("feat(pruning): add expert drop")
    assert any("unknown scope" in problem for problem in problems)


@pytest.mark.unit
def test_a_subject_without_a_type_is_rejected() -> None:
    problems = check("add expert drop")
    assert any("does not match" in problem for problem in problems)


@pytest.mark.unit
def test_an_overlong_subject_is_rejected() -> None:
    subject = "feat(prune): " + "x" * check_commit_message.MAX_SUBJECT
    problems = check(subject)
    assert any("the limit is" in problem for problem in problems)


@pytest.mark.unit
def test_a_trailing_period_is_rejected() -> None:
    assert any("period" in problem for problem in check("feat(prune): add expert drop."))


@pytest.mark.unit
def test_a_capitalized_summary_is_rejected() -> None:
    assert any("capital" in problem for problem in check("feat(prune): Add expert drop"))


@pytest.mark.unit
def test_a_breaking_change_needs_its_footer() -> None:
    subject = "feat(core)!: publish a release as a weights and harness triple"
    assert any("BREAKING CHANGE" in problem for problem in check(subject))
    assert check(subject, "BREAKING CHANGE: HoneRelease replaces ArtifactRef") == []


@pytest.mark.unit
@pytest.mark.parametrize(
    "subject",
    [
        "Merge branch 'main' into feat/expert-drop",
        'Revert "feat(prune): add expert drop"',
        "fixup! feat(prune): add expert drop",
    ],
)
def test_git_generated_subjects_pass_unchanged(subject: str) -> None:
    # git writes these on the author's behalf; holding them to the author's
    # rules would block an ordinary merge or revert.
    assert check(subject) == []


@pytest.mark.unit
def test_every_documented_type_and_scope_is_accepted(repo_root) -> None:
    """CONTRIBUTING.md's tables and this script must not drift apart."""
    contributing = (repo_root / "CONTRIBUTING.md").read_text(encoding="utf-8")
    for kind in check_commit_message.TYPES:
        assert f"| `{kind}` |" in contributing, f"type {kind!r} is not documented"
    for scope in check_commit_message.SCOPES:
        assert f"`{scope}`" in contributing, f"scope {scope!r} is not documented"
