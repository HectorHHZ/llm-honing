"""Contract tests for the Conventional Commits check.

The rules these pin are the ones CONTRIBUTING.md states, so a change to either
side without the other fails here.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tests.conftest import load_script

check_commit_message = load_script(".github/scripts/check_commit_message.py")
check = check_commit_message.check


@pytest.mark.unit
@pytest.mark.parametrize(
    "subject",
    [
        "feat(compression): add expert-drop scoring over router activation counts",
        "fix(train): reject a sparse-mask selection spanning pruned experts",
        "feat(campaign): refuse a quantize step whose scheme differs from qat",
        "docs(design): record the co-adaptation invariant for harness releases",
        "chore: update the pinned pre-commit hooks",  # scope is optional
        "ci(ci): pin the checkout action by major version",
    ],
)
def test_a_conforming_subject_passes(subject: str) -> None:
    assert check(subject) == []


@pytest.mark.unit
def test_an_unknown_type_is_rejected() -> None:
    problems = check("feature(compression): add expert drop")
    assert any("unknown type" in problem for problem in problems)


@pytest.mark.unit
@pytest.mark.parametrize("scope", ["pruning", "prune"])
def test_an_unknown_scope_is_rejected(scope: str) -> None:
    # `prune` was retired when the plane became Compression; a stale habit must fail loudly.
    problems = check(f"feat({scope}): add expert drop")
    assert any("unknown scope" in problem for problem in problems)


@pytest.mark.unit
def test_a_subject_without_a_type_is_rejected() -> None:
    problems = check("add expert drop")
    assert any("does not match" in problem for problem in problems)


@pytest.mark.unit
def test_an_overlong_subject_is_rejected() -> None:
    subject = "feat(compression): " + "x" * check_commit_message.MAX_SUBJECT
    problems = check(subject)
    assert any("the limit is" in problem for problem in problems)


@pytest.mark.unit
def test_a_trailing_period_is_rejected() -> None:
    assert any("period" in problem for problem in check("feat(compression): add expert drop."))


@pytest.mark.unit
def test_a_capitalized_summary_is_rejected() -> None:
    assert any("capital" in problem for problem in check("feat(compression): Add expert drop"))


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
        'Revert "feat(compression): add expert drop"',
        "fixup! feat(compression): add expert drop",
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


Commit = check_commit_message.Commit
range_problems = check_commit_message.range_problems


@pytest.mark.unit
def test_every_commit_a_pull_request_brings_is_checked() -> None:
    good = Commit("a" * 40, "feat(core): add a gate", "", 1)
    bad = Commit("b" * 40, "Add a gate", "", 1)
    assert range_problems([good]) == []
    problems = range_problems([good, bad])
    assert problems[0].startswith("bbbbbbb 'Add a gate': subject does not match")
    assert all(problem.startswith("bbbbbbb ") for problem in problems)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("commit", "says"),
    [
        (Commit("c" * 40, "Merge branch 'main' into feat/x", "", 2), "rebase the branch onto main"),
        (Commit("d" * 40, "fixup! feat(core): add a gate", "", 1), "is a fixup commit: fold it in"),
        (Commit("e" * 40, "squash! feat(core): add a gate", "", 1), "is a squash commit: fold it in"),
    ],
)
def test_a_merge_or_an_unfolded_fixup_in_a_pull_request_fails(commit: object, says: str) -> None:
    (problem,) = range_problems([commit])
    assert says in problem


@pytest.mark.integration
def test_the_range_is_read_from_git_oldest_first(tmp_path: Path) -> None:
    def git(*args: str) -> str:
        env = {
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
        }
        return subprocess.run(["git", *args], cwd=tmp_path, capture_output=True, text=True, check=True, env=env).stdout

    git("init", "-q", "-b", "main")
    git("commit", "-q", "--allow-empty", "-m", "chore: start")
    git("checkout", "-q", "-b", "feature")
    git("commit", "-q", "--allow-empty", "-m", "feat(core): add a gate", "-m", "Why it matters.")
    git("commit", "-q", "--allow-empty", "-m", "fix(core): close the gate")
    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        commits = check_commit_message.read_commits("main..feature")
    finally:
        os.chdir(cwd)
    assert [(c.subject, c.body.strip(), c.parents) for c in commits] == [
        ("feat(core): add a gate", "Why it matters.", 1),
        ("fix(core): close the gate", "", 1),
    ]
