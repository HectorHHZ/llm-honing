"""The pull request bots: area labels by path (.github/labeler.yml) and the needs-rebase label."""

from __future__ import annotations

import glob
import re
from pathlib import Path

import pytest

from tests.conftest import load_script

yaml = pytest.importorskip("yaml")
conflicts = load_script(".github/scripts/conflicts.py")


def labeler_rules(repo_root: Path) -> dict[str, list[str]]:
    config = yaml.safe_load((repo_root / ".github" / "labeler.yml").read_text())
    return {
        label: [g for rule in rules for entry in rule["changed-files"] for g in entry["any-glob-to-any-file"]]
        for label, rules in config.items()
    }


@pytest.mark.unit
def test_every_labeler_glob_matches_a_file(repo_root: Path) -> None:
    for label, globs in labeler_rules(repo_root).items():
        for pattern in globs:
            assert glob.glob(pattern, root_dir=repo_root, recursive=True), f"{label}: {pattern} matches no file"


@pytest.mark.unit
def test_the_labels_are_the_areas_contributing_lists_and_each_has_a_release_notes_section(repo_root: Path) -> None:
    areas = set(re.findall(r"^- `(area: [a-z]+)`", (repo_root / "CONTRIBUTING.md").read_text(), re.M))
    assert set(labeler_rules(repo_root)) == areas
    sections = yaml.safe_load((repo_root / ".github" / "release.yml").read_text())["changelog"]["categories"]
    assert {label for section in sections for label in section["labels"]} == areas | {"*"}
    assert sections[-1]["labels"] == ["*"]  # the catch-all goes last, or it would take every pull request


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mergeable", "labels", "told", "actions"),
    [
        ("CONFLICTING", [], False, ["label", "comment"]),
        ("CONFLICTING", ["needs-rebase"], True, []),  # labelled and told already
        ("CONFLICTING", [], True, ["label"]),  # conflicts again after a rebase: label, no second comment
        ("MERGEABLE", ["needs-rebase", "area: core"], True, ["unlabel"]),
        ("MERGEABLE", ["area: core"], False, []),
        ("UNKNOWN", ["needs-rebase"], True, []),  # GitHub is still computing: change nothing
    ],
)
def test_a_conflict_is_labelled_and_the_author_told_once(
    mergeable: str, labels: list[str], told: bool, actions: list[str]
) -> None:
    assert conflicts.decide(mergeable, labels, told) == actions


@pytest.mark.unit
def test_the_comment_asks_for_a_rebase_and_carries_its_mark() -> None:
    text = conflicts.message("lucas")
    assert text.startswith(conflicts.MARK) and "@lucas" in text and "rebase it onto `main`" in text


type_label = load_script(".github/scripts/type_label.py")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("title", "label"),
    [
        ("feat(data): add a Reef recipe", "type: feature"),
        ("feat!: drop the old plane", "type: feature"),
        ("fix(train): refuse a no-op grpo run", "type: bug"),
        ("docs: add AGENTS.md", "type: task"),
        ("ci(ci): pin every action", "type: task"),
        ("feature: not a Conventional Commits type", "type: task"),
        ("Add a gate", "type: task"),
    ],
)
def test_the_title_names_the_type_label(title: str, label: str) -> None:
    assert type_label.type_label(title) == label


@pytest.mark.unit
def test_a_retitled_pull_request_ends_with_one_type_label() -> None:
    assert type_label.changes("fix: x", ["type: feature", "area: core"]) == (["type: bug"], ["type: feature"])
    assert type_label.changes("fix: x", ["type: bug"]) == ([], [])
    assert set(type_label.LABELS) == {"type: feature", "type: bug", "type: task"}
