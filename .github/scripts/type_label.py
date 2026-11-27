"""Give a pull request the type label its Conventional Commits title names, and take off the other type labels.

``feat`` is a feature and ``fix`` a bug fix. Every other type (docs, test, ci, refactor, perf, build, chore) is a
task. The ``labeler`` workflow runs this from main's copy when a pull request opens or its title changes.
Usage: ``type_label.py`` with GH_TOKEN, GITHUB_REPOSITORY, NUMBER and TITLE set.
"""

from __future__ import annotations

import os
import re
import subprocess

TYPES = {"feat": "type: feature", "fix": "type: bug"}
TASK = "type: task"
LABELS = (*TYPES.values(), TASK)


def type_label(title: str) -> str:
    match = re.match(r"([a-z]+)[(!:]", title)
    return TYPES.get(match.group(1) if match else "", TASK)


def changes(title: str, labels: list[str]) -> tuple[list[str], list[str]]:
    """The type labels to add and to remove, so the pull request ends with exactly the one its title names."""
    want = type_label(title)
    return ([want] if want not in labels else []), [label for label in labels if label in LABELS and label != want]


def main() -> None:
    repo, number, title = os.environ["GITHUB_REPOSITORY"], os.environ["NUMBER"], os.environ["TITLE"]
    gh = ["gh", "pr", "edit", number, "-R", repo]
    labels = subprocess.run(
        ["gh", "pr", "view", number, "-R", repo, "--json", "labels", "--jq", ".labels[].name"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    add, remove = changes(title, labels)
    for label in add:
        subprocess.run([*gh, "--add-label", label], check=True)
    for label in remove:
        subprocess.run([*gh, "--remove-label", label], check=True)
    print(f"#{number}: add {add or '-'}, remove {remove or '-'}")


if __name__ == "__main__":
    main()
