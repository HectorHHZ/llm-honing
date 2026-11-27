"""Label the open pull requests that conflict with main ``needs-rebase``, and tell the author once.

A pull request updates by rebasing onto main (CONTRIBUTING.md, Commit messages), so the comment says
that. The label comes off once the pull request merges cleanly again. Adapted from vLLM's mergify
conflict rule (.github/mergify.yml there), without a GitHub App. The ``conflicts`` workflow runs this on
every push to main and every hour. Usage: ``conflicts.py [--dry-run]`` with GH_TOKEN and
GITHUB_REPOSITORY set.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

LABEL = "needs-rebase"
MARK = "<!-- needs-rebase -->"


def decide(mergeable: str, labels: list[str], told: bool) -> list[str]:
    """What to do for one pull request: any of "label", "comment" and "unlabel", in that order.

    ``mergeable`` is GitHub's MERGEABLE, CONFLICTING or UNKNOWN (still computing, so nothing changes),
    and ``told`` says whether the author already has this bot's comment.
    """
    if mergeable == "CONFLICTING":
        return (["label"] if LABEL not in labels else []) + (["comment"] if not told else [])
    if mergeable == "MERGEABLE" and LABEL in labels:
        return ["unlabel"]
    return []


def message(author: str) -> str:
    return (
        f"{MARK}\n@{author} this conflicts with `main` now. Please rebase it onto `main` "
        "(`git rebase origin/main`, then `git push --force-with-lease`) rather than merging `main` into it."
    )


def gh(*args: str) -> str:
    return subprocess.run(["gh", *args], capture_output=True, text=True, check=True).stdout


def main() -> None:
    repo, dry = os.environ["GITHUB_REPOSITORY"], "--dry-run" in sys.argv
    fields = "number,author,mergeable,labels,comments,isDraft"
    for pr in json.loads(gh("pr", "list", "-R", repo, "--state", "open", "--limit", "200", "--json", fields)):
        labels = [label["name"] for label in pr["labels"]]
        told = any(MARK in comment["body"] for comment in pr["comments"])
        actions = decide(pr["mergeable"], labels, told)
        print(f"#{pr['number']} {pr['mergeable']}: {', '.join(actions) or 'nothing'}")
        if dry:
            continue
        number = str(pr["number"])
        for action in actions:
            if action == "label":
                gh("pr", "edit", number, "-R", repo, "--add-label", LABEL)
            elif action == "unlabel":
                gh("pr", "edit", number, "-R", repo, "--remove-label", LABEL)
            elif action == "comment":
                gh("pr", "comment", number, "-R", repo, "--body", message(pr["author"]["login"]))


if __name__ == "__main__":
    main()
