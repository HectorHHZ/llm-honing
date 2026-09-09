"""Black, isort and ruff are pinned in two files, so the two must agree.

``pyproject.toml``'s ``dev`` extra is what a developer installs; the
``.pre-commit-config.yaml`` revs are what ``pre-commit run --all-files``
executes, and CI enforces formatting through the second. When a dependency
update moves one copy and not the other, CI formats with one version while the
developer's machine formats with another, and the diff that results looks like
a mistake by whoever touched the file last.

Dependabot updates the two through separate ecosystems (``pip`` and
``pre-commit``), so the drift is a matter of when, not whether. These tests make
it fail loudly at the point it is introduced.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

#: The pinned tool, and the pre-commit repository that carries the same version.
PINNED_TOOLS = {
    "black": "https://github.com/psf/black",
    "isort": "https://github.com/pycqa/isort",
    "ruff": "https://github.com/astral-sh/ruff-pre-commit",
}


def pyproject_pins(repo_root: Path) -> dict[str, str]:
    """The ``name==version`` pins in pyproject's ``dev`` extra, by tool name."""
    text = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    return {
        match["name"]: match["version"] for match in re.finditer(r'"(?P<name>[a-z0-9-]+)==(?P<version>[^"]+)"', text)
    }


def pre_commit_revs(repo_root: Path) -> dict[str, str]:
    """The ``rev`` of each pre-commit repository, by repository URL.

    A leading ``v`` is stripped: ruff tags its hook releases ``v0.16.6`` while
    its PyPI release is ``0.16.6``, and the two name the same version.
    """
    text = (repo_root / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    pattern = re.compile(r"-\s+repo:\s*(?P<repo>\S+)\s*\n\s+rev:\s*(?P<rev>\S+)", re.MULTILINE)
    return {match["repo"]: match["rev"].lstrip("v") for match in pattern.finditer(text)}


@pytest.mark.unit
@pytest.mark.parametrize("tool", sorted(PINNED_TOOLS))
def test_a_pinned_tool_has_the_same_version_in_both_files(repo_root: Path, tool: str) -> None:
    declared = pyproject_pins(repo_root)
    hooks = pre_commit_revs(repo_root)

    assert tool in declared, f"{tool} is not pinned in pyproject.toml's dev extra"
    repository = PINNED_TOOLS[tool]
    assert repository in hooks, f"{repository} is not a pre-commit repository"

    assert declared[tool] == hooks[repository], (
        f"{tool} is pinned to {declared[tool]} in pyproject.toml but "
        f"{hooks[repository]} in .pre-commit-config.yaml; update both"
    )


@pytest.mark.unit
def test_every_pinned_formatter_is_covered(repo_root: Path) -> None:
    """A newly pinned formatter must be added to PINNED_TOOLS, not silently skipped."""
    formatters = {"black", "isort", "ruff"}
    declared = set(pyproject_pins(repo_root))
    uncovered = (declared & formatters) - set(PINNED_TOOLS)
    assert not uncovered, f"pinned but unchecked: {sorted(uncovered)}"


@pytest.mark.unit
def test_dependabot_updates_the_pre_commit_ecosystem(repo_root: Path) -> None:
    """Both copies need an update channel, or one of them silently goes stale."""
    dependabot = (repo_root / ".github/dependabot.yml").read_text(encoding="utf-8")
    for ecosystem in ("pip", "pre-commit", "github-actions"):
        assert f"package-ecosystem: {ecosystem}" in dependabot, f"{ecosystem} updates are not configured"
