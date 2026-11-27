"""The release workflow (CONTRIBUTING.md, Releases): a version tag builds the package and publishes it."""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")


@pytest.mark.unit
def test_a_version_tag_builds_the_package_and_publishes_a_github_release(repo_root: Path) -> None:
    workflow = yaml.safe_load((repo_root / ".github" / "workflows" / "release.yml").read_text())
    assert workflow[True] == {"push": {"tags": ["v[0-9]*"]}}  # YAML reads the key `on` as True
    assert workflow["permissions"] == {"contents": "read"}
    build, release = workflow["jobs"]["build"], workflow["jobs"]["github-release"]
    checkout = next(step for step in build["steps"] if step.get("uses", "").startswith("actions/checkout@"))
    assert checkout["with"]["fetch-depth"] == 0  # setuptools-scm needs the tags
    assert any("${GITHUB_REF_NAME#v}" in step.get("run", "") for step in build["steps"])
    assert release["needs"] == "build" and release["permissions"] == {"contents": "write"}
    (create,) = [step["run"] for step in release["steps"] if "gh release create" in step.get("run", "")]
    assert "--verify-tag" in create and "--generate-notes" in create and "dist/*" in create
