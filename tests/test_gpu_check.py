"""The GPU check's path list and scripts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "gpu"))

from needs_gpu import PATTERNS, first_gpu_path, patterns  # noqa: E402


@pytest.mark.unit
@pytest.mark.parametrize(
    ("changed", "hit"),
    [
        (["README.md", "honing/backends/jev/nanojev_tree.py"], "honing/backends/jev/nanojev_tree.py"),
        (["honing/backends/jev/parity/sub/deep.py"], "honing/backends/jev/parity/sub/deep.py"),
        (["configs/e2e/tank_direction_edit_reef.json"], "configs/e2e/tank_direction_edit_reef.json"),
        (["scripts/gpu/check.sh"], "scripts/gpu/check.sh"),
        (["honing/core/candidate.py", "docs/x.md", "configs/e2e/tank_dry_run.json"], None),
        (["honing/backends/jevx.py"], None),
        ([], None),
    ],
)
def test_a_change_needs_the_gpu_check_only_under_the_listed_paths(changed: list[str], hit: str | None) -> None:
    assert first_gpu_path(changed, patterns(PATTERNS.read_text())) == hit


@pytest.mark.unit
def test_every_listed_pattern_matches_a_tracked_file(repo_root: Path) -> None:
    tracked = subprocess.run(["git", "ls-files"], cwd=repo_root, capture_output=True, text=True, check=True)
    files = tracked.stdout.split()
    for glob in patterns(PATTERNS.read_text()):
        assert first_gpu_path(files, [glob]), f"{glob} matches no tracked file"


@pytest.mark.unit
@pytest.mark.parametrize("script", ["setup.sh", "check.sh"])
def test_the_scripts_parse(script: str) -> None:
    subprocess.run(["bash", "-n", str(ROOT / "scripts" / "gpu" / script)], check=True)
