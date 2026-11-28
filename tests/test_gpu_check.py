"""The GPU check: every pull request waits for it, and its scripts parse."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.unit
def test_every_pull_request_waits_for_the_gpu_check() -> None:
    """The status job sets `gpu` to pending on every push and never to success: only the runner passes it."""
    workflow = (ROOT / ".github" / "workflows" / "gpu.yml").read_text()
    status = workflow.split("\n  status:\n", 1)[1].split("\n  queue:\n", 1)[0]
    assert "state=pending" in status and "context=gpu" in status
    assert "success" not in status


@pytest.mark.functional
@pytest.mark.parametrize("script", ["setup.sh", "check.sh"])
def test_the_scripts_parse(script: str) -> None:
    subprocess.run(["bash", "-n", str(ROOT / "scripts" / "gpu" / script)], check=True)
