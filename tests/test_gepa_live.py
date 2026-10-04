"""The live GEPA run (#100) on a real provider: ``configs/e2e/gepa_live.json`` as committed, and its faults.

Marked ``live``: each test calls the provider and costs money, so it is skipped
unless the config's key variable is set, and CI never sets it. A happy run is
about 200 agent episodes and 10 reflection calls; ``scripts/e2e/README.md`` has
the cost. The fault runs stage each failure with a bad key or a timeout too
short to answer in; a real 429 cannot be staged, so the rate-limit paths are
covered against the stub in ``tests/test_gepa_live_offline.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("gepa", reason="requires the optional 'gepa' extra")
pytest.importorskip("litellm", reason="requires the optional 'litellm' extra")

from tests.test_gepa_live_offline import Run

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/e2e/gepa_live.json"
KEY_ENV = json.loads(CONFIG.read_text())["requires"]["env"][0]
BAD_KEY_ENV = "HONING_LIVE_BAD_KEY"
BAD_KEY = "sk-honing-not-a-real-key-0000000000"
HARNESS_MOVE = {"kind": "harness", "label": "pre_training"}

pytestmark = [
    pytest.mark.live,
    pytest.mark.slow,
    pytest.mark.skipif(not os.environ.get(KEY_ENV), reason=f"set {KEY_ENV} to call the provider (costs money)"),
]


def live_config(**change: Any) -> dict[str, Any]:
    config = json.loads(CONFIG.read_text())
    components = config["components"]
    if "reflection" in change:
        components["method"]["args"]["reflection"].update(change["reflection"])
    if "runner" in change:
        components["runner"]["args"].update(change["runner"])
    if "schedule" in change:
        config["schedule"] = change["schedule"]
    return config


def run_live(config: dict[str, Any], tmp_path: Path) -> Run:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    work_dir = tmp_path / "run"
    result = subprocess.run(
        [sys.executable, "scripts/e2e/gepa_live.py", "--config", str(path), "--work-dir", str(work_dir)],
        cwd=REPO,
        env={**os.environ, BAD_KEY_ENV: BAD_KEY},
        capture_output=True,
        text=True,
        timeout=3600,
    )
    print(result.stdout)  # the summary, for the PR
    return Run(result.returncode, result.stdout, result.stderr, work_dir)


def assert_no_key(run: Run) -> None:
    key = os.environ[KEY_ENV]
    assert run.checks["8 no key written"]["passed"], run.checks["8 no key written"]
    assert key not in run.stdout and key not in run.stderr
    assert BAD_KEY not in run.stdout and BAD_KEY not in run.stderr


def test_the_live_run_passes_every_check(tmp_path: Path) -> None:
    run = run_live(live_config(), tmp_path)

    failed = [check for check in run.summary["checks"] if not check["passed"]]
    assert run.code == 0 and not failed, failed
    assert_no_key(run)


def test_a_bad_reflection_key_is_a_logged_skip(tmp_path: Path) -> None:
    config = live_config(reflection={"api_key_env": BAD_KEY_ENV}, schedule=[HARNESS_MOVE] * 2)

    run = run_live(config, tmp_path)

    assert run.summary["aborted"] is None
    assert [record["op"] for record in run.records] == ["publish", "skip", "skip"]
    assert run.checks["7a failed reflections"]["passed"], run.checks["7a failed reflections"]
    assert run.log.count("(status 401)") == 2
    assert_no_key(run)


def test_a_reflection_timeout_is_a_logged_skip(tmp_path: Path) -> None:
    config = live_config(reflection={"timeout_seconds": 0.001, "max_attempts": 1}, schedule=[HARNESS_MOVE])

    run = run_live(config, tmp_path)

    assert [record["op"] for record in run.records] == ["publish", "skip"]
    assert run.checks["7a failed reflections"]["passed"], run.checks["7a failed reflections"]
    assert "GEPA reflection on" in run.log and "Timeout" in run.log
    assert_no_key(run)


@pytest.mark.parametrize(
    ("runner", "kind"),
    [
        ({"api_key_env": BAD_KEY_ENV}, "auth"),
        ({"agent_timeout_seconds": 0.001, "agent_max_attempts": 1}, "timeout"),
    ],
    ids=["bad-key", "timeout"],
)
def test_an_unavailable_policy_abandons_the_move(tmp_path: Path, runner: dict[str, Any], kind: str) -> None:
    run = run_live(live_config(runner=runner, schedule=[HARNESS_MOVE] * 2), tmp_path)

    assert [step["result"] for step in run.summary["steps"]] == ["abandoned"] * 2
    assert [record["op"] for record in run.records] == ["publish"]
    assert {fault["kind"] for fault in run.rows("faults")} == {kind}
    for name in ("5c every episode accounted for", "5d abandoned moves charge nothing", "7b abandoned moves"):
        assert run.checks[name]["passed"], run.checks[name]
    assert_no_key(run)
