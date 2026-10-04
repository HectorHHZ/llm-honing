"""The live GEPA run (#100) against a loopback OpenAI-compatible stub: the whole script, its checks, and its faults.

``scripts/e2e/gepa_live.py`` runs with ``configs/e2e/gepa_live.json`` changed in
one place only: both models point at :class:`~tests.fakes.openai_stub.OpenAIStub`.
So CI runs the real runner, agent, grader, chat client and checks with no key and
no request leaving the machine. The stub's URL picks its behaviour, which is how
a rate limit, a bad key, a timeout, or an outage in the middle of a move is
staged for either model. ``tests/test_gepa_live.py`` runs the same script on a
real provider.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("gepa", reason="requires the optional 'gepa' extra")
pytest.importorskip("litellm", reason="requires the optional 'litellm' extra")

from tests.fakes.openai_stub import OpenAIStub

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/e2e/gepa_live.json"
KEY_ENV = "HONING_STUB_KEY"
KEY = "sk-stub-0123456789abcdef"
HARNESS_MOVE = {"kind": "harness", "label": "pre_training"}
STUB_MODEL = {"provider": "openai", "name": "stub-model"}


@pytest.fixture
def stub() -> Iterator[OpenAIStub]:
    with OpenAIStub() as server:
        yield server


def offline_config(
    stub: OpenAIStub,
    *,
    policy: str = "ok",
    reflection: str = "ok",
    schedule: list[dict[str, Any]] | None = None,
    reflection_settings: dict[str, Any] | None = None,
    runner_settings: dict[str, Any] | None = None,
    max_consecutive_faults: int = 3,
) -> dict[str, Any]:
    config = json.loads(CONFIG.read_text())
    config["requires"]["env"] = [KEY_ENV]
    config["model"] = STUB_MODEL  # the one pin both models reference
    components = config["components"]
    components["endpoints"]["args"]["base_url"] = stub.url(policy)
    components["runner"]["args"].update(
        {"api_key_env": KEY_ENV, "agent_timeout_seconds": 10, "agent_max_attempts": 2, **(runner_settings or {})}
    )
    components["method"]["args"]["reflection"] = {
        **components["method"]["args"]["reflection"],
        "api_base": stub.url(reflection),
        "api_key_env": KEY_ENV,
        "timeout_seconds": 10,
        "max_attempts": 2,
        "retry_backoff_seconds": 0.01,
        **(reflection_settings or {}),
    }
    config["live"]["max_consecutive_faults"] = max_consecutive_faults
    if schedule is not None:
        config["schedule"] = schedule
    return config


@dataclass(frozen=True)
class Run:
    code: int
    stdout: str
    stderr: str
    work_dir: Path

    @property
    def summary(self) -> dict[str, Any]:
        return json.loads((self.work_dir / "summary.json").read_text())  # type: ignore[no-any-return]

    @property
    def checks(self) -> dict[str, dict[str, Any]]:
        return {check["name"]: check for check in self.summary["checks"]}

    def rows(self, name: str) -> list[dict[str, Any]]:
        text = (self.work_dir / f"{name}.jsonl").read_text()
        return [json.loads(line) for line in text.split("\n") if line.strip()]

    @property
    def records(self) -> list[dict[str, Any]]:
        return self.rows("releases/commits")

    @property
    def log(self) -> str:
        return (self.work_dir / "run.log").read_text()


def run_script(config: dict[str, Any], tmp_path: Path) -> Run:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    work_dir = tmp_path / "run"
    result = subprocess.run(
        [sys.executable, "scripts/e2e/gepa_live.py", "--config", str(path), "--work-dir", str(work_dir)],
        cwd=REPO,
        env={**os.environ, KEY_ENV: KEY},
        capture_output=True,
        text=True,
        timeout=600,
    )
    return Run(result.returncode, result.stdout, result.stderr, work_dir)


def assert_no_key(run: Run) -> None:
    assert run.checks["8 no key written"]["passed"], run.checks["8 no key written"]
    assert KEY not in run.stdout and KEY not in run.stderr


# ---------------------------------------------------------------- the whole run


@pytest.mark.integration
@pytest.mark.slow
def test_the_run_passes_every_check_against_the_stub(stub: OpenAIStub, tmp_path: Path) -> None:
    run = run_script(offline_config(stub), tmp_path)

    failed = [check for check in run.summary["checks"] if not check["passed"]]
    assert not failed, failed
    assert run.code == 0, run.stdout + run.stderr
    ops = [record["op"] for record in run.records]
    assert "reject" in ops and "publish" in ops[1:] and "rollback" in ops
    # Harness moves until the first publish, one more, then a rollback to that publish.
    steps = run.summary["steps"]
    rollback = next(step for step in steps if step["move"] == "rollback")
    first = next(step for step in steps if step["result"] == "published")
    assert rollback["index"] == first["index"] + 2 and rollback["to"] == first["release"]
    assert rollback["exercised"]
    usage = run.summary["usage"]
    assert usage["policy"]["episodes"] == usage["policy"]["calls"] == len(run.rows("episodes"))
    assert usage["reflection"]["calls"] == len(run.rows("reflections")) > 0
    assert usage["policy"]["prompt_tokens"] == 100 * usage["policy"]["calls"]  # the stub's usage, summed
    assert run.summary["model"] == STUB_MODEL
    # Both models used the one pin: the agent sent its name, litellm sent the same name.
    assert {model for _, model in stub.calls} == {"stub-model"}
    assert_no_key(run)


# ---------------------------------------------------------------- a failed reflection call


@pytest.mark.integration
@pytest.mark.parametrize(
    ("mode", "settings", "error"),
    [
        ("401", {}, "AuthenticationError (status 401)"),
        ("429", {}, "RateLimitError (status 429)"),
        ("slow", {"timeout_seconds": 0.5, "max_attempts": 1}, "Timeout"),
    ],
    ids=["bad-key", "rate-limit", "timeout"],
)
def test_a_failed_reflection_is_a_logged_skip_that_charges_only_the_parent(
    stub: OpenAIStub, tmp_path: Path, mode: str, settings: dict[str, Any], error: str
) -> None:
    config = offline_config(stub, reflection=mode, reflection_settings=settings, schedule=[HARNESS_MOVE] * 2)

    run = run_script(config, tmp_path)

    assert run.code == 1  # the gate never ran, so the checks that need it are not exercised
    assert run.summary["aborted"] is None
    assert [record["op"] for record in run.records] == ["publish", "skip", "skip"]
    assert [record["episodes_spent"] for record in run.records[1:]] == [3, 3]
    assert {row["phase"] for row in run.rows("episodes")} == {"parent"}
    assert run.checks["7a failed reflections"]["passed"]
    assert run.checks["7a failed reflections"]["detail"].startswith("2 failed reflections")
    assert run.checks["5c every episode accounted for"]["passed"]
    lines = [line for line in run.log.split("\n") if "GEPA reflection on" in line]
    assert len(lines) == 2 and all("WARNING" in line and error in line for line in lines)
    assert "Traceback" not in run.log
    assert_no_key(run)  # the stub echoes the key in its error body


# ---------------------------------------------------------------- an unavailable policy


@pytest.mark.integration
@pytest.mark.parametrize(
    ("mode", "settings", "kind"),
    [
        ("401", {}, "auth"),
        ("429", {}, "rate_limit"),
        ("slow", {"agent_timeout_seconds": 0.5, "agent_max_attempts": 1}, "timeout"),
    ],
    ids=["bad-key", "rate-limit", "timeout"],
)
def test_an_unavailable_policy_abandons_moves_and_stops_after_three(
    stub: OpenAIStub, tmp_path: Path, mode: str, settings: dict[str, Any], kind: str
) -> None:
    run = run_script(offline_config(stub, policy=mode, runner_settings=settings), tmp_path)

    for name in ("5c every episode accounted for", "5d abandoned moves charge nothing", "7b abandoned moves"):
        assert run.checks[name]["passed"], run.checks[name]
    assert run.code == 3
    assert run.summary["aborted"].startswith("3 moves in a row abandoned")
    # Stopped by the count of abandoned moves, three of the schedule's thirteen; the budget never moved.
    assert [step["result"] for step in run.summary["steps"]] == ["abandoned"] * 3
    assert [record["op"] for record in run.records] == ["publish"]
    faults = run.rows("faults")
    assert [(fault["move"], fault["kind"], fault["phase"]) for fault in faults] == [
        (move, kind, "parent") for move in (1, 2, 3)
    ]
    assert sum(fault["episodes_started"] for fault in faults) == len(run.rows("episodes"))
    assert run.log.count("abandoned: policy unavailable") == 3
    assert run.summary["usage"]["abandoned_moves"]["calls"] == run.summary["usage"]["policy"]["calls"] > 0
    assert_no_key(run)


@pytest.mark.integration
def test_a_bad_policy_key_ends_by_the_schedule_when_faults_are_not_capped(stub: OpenAIStub, tmp_path: Path) -> None:
    config = offline_config(stub, policy="401", schedule=[HARNESS_MOVE] * 4, max_consecutive_faults=100)

    run = run_script(config, tmp_path)

    assert run.code == 1 and run.summary["aborted"] is None
    assert [step["result"] for step in run.summary["steps"]] == ["abandoned"] * 4
    # Three agents per minibatch. A 401 is not retried, so an agent makes at most one call, and the
    # first failure cancels its siblings: one killed before it called has no call at all.
    policy = run.summary["usage"]["policy"]
    assert policy["episodes"] == 12 and len(run.rows("episodes")) == 12
    assert 4 <= policy["calls"] == policy["failed_calls"] <= 12
    assert (policy["prompt_tokens"], policy["completion_tokens"]) == (0, 0)


@pytest.mark.integration
def test_an_outage_inside_a_move_abandons_only_that_move(stub: OpenAIStub, tmp_path: Path) -> None:
    # Policy calls 3 to 5 fail: move 1's child minibatch, after its parent ran and GEPA reflected.
    config = offline_config(
        stub, policy="window-3-6", runner_settings={"agent_max_attempts": 1}, schedule=[HARNESS_MOVE] * 3
    )

    run = run_script(config, tmp_path)

    # The next move starts from the state move 1 started from: nothing was charged for it.
    assert run.checks["5d abandoned moves charge nothing"]["passed"], run.checks["5d abandoned moves charge nothing"]
    assert run.checks["5d abandoned moves charge nothing"]["detail"].startswith("1 abandoned moves")
    steps = [step["result"] for step in run.summary["steps"]]
    assert steps[0] == "abandoned" and "abandoned" not in steps[1:]
    [fault] = run.rows("faults")
    assert (fault["move"], fault["phase"], fault["episodes_started"]) == (1, "child", 6)
    assert [row["move"] for row in run.rows("reflections")][:1] == [1]
    for name in ("5a metric calls per move", "5b metric calls on the final path", "5c every episode accounted for"):
        assert run.checks[name]["passed"], run.checks[name]
    assert_no_key(run)


@pytest.mark.integration
def test_moves_until_a_publish_stop_at_their_cap_and_leave_no_rollback_target(
    stub: OpenAIStub, tmp_path: Path
) -> None:
    # A bad reflection key: nothing is ever proposed, so nothing is published.
    schedule = [
        {"kind": "harness_until_publish", "label": "pre_training", "max_moves": 3},
        {"kind": "rollback", "to": "first_publish"},
    ]
    run = run_script(offline_config(stub, reflection="401", schedule=schedule), tmp_path)

    assert [(step["move"], step["result"]) for step in run.summary["steps"]] == [
        ("harness", "skipped"), ("harness", "skipped"), ("harness", "skipped"), ("rollback", "not runnable")
    ]  # fmt: skip
    assert run.summary["steps"][-1]["reason"] == "no harness release was published before the rollback"
    assert run.checks["6 rollback"]["passed"] is False
    # Nothing reached the gate either, so the checks that need it were not exercised, rejection included.
    assert {name for name, check in run.checks.items() if check["not_exercised"]} == {
        "1 extracted", "2 reflection prompt", "3 gate on real scores", "4 rejected stays", "6 rollback"
    }  # fmt: skip
    assert "not exercised, rerun" not in run.stdout  # checks 1 to 3 are not ones a rerun is for


@pytest.mark.integration
def test_the_rollback_goes_to_the_first_publish_when_later_moves_publish_too(stub: OpenAIStub, tmp_path: Path) -> None:
    schedule = [
        {"kind": "harness_until_publish", "label": "pre_training", "max_moves": 12},
        {"kind": "harness", "label": "pre_training", "repeat": 8},
        {"kind": "rollback", "to": "first_publish"},
    ]
    run = run_script(offline_config(stub, schedule=schedule), tmp_path)

    published = [step["release"] for step in run.summary["steps"] if step["result"] == "published"]
    assert len(published) >= 2, "a later move must publish too, or this test shows nothing"
    assert run.summary["steps"][-1]["to"] == published[0]
    assert run.checks["6 rollback"]["passed"], run.checks["6 rollback"]


@pytest.mark.integration
def test_a_schedule_the_live_run_cannot_run_is_refused_before_anything_runs(stub: OpenAIStub, tmp_path: Path) -> None:
    run = run_script(offline_config(stub, schedule=[{"kind": "train"}]), tmp_path)

    assert run.code == 2 and "cannot start: schedule step 1:" in run.stderr
    assert not run.work_dir.exists()


@pytest.mark.integration
def test_a_run_whose_only_gap_is_an_unexercised_rollback_says_rerun(stub: OpenAIStub, tmp_path: Path) -> None:
    # The stub's first move is rejected, so one move until a publish ends with nothing to roll back to.
    schedule = [
        {"kind": "harness_until_publish", "label": "pre_training", "max_moves": 1},
        {"kind": "rollback", "to": "first_publish"},
        {"kind": "harness", "label": "pre_training", "repeat": 40},
    ]
    run = run_script(offline_config(stub, schedule=schedule), tmp_path)

    failed = [check for check in run.summary["checks"] if not check["passed"]]
    assert [(check["name"], check["not_exercised"]) for check in failed] == [("6 rollback", True)]
    assert run.code == 1
    assert "[NOT EXERCISED] 6 rollback: not exercised: move 2: not runnable" in run.stdout
    assert "not exercised, rerun: 6 rollback." in run.stdout


@pytest.mark.integration
def test_a_rollback_to_a_release_that_does_not_exist_is_a_real_failure(stub: OpenAIStub, tmp_path: Path) -> None:
    schedule = [{"kind": "harness", "label": "pre_training"}, {"kind": "rollback", "to": "r0099"}]
    run = run_script(offline_config(stub, schedule=schedule), tmp_path)

    rollback = run.checks["6 rollback"]
    assert (rollback["passed"], rollback["not_exercised"]) == (False, False)
    assert "no release 'r0099'" in rollback["detail"]
