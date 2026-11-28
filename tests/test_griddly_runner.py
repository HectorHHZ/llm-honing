"""GriddlyEpisodeRunner and GriddlyGrader driving the real match and replay commands on a stand-in engine.

The commands are ``tests/fakes/griddly_match.py``, which runs Honing's match and
replay code on ``tests/fakes/griddly_engine.py``, and the policy is the
stand-in decision server in ``tests/fakes/griddly.py``. So the runner's own
part runs for real: the arguments it passes, the harness it delivers, how a
match process's failure reaches the campaign, and that a cancelled episode
leaves no process behind.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from honing.backends.griddly import GriddlyEpisodeRunner, GriddlyGrader, GriddlySuite, seed_harness
from honing.backends.griddly.replay import audit_run
from honing.backends.griddly.runner import runtime_python
from honing.backends.griddly.scenarios import GriddlyScenario
from honing.core import (
    CampaignRuleError,
    ContractError,
    HarnessSpec,
    PolicyEndpoint,
    PolicyRef,
    Task,
    run_graded,
)
from tests.fakes.griddly import StandInEndpoints
from tests.fakes.model import FakeModelSource, FakeWorld

ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "integrations/griddly"
SEED = seed_harness(INTEGRATION)
COMMAND = [sys.executable, str(ROOT / "tests/fakes/griddly_match.py")]
SUITE = GriddlySuite(
    (
        {
            "task_id": "spiders-val",
            "split": "val",
            "prompt": "Reach the gem.",
            "griddly": GriddlyScenario(seed=3, level_id=1, max_steps=12).record(),
        },
    )
)
TASK = SUITE.tasks("val")[0]


@pytest.fixture
def endpoints() -> Iterator[StandInEndpoints]:
    value = StandInEndpoints()
    try:
        yield value
    finally:
        value.close()
        value.server.server_close()  # close() stops serving but leaves the socket open


@pytest.fixture
def policy(tmp_path: Path) -> PolicyRef:
    snapshot = FakeModelSource(FakeWorld()).materialize(tmp_path / "model")
    return PolicyRef(snapshot.checkpoint, snapshot.architecture, "master")


def runner(endpoints: Any, tmp_path: Path, **options: Any) -> GriddlyEpisodeRunner:
    return GriddlyEpisodeRunner(
        suite=SUITE, endpoints=endpoints, integration_dir=INTEGRATION, runs_dir=tmp_path / "runs", **options
    )


@pytest.mark.functional
def test_an_episode_runs_the_match_with_the_task_the_policy_and_the_harness_and_grades_it(
    tmp_path: Path, endpoints: StandInEndpoints, policy: PolicyRef
) -> None:
    grader = GriddlyGrader(suite=SUITE, integration_dir=INTEGRATION, command=[*COMMAND, "--replay"])

    record = asyncio.run(run_graded(runner(endpoints, tmp_path, command=COMMAND), grader, SEED, TASK, policy))

    outcome = record.outcome
    run = Path(str(outcome.workspace_location))
    assert (outcome.exit_code, outcome.task_id, outcome.spec_id) == (0, TASK.task_id, SEED.spec_id)
    assert outcome.policy_checksum == policy.checkpoint.checksum
    assert outcome.transcript_location == str(run / "decisions.jsonl")
    assert json.loads(outcome.final_output) == json.loads((run / "result.json").read_text())
    manifest, _rows, result = audit_run(run)
    assert (manifest["controller"], manifest["kind"], manifest["sample"]) == ("model", "evaluation", False)
    assert manifest["scenario"] == SUITE.scenario(TASK.task_id).record()
    assert manifest["policy_checksum"] == policy.checkpoint.checksum
    assert manifest["spec_id"] == SEED.spec_id
    # The harness is delivered beside the run, and the match's output goes to a log beside it.
    assert (run.parent / f"{run.name}.harness" / "index.json").is_file()
    assert (run.parent / f"{run.name}.log").is_file()
    assert result["success"] is True
    assert record.grade.score == 1.0
    assert record.grade.feedback.startswith(f"Ended as won after {result['steps']} ticks.")


@pytest.mark.functional
def test_a_sampling_runner_draws_its_actions_with_a_fresh_policy_seed(
    tmp_path: Path, endpoints: StandInEndpoints, policy: PolicyRef
) -> None:
    sampling = runner(endpoints, tmp_path, command=COMMAND).sampling()

    seeds = []
    for _ in range(2):
        outcome = asyncio.run(sampling.run(SEED, TASK, policy))
        manifest = audit_run(Path(str(outcome.workspace_location)))[0]
        assert (manifest["kind"], manifest["sample"]) == ("rollout", True)
        seeds.append(manifest["policy_seed"])

    assert seeds[0] != seeds[1]


@pytest.mark.functional
def test_a_harness_the_match_refuses_is_a_harness_error_for_the_campaign(
    tmp_path: Path, endpoints: StandInEndpoints, policy: PolicyRef
) -> None:
    refusing = runner(endpoints, tmp_path, command=COMMAND, environment={"FAKE_GRIDDLY_HARNESS_ERROR": "1"})

    with pytest.raises(CampaignRuleError, match="stand-in refusal of the delivered harness") as raised:
        asyncio.run(refusing.run(SEED, TASK, policy))

    assert raised.value.rule == "harness_readable"


@pytest.mark.functional
def test_a_crashed_match_is_an_outcome_with_its_exit_code_and_no_result(
    tmp_path: Path, endpoints: StandInEndpoints, policy: PolicyRef
) -> None:
    crashing = runner(endpoints, tmp_path, command=COMMAND, environment={"FAKE_GRIDDLY_CRASH": "1"})

    outcome = asyncio.run(crashing.run(SEED, TASK, policy))

    assert outcome.exit_code != 0 and outcome.final_output == "{}"
    error = json.loads((Path(str(outcome.workspace_location)) / "error.json").read_text())
    assert (error["kind"], error["error"]) == ("match", "RuntimeError: stand-in engine crash")
    grader = GriddlyGrader(suite=SUITE, integration_dir=INTEGRATION, command=[*COMMAND, "--replay"])
    with pytest.raises(ContractError, match="Griddly did not complete this task"):
        asyncio.run(grader.grade(TASK, outcome))


@pytest.mark.functional
def test_cancelling_an_episode_stops_its_match_process(
    tmp_path: Path, endpoints: StandInEndpoints, policy: PolicyRef
) -> None:
    started = tmp_path / "pid"
    hanging = runner(endpoints, tmp_path, command=COMMAND, environment={"FAKE_GRIDDLY_HANG": str(started)})

    async def cancel_once_started() -> None:
        episode = asyncio.ensure_future(hanging.run(SEED, TASK, policy))
        deadline = time.monotonic() + 30
        while not started.exists() and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        assert started.exists(), "the match process never started"
        episode.cancel()
        with pytest.raises(asyncio.CancelledError):
            await episode

    asyncio.run(cancel_once_started())

    with pytest.raises(ProcessLookupError):
        os.kill(int(started.read_text()), 0)


@pytest.mark.functional
def test_a_policy_without_a_typed_decision_endpoint_is_refused_before_a_match_starts(
    tmp_path: Path, policy: PolicyRef
) -> None:
    class ChatEndpoints:
        def resolve(self, policy: PolicyRef) -> PolicyEndpoint:
            return PolicyEndpoint("http://127.0.0.1:1", "tokenizer", "m", runtime="sglang")

    with pytest.raises(ContractError, match="Griddly needs a typed Jev decision endpoint"):
        asyncio.run(runner(ChatEndpoints(), tmp_path, command=COMMAND).run(SEED, TASK, policy))

    assert not (tmp_path / "runs").exists()


@pytest.mark.unit
def test_a_task_outside_the_suite_and_a_foreign_harness_are_refused_before_a_match_starts(
    tmp_path: Path, policy: PolicyRef
) -> None:
    griddly = runner(None, tmp_path, command=COMMAND)

    with pytest.raises(ContractError, match="unknown Griddly task 'other'"):
        asyncio.run(griddly.run(SEED, Task("other", "prompt"), policy))
    with pytest.raises(CampaignRuleError, match="Griddly accepts its own agent"):
        griddly.check_spec(HarnessSpec("claude_code", SEED.files))
    assert not (tmp_path / "runs").exists()


@pytest.mark.unit
@pytest.mark.parametrize("concurrency", [0, -1, 1.5, True])
def test_concurrency_must_be_a_positive_integer(tmp_path: Path, concurrency: object) -> None:
    with pytest.raises(ContractError, match="concurrency must be a positive integer"):
        runner(None, tmp_path, concurrency=concurrency)


@pytest.mark.unit
def test_by_default_the_match_runs_in_the_integration_s_own_environment(tmp_path: Path) -> None:
    griddly = runner(None, tmp_path, concurrency=3, environment={"A": "1"})
    sampling = griddly.sampling()

    assert runtime_python(INTEGRATION) == str(INTEGRATION / ".venv" / "bin" / "python")
    assert griddly._command == [runtime_python(INTEGRATION), str(INTEGRATION / "scripts/match.py")]
    assert (sampling._command, sampling._environment, sampling._concurrency) == (griddly._command, {"A": "1"}, 3)
    assert (sampling._sample, griddly._sample) == (True, False)
    assert sampling.runs_dir == griddly.runs_dir == (tmp_path / "runs").resolve()


@pytest.mark.functional
@pytest.mark.parametrize(
    ("forge", "command", "message"),
    [
        (
            {"policy_checksum": "sha256:another"},
            COMMAND,
            "Griddly run does not match the task, policy and harness",
        ),
        ({}, [sys.executable, "-c", "import sys; sys.exit(1)"], "native Griddly replay did not verify"),
        ({}, [sys.executable, "-c", "print('{\"verified\": true}')"], "native Griddly replay did not verify"),
    ],
    ids=["other-policy", "replay-fails", "replay-of-another-episode"],
)
def test_the_grader_scores_only_an_episode_it_can_trace_and_replay(
    tmp_path: Path,
    endpoints: StandInEndpoints,
    policy: PolicyRef,
    forge: dict[str, str],
    command: list[str],
    message: str,
) -> None:
    outcome = asyncio.run(runner(endpoints, tmp_path, command=COMMAND).run(SEED, TASK, policy))
    grader = GriddlyGrader(suite=SUITE, integration_dir=INTEGRATION, command=[*command, "--replay"])

    with pytest.raises(ContractError, match=message):
        asyncio.run(grader.grade(TASK, dataclasses.replace(outcome, **forge)))
