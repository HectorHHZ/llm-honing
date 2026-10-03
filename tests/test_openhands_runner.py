"""The OpenHands runner on a CPU: episode layout, tool allowlists, keys, timeouts, and the delivery probe.

A stand-in worker (tests/fakes/openhands_worker.py) follows the real worker's file contract, so these
tests need neither the OpenHands environment nor a model. The real worker is checked end to end by
scripts/e2e/check_runner.py on configs/e2e/openhands.json; the last test here runs that when the
OpenHands environment exists.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from honing.backends.hosted import HostedEndpoints, HostedModel
from honing.backends.openhands import (
    AGENT_SETTINGS,
    MCP_SETTINGS,
    OpenHandsDeliveryProbe,
    OpenHandsEpisodeRunner,
    ToolPolicy,
)
from honing.core import CampaignRuleError, ContractError, HarnessFile, HarnessSpec, PolicyRef, Task, protocols

REPO = Path(__file__).resolve().parents[1]
FAKE_WORKER = REPO / "tests/fakes/openhands_worker.py"
MODEL = HostedModel(
    base_url="https://example.invalid/api/v1",
    model="vendor/cheap-model",
    api_key_env="HONING_TEST_KEY",
    input_usd_per_million=0.03,
    output_usd_per_million=0.06,
)
TASK = Task(task_id="q1", prompt="What is the figure?")
SEED = HarnessSpec(
    agent="openhands",
    files=(
        ("AGENTS.md", HarnessFile("instructions", "Answer from the documents.\n")),
        ("skills/search/SKILL.md", HarnessFile("skill", "---\nname: search\ndescription: d\n---\nUse grep.\n")),
        (AGENT_SETTINGS, HarnessFile("settings", json.dumps({"tools": ["terminal"]}))),
    ),
)


def _policy() -> PolicyRef:
    snapshot = MODEL.snapshot()
    return PolicyRef(snapshot.checkpoint, snapshot.architecture, "master")


def _runner(tmp_path: Path, **overrides: object) -> OpenHandsEpisodeRunner:
    corpus = tmp_path / "corpus"
    corpus.mkdir(exist_ok=True)
    settings: dict[str, object] = {
        "python": Path(sys.executable),
        "worker": FAKE_WORKER,
        "workspaces": tmp_path / "workspaces",
        "endpoints": HostedEndpoints(MODEL),
        "tools": ToolPolicy(
            allowed=frozenset({"terminal", "file_editor", "grep"}),
            default=("terminal", "file_editor"),
            mcp_registry={"fetch": {"command": "uvx", "args": ["mcp-server-fetch"]}},
        ),
        "mounts": {"corpus": corpus},
        **overrides,
    }
    return OpenHandsEpisodeRunner(**settings)  # type: ignore[arg-type]


def _run(runner: OpenHandsEpisodeRunner, spec: HarnessSpec = SEED):  # type: ignore[no-untyped-def]
    return asyncio.run(runner.run(spec, TASK, _policy()))


@pytest.mark.unit
def test_the_runner_satisfies_the_episode_runner_protocol(tmp_path: Path) -> None:
    runner = _runner(tmp_path)

    assert isinstance(runner, protocols.EpisodeRunner)
    assert runner.support.agents == frozenset({"openhands"})
    assert runner.support.file_kinds == frozenset({"instructions", "skill", "settings"})
    assert runner.support.local_policy_only is False


@pytest.mark.unit
def test_an_episode_directory_holds_the_harness_the_task_and_the_mounts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HONING_TEST_KEY", "secret-value-123")

    outcome = _run(_runner(tmp_path))

    root = Path(outcome.workspace_location or "")
    assert (outcome.task_id, outcome.spec_id, outcome.policy_checksum) == ("q1", SEED.spec_id, MODEL.checksum)
    assert outcome.exit_code == 0
    for path, file in SEED.files:
        assert (root / "harness" / path).read_text() == file.content
    assert (root / "work" / "corpus").is_symlink()
    assert outcome.transcript_location == str(root / "transcript.jsonl")
    answer = json.loads(outcome.final_output)
    assert answer == {"task": "q1", "cwd": str((root / "work").resolve()), "key_present": True}
    episode = json.loads((root / "episode.json").read_text())
    assert episode["task"] == {"task_id": "q1", "prompt": TASK.prompt}
    assert episode["policy"]["model"] == "vendor/cheap-model"
    assert episode["policy"]["api_key_env"] == "HONING_TEST_KEY"
    assert episode["policy"]["input_cost_per_token"] == pytest.approx(0.03e-6)
    assert episode["tools"] == {"allowed": ["file_editor", "grep", "terminal"], "default": ["terminal", "file_editor"]}
    assert episode["mcp_registry"] == {"fetch": {"command": "uvx", "args": ["mcp-server-fetch"]}}


@pytest.mark.unit
def test_the_api_key_is_passed_through_the_environment_and_never_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HONING_TEST_KEY", "secret-value-123")

    outcome = _run(_runner(tmp_path))

    root = Path(outcome.workspace_location or "")
    written = [p for p in root.rglob("*") if p.is_file() and "secret-value-123" in p.read_text(errors="ignore")]
    assert written == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("path", "content", "message"),
    [
        (AGENT_SETTINGS, json.dumps({"tools": ["terminal", "browser_use"]}), "enables tools \\['browser_use'\\]"),
        (AGENT_SETTINGS, json.dumps({"tools": ["terminal"], "shell": "bash"}), "unknown keys \\['shell'\\]"),
        (AGENT_SETTINGS, json.dumps({"max_iterations": 0}), "max_iterations must be a positive integer"),
        (AGENT_SETTINGS, "{not json", "is not JSON"),
        (MCP_SETTINGS, json.dumps({"enabled": ["fetch", "shell"]}), "enables MCP servers \\['shell'\\]"),
        (MCP_SETTINGS, json.dumps({"mcpServers": {"x": {"command": "rm"}}}), "enables registry servers by name only"),
    ],
)
def test_a_harness_asking_for_what_the_operator_forbids_is_refused_before_it_runs(
    tmp_path: Path, path: str, content: str, message: str
) -> None:
    runner = _runner(tmp_path)
    spec = SEED.with_file(path, HarnessFile("settings", content))

    with pytest.raises(CampaignRuleError, match=message) as refused:
        _run(runner, spec)

    assert refused.value.rule == "harness_tools_allowed"
    assert not (tmp_path / "workspaces").exists()


@pytest.mark.unit
def test_a_harness_may_enable_registry_servers_and_allowed_tools(tmp_path: Path) -> None:
    spec = SEED.with_file(MCP_SETTINGS, HarnessFile("settings", json.dumps({"enabled": ["fetch"]}))).with_file(
        AGENT_SETTINGS, HarnessFile("settings", json.dumps({"tools": ["grep", "terminal"], "max_iterations": 5}))
    )

    assert _run(_runner(tmp_path), spec).exit_code == 0


@pytest.mark.unit
def test_another_agent_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="not 'claude_code'"):
        _runner(tmp_path).check_spec(HarnessSpec(agent="claude_code"))


@pytest.mark.unit
def test_a_timeout_kills_the_worker_and_its_children(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_WORKER_SLEEP", "30")
    monkeypatch.setenv("FAKE_WORKER_CHILD", "1")

    started = time.monotonic()
    outcome = _run(_runner(tmp_path, timeout_seconds=2))

    assert outcome.exit_code == -9
    assert time.monotonic() - started < 15
    pid = int((Path(outcome.workspace_location or "") / "child.pid").read_text())
    time.sleep(0.2)
    assert subprocess.run(["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True).stdout.strip() in (
        "",
        "Z",
    )


@pytest.mark.unit
def test_concurrent_episodes_get_separate_directories(tmp_path: Path) -> None:
    runner = _runner(tmp_path, concurrency=2)

    async def three() -> list[object]:
        return list(await asyncio.gather(*(runner.run(SEED, TASK, _policy()) for _ in range(3))))

    outcomes = asyncio.run(three())

    assert len({o.workspace_location for o in outcomes}) == 3  # type: ignore[attr-defined]


@pytest.mark.unit
def test_the_probe_counts_a_file_only_when_the_worker_reports_its_content(tmp_path: Path) -> None:
    outcome = _run(_runner(tmp_path))
    probe = OpenHandsDeliveryProbe()

    assert probe.delivered(outcome, SEED) == {path: True for path, _ in SEED.files}
    changed = SEED.with_file("AGENTS.md", HarnessFile("instructions", "something else\n"))
    assert probe.delivered(outcome, changed)["AGENTS.md"] is False
    (Path(outcome.workspace_location or "") / "delivery.json").unlink()
    assert set(probe.delivered(outcome, SEED).values()) == {False}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"concurrency": 0}, "must be positive"),
        ({"max_budget_usd": 0.0}, "max_budget_usd must be positive"),
        ({"mounts": {"a/b": Path(".")}}, "one path segment"),
        ({"mounts": {"data": Path("/no/such/dir")}}, "not a directory"),
    ],
)
def test_bad_settings_are_refused_at_construction(tmp_path: Path, overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ContractError, match=message):
        _runner(tmp_path, **overrides)


@pytest.mark.unit
def test_a_default_tool_must_be_allowed() -> None:
    with pytest.raises(ContractError, match="must be allowed"):
        ToolPolicy(allowed=frozenset({"terminal"}), default=("terminal", "file_editor"))


# ---------------------------------------------------------------- the hosted model behind the runner


@pytest.mark.unit
def test_a_hosted_model_is_one_stable_policy() -> None:
    same = HostedModel(base_url="https://example.invalid/api/v1/", model="vendor/cheap-model")
    other = HostedModel(base_url="https://example.invalid/api/v1", model="vendor/other")

    assert MODEL.checksum == same.checksum != other.checksum
    assert MODEL.snapshot().checkpoint.location == "hosted://example.invalid/api/v1/vendor/cheap-model"
    endpoints = HostedEndpoints(MODEL)
    assert isinstance(endpoints, protocols.PolicyEndpoints)
    assert endpoints.resolve(_policy()).served_model == "vendor/cheap-model"
    with pytest.raises(ContractError, match="no hosted model serves"):
        HostedEndpoints(other).resolve(_policy())


# ---------------------------------------------------------------- the real worker, when its environment exists


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(not (REPO / "integrations/openhands/.venv/bin/python").exists(), reason="no OpenHands environment")
def test_the_real_worker_passes_the_runner_checks_against_the_stand_in_policy(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/e2e/check_runner.py",
            *("--config", "configs/e2e/openhands.json"),
            *("--work-dir", str(tmp_path / "check")),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ},
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "runner check: ok" in result.stdout


@pytest.mark.unit
def test_a_cancelled_episode_kills_its_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_WORKER_SLEEP", "30")
    monkeypatch.setenv("FAKE_WORKER_CHILD", "1")
    runner = _runner(tmp_path)

    async def cancel_after_start() -> None:
        episode = asyncio.ensure_future(runner.run(SEED, TASK, _policy()))
        await asyncio.sleep(1.5)
        episode.cancel()
        with pytest.raises(asyncio.CancelledError):
            await episode

    asyncio.run(cancel_after_start())

    (root,) = (tmp_path / "workspaces").iterdir()
    pid = int((root / "child.pid").read_text())
    time.sleep(0.2)
    assert subprocess.run(["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True).stdout.strip() in (
        "",
        "Z",
    )
