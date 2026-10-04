"""The OpenHands runner on a CPU: the episode directory, keys, limits, process control, and the delivery probe.

A stand-in worker (tests/fakes/openhands_worker.py) follows the real worker's file contract, so these tests
need neither OpenHands nor a model. The last test runs the real worker when its environment exists.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from honing.backends.hosted import HostedEndpoints, HostedModel, endpoints, model_source
from honing.backends.openhands import (
    AGENT_SETTINGS,
    OpenHandsDeliveryProbe,
    OpenHandsEpisodeRunner,
    ToolPolicy,
    agent_settings,
)
from honing.core import (
    CampaignRuleError,
    ContractError,
    EpisodeOutcome,
    HarnessFile,
    HarnessSpec,
    PolicyRef,
    Task,
    protocols,
)
from honing.e2e.components import PipelineContext

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
        (AGENT_SETTINGS, agent_settings(["terminal"])),
    ),
)


@pytest.fixture(autouse=True)
def _key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HONING_TEST_KEY", "secret-value-123")


def _policy(model: HostedModel = MODEL) -> PolicyRef:
    snapshot = model.snapshot()
    return PolicyRef(snapshot.checkpoint, snapshot.architecture, "master")


def _runner(tmp_path: Path, **overrides: object) -> OpenHandsEpisodeRunner:
    corpus = tmp_path / "corpus"
    corpus.mkdir(exist_ok=True)
    settings: dict[str, object] = {
        "python": Path(sys.executable),
        "worker": FAKE_WORKER,
        "workspaces": tmp_path / "workspaces",
        "endpoints": HostedEndpoints(MODEL),
        "tools": ToolPolicy(allowed=frozenset({"terminal", "file_editor", "grep"})),
        "mounts": {"corpus": corpus},
        **overrides,
    }
    return OpenHandsEpisodeRunner(**settings)  # type: ignore[arg-type]


def _run(runner: OpenHandsEpisodeRunner, spec: HarnessSpec = SEED) -> EpisodeOutcome:
    return asyncio.run(runner.run(spec, TASK, _policy()))


def _gone(pid: int) -> bool:
    time.sleep(0.2)
    state = subprocess.run(["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True).stdout.strip()
    return state in ("", "Z")


@pytest.mark.unit
def test_the_runner_satisfies_the_episode_runner_protocol(tmp_path: Path) -> None:
    runner = _runner(tmp_path)

    assert isinstance(runner, protocols.EpisodeRunner)
    assert runner.support.file_kinds == frozenset({"instructions", "skill", "settings"})
    assert runner.support.local_policy_only is False


@pytest.mark.unit
def test_an_episode_directory_holds_the_harness_the_plan_and_the_mounts(tmp_path: Path) -> None:
    outcome = _run(_runner(tmp_path))

    root = Path(outcome.workspace_location or "")
    assert (outcome.task_id, outcome.spec_id, outcome.policy_checksum) == ("q1", SEED.spec_id, MODEL.checksum)
    assert outcome.exit_code == 0
    assert outcome.transcript_location == str(root / "transcript.jsonl")
    for path, file in SEED.files:
        assert (root / "harness" / path).read_text() == file.content
    assert (root / "work" / "corpus").is_symlink()
    assert json.loads(outcome.final_output) == {
        "task": "q1",
        "cwd": str((root / "work").resolve()),
        "key_present": True,
        "tools": ["terminal"],
    }
    episode = json.loads((root / "episode.json").read_text())
    assert episode["policy"] == {
        "base_url": "https://example.invalid/api/v1",
        "model": "vendor/cheap-model",
        "api_key_env": "HONING_TEST_KEY",
        "input_cost_per_token": pytest.approx(0.03e-6),
        "output_cost_per_token": pytest.approx(0.06e-6),
    }
    assert episode["agent"]["routes"]["skills/search/SKILL.md"] == "skill:search"
    assert episode["limits"] == {"max_budget_usd": None, "llm_timeout_seconds": 120}


@pytest.mark.unit
def test_the_api_key_is_passed_through_the_environment_and_never_written(tmp_path: Path) -> None:
    outcome = _run(_runner(tmp_path))

    root = Path(outcome.workspace_location or "")
    assert [p for p in root.rglob("*") if p.is_file() and "secret-value-123" in p.read_text(errors="ignore")] == []


@pytest.mark.unit
def test_a_missing_api_key_is_refused_before_the_agent_starts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HONING_TEST_KEY")

    with pytest.raises(ContractError, match="HONING_TEST_KEY"):
        _run(_runner(tmp_path))

    assert not (tmp_path / "workspaces").exists()


@pytest.mark.unit
def test_a_harness_the_policy_forbids_is_refused_before_the_agent_starts(tmp_path: Path) -> None:
    spec = SEED.with_file(AGENT_SETTINGS, agent_settings(["browser_use"]))

    with pytest.raises(CampaignRuleError) as refused:
        _run(_runner(tmp_path), spec)

    assert refused.value.rule == "harness_tools_allowed"
    assert not (tmp_path / "workspaces").exists()


@pytest.mark.unit
def test_a_timeout_kills_the_worker_and_its_children(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_WORKER_SLEEP", "30")
    monkeypatch.setenv("FAKE_WORKER_CHILD", "1")

    started = time.monotonic()
    outcome = _run(_runner(tmp_path, timeout_seconds=2))

    assert outcome.exit_code == -9
    assert time.monotonic() - started < 15
    assert _gone(int((Path(outcome.workspace_location or "") / "child.pid").read_text()))


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
    assert _gone(int((root / "child.pid").read_text()))


@pytest.mark.unit
def test_concurrent_episodes_get_separate_directories(tmp_path: Path) -> None:
    runner = _runner(tmp_path, concurrency=2)

    async def three() -> list[EpisodeOutcome]:
        return list(await asyncio.gather(*(runner.run(SEED, TASK, _policy()) for _ in range(3))))

    assert len({o.workspace_location for o in asyncio.run(three())}) == 3


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
def test_a_hosted_model_is_one_stable_policy(tmp_path: Path) -> None:
    same = HostedModel(base_url="https://example.invalid/api/v1/", model="vendor/cheap-model")
    other = HostedModel(base_url="https://example.invalid/api/v1", model="vendor/other")
    context = PipelineContext(root=tmp_path, work_dir=tmp_path)

    assert MODEL.checksum == same.checksum != other.checksum
    assert MODEL.snapshot().checkpoint.location == "hosted://example.invalid/api/v1/vendor/cheap-model"
    assert HostedEndpoints(MODEL).resolve(_policy()).served_model == "vendor/cheap-model"
    with pytest.raises(ContractError, match="no hosted model serves"):
        HostedEndpoints(other).resolve(_policy())
    fields = {"base_url": "https://example.invalid/api/v1", "model": "m"}
    snapshot = model_source(context, **fields).materialize(tmp_path)
    assert endpoints(context, **fields).resolve(PolicyRef(snapshot.checkpoint, snapshot.architecture, "master"))
    with pytest.raises(ContractError, match="two different ways"):
        endpoints(context, **fields, input_usd_per_million=1.0)


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
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "runner check: ok" in result.stdout
