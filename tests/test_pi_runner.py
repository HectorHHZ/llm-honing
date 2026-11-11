"""The pi runner on a CPU: the episode directory, models.json, keys, failures, process control, and the delivery probe.

A stand-in worker (tests/fakes/pi_worker.py) follows the real worker's file contract, so these tests need neither
Node.js, pi, nor a model. The last test runs the real worker when pi is installed.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from honing.backends.llm_gateway import GatewayEndpoints, GatewayModel
from honing.backends.pi import (
    REFUSED,
    SETTINGS,
    TIMED_OUT,
    PiDeliveryProbe,
    PiEpisodeRunner,
    ToolPolicy,
    settings_file,
)
from honing.backends.pi.e2e import runner as runner_factory
from honing.core import (
    ContractError,
    EpisodeOutcome,
    EpisodeRecord,
    Grade,
    HarnessFile,
    HarnessSpec,
    HoneRelease,
    PolicyEndpoint,
    PolicyRef,
    Task,
    UpdateCandidate,
    protocols,
    run_graded,
)
from honing.e2e.components import PipelineContext
from honing.release import EpisodeEvaluator, Gate, WinLossSelector
from tests.fakes.harness import FakeTaskGrader
from tests.fakes.tool_policy import ToolCallingPolicy

REPO = Path(__file__).resolve().parents[1]
FAKE_WORKER = REPO / "tests/fakes/pi_worker.py"
PI = REPO / "integrations/pi/node_modules/@earendil-works/pi-coding-agent/package.json"
needs_pi = pytest.mark.skipif(not PI.exists() or shutil.which("node") is None, reason="pi is not installed")
MODEL = GatewayModel(
    base_url="https://example.invalid/api/v1",
    model="vendor/cheap-model",
    api_key_env="HONING_TEST_KEY",
    input_usd_per_million=0.03,
    output_usd_per_million=0.06,
)
TASK = Task(task_id="q1", prompt="What is the figure?")
HANG = {"FAKE_PI_SLEEP": "30", "FAKE_PI_CHILD": "1"}
SEED = HarnessSpec(
    agent="pi",
    files=(
        ("AGENTS.md", HarnessFile("instructions", "Answer from the documents.\n")),
        ("skills/search/SKILL.md", HarnessFile("skill", "---\nname: search\ndescription: d\n---\nUse grep.\n")),
        (SETTINGS, settings_file(["read", "bash"])),
    ),
)
FORBIDDEN = SEED.with_file(SETTINGS, settings_file(["read", "browser"]))


@pytest.fixture(autouse=True)
def _key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HONING_TEST_KEY", "secret-value-123")


def _policy(model: GatewayModel = MODEL) -> PolicyRef:
    snapshot = model.snapshot()
    return PolicyRef(snapshot.checkpoint, snapshot.architecture, "master")


def _runner(tmp_path: Path, **overrides: object) -> PiEpisodeRunner:
    corpus = tmp_path / "corpus"
    corpus.mkdir(exist_ok=True)
    settings: dict[str, object] = {
        "node": Path(sys.executable),
        "worker": FAKE_WORKER,
        "workspaces": tmp_path / "workspaces",
        "endpoints": GatewayEndpoints(MODEL),
        "tools": ToolPolicy(allowed=frozenset({"read", "bash", "edit", "write", "grep"})),
        "mounts": {"corpus": corpus},
        **overrides,
    }
    return PiEpisodeRunner(**settings)  # type: ignore[arg-type]


def _run(runner: PiEpisodeRunner, spec: HarnessSpec = SEED) -> EpisodeOutcome:
    return asyncio.run(runner.run(spec, TASK, _policy()))


def _graded(runner: PiEpisodeRunner) -> EpisodeRecord:
    return asyncio.run(run_graded(runner, FakeTaskGrader(), SEED, TASK, _policy()))


def _gone(pid: int) -> bool:
    time.sleep(0.2)
    state = subprocess.run(["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True).stdout.strip()
    return state in ("", "Z")


@pytest.mark.unit
def test_the_runner_satisfies_the_episode_runner_protocol(tmp_path: Path) -> None:
    runner = _runner(tmp_path)

    assert isinstance(runner, protocols.EpisodeRunner)
    assert runner.support.agents == frozenset({"pi"})
    assert runner.support.file_kinds == frozenset({"instructions", "skill", "settings"})
    assert runner.support.local_policy_only is False


@pytest.mark.unit
def test_an_episode_directory_holds_the_harness_the_plan_the_models_and_the_mounts(tmp_path: Path) -> None:
    outcome = _run(_runner(tmp_path, model_settings={"contextWindow": 32768}))

    root = Path(outcome.workspace_location or "")
    assert (outcome.task_id, outcome.spec_id, outcome.policy_checksum) == ("q1", SEED.spec_id, MODEL.checksum)
    assert outcome.exit_code == 0
    assert outcome.transcript_location == str(root / "transcript.jsonl")
    for path, file in SEED.files:
        assert (root / "harness" / path).read_text() == file.content
    assert (root / "work" / "corpus").is_dir() and not (root / "work" / "corpus").is_symlink()
    answer = json.loads(outcome.final_output)
    assert (answer["task"], answer["cwd"], answer["key_present"]) == ("q1", str((root / "work").resolve()), True)
    assert answer["tools"] == ["read", "bash"]
    assert answer["models"] == {
        "providers": {
            "honing": {
                "baseUrl": "https://example.invalid/api/v1",
                "api": "openai-completions",
                "apiKey": "$HONING_TEST_KEY",
                "models": [
                    {
                        "id": "vendor/cheap-model",
                        "contextWindow": 32768,
                        "cost": {"input": 0.03, "output": 0.06, "cacheRead": 0.03, "cacheWrite": 0.03},
                    }
                ],
            }
        }
    }
    episode = json.loads((root / "episode.json").read_text())
    assert episode["model"] == {"provider": "honing", "id": "vendor/cheap-model", "api_key_env": "HONING_TEST_KEY"}
    assert episode["agent"]["routes"]["skills/search/SKILL.md"] == "skill:search"


@pytest.mark.unit
def test_each_episode_gets_its_own_copy_of_a_mount(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    (tmp_path / "corpus" / "doc.txt").write_text("original\n")

    first = _run(runner)
    (Path(first.workspace_location or "") / "work" / "corpus" / "doc.txt").write_text("edited by the agent\n")
    second = _run(runner)

    assert (tmp_path / "corpus" / "doc.txt").read_text() == "original\n"
    assert (Path(second.workspace_location or "") / "work" / "corpus" / "doc.txt").read_text() == "original\n"


@pytest.mark.unit
def test_link_mode_shares_the_mount(tmp_path: Path) -> None:
    outcome = _run(_runner(tmp_path, mount_mode="link"))

    linked = Path(outcome.workspace_location or "") / "work" / "corpus"
    assert linked.is_symlink() and linked.resolve() == (tmp_path / "corpus").resolve()


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


class _Endpoints:
    """Resolves every policy to one endpoint of the given runtime."""

    def __init__(self, runtime: str) -> None:
        self.runtime = runtime

    def resolve(self, policy: PolicyRef) -> PolicyEndpoint:
        return PolicyEndpoint("http://127.0.0.1:1/v1", "tokenizer", "local-model", runtime=self.runtime)  # type: ignore[arg-type]


@pytest.mark.unit
def test_a_local_sglang_model_needs_no_key(tmp_path: Path) -> None:
    outcome = _run(_runner(tmp_path, endpoints=_Endpoints("sglang")))

    provider = json.loads(outcome.final_output)["models"]["providers"]["honing"]
    assert (provider["baseUrl"], provider["apiKey"], provider["models"]) == (
        "http://127.0.0.1:1/v1",
        "unused",
        [{"id": "local-model"}],
    )


@pytest.mark.unit
def test_a_decision_server_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="OpenAI-compatible"):
        _run(_runner(tmp_path, endpoints=_Endpoints("jev_decision")))

    assert not (tmp_path / "workspaces").exists()


@pytest.mark.unit
def test_a_harness_the_policy_forbids_is_refused_without_starting_the_agent(tmp_path: Path) -> None:
    outcome = _run(_runner(tmp_path), FORBIDDEN)

    assert outcome.exit_code == REFUSED
    assert "harness_tools_allowed" in outcome.final_output and "browser" in outcome.final_output
    root = Path(outcome.workspace_location or "")
    assert json.loads((root / "result.json").read_text())["status"] == "refused"
    assert not (root / "episode.json").exists()


@pytest.mark.unit
def test_the_gate_rejects_a_refused_harness_instead_of_stopping(tmp_path: Path) -> None:
    class _FinishedGrader:
        async def grade(self, task: Task, outcome: EpisodeOutcome) -> Grade:
            return Grade(score=float(outcome.exit_code == 0))

    snapshot = MODEL.snapshot()
    incumbent = HoneRelease("r0000", UpdateCandidate("init", None, snapshot, harness=SEED), None)
    candidate = UpdateCandidate("harness", "r0000", snapshot, harness=FORBIDDEN)
    gate = Gate(EpisodeEvaluator(_runner(tmp_path), _FinishedGrader(), [TASK]), WinLossSelector())

    evaluation, decision = gate.judge(candidate, incumbent)

    assert (evaluation.candidate_scores, evaluation.incumbent_scores) == ((0.0,), (1.0,))
    assert not decision.selected


@pytest.mark.unit
def test_the_worker_sees_only_allowed_environment_variables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf-secret")
    monkeypatch.setenv("LC_ALL", "C")
    monkeypatch.setenv("PI_CODING_AGENT_DIR", "/home/someone/.pi/agent")

    outcome = _run(_runner(tmp_path, environment={"EXTRA_SETTING": "1"}))

    seen = set(json.loads(outcome.final_output)["environment"])
    assert {"PATH", "LC_ALL", "HONING_TEST_KEY", "EXTRA_SETTING", "PI_OFFLINE", "PI_CODING_AGENT_DIR"} <= seen
    assert "HF_TOKEN" not in seen


@pytest.mark.unit
def test_a_proxy_reaches_the_worker_only_when_named(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:3128")

    default = set(json.loads(_run(_runner(tmp_path)).final_output)["environment"])
    named = set(json.loads(_run(_runner(tmp_path, inherit_environment=["HTTPS_PROXY"])).final_output)["environment"])

    assert "HTTPS_PROXY" not in default
    assert "HTTPS_PROXY" in named


@pytest.mark.unit
def test_a_timeout_kills_the_worker_and_its_children_and_scores_zero_with_feedback(tmp_path: Path) -> None:
    started = time.monotonic()
    record = _graded(_runner(tmp_path, timeout_seconds=2, environment=HANG))

    assert record.outcome.exit_code == TIMED_OUT
    assert time.monotonic() - started < 15
    assert _gone(int((Path(record.outcome.workspace_location or "") / "child.pid").read_text()))
    assert record.outcome.final_output.startswith("The agent did not finish within 2 seconds")
    assert record.grade.score == 0.0
    assert "did not finish within 2 seconds" in (record.grade.feedback or "")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status", "error", "reason"),
    [
        ("aborted", None, "The agent stopped (aborted)."),
        ("error", "429: rate limited", "The agent stopped before finishing (error): 429: rate limited"),
    ],
)
def test_an_aborted_run_or_a_provider_error_scores_zero_with_feedback(
    tmp_path: Path, status: str, error: str | None, reason: str
) -> None:
    environment = {"FAKE_PI_STATUS": status, **({"FAKE_PI_ERROR": error} if error else {})}

    record = _graded(_runner(tmp_path, environment=environment))

    assert record.outcome.exit_code == 2
    assert record.outcome.final_output.startswith(reason)
    assert record.grade.score == 0.0
    assert reason in (record.grade.feedback or "")


@pytest.mark.unit
def test_a_cancelled_episode_kills_its_worker(tmp_path: Path) -> None:
    runner = _runner(tmp_path, environment=HANG)

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
    probe = PiDeliveryProbe()

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
        ({"timeout_seconds": 0}, "must be positive"),
        ({"mounts": {"a/b": Path(".")}}, "one path segment"),
        ({"mounts": {"data": Path("/no/such/dir")}}, "not a directory"),
        ({"mount_mode": "bind"}, "mount_mode must be"),
        ({"inherit_environment": ["HTTPS_PROXY=http://p"]}, "names variables"),
    ],
)
def test_bad_settings_are_refused_at_construction(tmp_path: Path, overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ContractError, match=message):
        _runner(tmp_path, **overrides)


@pytest.mark.unit
def test_the_factory_refuses_a_node_it_cannot_find(tmp_path: Path) -> None:
    context = PipelineContext(root=tmp_path, work_dir=tmp_path)

    with pytest.raises(ContractError, match=r"no Node\.js executable"):
        runner_factory(context, node="no-such-node-executable")


# The real worker, with pi, against a stand-in model that has the agent run one shell command.


def _real_runner(tmp_path: Path, server: ToolCallingPolicy, **overrides: object) -> tuple[PiEpisodeRunner, PolicyRef]:
    model = GatewayModel(base_url=server.url, model="stand-in", api_key_env="HONING_TEST_KEY")
    settings: dict[str, object] = {
        "node": Path(shutil.which("node") or "node"),
        # Relative on purpose: the worker starts in the episode's working directory.
        "worker": Path(os.path.relpath(REPO / "integrations/pi/worker.mjs")),
        "workspaces": tmp_path / "workspaces",
        "endpoints": GatewayEndpoints(model),
        "timeout_seconds": 60,
        **overrides,
    }
    return PiEpisodeRunner(**settings), _policy(model)  # type: ignore[arg-type]


def _alive(pid: int) -> bool:
    return not _gone(pid)


@pytest.mark.integration
@needs_pi
@pytest.mark.parametrize("stop", ["timeout", "cancel"])
def test_stopping_the_real_worker_stops_the_agents_shell_commands(tmp_path: Path, stop: str) -> None:
    # pi starts each shell command in a process group of its own, which a signal to the worker's group misses.
    with ToolCallingPolicy("echo $$ > ../sleeper.pid; exec sleep 613") as server:
        runner, policy = _real_runner(tmp_path, server, timeout_seconds=4 if stop == "timeout" else 60)

        async def episode() -> None:
            running = asyncio.ensure_future(runner.run(SEED, TASK, policy))
            if stop == "timeout":
                assert (await running).exit_code == TIMED_OUT
                return
            pid_file = await _appears(tmp_path, "sleeper.pid")
            assert _alive(int(pid_file.read_text()))
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running

        asyncio.run(episode())

    (root,) = (tmp_path / "workspaces").iterdir()
    assert _gone(int((root / "sleeper.pid").read_text()))


async def _appears(tmp_path: Path, name: str) -> Path:
    for _ in range(300):
        found = list((tmp_path / "workspaces").glob(f"*/{name}"))
        if found and found[0].read_text().strip():
            return found[0]
        await asyncio.sleep(0.1)
    raise AssertionError(f"no {name} appeared")


@pytest.mark.integration
@needs_pi
def test_the_api_key_reaches_the_model_but_not_the_agents_shell(tmp_path: Path) -> None:
    with ToolCallingPolicy("env") as server:
        runner, policy = _real_runner(tmp_path, server)
        outcome = asyncio.run(runner.run(SEED, TASK, policy))

    assert outcome.exit_code == 0
    assert server.requests[0]["headers"]["authorization"] == "Bearer secret-value-123"
    root = Path(outcome.workspace_location or "")
    transcript = (root / "transcript.jsonl").read_text()
    assert "PI_CODING_AGENT_DIR" in transcript  # the command's output, the environment, is in the transcript
    assert [p for p in root.rglob("*") if p.is_file() and "secret-value-123" in p.read_text(errors="ignore")] == []


@pytest.mark.integration
@needs_pi
def test_a_harness_that_does_not_reach_pi_stops_before_the_model_is_called(tmp_path: Path) -> None:
    # The frontmatter passes Honing's check but is not YAML, so pi drops the skill.
    spec = SEED.with_file("s/SKILL.md", HarnessFile("skill", "---\nname: s\ndescription: [unclosed\n---\nbody\n"))
    with ToolCallingPolicy("true") as server:
        runner, policy = _real_runner(tmp_path, server)
        record = asyncio.run(run_graded(runner, FakeTaskGrader(), spec, TASK, policy))

    assert server.requests == []
    assert record.outcome.exit_code == 2
    assert record.outcome.final_output.startswith(
        "The agent stopped before finishing (undelivered): "
        "the harness did not reach pi: s/SKILL.md (skill:s); pi said:"
    )
    assert record.grade.score == 0.0
    assert PiDeliveryProbe().delivered(record.outcome, spec)["s/SKILL.md"] is False


@pytest.mark.integration
@needs_pi
def test_a_thinking_level_reaches_a_reasoning_model(tmp_path: Path) -> None:
    spec = SEED.with_file(SETTINGS, settings_file(["read", "bash"], "high"))
    with ToolCallingPolicy("true") as server:
        runner, policy = _real_runner(tmp_path, server, model_settings={"reasoning": True})
        outcome = asyncio.run(runner.run(spec, TASK, policy))

    assert outcome.exit_code == 0
    assert (
        json.loads((Path(outcome.workspace_location or "") / "delivery.json").read_text())["thinking_level"] == "high"
    )
    assert PiDeliveryProbe().delivered(outcome, spec)[SETTINGS] is True
    assert outcome.final_output == "done"


@pytest.mark.integration
@pytest.mark.slow
@needs_pi
def test_the_real_worker_passes_the_runner_checks_against_the_stand_in_policy(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/e2e/check_runner.py",
            *("--config", "configs/e2e/pi.json"),
            *("--work-dir", str(tmp_path / "check")),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "runner check: ok" in result.stdout
