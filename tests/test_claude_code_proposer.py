"""The Claude Code proposer, driven against a fake ``claude`` executable that records how it was called."""

from __future__ import annotations

import json
import os
import stat
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

from honing.core import ContractError, ProposerAgent
from honing.e2e.components import PipelineContext
from honing.e2e.meta_harness import claude_code, method
from honing.harness.proposers import DEFAULT_TOOLS, TIMEOUT_EXIT_CODE, ClaudeCodeProposer

FAKE_CLAUDE = """#!/bin/sh
pwd -P > "$FAKE_CLAUDE_RECORD.cwd"
for arg in "$@"; do printf '%s\\0' "$arg"; done > "$FAKE_CLAUDE_RECORD.argv"
env > "$FAKE_CLAUDE_RECORD.env"
echo '{"type": "result"}'
echo 'a warning' >&2
if [ -n "$FAKE_CLAUDE_CHILD" ]; then sleep 30 & echo $! > "$FAKE_CLAUDE_RECORD.child"; fi
if [ -n "$FAKE_CLAUDE_SLEEP" ]; then sleep "$FAKE_CLAUDE_SLEEP"; fi
exit "${FAKE_CLAUDE_EXIT:-0}"
"""
# The fake's own controls; the proposer passes nothing else through unless told to.
FAKE_VARIABLES = ("FAKE_CLAUDE_RECORD", "FAKE_CLAUDE_SLEEP", "FAKE_CLAUDE_EXIT", "FAKE_CLAUDE_CHILD")


def _proposer(tmp_path: Path, **options: Any) -> ClaudeCodeProposer:
    return ClaudeCodeProposer("opus", tmp_path / "logs", **{"inherit_environment": FAKE_VARIABLES, **options})


@pytest.fixture
def record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Put the fake ``claude`` first on PATH; it writes its argv and cwd beside the returned path."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    claude = bin_dir / "claude"
    claude.write_text(FAKE_CLAUDE)
    claude.chmod(claude.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    path = tmp_path / "record"
    monkeypatch.setenv("FAKE_CLAUDE_RECORD", str(path))
    return path


def _argv(record: Path) -> list[str]:
    return record.with_suffix(".argv").read_text().split("\0")[:-1]


def _env(record: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in record.with_suffix(".env").read_text().splitlines() if "=" in line)


@pytest.mark.integration
def test_the_cli_runs_headless_in_the_workdir_with_the_skill_appended(tmp_path: Path, record: Path) -> None:
    settings = tmp_path / "settings.json"
    proposer = _proposer(tmp_path, effort="max", settings=settings)
    workdir = tmp_path / "archive"
    workdir.mkdir()

    assert isinstance(proposer, ProposerAgent)
    assert proposer.run("Run iteration 1.", workdir, instructions="# Meta-Harness") == 0

    assert Path(record.with_suffix(".cwd").read_text().strip()) == workdir.resolve()
    argv = _argv(record)
    assert argv[:3] == ["-p", "Run iteration 1.", "--output-format"]
    # Tools are approved by name, so permission rules and the sandbox still apply.
    assert "--dangerously-skip-permissions" not in argv
    assert argv[argv.index("--settings") + 1] == str(settings)
    assert argv[argv.index("--model") + 1] == "opus"
    assert argv[argv.index("--setting-sources") + 1] == ""
    assert argv[argv.index("--tools") + 1] == ",".join(DEFAULT_TOOLS)
    start = argv.index("--allowedTools") + 1
    assert tuple(argv[start : start + len(DEFAULT_TOOLS)]) == DEFAULT_TOOLS
    assert {"--disable-slash-commands", "--strict-mcp-config", "--verbose"} <= set(argv)
    assert argv[argv.index("--append-system-prompt") + 1] == (
        "Follow these skill instructions:\n\n## Skill: meta-harness\n# Meta-Harness\n\n"
    )
    assert argv[argv.index("--effort") + 1] == "max"
    assert (tmp_path / "logs/session-0.jsonl").read_text() == '{"type": "result"}\n'
    assert (tmp_path / "logs/session-0.stderr").read_text() == "a warning\n"


@pytest.mark.integration
def test_each_session_gets_its_own_log_and_returns_the_cli_exit_code(
    tmp_path: Path, record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proposer = _proposer(tmp_path)
    proposer.run("first", tmp_path, instructions="")
    monkeypatch.setenv("FAKE_CLAUDE_EXIT", "3")

    assert proposer.run("second", tmp_path, instructions="") == 3
    assert (tmp_path / "logs/session-1.jsonl").exists()
    assert "--effort" not in _argv(record) and "--settings" not in _argv(record)


@pytest.mark.integration
def test_a_session_past_its_timeout_is_killed_with_what_it_started_and_reports_124(
    tmp_path: Path, record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_CLAUDE_SLEEP", "5")
    monkeypatch.setenv("FAKE_CLAUDE_CHILD", "1")
    proposer = _proposer(tmp_path, timeout_seconds=0.5)

    assert proposer.run("slow", tmp_path, instructions="") == TIMEOUT_EXIT_CODE

    # A command the agent started, here a background sleep, dies with the session.
    child = int(record.with_suffix(".child").read_text())
    deadline = time.monotonic() + 5
    while _alive(child) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(child)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.integration
def test_the_cli_sees_only_a_minimal_environment(
    tmp_path: Path, record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("AWS_SECRET_ACCESS_KEY", "WANDB_API_KEY", "HF_TOKEN", "CLAUDECODE"):
        monkeypatch.setenv(name, "secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "the proposer's own key")
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "0")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy")

    _proposer(tmp_path, inherit_environment=(*FAKE_VARIABLES, "HTTPS_PROXY")).run("x", tmp_path, instructions="")

    env = _env(record)
    assert not {"AWS_SECRET_ACCESS_KEY", "WANDB_API_KEY", "HF_TOKEN", "CLAUDECODE"} & set(env)
    assert env["ANTHROPIC_API_KEY"] == "the proposer's own key" and env["CLAUDE_CODE_USE_BEDROCK"] == "0"
    assert env["HTTPS_PROXY"] == "http://proxy" and env["HOME"] == os.environ["HOME"]


@pytest.mark.integration
def test_a_base_url_points_the_cli_at_another_api(tmp_path: Path, record: Path) -> None:
    _proposer(tmp_path, base_url="http://127.0.0.1:30000").run("x", tmp_path, instructions="")

    assert _env(record)["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:30000"


@pytest.mark.unit
def test_inherit_environment_takes_names_only(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="names variables"):
        ClaudeCodeProposer("opus", tmp_path, inherit_environment=("KEY=value",))


@pytest.mark.unit
def test_a_missing_cli_is_a_contract_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))

    with pytest.raises(ContractError, match="claude"):
        ClaudeCodeProposer("opus", tmp_path / "logs").run("x", tmp_path, instructions="")


@pytest.mark.unit
def test_the_pipeline_factories_write_the_settings_and_place_the_archive(tmp_path: Path) -> None:
    context = PipelineContext(root=tmp_path / "repo", work_dir=tmp_path / "run-7")
    context.work_dir.mkdir()
    deny = {"permissions": {"deny": ["Read(~/**)", "Read(/{root}/**)"]}, "sandbox": {"enabled": True}}

    proposer = claude_code(context, "opus", settings=deny)
    built = method(
        context,
        {"factory": "honing.e2e.meta_harness:claude_code", "args": {"model": "opus"}},
        work_dir="{tmp}/meta-harness-{run}",
    )

    assert proposer.settings == tmp_path / "run-7/proposer-settings.json"
    written = json.loads(proposer.settings.read_text())
    assert written == {
        "permissions": {"deny": ["Read(~/**)", f"Read(/{tmp_path}/repo/**)"]},
        "sandbox": {"enabled": True},
    }
    assert built.work_dir == Path(tempfile.gettempdir()) / "meta-harness-run-7"
    assert isinstance(built.proposer, ClaudeCodeProposer)
