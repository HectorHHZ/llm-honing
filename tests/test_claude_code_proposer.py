"""The Claude Code proposer, driven against a fake ``claude`` executable that records how it was called."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from honing.core import ContractError, ProposerAgent
from honing.harness.proposers import DEFAULT_TOOLS, TIMEOUT_EXIT_CODE, ClaudeCodeProposer

FAKE_CLAUDE = """#!/bin/sh
pwd -P > "$FAKE_CLAUDE_RECORD.cwd"
for arg in "$@"; do printf '%s\\0' "$arg"; done > "$FAKE_CLAUDE_RECORD.argv"
echo '{"type": "result"}'
echo 'a warning' >&2
if [ -n "$FAKE_CLAUDE_SLEEP" ]; then sleep "$FAKE_CLAUDE_SLEEP"; fi
exit "${FAKE_CLAUDE_EXIT:-0}"
"""


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


@pytest.mark.integration
def test_the_cli_runs_headless_in_the_workdir_with_the_skill_appended(tmp_path: Path, record: Path) -> None:
    settings = tmp_path / "settings.json"
    proposer = ClaudeCodeProposer("opus", tmp_path / "logs", effort="max", settings=settings)
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
    proposer = ClaudeCodeProposer("opus", tmp_path / "logs")
    proposer.run("first", tmp_path, instructions="")
    monkeypatch.setenv("FAKE_CLAUDE_EXIT", "3")

    assert proposer.run("second", tmp_path, instructions="") == 3
    assert (tmp_path / "logs/session-1.jsonl").exists()
    assert "--effort" not in _argv(record) and "--settings" not in _argv(record)


@pytest.mark.integration
def test_a_session_past_its_timeout_is_killed_and_reports_124(
    tmp_path: Path, record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_CLAUDE_SLEEP", "5")
    proposer = ClaudeCodeProposer("opus", tmp_path / "logs", timeout_seconds=0.5)

    assert proposer.run("slow", tmp_path, instructions="") == TIMEOUT_EXIT_CODE


@pytest.mark.unit
def test_a_missing_cli_is_a_contract_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))

    with pytest.raises(ContractError, match="claude"):
        ClaudeCodeProposer("opus", tmp_path / "logs").run("x", tmp_path, instructions="")
