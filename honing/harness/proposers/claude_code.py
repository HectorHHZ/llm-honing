"""Claude Code as a proposer agent: the ``claude`` CLI run headless, as upstream Meta-Harness runs it.

The command follows upstream's ``claude_wrapper.build_command``
(``reference_examples/terminal_bench_2/claude_wrapper.py``): the user's
settings, slash commands, MCP servers, and plugins are all shut out, and the
method's skill is appended to the system prompt rather than installed.

One deliberate difference: upstream skips every permission check
(``--dangerously-skip-permissions``). Here the proposer's tools are approved
by name with ``--allowedTools`` instead, so permission rules still apply, and
``settings`` (a file passed with ``--settings``) can deny paths outside the
archive and turn on Claude Code's Bash sandbox. Without such settings the
agent can read anything the user can, including other runs' ground truth.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

from honing.core import ContractError

#: Upstream's ``PROPOSER_ALLOWED_TOOLS``.
DEFAULT_TOOLS = ("Read", "Glob", "Grep", "Agent", "Write", "Edit", "Bash")
#: The exit code a timed-out session reports, as upstream's ``run_cmd`` does.
TIMEOUT_EXIT_CODE = 124


class ClaudeCodeProposer:
    """Runs ``claude -p`` in the working directory and logs its stream-json events under ``log_dir``.

    Each session writes ``session-<k>.jsonl`` (stdout) and ``session-<k>.stderr``,
    ``<k>`` the next unused index.
    """

    def __init__(
        self,
        model: str,
        log_dir: Path,
        *,
        tools: Sequence[str] = DEFAULT_TOOLS,
        effort: str | None = None,
        timeout_seconds: float | None = None,
        settings: Path | None = None,
    ) -> None:
        self.model = model
        self.log_dir = log_dir
        self.tools = tuple(tools)
        self.effort = effort
        self.timeout_seconds = timeout_seconds
        self.settings = settings

    def command(self, prompt: str, instructions: str, plugin_dir: Path) -> list[str]:
        command = [
            "claude",
            "-p",
            prompt,
            "--output-format",
            "stream-json",
            "--verbose",
            "--model",
            self.model,
            "--setting-sources",
            "",
        ]
        if self.tools:
            command.extend(["--tools", ",".join(self.tools), "--allowedTools", *self.tools])
        command.extend(["--disable-slash-commands", "--strict-mcp-config", "--plugin-dir", str(plugin_dir)])
        command.extend(
            [
                "--append-system-prompt",
                f"Follow these skill instructions:\n\n## Skill: meta-harness\n{instructions}\n\n",
            ]
        )
        if self.settings is not None:
            command.extend(["--settings", str(self.settings)])
        if self.effort:
            command.extend(["--effort", self.effort])
        return command

    def run(self, prompt: str, workdir: Path, *, instructions: str) -> int:
        executable = shutil.which("claude")
        if executable is None:
            raise ContractError("the Claude Code proposer needs the 'claude' CLI on PATH")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        index = 0
        while (self.log_dir / f"session-{index}.jsonl").exists():
            index += 1
        with tempfile.TemporaryDirectory() as plugin_dir:
            command = [executable, *self.command(prompt, instructions, Path(plugin_dir))[1:]]
            with (
                (self.log_dir / f"session-{index}.jsonl").open("w", encoding="utf-8") as stdout,
                (self.log_dir / f"session-{index}.stderr").open("w", encoding="utf-8") as stderr,
            ):
                try:
                    completed = subprocess.run(
                        command,
                        cwd=workdir,
                        stdin=subprocess.DEVNULL,
                        stdout=stdout,
                        stderr=stderr,
                        timeout=self.timeout_seconds,
                        check=False,
                    )
                except subprocess.TimeoutExpired:
                    return TIMEOUT_EXIT_CODE
        return completed.returncode
