"""Claude Code as a proposer agent: the ``claude`` CLI run headless, as upstream Meta-Harness runs it.

The command follows upstream's ``claude_wrapper.build_command``
(``reference_examples/terminal_bench_2/claude_wrapper.py``): the user's
settings, slash commands, MCP servers, and plugins are all shut out, and the
method's skill is appended to the system prompt rather than installed.

Deliberate differences from upstream:

- Upstream skips every permission check (``--dangerously-skip-permissions``).
  Here the proposer's tools are approved by name with ``--allowedTools``
  instead, so permission rules still apply, and ``settings`` (a file passed
  with ``--settings``) can deny paths outside the archive and turn on Claude
  Code's Bash sandbox. Without such settings the agent can read any file the
  user can, including other runs' ground truth.
- The CLI gets a minimal environment, not Honing's: cloud and tracking
  credentials stay out of reach of the agent's Bash.
- The CLI runs in its own process group, so a timeout also kills the
  commands and subagents it started.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from honing.core import ContractError, require

#: Upstream's ``PROPOSER_ALLOWED_TOOLS``.
DEFAULT_TOOLS = ("Read", "Glob", "Grep", "Agent", "Write", "Edit", "Bash")
#: The exit code a timed-out session reports, as upstream's ``run_cmd`` does.
TIMEOUT_EXIT_CODE = 124

# What the CLI inherits from Honing's environment: enough to run and to find its own login. Everything else,
# such as AWS_* or WANDB_API_KEY, stays out of reach of the agent's Bash.
_INHERITED = frozenset({"PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "TMPDIR", "TERM"})
_INHERITED_PREFIXES = ("LC_", "ANTHROPIC_", "CLAUDE_CODE_")


class ClaudeCodeProposer:
    """Runs ``claude -p`` in the working directory and logs its stream-json events under ``log_dir``.

    Each session writes ``session-<k>.jsonl`` (stdout) and ``session-<k>.stderr``,
    ``<k>`` the next unused index.

    Args:
        model: The model the CLI runs, by alias or full name.
        log_dir: Where sessions are logged.
        tools: The tools the agent may use, each approved by name.
        effort: The CLI's ``--effort``, if any.
        timeout_seconds: A session still running after this is killed with
            every process it started, and reports exit code 124.
        settings: A settings file passed with ``--settings``.
        base_url: An Anthropic-compatible API root to call instead of
            Anthropic's, such as an engine Honing serves; it becomes the CLI's
            ``ANTHROPIC_BASE_URL``.
        inherit_environment: Names of further variables the CLI takes from
            Honing's environment, such as ``HTTPS_PROXY``. The CLI otherwise
            sees only ``PATH``, ``HOME``, ``USER``, ``LOGNAME``, ``SHELL``,
            ``LANG``, ``TMPDIR``, ``TERM``, ``LC_*``, ``ANTHROPIC_*`` and
            ``CLAUDE_CODE_*``.
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
        base_url: str | None = None,
        inherit_environment: Iterable[str] = (),
    ) -> None:
        self.model = model
        self.log_dir = log_dir
        self.tools = tuple(tools)
        self.effort = effort
        self.timeout_seconds = timeout_seconds
        self.settings = settings
        self.base_url = base_url
        self.inherit = frozenset(inherit_environment)
        for name in self.inherit:
            require(bool(name) and "=" not in name, f"inherit_environment names variables, got {name!r}")

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

    def environment(self, source: Mapping[str, str]) -> dict[str, str]:
        """The CLI's environment, taken from ``source`` (normally ``os.environ``)."""
        inherited = {
            name: value
            for name, value in source.items()
            if name in _INHERITED or name.startswith(_INHERITED_PREFIXES) or name in self.inherit
        }
        if self.base_url is not None:
            inherited["ANTHROPIC_BASE_URL"] = self.base_url
        return inherited

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
                process = subprocess.Popen(
                    command,
                    cwd=workdir,
                    env=self.environment(os.environ),
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
                try:
                    return process.wait(timeout=self.timeout_seconds)
                except subprocess.TimeoutExpired:
                    _kill_group(process.pid)
                    process.wait()
                    return TIMEOUT_EXIT_CODE
                except BaseException:
                    _kill_group(process.pid)
                    process.wait()
                    raise


def _kill_group(pid: int) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGKILL)
