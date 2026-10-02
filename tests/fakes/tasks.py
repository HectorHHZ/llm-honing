"""Task suite fakes: a manifest written into a temporary directory, and a sandbox that replays one result."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path

from honing.harness.graders import CommandResult


def task_line(task_id: str = "t1", split: str = "train", **overrides: object) -> dict[str, object]:
    """One valid manifest line; its workspace is ``workspaces/<task_id>``."""
    return {
        "task_id": task_id,
        "prompt": f"Fix {task_id}.",
        "workspace": f"workspaces/{task_id}",
        "grader": ["python3", "grade.py"],
        "split": split,
        **overrides,
    }


def write_manifest(root: Path, lines: Sequence[Mapping[str, object] | str]) -> Path:
    """Write ``lines`` as a manifest under ``root``, creating every ``workspaces/<id>`` a line names.

    A string is written as-is, so a test can write a line that is not JSON.
    """
    for line in lines:
        if isinstance(line, Mapping) and isinstance(line.get("task_id"), str):
            (root / "workspaces" / str(line["task_id"])).mkdir(parents=True, exist_ok=True)
    path = root / "manifest.jsonl"
    path.write_text("".join((line if isinstance(line, str) else json.dumps(line)) + "\n" for line in lines))
    return path


class FakeCommandSandbox:
    """Records what a grader asked of it and returns one scripted result for every command."""

    def __init__(self, result: CommandResult) -> None:
        self.result = result
        self.opened: list[Path] = []
        self.copied: list[Path] = []
        self.files: dict[str, bytes] = {}
        self.calls: list[tuple[tuple[str, ...], dict[str, str]]] = []

    async def copy_in(self, source: Path) -> None:
        self.copied.append(source)

    async def write_file(self, name: str, content: bytes) -> str:
        self.files[name] = content
        return f"/sandbox/scratch/{name}"

    async def exec(
        self, argv: Sequence[str], *, env: Mapping[str, str], timeout_seconds: float, max_output_bytes: int
    ) -> CommandResult:
        self.calls.append((tuple(argv), dict(env)))
        return self.result

    def opener(self) -> Callable[[Path], AbstractAsyncContextManager[FakeCommandSandbox]]:
        @asynccontextmanager
        async def open_sandbox(workspace: Path) -> AsyncGenerator[FakeCommandSandbox, None]:
            self.opened.append(workspace)
            yield self

        return open_sandbox
