"""A task suite manifest: one JSON line per task, with its workspace, its grader, and its split.

Each line is one JSON object with exactly these keys:

- ``task_id``: unique within the suite;
- ``prompt``: what the agent is asked to do;
- ``workspace``: a directory copied into the sandbox before the episode;
- ``grader``: the grader command, as an argv list, run in the sandbox after the episode;
- ``split``: ``train`` (a method's minibatches), ``val`` (the gate), or ``heldout``
  (reported only, never used to select);
- ``grader_files`` (optional): a directory of hidden tests and grading scripts. The
  grader gets its own copy, outside the agent's workspace, and runs there; it finds the
  agent's files through ``HONING_WORKSPACE``. Without it, the grader runs in the workspace;
- ``metadata`` (optional): a JSON object handed to the grader, for task arguments that do
  not belong in the prompt.

``workspace`` and ``grader_files`` are relative to the manifest's directory. Episode
runners and graders receive a core :class:`~honing.core.Task`, which holds only the id
and the prompt, and look the rest up in the suite by ``task_id``.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Literal, get_args

from honing.core import ContractError, Task, require, require_choice

Split = Literal["train", "val", "heldout"]

_REQUIRED_KEYS = ("task_id", "prompt", "workspace", "grader", "split")
_OPTIONAL_KEYS = ("grader_files", "metadata")


@dataclass(frozen=True)
class ManifestTask:
    """One manifest line, with its directories resolved and its content hashed.

    ``content_hash`` covers the line itself and every file under ``workspace`` and
    ``grader_files``, so a task whose files change under the same ``task_id`` gets a
    different hash.
    """

    task_id: str
    prompt: str
    workspace: Path
    grader: tuple[str, ...]
    split: Split
    content_hash: str
    grader_files: Path | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require(bool(self.task_id), "task_id must be a non-empty string")
        require(bool(self.prompt), "prompt must be a non-empty string")
        require_choice(self.split, get_args(Split), "split")
        require(len(self.grader) > 0, "grader must name a command")

    @property
    def task(self) -> Task:
        return Task(task_id=self.task_id, prompt=self.prompt)


@dataclass(frozen=True)
class TaskSuite:
    """The tasks of one manifest, in file order."""

    entries: tuple[ManifestTask, ...]

    def __post_init__(self) -> None:
        require(len(self.entries) > 0, "a task suite needs at least one task")
        ids = [entry.task_id for entry in self.entries]
        require(len(set(ids)) == len(ids), "a task suite names each task once")

    def tasks(self, split: Split, *, limit: int | None = None) -> tuple[Task, ...]:
        """The split's tasks in file order, so a gate and a method see the same order every run."""
        require_choice(split, get_args(Split), "split")
        require(limit is None or limit >= 0, f"limit must be >= 0, got {limit}")
        chosen = [entry.task for entry in self.entries if entry.split == split]
        return tuple(chosen[:limit])

    def entry(self, task_id: str) -> ManifestTask:
        for entry in self.entries:
            if entry.task_id == task_id:
                return entry
        raise ContractError(f"task {task_id!r} is not in the task suite")

    def split_of(self, task_id: str) -> Split:
        return self.entry(task_id).split

    @property
    def checksum(self) -> str:
        """A content address for the whole suite: its task ids, in order, and their content hashes."""
        canonical = json.dumps([[entry.task_id, entry.content_hash] for entry in self.entries], separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def assert_disjoint(self, *selections: Sequence[Task]) -> None:
        """Refuse task selections that share a task, name a task not in the suite, or use a held-out task.

        Call it with the tasks a campaign is about to use, for example a method's
        minibatch tasks and the gate's tasks: ids are unique within a suite, so the
        splits themselves cannot overlap, but two selections drawn from one split can.
        """
        owner: dict[str, int] = {}
        for index, selection in enumerate(selections):
            for task in selection:
                entry = self.entry(task.task_id)
                require(entry.split != "heldout", f"task {task.task_id!r} is held out and must not be selected on")
                first = owner.setdefault(task.task_id, index)
                require(
                    first == index,
                    f"task {task.task_id!r} is in selections {first} and {index}; selections must be disjoint",
                )


def _relative_directory(record: Mapping[str, object], key: str, root: Path) -> Path:
    value = record[key]
    if not isinstance(value, str) or not value:
        raise ContractError(f"{key} must be a non-empty path string")
    posix = PurePosixPath(value)
    require(not posix.is_absolute(), f"{key} must be relative to the manifest, got {value!r}")
    require(".." not in posix.parts, f"{key} may not leave the manifest's directory, got {value!r}")
    path = root / value
    require(path.is_dir(), f"{key} {value!r} is not a directory next to the manifest")
    return path


def _tree_digest(root: Path) -> list[list[str]]:
    """Every entry under ``root``, sorted, with a file's sha256 or a link's target."""
    entries: list[list[str]] = []
    for directory, subdirectories, files in os.walk(root):
        subdirectories.sort()
        base = Path(directory)
        for name in sorted([*subdirectories, *files]):
            path = base / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                entries.append([relative, "link", os.readlink(path)])
            elif path.is_dir():
                entries.append([relative, "dir", ""])
            else:
                entries.append([relative, "file", hashlib.sha256(path.read_bytes()).hexdigest()])
    return entries


def _parse_line(record: object, root: Path) -> ManifestTask:
    if not isinstance(record, dict):
        raise ContractError("a manifest line must be a JSON object")
    missing = [key for key in _REQUIRED_KEYS if key not in record]
    require(not missing, f"missing keys {missing}")
    unknown = sorted(set(record) - {*_REQUIRED_KEYS, *_OPTIONAL_KEYS})
    require(not unknown, f"unknown keys {unknown}")
    for key in ("task_id", "prompt"):
        require(isinstance(record[key], str), f"{key} must be a string")
    require_choice(record["split"], get_args(Split), "split")
    grader = record["grader"]
    require(
        isinstance(grader, list) and len(grader) > 0 and all(isinstance(arg, str) and arg for arg in grader),
        "grader must be a non-empty list of non-empty strings (an argv, not a shell command)",
    )
    metadata = record.get("metadata", {})
    require(isinstance(metadata, dict), "metadata must be a JSON object")
    workspace = _relative_directory(record, "workspace", root)
    grader_files = _relative_directory(record, "grader_files", root) if "grader_files" in record else None
    content = {
        "line": record,
        "workspace": _tree_digest(workspace),
        "grader_files": None if grader_files is None else _tree_digest(grader_files),
    }
    content_hash = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return ManifestTask(
        task_id=record["task_id"],
        prompt=record["prompt"],
        workspace=workspace,
        grader=tuple(grader),
        split=record["split"],
        content_hash=content_hash.hexdigest(),
        grader_files=grader_files,
        metadata=metadata,
    )


def load_task_suite(path: Path) -> TaskSuite:
    """Read a manifest, refusing a bad line with its file and line number. Blank lines are skipped."""
    root = path.parent
    entries: list[ManifestTask] = []
    first_line: dict[str, int] = {}
    # Only "\n" ends a line: ``str.splitlines`` also splits on characters such as
    # U+2028 that JSON allows unescaped inside a string.
    for number, raw in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1):
        line = raw.removesuffix("\r")
        if not line.strip():
            continue
        try:
            try:
                record = json.loads(line)
            except ValueError as error:  # JSONDecodeError, or an integer longer than Python will convert
                raise ContractError(f"not valid JSON: {error}") from None
            entry = _parse_line(record, root)
            if entry.task_id in first_line:
                raise ContractError(f"duplicate task_id {entry.task_id!r}, first on line {first_line[entry.task_id]}")
        except ContractError as error:
            raise ContractError(f"{path}:{number}: {error}") from None
        first_line[entry.task_id] = number
        entries.append(entry)
    if not entries:
        raise ContractError(f"{path}: the manifest has no tasks")
    return TaskSuite(tuple(entries))
