"""Log an end-to-end run to Weights & Biases when its config asks for it (#100).

A config without a ``logging`` block logs nothing and never imports ``wandb``.
With ``{"logging": {"wandb": {"project": ..., "tags": [...]}}}`` (and an
optional ``entity``; without one, the key's default) a run logs one W&B step per
schedule step (moves, rollbacks, and abandoned moves), keeps the pinned model,
the git commit and the config as the run's config, and uploads ``summary.json``
and ``releases/commits.jsonl`` when it ends. The uploaded copies name the work
directory, the repository and the home directory by placeholders, not by their
absolute paths, since a run may be public.
``"mode": "offline"`` writes the run under the work directory for a later
``wandb sync`` and needs no key; the default, ``"online"``, needs ``WANDB_API_KEY``.

Only what :func:`step_values` names is logged: each step's kind, label, result
and release; the numbers in the chain record the move wrote (the candidate's
gate, size and ``method_episodes`` metrics, and the selector's metrics); the
gate's episode count; and the API calls and tokens the caller passes. Keys and
environment variables are never logged. Neither are task prompts, agent outputs,
grader feedback, reflection prompts and replies, or decision reasons, because
any of them can quote private data. wandb's own capture of the console, code,
git state, installed packages and machine metadata is turned off. The files are
uploaded only if no value of the named key variables is in any file under the
work directory.

Only this module imports ``wandb`` (``tests/contracts/test_boundaries.py``), and
only when a run has a ``logging`` block.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from honing.core import ContractError

WANDB_KEY_ENV = "WANDB_API_KEY"
Mode = Literal["online", "offline"]
MODES: tuple[Mode, ...] = ("online", "offline")
UPLOADED = ("summary.json", "releases/commits.jsonl")
_WANDB_KEYS = frozenset({"project", "entity", "tags", "mode"})
# A step's fields, by the name each is logged under.
_STEP_FIELDS = {"move": "kind", "label": "label", "result": "result", "release": "release"}


@dataclass(frozen=True)
class WandbSettings:
    """Where a run is logged, from the config's ``logging.wandb`` block."""

    project: str
    entity: str | None = None
    tags: tuple[str, ...] = ()
    mode: Mode = "online"

    @property
    def required_env(self) -> tuple[str, ...]:
        return (WANDB_KEY_ENV,) if self.mode == "online" else ()


def logging_settings(block: object) -> WandbSettings | None:
    """The settings in a config's ``logging`` block, or ``None`` without one; refuses a block it cannot honour."""
    if block is None:
        return None
    if not isinstance(block, Mapping) or set(block) != {"wandb"}:
        raise ContractError('logging must be {"wandb": {...}}; wandb is the one supported backend')
    entry = block["wandb"]
    if not isinstance(entry, Mapping):
        raise ContractError("logging.wandb must be an object")
    unknown = set(entry) - _WANDB_KEYS
    if unknown:
        raise ContractError(f"unknown logging.wandb keys {sorted(unknown)}; use {sorted(_WANDB_KEYS)}")
    if not isinstance(entry.get("project"), str) or not entry["project"]:
        raise ContractError("logging.wandb.project must be a non-empty string")
    if "entity" in entry and (not isinstance(entry["entity"], str) or not entry["entity"]):
        raise ContractError("logging.wandb.entity, when given, must be a non-empty string")
    tags = entry.get("tags", [])
    if not isinstance(tags, list) or not all(isinstance(tag, str) and tag for tag in tags):
        raise ContractError("logging.wandb.tags must be a list of non-empty strings")
    mode = next((known for known in MODES if known == entry.get("mode", "online")), None)
    if mode is None:
        raise ContractError(f"logging.wandb.mode must be one of {list(MODES)}, got {entry['mode']!r}")
    # Checked without importing it, so a run that cannot log fails before its campaign starts.
    if "wandb" not in sys.modules and importlib.util.find_spec("wandb") is None:
        raise ContractError("the logging block needs the optional 'wandb' extra: pip install -e '.[wandb]'")
    return WandbSettings(project=entry["project"], entity=entry.get("entity"), tags=tuple(tags), mode=mode)


def git_commit(root: Path) -> str:
    """``HEAD`` of the checkout at ``root``, with ``-dirty`` if it has changes; ``unknown`` outside one."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return commit + ("-dirty" if dirty else "")


def run_config(config: Mapping[str, Any], root: Path, script: str) -> dict[str, Any]:
    """The W&B run config: the script, the git commit, the pinned model, and the config (names, never values)."""
    return {"script": script, "git_commit": git_commit(root), "model": config.get("model"), "config": dict(config)}


class ChainRecords:
    """The records a release chain appends, read a step at a time from where the last read stopped."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._offset = path.stat().st_size if path.is_file() else 0

    def new(self) -> list[dict[str, Any]]:
        if not self._path.is_file():
            return []
        with self._path.open("rb") as handle:
            handle.seek(self._offset)
            data = handle.read()
        complete = data[: data.rfind(b"\n") + 1]  # a line still being written is read next time
        self._offset += len(complete)
        return [json.loads(line) for line in complete.decode("utf-8").split("\n") if line.strip()]


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return float(value)


def step_values(
    step: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]] = (),
    usage: Mapping[str, object] | None = None,
) -> dict[str, float | str]:
    """What one step logs: named fields of the step, the numbers in its chain records, and its usage.

    Nothing else in ``step`` or a record is copied, so a field added to either
    later is not logged until it is named here.
    """
    values: dict[str, float | str] = {
        f"move/{logged}": step[name] for name, logged in _STEP_FIELDS.items() if isinstance(step.get(name), str)
    }
    if (episodes := _number(step.get("episodes"))) is not None:
        values["gate/episodes"] = episodes
    for record in records:
        op = record.get("op")
        if op == "skip":
            if (spent := _number(record.get("episodes_spent"))) is not None:
                values["candidate/method_episodes"] = spent
            continue
        if op not in ("publish", "reject"):
            continue
        values["move/published"] = 1.0 if op == "publish" else 0.0
        for prefix, source in (("candidate", record.get("candidate")), ("selector", record.get("decision"))):
            metrics = source.get("metrics", {}) if isinstance(source, Mapping) else {}
            for name, value in metrics.items():
                if (number := _number(value)) is not None:
                    values[f"{prefix}/{name}"] = number
    for name, value in (usage or {}).items():
        if (number := _number(value)) is not None:
            values[f"usage/{name}"] = number
    return values


def without_local_paths(text: str, work_dir: Path, root: Path) -> str:
    """``text`` with the work directory, the repository and the home directory named by placeholders."""
    for path, name in ((work_dir, "<work_dir>"), (root, "<repo>"), (Path.home(), "~")):
        for spelling in dict.fromkeys((str(path), str(path.resolve()))):
            text = text.replace(spelling, name)
    return text


def key_hits(directory: Path, names: Sequence[str]) -> list[str]:
    """Files under ``directory`` holding the value of any named variable (8 characters or more). Never prints one."""
    values = [os.environ[name].encode() for name in dict.fromkeys(names) if len(os.environ.get(name, "")) >= 8]
    if not values or not directory.is_dir():
        return []
    return [
        str(path.relative_to(directory))
        for path in sorted(directory.rglob("*"))
        if path.is_file() and not path.is_symlink() and any(value in path.read_bytes() for value in values)
    ]


class RunTracker(Protocol):
    """Where a run's steps go. ``finish`` returns what went wrong at the end, if anything."""

    @property
    def url(self) -> str | None: ...

    def log_step(self, index: int, values: Mapping[str, float | str]) -> None: ...

    def finish(self, work_dir: Path, key_names: Sequence[str], *, failed: bool) -> list[str]: ...


class NullTracker:
    """A run without a ``logging`` block: logs nothing."""

    @property
    def url(self) -> str | None:
        return None

    def log_step(self, index: int, values: Mapping[str, float | str]) -> None:
        return None

    def finish(self, work_dir: Path, key_names: Sequence[str], *, failed: bool) -> list[str]:
        return []


class WandbTracker:
    """Logs a run to W&B, with wandb's own capture of the machine turned off."""

    def __init__(
        self, settings: WandbSettings, *, name: str, work_dir: Path, root: Path, config: Mapping[str, Any]
    ) -> None:
        import wandb  # only here: a run without a logging block never imports it

        self._root = root

        self._run: Any = wandb.init(
            project=settings.project,
            entity=settings.entity,
            name=name,
            tags=list(settings.tags),
            mode=settings.mode,
            dir=str(work_dir),
            config=dict(config),
            settings=wandb.Settings(
                console="off",  # stdout and stderr carry step reasons and log lines
                save_code=False,
                disable_code=True,
                disable_git=True,  # a remote URL can hold a token; the commit is in the config
                x_disable_meta=True,  # host, user, argv, executable
                x_disable_machine_info=True,
                x_save_requirements=False,
            ),
        )

    @property
    def url(self) -> str | None:
        return getattr(self._run, "url", None)

    def log_step(self, index: int, values: Mapping[str, float | str]) -> None:
        self._run.log(dict(values), step=index)

    def finish(self, work_dir: Path, key_names: Sequence[str], *, failed: bool) -> list[str]:
        """Upload the run's files unless a key is in a file under ``work_dir``, then end the run."""
        names = [*key_names, WANDB_KEY_ENV]
        hits = key_hits(work_dir, names)
        problems = [f"{hit} holds a key; nothing was uploaded" for hit in hits]
        try:
            if not hits:
                upload = work_dir / "wandb-upload"
                for relative in UPLOADED:
                    if (work_dir / relative).is_file():
                        copy = upload / relative
                        copy.parent.mkdir(parents=True, exist_ok=True)
                        text = (work_dir / relative).read_text(encoding="utf-8")
                        copy.write_text(without_local_paths(text, work_dir, self._root), encoding="utf-8")
                        self._run.save(str(copy), base_path=str(upload), policy="now")
        finally:
            self._run.finish(exit_code=1 if failed or hits else 0)
        # wandb's own files are written until the run ends, so they are checked again after it.
        problems += [f"{hit} holds a key after the run ended" for hit in key_hits(work_dir, names) if hit not in hits]
        return problems


def start_tracker(
    settings: WandbSettings | None, *, name: str, work_dir: Path, root: Path, config: Mapping[str, Any]
) -> RunTracker:
    """A :class:`WandbTracker` for a config with a ``logging`` block, else a :class:`NullTracker`."""
    if settings is None:
        return NullTracker()
    return WandbTracker(settings, name=name, work_dir=work_dir, root=root, config=config)


__all__ = [
    "MODES",
    "UPLOADED",
    "WANDB_KEY_ENV",
    "ChainRecords",
    "Mode",
    "NullTracker",
    "RunTracker",
    "WandbSettings",
    "WandbTracker",
    "git_commit",
    "key_hits",
    "logging_settings",
    "run_config",
    "start_tracker",
    "step_values",
    "without_local_paths",
]
