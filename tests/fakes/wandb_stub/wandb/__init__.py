"""A stand-in for the ``wandb`` package: records every call as a JSON line, and sends nothing anywhere.

Put ``tests/fakes/wandb_stub`` first on ``sys.path`` (or ``PYTHONPATH``, for a
subprocess) and set ``HONING_FAKE_WANDB_LOG`` to the file the calls go to.
``HONING_FAKE_WANDB_FAIL`` makes ``init`` raise, and ``HONING_FAKE_WANDB_LEAK``
(``init`` or ``finish``) writes ``WANDB_API_KEY`` into the run directory then,
as a wandb that wrote the key to disk would.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

__version__ = "0.0.0-fake"


def _record(call: str, **fields: Any) -> None:
    with open(os.environ["HONING_FAKE_WANDB_LOG"], "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"call": call, **fields}, default=repr) + "\n")


def _leak(when: str, directory: str) -> None:
    if os.environ.get("HONING_FAKE_WANDB_LEAK") == when:
        leak = Path(directory) / "wandb" / f"leak-{when}.txt"
        leak.parent.mkdir(parents=True, exist_ok=True)
        leak.write_text(os.environ["WANDB_API_KEY"], encoding="utf-8")


class Settings:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class Run:
    url = "https://wandb.invalid/fake-entity/fake-project/runs/fake"

    def __init__(self, directory: str) -> None:
        self._directory = directory

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        _record("log", data=data, step=step)

    def save(self, glob_str: str, base_path: str | None = None, policy: str = "live") -> None:
        _record("save", path=glob_str, base_path=base_path, policy=policy, content=Path(glob_str).read_text())

    def finish(self, exit_code: int | None = None) -> None:
        _leak("finish", self._directory)
        _record("finish", exit_code=exit_code)


def init(**kwargs: Any) -> Run:
    settings = kwargs.pop("settings", None)
    _record("init", settings=None if settings is None else settings.kwargs, **kwargs)
    if os.environ.get("HONING_FAKE_WANDB_FAIL"):
        raise RuntimeError("fake wandb: init failed")
    _leak("init", kwargs["dir"])
    return Run(kwargs["dir"])
