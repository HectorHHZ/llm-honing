"""Serve NanoJev checkpoints on demand in a campaign: the pinned release and the checkpoints its trainers write.

:class:`NanoJevModelSource` materializes the pinned release as the campaign's
first model. :func:`nanojev_endpoints` is a
:class:`~honing.core.PolicyEndpoints` that starts Honing's decision server
(``honing.backends.jev.server --family nanojev``) for each checkpoint a runner
asks for, on a free GPU, reusing
:class:`~honing.backends.sglang.endpoints.SGLangPolicyEndpoints`, which
already keeps one server per checksum and stops the least recently used one
when the GPUs run out. A trained checkpoint is recognised by the
``checkpoint.json`` manifest its trainer wrote next to it
(:class:`~honing.backends.jev.nanojev_trainer.NanoJevTrainer`,
:class:`~honing.backends.jev.direction_edit.NanoJevDirectionEditTrainer`); the
pinned release needs none.
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import honing
from honing.backends.jev.nanojev import nanojev_snapshot, verify_published, weights_checksum
from honing.backends.jev.nanojev_trainer import release_path
from honing.backends.sglang.endpoints import ServingSettings, SGLangPolicyEndpoints, _stop_server
from honing.backends.sglang.engine import die_with_parent
from honing.core import ContractError, ModelSnapshot

REPO = Path(honing.__file__).resolve().parents[1]


@dataclass(frozen=True)
class NanoJevServing:
    """How NanoJev servers are started: the pinned source and release, the GPUs, and the server's settings."""

    pins_path: Path
    source_dir: Path
    gpus: tuple[str, ...]
    model: str = "nanojev"
    python: Path = Path(sys.executable)
    max_length: int = 4096
    inference: str = "tree"
    log_dir: Path | None = None

    def __post_init__(self) -> None:
        if not self.gpus:
            raise ValueError("NanoJevServing needs at least one GPU")


class NanoJevModelSource:
    """Satisfies :class:`~honing.core.ModelSource`: the pinned NanoJev release, checked against its hashes."""

    def __init__(self, pins_path: Path, model: str = "nanojev") -> None:
        self._pins_path = pins_path
        self._model = model

    def materialize(self, work_dir: Path) -> ModelSnapshot:
        pins = json.loads(self._pins_path.read_text())
        checkpoint = release_path(pins, self._model)
        verify_published(checkpoint)
        return nanojev_snapshot(checkpoint)


def trained_manifest(checkpoint: Path) -> Path | None:
    """The manifest a NanoJev trainer wrote for ``checkpoint``, if it is one of theirs."""
    manifest = checkpoint.parent / "checkpoint.json"
    if not manifest.exists():
        return None
    try:
        record: dict[str, Any] = json.loads(manifest.read_text())
    except json.JSONDecodeError:
        return None
    if record.get("kind") != "trained-nanojev" or Path(str(record.get("path"))).resolve() != checkpoint.resolve():
        return None
    return manifest


def start_nanojev_server(serving: NanoJevServing, checkpoint: Path, key: str, gpu: str, stack: ExitStack) -> str:
    """Start the decision server for one checkpoint on one GPU; register its shutdown on ``stack``; return its URL."""
    if weights_checksum(checkpoint) != key:
        raise ContractError(f"{checkpoint} does not hold the weights with checksum {key[:12]}")
    manifest = trained_manifest(checkpoint)
    if manifest is None:
        # The release ships its published checksums; a trained checkpoint ships a manifest. Nothing is
        # downloaded here: this runs inside a runner's event loop, where a download's thread pool would
        # outlive the campaign.
        if not (checkpoint / "SHA256_MANIFEST.json").exists():
            raise ContractError(
                f"{checkpoint} is neither the pinned NanoJev release nor a trained checkpoint with a manifest"
            )
        verify_published(checkpoint)
    command = [
        str(serving.python), "-m", "honing.backends.jev.server",
        "--family", "nanojev", "--pins", str(serving.pins_path), "--nanojev-source", str(serving.source_dir),
        "--model", serving.model, "--gpu", gpu, "--nanojev-inference", serving.inference,
        "--max-len", str(serving.max_length), "--port", "0",
        *(() if manifest is None else ("--checkpoint-manifest", str(manifest))),
    ]  # fmt: skip
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")]))}
    log = None
    if serving.log_dir is not None:
        serving.log_dir.mkdir(parents=True, exist_ok=True)
        log = serving.log_dir / f"decision-{key[:16]}.log"
    errors = log.open("ab") if log is not None else subprocess.DEVNULL
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=errors, env=env, text=True, preexec_fn=die_with_parent
    )
    stack.callback(_stop_server, process)
    line = process.stdout.readline() if process.stdout else ""
    try:
        ready: dict[str, Any] = json.loads(line)
    except json.JSONDecodeError:
        raise ContractError(f"the NanoJev decision server for {key[:12]} did not start; see {log}") from None
    url: str = ready["url"]
    return url


def nanojev_endpoints(serving: NanoJevServing) -> SGLangPolicyEndpoints:
    """Endpoints that start a NanoJev decision server per checkpoint. Close them, or use them as a context manager."""
    settings = ServingSettings(
        python=serving.python, gpus=serving.gpus, runtime="jev_decision", verify=False, log_dir=serving.log_dir
    )
    return SGLangPolicyEndpoints(settings, start=functools.partial(start_nanojev_server, serving))


__all__ = ["NanoJevModelSource", "NanoJevServing", "nanojev_endpoints", "start_nanojev_server", "trained_manifest"]
