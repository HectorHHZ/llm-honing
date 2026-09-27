"""Load a pinned Laya checkpoint and answer typed questions with it.

This is the one implementation of Laya inference. The HTTP server and the RA2
integration's stdio worker both use it, so the two paths cannot drift apart.
Every check the original worker made is kept: the Laya source and model must
match their pins, a trained checkpoint must match its recorded hash, and the
GPU must be free and locked for this process.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from honing.backends.jev.token_budget import prepare_request

HEAD_MAX_LEN = 512


@dataclass(frozen=True)
class LayaSettings:
    """Everything needed to load one Laya checkpoint on one GPU."""

    pins_path: Path
    model_type: str
    model_config_dir: Path
    source_dir: Path
    gpu: str = "0"
    checkpoint_manifest: Path | None = None
    lock_dir: Path | None = None
    max_len: int = 2048

    def __post_init__(self) -> None:
        if self.max_len <= 0:
            raise ValueError("max_len must be positive")


def _sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def _gpu_uuid(gpu: str) -> str:
    devices = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    matches = [row.split(",")[1].strip() for row in devices.splitlines() if row.split(",")[0].strip() == gpu]
    if not matches:
        raise ValueError(f"GPU index {gpu} not found")
    return matches[0]


def _resolve_model(settings: LayaSettings, pins: Mapping[str, Any], revision: str) -> dict[str, Any]:
    if settings.model_type not in pins["models"]:
        raise ValueError("Unknown Laya model type")
    model: dict[str, Any] = json.loads((settings.model_config_dir / f"{settings.model_type}.json").read_text())
    for key in ("repo", "revision"):
        if model[key] != pins["models"][settings.model_type][key]:
            raise ValueError("Model config differs from pins.json")
    if settings.checkpoint_manifest is None:
        return model
    trained = json.loads(settings.checkpoint_manifest.read_text())
    if trained.get("kind") != "trained-laya" or trained.get("sourceRevision") != revision:
        raise ValueError("Unrecognized trained checkpoint manifest")
    if trained.get("parent", {}).get("revision") != pins["models"][settings.model_type]["revision"]:
        raise ValueError("Trained checkpoint parent does not match selected model")
    if _sha256(Path(trained["path"]) / "model.safetensors") != trained["weightsSha256"]:
        raise ValueError("Trained checkpoint weight hash mismatch")
    model = {key: trained[key] for key in ("repo", "revision", "path", "weightsSha256", "parent")}
    model["trainingManifestSha256"] = hashlib.sha256(settings.checkpoint_manifest.read_bytes()).hexdigest()
    return model


class LayaDecisionModel:
    """A loaded Laya agent. Create it with :meth:`load`, which holds the GPU lock while in use."""

    def __init__(self, agent: Any, metadata: Mapping[str, object], max_len: int) -> None:
        self._agent = agent
        self._metadata = dict(metadata)
        self._max_len = max_len
        from laya.common import build_sequence, render_options

        self._build_sequence = build_sequence
        self._render_options = render_options

    @property
    def metadata(self) -> Mapping[str, object]:
        return self._metadata

    @classmethod
    @contextlib.contextmanager
    def load(cls, settings: LayaSettings) -> Iterator[LayaDecisionModel]:
        os.environ.update(USE_TF="0", TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1")
        pins = json.loads(settings.pins_path.read_text())
        source = settings.source_dir
        revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if revision != pins["layaSource"]["revision"]:
            raise ValueError("The Laya source must use the revision in pins.json")
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        uuid = _gpu_uuid(settings.gpu)
        lock_dir = settings.lock_dir or Path(tempfile.gettempdir()) / f"laya-{os.getuid()}"
        lock_dir.mkdir(parents=True, exist_ok=True)
        with (lock_dir / f"{uuid}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            used = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True
            )
            if uuid in used:
                raise RuntimeError(f"GPU {settings.gpu} already has a compute process")
            os.environ["CUDA_VISIBLE_DEVICES"] = uuid
            model = _resolve_model(settings, pins, revision)
            with contextlib.redirect_stdout(sys.stderr):
                import laya
                import torch
                import transformers

                torch.set_num_threads(4)
                torch.manual_seed(1)
                agent = laya.load(model["path"], device="cuda")
            metadata = {
                **model,
                "sourceCommit": revision,
                "layaVersion": laya.__version__,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "gpu": settings.gpu,
                "gpuUuid": uuid,
                "dtype": str(agent.dtype),
                "maxLen": settings.max_len,
                "headMaxLen": HEAD_MAX_LEN,
                "checkpointMaxLen": agent.cfg.get("max_len"),
                "temperatureByOptions": agent.temperature_by_options,
                "python": sys.version.split()[0],
            }
            yield cls(agent, metadata, settings.max_len)

    def predict(self, parts: Mapping[str, object], questions: Mapping[str, object]) -> Mapping[str, object]:
        import torch

        started = time.perf_counter()
        state, audit, trimmed = prepare_request(
            self._agent, parts, questions, self._max_len, HEAD_MAX_LEN, self._build_sequence, self._render_options
        )
        with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
            result = self._agent.predict(state, questions, max_len=self._max_len, head_max_len=HEAD_MAX_LEN)
        return {
            "answers": result["answers"],
            "modelResult": result,
            "stateText": state,
            "tokenAudit": audit,
            "trimmed": trimmed,
            "latencyMs": (time.perf_counter() - started) * 1000,
        }
