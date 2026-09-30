"""Load a pinned Kev checkpoint and answer typed questions with it.

Kev (``jaredpalmer/kev``) is a decision model like Laya: one state and a set of
typed questions in, a probability for every option out, in one forward pass. A
checkpoint is a LoRA adapter plus a pointer head (``head.pt``) on a Qwen base.
This module is the one implementation of Kev inference in Honing; the HTTP
decision server and the RA2 integration's stdio worker both use it.

Inference goes through Kev's own code at the commit in ``pins.json``
(``kevSource``): :class:`kev.checkpoint.Checkpoint` loads the checkpoint and
:func:`kev.api.to_record` and :func:`kev.api.to_answers` map the typed questions
to and from the model, as ``kev.serve`` does. What Honing adds is what Laya's
worker already does for RA2: the Kev source, model, and base must match their
pins, a trained checkpoint must match its recorded hashes, the GPU is locked for
this process, and the state is packed from its four sections and trimmed by
whole entries (never truncated) to fit the model's context, with every trim
reported. Each request runs on its own, with no state cached across requests,
so an answer does not depend on the requests before it.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from honing.backends.jev.laya import gpu_lock
from honing.backends.jev.token_budget import SECTIONS
from honing.core import ArchitectureSpec, CheckpointRef, LayerShape, ModelSnapshot

FAMILY = "kev"
# The same wording Laya's worker uses, so the RA2 grader scores an overrun the same way for either family.
QUESTION_OVERRUN = "Question {key} exceeds lossless option/instruction budget ({detail})"
STATE_OVERRUN = "Essential state exceeds token budget; refusing silent truncation"


@dataclass(frozen=True)
class KevSettings:
    """Everything needed to load one Kev checkpoint on one GPU.

    ``model`` names an entry of ``pins.json``'s ``kevModels``. ``dtype`` is
    ``bf16`` (``kev.serve``'s CUDA default) or ``fp32`` (the path Kev's
    reported numbers use). ``max_state`` and ``max_branch`` default to Kev's
    serving context.
    """

    pins_path: Path
    source_dir: Path
    model: str = "kev-0.8b"
    gpu: str = "0"
    checkpoint_manifest: Path | None = None
    lock_dir: Path | None = None
    dtype: str = "bf16"
    max_state: int | None = None
    max_branch: int | None = None

    def __post_init__(self) -> None:
        if self.dtype not in ("bf16", "fp32"):
            raise ValueError("dtype must be bf16 or fp32")


def _sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def weights_checksum(checkpoint: Path) -> str:
    """One checksum for a Kev checkpoint's trained weights: the adapter and the pointer head together."""
    parts = f"{_sha256(checkpoint / 'adapter_model.safetensors')}:{_sha256(checkpoint / 'head.pt')}"
    return hashlib.sha256(parts.encode()).hexdigest()


def import_kev(source_dir: Path, pins: Mapping[str, Any]) -> str:
    """Put the pinned Kev source on the import path and return its revision; refuse any other revision."""
    revision = subprocess.check_output(["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True).strip()
    if revision != pins["kevSource"]["revision"]:
        raise ValueError("The Kev source must use the revision in pins.json")
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    return revision


def resolve_checkpoint(settings: KevSettings, pins: Mapping[str, Any], revision: str) -> dict[str, Any]:
    """The checkpoint directory and what it is: the pinned release, or a trained checkpoint named by its manifest."""
    from kev.checkpoint import read_meta, resolve_run

    if settings.model not in pins.get("kevModels", {}):
        raise ValueError(f"Unknown Kev model {settings.model!r}")
    pin = pins["kevModels"][settings.model]
    if settings.checkpoint_manifest is None:
        path = Path(resolve_run(f"{pin['repo']}@{pin['revision']}"))
        model: dict[str, Any] = {"repo": pin["repo"], "revision": pin["revision"], "path": str(path)}
    else:
        trained = json.loads(settings.checkpoint_manifest.read_text())
        if trained.get("kind") != "trained-kev" or trained.get("sourceRevision") != revision:
            raise ValueError("Unrecognized trained checkpoint manifest")
        if trained.get("parent", {}).get("revision") != pin["revision"]:
            raise ValueError("Trained checkpoint parent does not match selected model")
        path = Path(trained["path"])
        model = {key: trained[key] for key in ("repo", "revision", "path", "parent")}
        model["trainingManifestSha256"] = hashlib.sha256(settings.checkpoint_manifest.read_bytes()).hexdigest()
    checksum = weights_checksum(path)
    if settings.checkpoint_manifest is not None and checksum != trained["weightsSha256"]:
        raise ValueError("Trained checkpoint weight hash mismatch")
    meta = read_meta(path)
    if (meta.base, meta.base_revision) != (pin["base"], pin["baseRevision"]):
        raise ValueError("The Kev checkpoint's base differs from pins.json")
    return {**model, "weightsSha256": checksum, "base": meta.base, "baseRevision": meta.base_revision}


def prepare_request(
    tok: Any, parts: Mapping[str, Any], questions: Mapping[str, Any], max_state: int, max_branch: int
) -> tuple[str, dict[str, dict[str, Any]], list[dict[str, str]]]:
    """Return the state text, a per-question token audit, and what was trimmed; fail rather than lose content.

    The state packs the four sections in order, as for Laya. When it does not
    fit, whole entries are removed from ``logs`` (oldest first), then
    ``memory`` and ``details`` (newest first). A question fits when its branch
    and the state together stay within Kev's row limit, and the state within
    its state limit.
    """
    from kev.api import SystemOneRequest, to_record
    from kev.model import user_tokens

    record, _ = to_record(SystemOneRequest(state="", questions=dict(questions)))
    keys = list(questions)
    branches = {}
    for key, question in zip(keys, record["questions"], strict=True):
        length = (
            2
            + len(user_tokens(tok, question["instr"]))
            + sum(2 + len(user_tokens(tok, o)) for o in question["options"])
        )
        if length >= max_branch:
            raise ValueError(QUESTION_OVERRUN.format(key=key, detail=f"question {length}/{max_branch - 1} tokens"))
        branches[key] = (length, len(question["options"]))
    retained = {key: list(parts.get(key, [])) for key in SECTIONS}
    trimmed: list[dict[str, str]] = []
    while True:
        state = "\n".join(text for key in SECTIONS for text in retained[key])
        state_tokens = 1 + len(user_tokens(tok, state))
        audit = {}
        for key, (length, options) in branches.items():
            room = min(max_state, max_branch - length)
            audit[key] = {
                "stateTokens": state_tokens,
                "stateRoom": room,
                "inputTokens": state_tokens + length,
                "options": options,
                "stateTruncated": state_tokens > room,
            }
        if not any(entry["stateTruncated"] for entry in audit.values()):
            return state, audit, trimmed
        for section in ("logs", "memory", "details"):
            if retained[section]:
                text = retained[section].pop(0 if section == "logs" else -1)
                trimmed.append({"section": section, "text": text})
                break
        else:
            raise ValueError(STATE_OVERRUN)


class KevDecisionModel:
    """A loaded Kev checkpoint. Create it with :meth:`load`, which holds the GPU lock while in use."""

    def __init__(self, tok: Any, model: Any, metadata: Mapping[str, object], max_state: int, max_branch: int) -> None:
        self._tok = tok
        self._model = model
        self._metadata = dict(metadata)
        self._max_state = max_state
        self._max_branch = max_branch

    @property
    def metadata(self) -> Mapping[str, object]:
        return self._metadata

    @classmethod
    @contextlib.contextmanager
    def load(cls, settings: KevSettings) -> Iterator[KevDecisionModel]:
        os.environ.update(TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1")
        pins = json.loads(settings.pins_path.read_text())
        revision = import_kev(settings.source_dir, pins)
        with gpu_lock(settings.gpu, settings.lock_dir) as uuid:
            model_record = resolve_checkpoint(settings, pins, revision)
            with contextlib.redirect_stdout(sys.stderr):
                import peft
                import torch
                import transformers
                from kev.checkpoint import Checkpoint, LoadOptions
                from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE

                torch.manual_seed(1)
                dtype = torch.bfloat16 if settings.dtype == "bf16" else torch.float32
                checkpoint = Checkpoint(model_record["path"])
                tok, model = checkpoint.load("cuda", LoadOptions(dtype=dtype, backend="torch", cuda_graphs=False))
            max_state = settings.max_state or SERVE_MAX_STATE
            max_branch = settings.max_branch or SERVE_MAX_BRANCH
            metadata = {
                "family": FAMILY,
                "model": settings.model,
                **model_record,
                "sourceCommit": revision,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "peft": peft.__version__,
                "gpu": settings.gpu,
                "gpuUuid": uuid,
                "dtype": model.dtype,
                "temperature": model.head.temperature,
                "maxState": max_state,
                "maxBranch": max_branch,
                "python": sys.version.split()[0],
            }
            yield cls(tok, model, metadata, max_state, max_branch)

    def predict(self, parts: Mapping[str, object], questions: Mapping[str, object]) -> Mapping[str, object]:
        import torch
        from kev.api import SystemOneRequest, to_answers, to_record

        started = time.perf_counter()
        state, audit, trimmed = prepare_request(self._tok, parts, questions, self._max_state, self._max_branch)
        record, meta = to_record(SystemOneRequest(state=state, questions=dict(questions)))
        with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
            encoded = self._model.encode(
                self._tok, record, max_state=self._max_state, max_branch=self._max_branch, strict=True
            )
            probabilities, _ = self._model.probs_and_prefix(encoded)
        rows = [p.tolist() for p in probabilities]
        answers = to_answers(rows, meta)
        return {
            "answers": answers,
            "modelResult": {"answers": answers, "probabilities": rows, "temperature": self._model.head.temperature},
            "stateText": state,
            "tokenAudit": audit,
            "trimmed": trimmed,
            "latencyMs": (time.perf_counter() - started) * 1000,
        }


def kev_snapshot(checkpoint: Path, base_config: Path) -> ModelSnapshot:
    """A Kev checkpoint as a ModelSnapshot: the base's layer shape, and one checksum over adapter and head.

    ``base_config`` is the base model's ``config.json``. ``parameters`` counts
    the adapter's parameters, the part a trainer changes.
    """
    from honing.backends.jev.laya import safetensors_parameters

    config = json.loads(base_config.read_text())
    text = config.get("text_config", config)
    layer = LayerShape(attention_heads=int(text["num_attention_heads"]), ffn_size=int(text["intermediate_size"]))
    return ModelSnapshot(
        checkpoint=CheckpointRef(
            location=str(checkpoint), weight_format="hf_safetensors", checksum=weights_checksum(checkpoint)
        ),
        architecture=ArchitectureSpec(
            hidden_size=int(text["hidden_size"]), layers=(layer,) * int(text["num_hidden_layers"])
        ),
        parameters=safetensors_parameters(checkpoint / "adapter_model.safetensors"),
    )
