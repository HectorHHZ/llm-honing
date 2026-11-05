"""Train a Laya decision model on typed decisions, on one local GPU.

The training procedure is the one the RA2 experiment used (#49's
``train_laya.py``): every question of every decision becomes one example,
encoded exactly as inference encodes it; identical inputs are deduplicated and
must agree on their label; examples from validation groups are held out and
must not also appear in training; then full-weight cross-entropy over the
option logits, AdamW with separate encoder and head learning rates, bf16
autocast, gradient clipping at 1.0, and the checkpoint with the best validation
loss is kept. The data handling, the losses and the loops are the ones every
decision-model trainer shares (:mod:`honing.backends.jev.decision_training`);
this module loads, encodes and saves Laya checkpoints.

It sits behind :class:`~honing.core.Trainer`, so another trainer can replace it
through configuration. The trained checkpoint's manifest is one
:class:`~honing.backends.jev.laya.LayaDecisionModel` can serve.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from honing.backends.jev.decision_training import (
    Example,
    Forward,
    decision_examples,
    deduplicated_split,
    group_advantages,
    policy_gradient_epochs,
    policy_gradient_record,
    record_start,
    supervised_epochs,
)
from honing.backends.jev.laya import HEAD_MAX_LEN, gpu_lock
from honing.backends.jev.token_budget import prepare_request
from honing.core import (
    BackendSupport,
    CheckpointRef,
    ContractError,
    DecisionSample,
    ModelSnapshot,
    TrainingBatch,
    TrainJobSpec,
    TrainResult,
)

MAX_LEN = 2048
SUPPORT = BackendSupport(
    name="local-jev",
    architectures="uniform",
    train_numerics=frozenset({"bf16"}),
    weight_formats=frozenset({"hf_safetensors"}),
    algorithms=(("sft", "native"), ("grpo", "native")),
    contracts=frozenset({"typed_decisions"}),
)


@dataclass(frozen=True)
class JevTrainerSettings:
    """Where and how the trainer runs.

    ``model_record`` is the prepared record of the pinned base model (for example
    ``integrations/ra2/.cache/models/typed-decisions.json``); every trained
    checkpoint names it as its parent. Decisions whose ``group`` matches
    ``validation_groups`` are held out for validation; the default holds out seed
    101, as the RA2 experiment did (groups are ``map:seed:seat``).
    ``deterministic`` makes PyTorch use deterministic kernels, so the same data
    and seed give the same weights bit for bit; it is slower, and off by default.
    """

    laya_source: Path
    pins_path: Path
    model_record: Path
    work_dir: Path
    gpu: str = "0"
    batch_size: int = 8
    seed: int = 7
    encoder_lr: float = 1e-5
    head_lr: float = 1e-4
    validation_groups: str = r"^[^:]+:101:"
    min_validation: int = 50
    lock_dir: Path | None = None
    deterministic: bool = False
    clip_ratio: float = 0.2


def _sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def collate(items: Sequence[Mapping[str, Any]], pad: int, torch: Any, device: str) -> tuple[Any, ...]:
    n, length, width = len(items), max(len(x["ids"]) for x in items), max(len(x["markers"]) for x in items)
    ids = torch.full((n, length), pad, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    positions = torch.zeros((n, width), dtype=torch.long, device=device)
    mask = torch.zeros((n, width), dtype=torch.bool, device=device)
    for i, x in enumerate(items):
        ids[i, : len(x["ids"])] = torch.tensor(x["ids"], device=device)
        attention[i, : len(x["ids"])] = 1
        positions[i, : len(x["markers"])] = torch.tensor(x["markers"], device=device)
        mask[i, : len(x["markers"])] = True
    return (
        ids,
        attention,
        positions,
        mask,
        torch.zeros(n, dtype=torch.long, device=device),
        torch.tensor([x["label"] for x in items], device=device),
    )


class LocalJevTrainer:
    """Satisfies :class:`~honing.core.Trainer` for Laya decision models on ``typed_decisions`` data.

    ``job.steps`` is the number of passes (epochs) over the batch. ``full``
    updates every weight; ``param_subset`` updates only parameters whose names
    match ``trainable_patterns`` and leaves the rest bit-for-bit unchanged.

    ``sft`` learns the chosen options (see the module docstring). ``grpo`` runs
    policy gradient on rollout decisions with a group baseline and a clipped
    ratio (:mod:`honing.backends.jev.decision_training`); it keeps the final
    weights, since it has no validation split.
    """

    support = SUPPORT

    def __init__(self, settings: JevTrainerSettings) -> None:
        self._settings = settings

    def train(self, job: TrainJobSpec, base: ModelSnapshot, batch: TrainingBatch | None) -> TrainResult:
        if job.algorithm not in ("sft", "grpo") or job.contract != "typed_decisions":
            raise ContractError("the local Jev trainer runs sft or grpo on typed_decisions only")
        if job.update_policy.kind not in ("full", "param_subset"):
            raise ContractError(f"the local Jev trainer cannot apply a {job.update_policy.kind} update")
        if batch is None or batch.contract.kind != "typed_decisions":
            raise ContractError("the local Jev trainer needs a typed_decisions batch")
        run = self._settings.work_dir / f"train-{uuid.uuid4().hex[:12]}"
        run.mkdir(parents=True)
        try:
            with gpu_lock(self._settings.gpu, self._settings.lock_dir):
                if job.algorithm == "grpo":
                    return self._policy_gradient(job, base, batch, run)
                return self._train(job, base, batch, run)
        except Exception as error:
            (run / "error.json").write_text(json.dumps({"error": str(error)}) + "\n")
            raise

    def _examples(
        self,
        decisions: Sequence[DecisionSample],
        tok: Any,
        agent: Any,
        qtypes: Mapping[str, int],
        helpers: Any,
        advantages: Sequence[float] | None = None,
    ) -> tuple[list[Example], list[Example], dict[str, int]]:
        """Every question of every decision as one Laya input, packed as inference packs it."""
        build_sequence, render_options = helpers
        formatter = SimpleNamespace(tok=tok, _to_internal=agent._to_internal)

        def pack(parts: dict[str, list[str]], questions: dict[str, dict[str, Any]]) -> tuple[Any, Any, Any]:
            return prepare_request(formatter, parts, questions, MAX_LEN, HEAD_MAX_LEN, build_sequence, render_options)

        def encode(
            state: Any, questions: dict[str, dict[str, Any]], chosen: dict[str, str]
        ) -> Iterator[tuple[str, Example]]:
            for key, q in questions.items():
                ids, markers = build_sequence(tok, state, agent._to_internal(q), MAX_LEN, HEAD_MAX_LEN)
                keys = list(q["criteria"])
                if len(markers) != len(keys):
                    raise ContractError("Option marker loss")
                yield key, {
                    "ids": ids,
                    "markers": markers,
                    "label": keys.index(chosen[key]),
                    "qtype": qtypes["choice"],
                }

        return decision_examples(
            decisions,
            pack=pack,
            encode=encode,
            validation_groups=self._settings.validation_groups,
            advantages=advantages,
        )

    @staticmethod
    def _forward(torch: Any, model: Any, pad: int, device: str) -> Forward:
        """A chunk's masked option logits, labels, and the head's auxiliary output, which stays in the graph."""

        def forward(chunk: Sequence[Example]) -> tuple[Any, Any, Any]:
            tensors = collate(chunk, pad, torch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, aux = model(*tensors[:5])
            return logits.float().masked_fill(~tensors[3], -1e4), tensors[5], aux

        return forward

    @staticmethod
    def _save(checkpoint: Path, torch: Any, model: Any, tok: Any, cfg: dict[str, Any]) -> None:
        """Laya's checkpoint layout, with the temperature reset to 1.0: raw probabilities."""
        from safetensors.torch import save_file

        checkpoint.mkdir(exist_ok=True)
        save_file(
            {k: v.detach().to(torch.bfloat16).contiguous().cpu() for k, v in model.state_dict().items()},
            checkpoint / "model.safetensors",
        )
        model.encoder.config.save_pretrained(checkpoint / "encoder")
        tok.save_pretrained(checkpoint / "tokenizer")
        cfg["temperature"] = [1.0, 1.0, 1.0]
        cfg.pop("temperature_by_options", None)
        (checkpoint / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    def _train(self, job: TrainJobSpec, base: ModelSnapshot, batch: TrainingBatch, run: Path) -> TrainResult:
        settings = self._settings
        os.environ.update(HF_HUB_OFFLINE="1", USE_TF="0", TOKENIZERS_PARALLELISM="false")
        if settings.deterministic:
            # cuBLAS reads this when CUDA initializes, which has not happened yet.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        pins = json.loads(settings.pins_path.read_text())
        source = settings.laya_source
        revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if revision != pins["layaSource"]["revision"]:
            raise ContractError("the Laya source differs from its pin")
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        import torch
        from laya.agent import Agent
        from laya.common import QTYPES, build_model, build_sequence, render_options
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        torch.set_num_threads(4)
        if settings.deterministic:
            torch.use_deterministic_algorithms(True)
        torch.manual_seed(settings.seed)
        random.seed(settings.seed)
        record = json.loads(settings.model_record.read_text())
        parent = Path(base.checkpoint.location)
        if _sha256(parent / "model.safetensors") != base.checkpoint.checksum:
            raise ContractError("the base checkpoint's weights differ from its checksum")
        cfg = json.loads((parent / "rl_agent_config.json").read_text())
        cfg.update(max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN)
        tok = AutoTokenizer.from_pretrained(parent / "tokenizer", local_files_only=True)
        items, validation, packing = self._examples(
            batch.decisions, tok, Agent, QTYPES, (build_sequence, render_options)
        )
        items, validation, overlap = deduplicated_split(items, validation, settings.min_validation)
        device = "cuda"
        model = build_model(cfg, encoder_dir=str(parent / "encoder"))
        model.load_state_dict(load_file(parent / "model.safetensors"), strict=True)
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = True
        model.to(device)
        if job.update_policy.kind == "param_subset":
            patterns = [re.compile(p) for p in job.update_policy.trainable_patterns]
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(any(p.search(name) for p in patterns))
        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if not trainable:
            raise ContractError("no parameter matches the update policy")
        encoder = [p for n, p in trainable if n.startswith("encoder.")]
        head = [p for n, p in trainable if not n.startswith("encoder.")]
        groups = [
            {"params": params, "lr": lr}
            for params, lr in ((encoder, settings.encoder_lr), (head, settings.head_lr))
            if params or job.update_policy.kind == "full"
        ]
        optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
        started = time.time()
        checkpoint = run / "checkpoint"
        result = supervised_epochs(
            torch,
            model,
            optimizer,
            items,
            validation,
            forward=self._forward(torch, model, tok.pad_token_id, device),
            epochs=job.steps,
            batch_size=settings.batch_size,
            seed=settings.seed,
            save=lambda: self._save(checkpoint, torch, model, tok, cfg),
        )
        fields, metrics = result.record(items, validation, overlap)
        weight_hash = _sha256(checkpoint / "model.safetensors")
        if weight_hash == base.checkpoint.checksum:
            raise ContractError("training did not change the weights")
        manifest = {
            "kind": "trained-laya",
            "repo": "local:honing-jev/" + run.name,
            "revision": weight_hash,
            "path": str(checkpoint.resolve()),
            "weightsSha256": weight_hash,
            "parent": record,
            "base": {"location": base.checkpoint.location, "checksum": base.checkpoint.checksum},
            "sourceRevision": revision,
            "batchId": batch.batch_id,
            "updatePolicy": job.update_policy.kind,
            "packing": packing,
            **fields,
            "history": result.history,
            "settings": {
                "epochs": job.steps,
                "batchSize": settings.batch_size,
                "seed": settings.seed,
                "encoderLr": settings.encoder_lr,
                "headLr": settings.head_lr,
                "validationGroups": settings.validation_groups,
                "deterministic": settings.deterministic,
            },
            "elapsedSeconds": time.time() - started,
        }
        (run / "checkpoint.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return TrainResult(
            snapshot=ModelSnapshot(
                checkpoint=CheckpointRef(
                    location=str(checkpoint), weight_format="hf_safetensors", checksum=weight_hash
                ),
                architecture=base.architecture,
                parameters=base.parameters,
            ),
            metrics=metrics,
        )

    # -------------------------------------------------------------- policy gradient

    def _setup(self, base: ModelSnapshot) -> tuple[Any, ...]:
        """Load the pinned Laya source, tokenizer, and base model, as training does."""
        settings = self._settings
        os.environ.update(HF_HUB_OFFLINE="1", USE_TF="0", TOKENIZERS_PARALLELISM="false")
        if settings.deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        pins = json.loads(settings.pins_path.read_text())
        revision = subprocess.check_output(
            ["git", "-C", str(settings.laya_source), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != pins["layaSource"]["revision"]:
            raise ContractError("the Laya source differs from its pin")
        if str(settings.laya_source) not in sys.path:
            sys.path.insert(0, str(settings.laya_source))
        import torch
        from laya.agent import Agent
        from laya.common import QTYPES, build_model, build_sequence, render_options
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        torch.set_num_threads(4)
        if settings.deterministic:
            torch.use_deterministic_algorithms(True)
        torch.manual_seed(settings.seed)
        random.seed(settings.seed)
        parent = Path(base.checkpoint.location)
        if _sha256(parent / "model.safetensors") != base.checkpoint.checksum:
            raise ContractError("the base checkpoint's weights differ from its checksum")
        cfg = json.loads((parent / "rl_agent_config.json").read_text())
        cfg.update(max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN)
        tok = AutoTokenizer.from_pretrained(parent / "tokenizer", local_files_only=True)
        model = build_model(cfg, encoder_dir=str(parent / "encoder"))
        model.load_state_dict(load_file(parent / "model.safetensors"), strict=True)
        model.to("cuda")
        return torch, Agent, QTYPES, build_sequence, render_options, tok, model, cfg, revision

    def _policy_gradient(self, job: TrainJobSpec, base: ModelSnapshot, batch: TrainingBatch, run: Path) -> TrainResult:
        settings = self._settings
        torch, agent, qtypes, build_sequence, render_options, tok, model, cfg, revision = self._setup(base)
        advantages = group_advantages(batch.decisions)
        examples, _, _ = self._examples(
            batch.decisions, tok, agent, qtypes, (build_sequence, render_options), advantages
        )
        forward = self._forward(torch, model, tok.pad_token_id, "cuda")
        behavior_gap = record_start(torch, model, examples, forward=forward, batch_size=settings.batch_size)
        if job.update_policy.kind == "param_subset":
            patterns = [re.compile(p) for p in job.update_policy.trainable_patterns]
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(any(p.search(name) for p in patterns))
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise ContractError("no parameter matches the update policy")
        optimizer = torch.optim.AdamW(trainable, lr=settings.encoder_lr, weight_decay=0.0)
        started = time.time()
        history = policy_gradient_epochs(
            torch,
            model,
            optimizer,
            examples,
            forward=forward,
            epochs=job.steps,
            batch_size=settings.batch_size,
            seed=settings.seed,
            clip_ratio=settings.clip_ratio,
        )
        checkpoint = run / "checkpoint"
        self._save(checkpoint, torch, model, tok, cfg)
        fields, metrics = policy_gradient_record(batch, advantages, examples, history, behavior_gap)
        weight_hash = _sha256(checkpoint / "model.safetensors")
        manifest = {
            "kind": "trained-laya",
            "repo": "local:honing-jev/" + run.name,
            "revision": weight_hash,
            "path": str(checkpoint.resolve()),
            "weightsSha256": weight_hash,
            "parent": json.loads(settings.model_record.read_text()),
            "base": {"location": base.checkpoint.location, "checksum": base.checkpoint.checksum},
            "sourceRevision": revision,
            "batchId": batch.batch_id,
            "updatePolicy": job.update_policy.kind,
            **fields,
            "history": history,
            "settings": {
                "epochs": job.steps,
                "batchSize": settings.batch_size,
                "seed": settings.seed,
                "lr": settings.encoder_lr,
                "clipRatio": settings.clip_ratio,
                "deterministic": settings.deterministic,
            },
            "elapsedSeconds": time.time() - started,
        }
        (run / "checkpoint.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return TrainResult(
            snapshot=ModelSnapshot(
                checkpoint=CheckpointRef(
                    location=str(checkpoint), weight_format="hf_safetensors", checksum=weight_hash
                ),
                architecture=base.architecture,
                parameters=base.parameters,
            ),
            metrics=metrics,
        )
