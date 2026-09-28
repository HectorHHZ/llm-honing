"""Train a Laya decision model on typed decisions, on one local GPU.

The training procedure is the one the RA2 experiment used (#49's
``train_laya.py``): every question of every decision becomes one example,
encoded exactly as inference encodes it; identical inputs are deduplicated and
must agree on their label; examples from validation groups are held out and
must not also appear in training; then full-weight cross-entropy over the
option logits, AdamW with separate encoder and head learning rates, bf16
autocast, gradient clipping at 1.0, and the checkpoint with the best validation
loss is kept.

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
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
    algorithms=(("sft", "native"),),
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


def _sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def identity(item: Mapping[str, Any]) -> str:
    """One encoded model input: its token ids and option markers."""
    value = json.dumps([item["ids"], item["markers"]], separators=(",", ":"))
    return hashlib.sha256(value.encode()).hexdigest()


def separate_unique(
    training: Sequence[dict[str, Any]], validation: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Deduplicate inputs within each split, and keep only validation inputs training never saw."""

    def unique(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            key = identity(row)
            if key in result and result[key]["label"] != row["label"]:
                raise ContractError("Conflicting labels for the same complete model input")
            result.setdefault(key, row)
        return result

    train, valid = unique(training), unique(validation)
    overlap = set(train) & set(valid)
    for key in overlap:
        if train[key]["label"] != valid[key]["label"]:
            raise ContractError("Conflicting labels across episode splits")
    novel = [row for key, row in valid.items() if key not in train]
    audit = {
        "rawTraining": len(training),
        "rawValidation": len(validation),
        "uniqueTraining": len(train),
        "uniqueValidation": len(valid),
        "removedOverlap": len(overlap),
        "novelValidation": len(novel),
    }
    return list(train.values()), novel, audit


def laya_question(question: Any) -> dict[str, Any]:
    """A DecisionQuestion as the question object Laya reads."""
    if question.kind != "choice":
        raise ContractError(f"the Laya trainer handles choice questions only, not {question.kind}")
    return {"type": "choice", "instructions": question.instructions, "criteria": dict(question.options)}


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


@contextmanager
def training_attention() -> Iterator[None]:
    """Keep masked SDPA off the cuDNN backward path, including checkpoint recomputation.

    cuDNN can produce NaN query gradients for lengths such as 192 on H100:
    https://github.com/pytorch/pytorch/issues/196678. Keep the other fused
    implementations available, with math as a fallback. Restore the caller's
    backend settings on exit, including failures.
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        yield


def backward_step(loss: Any, parameters: Sequence[Any], optimizer: Any, where: str) -> float:
    """Reject invalid gradients before clipping can corrupt weights or optimizer state."""
    import torch

    if not torch.isfinite(loss):
        raise ContractError(f"nonfinite training loss ({where})")
    loss.backward()
    try:
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    except RuntimeError as error:
        if "non-finite" not in str(error):
            raise
        raise ContractError(f"nonfinite training gradients ({where}); optimizer step was not applied") from error
    optimizer.step()
    return float(norm)


class LocalJevTrainer:
    """Satisfies :class:`~honing.core.Trainer` for Laya decision models on ``typed_decisions`` data.

    ``job.steps`` is the number of passes (epochs) over the batch. ``full``
    updates every weight; ``param_subset`` updates only parameters whose names
    match ``trainable_patterns`` and leaves the rest bit-for-bit unchanged.
    """

    support = SUPPORT

    def __init__(self, settings: JevTrainerSettings) -> None:
        self._settings = settings

    def train(self, job: TrainJobSpec, base: ModelSnapshot, batch: TrainingBatch | None) -> TrainResult:
        if job.algorithm != "sft" or job.contract != "typed_decisions":
            raise ContractError("the local Jev trainer runs sft on typed_decisions only")
        if job.update_policy.kind not in ("full", "param_subset"):
            raise ContractError(f"the local Jev trainer cannot apply a {job.update_policy.kind} update")
        if batch is None or batch.contract.kind != "typed_decisions":
            raise ContractError("the local Jev trainer needs a typed_decisions batch")
        run = self._settings.work_dir / f"train-{uuid.uuid4().hex[:12]}"
        run.mkdir(parents=True)
        try:
            with gpu_lock(self._settings.gpu, self._settings.lock_dir), training_attention():
                return self._train(job, base, batch, run)
        except Exception as error:
            (run / "error.json").write_text(json.dumps({"error": str(error)}) + "\n")
            raise

    def _examples(
        self, decisions: Sequence[DecisionSample], tok: Any, agent: Any, qtypes: Mapping[str, int], helpers: Any
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
        build_sequence, render_options = helpers
        formatter = SimpleNamespace(tok=tok, _to_internal=agent._to_internal)
        held_out = re.compile(self._settings.validation_groups)
        items: list[dict[str, Any]] = []
        validation: list[dict[str, Any]] = []
        packing = {"maxInputTokens": 0, "trimmedRequests": 0}
        for decision in decisions:
            parts = {name: list(lines) for name, lines in decision.parts}
            questions = {key: laya_question(q) for key, q in decision.questions}
            state, audit, trimmed = prepare_request(
                formatter, parts, questions, MAX_LEN, HEAD_MAX_LEN, build_sequence, render_options
            )
            packing["maxInputTokens"] = max(packing["maxInputTokens"], max(v["inputTokens"] for v in audit.values()))
            packing["trimmedRequests"] += bool(trimmed)
            chosen = dict(decision.chosen)
            target = validation if decision.group is not None and held_out.search(decision.group) else items
            for key, q in questions.items():
                ids, markers = build_sequence(tok, state, agent._to_internal(q), MAX_LEN, HEAD_MAX_LEN)
                keys = list(q["criteria"])
                if len(markers) != len(keys):
                    raise ContractError("Option marker loss")
                target.append(
                    {"ids": ids, "markers": markers, "label": keys.index(chosen[key]), "qtype": qtypes["choice"]}
                )
        return items, validation, packing

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
        from safetensors.torch import load_file, save_file
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
        items, validation, overlap = separate_unique(items, validation)
        if not items or len(validation) < settings.min_validation:
            raise ContractError(
                f"need at least {settings.min_validation} novel validation examples after deduplication, "
                f"got {len(validation)}"
            )
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
        pad = tok.pad_token_id

        def evaluate() -> tuple[float, float]:
            model.eval()
            losses: list[float] = []
            correct = 0
            with torch.no_grad():
                for start in range(0, len(validation), settings.batch_size):
                    tensors = collate(validation[start : start + settings.batch_size], pad, torch, device)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits, _ = model(*tensors[:5])
                    logits = logits.float().masked_fill(~tensors[3], -1e4)
                    batch_losses = torch.nn.functional.cross_entropy(logits, tensors[5], reduction="none")
                    if not torch.isfinite(batch_losses).all():
                        raise ContractError(f"nonfinite validation loss (batch={start // settings.batch_size + 1})")
                    losses.extend(batch_losses.tolist())
                    correct += int((logits.argmax(-1) == tensors[5]).sum())
            return sum(losses) / len(losses), correct / len(validation)

        started = time.time()
        initial = evaluate()
        best = float("inf")
        history = []
        checkpoint = run / "checkpoint"
        for epoch in range(job.steps):
            random.Random(settings.seed + epoch).shuffle(items)
            model.train()
            total, steps = 0.0, 0
            for start in range(0, len(items), settings.batch_size):
                tensors = collate(items[start : start + settings.batch_size], pad, torch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, aux = model(*tensors[:5])
                loss = (
                    torch.nn.functional.cross_entropy(logits.float().masked_fill(~tensors[3], -1e4), tensors[5])
                    + 0 * aux.float().sum()
                )
                backward_step(loss, [p for _, p in trainable], optimizer, f"epoch={epoch + 1}, batch={steps + 1}")
                total += float(loss.detach())
                steps += 1
            val, accuracy = evaluate()
            history.append(
                {
                    "epoch": epoch + 1,
                    "trainLoss": total / max(1, steps),
                    "validationLoss": val,
                    "validationAccuracy": accuracy,
                }
            )
            if val < best:
                best = val
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
            "dataOverlapAudit": overlap,
            "trainExamples": len(items),
            "validationExamples": len(validation),
            "initialValidation": initial,
            "history": history,
            "settings": {
                "epochs": job.steps,
                "batchSize": settings.batch_size,
                "seed": settings.seed,
                "encoderLr": settings.encoder_lr,
                "headLr": settings.head_lr,
                "validationGroups": settings.validation_groups,
                "deterministic": settings.deterministic,
                "sdpaBackends": ["flash", "efficient", "math"],
            },
            "elapsedSeconds": time.time() - started,
        }
        (run / "checkpoint.json").write_text(json.dumps(manifest, indent=2) + "\n")
        final = history[-1]
        return TrainResult(
            snapshot=ModelSnapshot(
                checkpoint=CheckpointRef(
                    location=str(checkpoint), weight_format="hf_safetensors", checksum=weight_hash
                ),
                architecture=base.architecture,
                parameters=base.parameters,
            ),
            metrics={
                "initial_validation_loss": initial[0],
                "validation_loss": best,
                "final_validation_accuracy": final["validationAccuracy"],
                "train_examples": float(len(items)),
                "validation_examples": float(len(validation)),
            },
        )
