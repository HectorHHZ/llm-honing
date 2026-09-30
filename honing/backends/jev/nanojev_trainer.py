"""Train a NanoJev decision model on typed decisions, on one local GPU.

The model is loaded with NanoJev's own ``DecisionPredictor`` (fp32 parameters)
and trained the way NanoJev's released checkpoint was
(``scripts/train_unified_games.py``, per its ``config.json``): the backbone and
the heads as two AdamW groups (backbone learning rate 1e-5, heads 1e-4, weight
decay 0.01), a bf16 autocast forward, gradient checkpointing on the backbone,
and gradients clipped at 1.0. Examples are NanoJev's candidate paths from
``prepare_examples``, packed exactly as inference packs them
(:func:`honing.backends.jev.nanojev.pack_request`).

The data handling is the Laya trainer's (:mod:`honing.backends.jev.trainer`):
every question of every decision is one example; identical inputs are
deduplicated and must agree on their label; examples from validation groups
are held out and must not also appear in training. ``sft`` keeps the
checkpoint with the best validation loss; ``grpo`` runs the same clipped
policy gradient as the other trainers and keeps the final weights. The trained
checkpoint has NanoJev's layout (``best.safetensors``, ``config.json``,
``backbone_config/``, ``tokenizer/``), so ``DecisionPredictor`` and
:class:`~honing.backends.jev.nanojev.NanoJevDecisionModel` load it directly.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import random
import re
import shutil
import statistics
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from honing.backends.jev.laya import gpu_lock
from honing.backends.jev.nanojev import (
    DOWNLOAD_PATTERNS,
    import_nanojev,
    measure,
    nanojev_snapshot,
    pack_request,
    payload,
    weights_checksum,
)
from honing.backends.jev.trainer import laya_question, separate_unique
from honing.core import (
    BackendSupport,
    ContractError,
    DecisionSample,
    ModelSnapshot,
    TrainingBatch,
    TrainJobSpec,
    TrainResult,
)

SUPPORT = BackendSupport(
    name="local-nanojev",
    architectures="uniform",
    train_numerics=frozenset({"bf16"}),
    weight_formats=frozenset({"hf_safetensors"}),
    algorithms=(("sft", "native"), ("grpo", "native")),
    contracts=frozenset({"typed_decisions"}),
)


@dataclass(frozen=True)
class NanoJevTrainerSettings:
    """Where and how the trainer runs; the defaults are the released checkpoint's recipe."""

    pins_path: Path
    source_dir: Path
    work_dir: Path
    model: str = "nanojev"
    gpu: str = "0"
    batch_questions: int = 4
    seed: int = 7
    backbone_lr: float = 1e-5
    head_lr: float = 1e-4
    weight_decay: float = 0.01
    max_length: int = 2048
    validation_groups: str = r"^[^:]+:101:"
    min_validation: int = 50
    lock_dir: Path | None = None
    deterministic: bool = False
    clip_ratio: float = 0.2


@dataclass(frozen=True)
class _Loaded:
    torch: Any
    predictor: Any
    prepare_examples: Any
    revision: str
    pin: dict[str, Any]


def _identity(example: dict[str, Any]) -> str:
    value = json.dumps(example["leaf_tokens"], separators=(",", ":"))
    return hashlib.sha256(value.encode()).hexdigest()


class NanoJevTrainer:
    """Satisfies :class:`~honing.core.Trainer` for NanoJev decision models on ``typed_decisions`` data.

    ``job.steps`` is the number of passes (epochs) over the batch. ``full``
    trains every weight; ``param_subset`` only those whose names match
    ``trainable_patterns`` (for example the heads, ``^(norm|scalar|set_)``).
    """

    support = SUPPORT

    def __init__(self, settings: NanoJevTrainerSettings) -> None:
        self._settings = settings

    def train(self, job: TrainJobSpec, base: ModelSnapshot, batch: TrainingBatch | None) -> TrainResult:
        if job.algorithm not in ("sft", "grpo") or job.contract != "typed_decisions":
            raise ContractError("the local NanoJev trainer runs sft or grpo on typed_decisions only")
        if job.update_policy.kind not in ("full", "param_subset"):
            raise ContractError(f"the local NanoJev trainer cannot apply a {job.update_policy.kind} update")
        if batch is None or batch.contract.kind != "typed_decisions":
            raise ContractError("the local NanoJev trainer needs a typed_decisions batch")
        run = self._settings.work_dir / f"train-{uuid.uuid4().hex[:12]}"
        run.mkdir(parents=True)
        try:
            with gpu_lock(self._settings.gpu, self._settings.lock_dir):
                loaded = self._load(base)
                if job.algorithm == "grpo":
                    return self._policy_gradient(job, base, batch, run, loaded)
                return self._supervised(job, base, batch, run, loaded)
        except Exception as error:
            (run / "error.json").write_text(json.dumps({"error": str(error)}) + "\n")
            raise

    # -------------------------------------------------------------- setup

    def _load(self, base: ModelSnapshot) -> _Loaded:
        settings = self._settings
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        if settings.deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        pins = json.loads(settings.pins_path.read_text())
        upstream, revision = import_nanojev(settings.source_dir, pins)
        import torch

        if settings.deterministic:
            torch.use_deterministic_algorithms(True)
        torch.manual_seed(settings.seed)
        random.seed(settings.seed)
        parent = Path(base.checkpoint.location)
        if weights_checksum(parent) != base.checkpoint.checksum:
            raise ContractError("the base checkpoint's weights differ from its checksum")
        pin = pins["nanojevModels"][settings.model]
        run_config = json.loads((parent / "config.json").read_text())
        if (run_config.get("model"), run_config.get("resolved_model_revision")) != (pin["base"], pin["baseRevision"]):
            raise ContractError("the base checkpoint's Qwen base differs from pins.json")
        predictor = upstream.DecisionPredictor(
            str(parent), max_length=settings.max_length, device_name="cuda:0", precision="bf16"
        )
        predictor.model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return _Loaded(torch, predictor, upstream.prepare_examples, revision, pin)

    def _examples(
        self, decisions: Sequence[DecisionSample], loaded: _Loaded, advantages: Sequence[float] | None = None
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
        """Every question of every decision as NanoJev's example (candidate paths), packed as inference packs it."""
        tokenizer, limit = loaded.predictor.tokenizer, loaded.predictor.limit
        held_out = re.compile(self._settings.validation_groups)
        items: list[dict[str, Any]] = []
        validation: list[dict[str, Any]] = []
        packing = {"maxInputTokens": 0, "trimmedRequests": 0}
        for index, decision in enumerate(decisions):
            if advantages is not None and advantages[index] == 0.0:
                continue
            parts = {name: list(lines) for name, lines in decision.parts}
            questions = {key: laya_question(q) for key, q in decision.questions}
            state, audit, trimmed = pack_request(
                parts,
                questions,
                functools.partial(measure, loaded.prepare_examples, tokenizer, questions=questions),
                limit,
            )
            packing["maxInputTokens"] = max(packing["maxInputTokens"], max(v["inputTokens"] for v in audit.values()))
            packing["trimmedRequests"] += bool(trimmed)
            chosen = dict(decision.chosen)
            behavior = dict(decision.behavior_probs or ())
            target = (
                validation
                if advantages is None and decision.group is not None and held_out.search(decision.group)
                else items
            )
            for example in loaded.prepare_examples(payload(state, questions), tokenizer, limit):
                key = example["qid"]
                label = example["candidate_ids"].index(chosen[key])
                item = {**example, "label": label, "ids": _identity(example), "markers": []}
                if advantages is not None:
                    item["advantage"] = advantages[index]
                    item["behavior"] = dict(behavior.get(key, ())).get(chosen[key])
                target.append(item)
        return items, validation, packing

    def _trainable(self, job: TrainJobSpec, model: Any) -> None:
        if job.update_policy.kind == "param_subset":
            patterns = [re.compile(p) for p in job.update_policy.trainable_patterns]
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(any(p.search(name) for p in patterns))
        if not any(p.requires_grad for p in model.parameters()):
            raise ContractError("no parameter matches the update policy")

    def _optimizer(self, torch: Any, model: Any) -> Any:
        settings = self._settings
        body = [p for p in model.backbone.parameters() if p.requires_grad]
        head = [p for name, p in model.named_parameters() if not name.startswith("backbone.") and p.requires_grad]
        groups = [
            {"params": params, "lr": lr}
            for params, lr in ((body, settings.backbone_lr), (head, settings.head_lr))
            if params
        ]
        return torch.optim.AdamW(groups, weight_decay=settings.weight_decay)

    def _logits(self, loaded: _Loaded, chunk: Sequence[dict[str, Any]]) -> Any:
        torch, predictor = loaded.torch, loaded.predictor
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = predictor.model(list(chunk), predictor.tokenizer.pad_token_id)
        return logits.float()

    # -------------------------------------------------------------- training

    def _supervised(
        self, job: TrainJobSpec, base: ModelSnapshot, batch: TrainingBatch, run: Path, loaded: _Loaded
    ) -> TrainResult:
        settings, torch, model = self._settings, loaded.torch, loaded.predictor.model
        items, validation, packing = self._examples(batch.decisions, loaded)
        items, validation, overlap = separate_unique(items, validation)
        if not items or len(validation) < settings.min_validation:
            raise ContractError(
                f"need at least {settings.min_validation} novel validation examples after deduplication, "
                f"got {len(validation)}"
            )
        self._trainable(job, model)
        optimizer = self._optimizer(torch, model)

        def labels(chunk: Sequence[dict[str, Any]]) -> Any:
            return torch.tensor([x["label"] for x in chunk], device="cuda")

        def evaluate() -> tuple[float, float]:
            model.eval()
            losses: list[float] = []
            correct = 0
            with torch.no_grad():
                for start in range(0, len(validation), settings.batch_questions):
                    chunk = validation[start : start + settings.batch_questions]
                    logits = self._logits(loaded, chunk)
                    losses.extend(torch.nn.functional.cross_entropy(logits, labels(chunk), reduction="none").tolist())
                    correct += int((logits.argmax(-1) == labels(chunk)).sum())
            return sum(losses) / len(losses), correct / len(validation)

        started = time.time()
        initial = evaluate()
        best, history = float("inf"), []
        checkpoint = run / "checkpoint"
        for epoch in range(job.steps):
            random.Random(settings.seed + epoch).shuffle(items)
            model.train()
            total, count = 0.0, 0
            for start in range(0, len(items), settings.batch_questions):
                chunk = items[start : start + settings.batch_questions]
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(self._logits(loaded, chunk), labels(chunk))
                if not torch.isfinite(loss):
                    raise ContractError("nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += float(loss.detach()) * len(chunk)
                count += len(chunk)
            val, accuracy = evaluate()
            history.append(
                {"epoch": epoch + 1, "trainLoss": total / count, "validationLoss": val, "validationAccuracy": accuracy}
            )
            if val < best:
                best = val
                self._save(checkpoint, Path(base.checkpoint.location), loaded)
        return self._result(
            job,
            base,
            batch,
            run,
            checkpoint,
            loaded,
            packing,
            history=history,
            started=started,
            extra={
                "dataOverlapAudit": overlap,
                "trainExamples": len(items),
                "validationExamples": len(validation),
                "initialValidation": initial,
            },
            metrics={
                "initial_validation_loss": initial[0],
                "validation_loss": best,
                "final_validation_accuracy": history[-1]["validationAccuracy"],
                "train_examples": float(len(items)),
                "validation_examples": float(len(validation)),
            },
        )

    def _policy_gradient(
        self, job: TrainJobSpec, base: ModelSnapshot, batch: TrainingBatch, run: Path, loaded: _Loaded
    ) -> TrainResult:
        from honing.backends.jev.policy_gradient import group_advantages

        settings, torch, model = self._settings, loaded.torch, loaded.predictor.model
        advantages = group_advantages(batch.decisions)
        examples, _, packing = self._examples(batch.decisions, loaded, advantages)
        if not examples:
            raise ContractError("every group's rewards are equal; there is nothing to learn from this batch")

        def chosen_log_probs(chunk: Sequence[dict[str, Any]]) -> Any:
            logits = self._logits(loaded, chunk)
            chosen = torch.tensor([x["label"] for x in chunk], device="cuda")
            return torch.log_softmax(logits, dim=-1).gather(1, chosen[:, None]).squeeze(1)

        model.eval()
        with torch.no_grad():
            for start in range(0, len(examples), settings.batch_questions):
                chunk = examples[start : start + settings.batch_questions]
                for item, value in zip(chunk, chosen_log_probs(chunk).tolist(), strict=True):
                    item["old"] = value
        behavior_gap = [abs(math.exp(x["old"]) - x["behavior"]) for x in examples if x["behavior"] is not None]
        self._trainable(job, model)
        optimizer = self._optimizer(torch, model)
        eps = settings.clip_ratio
        started, history = time.time(), []
        for epoch in range(job.steps):
            random.Random(settings.seed + epoch).shuffle(examples)
            model.train()
            totals = {"loss": 0.0, "clipped": 0.0, "kl": 0.0, "ratio": 0.0}
            count = 0
            for start in range(0, len(examples), settings.batch_questions):
                chunk = examples[start : start + settings.batch_questions]
                optimizer.zero_grad(set_to_none=True)
                log_probs = chosen_log_probs(chunk)
                old = torch.tensor([x["old"] for x in chunk], device="cuda")
                advantage = torch.tensor([x["advantage"] for x in chunk], device="cuda")
                ratio = torch.exp(log_probs - old)
                surrogate = torch.minimum(ratio * advantage, torch.clamp(ratio, 1 - eps, 1 + eps) * advantage)
                loss = -surrogate.mean()
                if not torch.isfinite(loss):
                    raise ContractError("nonfinite policy-gradient loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                n = len(chunk)
                totals["loss"] += float(loss.detach()) * n
                totals["clipped"] += float(((ratio - 1).abs() > eps).float().sum())
                totals["kl"] += float((old - log_probs.detach()).sum())
                totals["ratio"] += float(ratio.detach().sum())
                count += n
            history.append({"epoch": epoch + 1, **{k: v / count for k, v in totals.items()}})
        checkpoint = run / "checkpoint"
        self._save(checkpoint, Path(base.checkpoint.location), loaded)
        skipped = sum(1 for a in advantages if a == 0.0)
        final = history[-1]
        return self._result(
            job,
            base,
            batch,
            run,
            checkpoint,
            loaded,
            packing,
            history=history,
            started=started,
            extra={
                "algorithm": "grpo",
                "sourceReleases": sorted(batch.source_releases),
                "sourceHarnesses": sorted(batch.source_harnesses),
                "decisions": len(batch.decisions),
                "decisionsWithoutSignal": skipped,
                "examples": len(examples),
                "meanBehaviorGap": statistics.fmean(behavior_gap) if behavior_gap else None,
            },
            metrics={
                "policy_loss": final["loss"],
                "approx_kl": final["kl"],
                "clipped_fraction": final["clipped"],
                "examples": float(len(examples)),
                "decisions_without_signal": float(skipped),
            },
        )

    # -------------------------------------------------------------- output

    def _save(self, checkpoint: Path, parent: Path, loaded: _Loaded) -> None:
        """NanoJev's checkpoint layout: the weights, and the parent's run config, backbone config and tokenizer."""
        from safetensors.torch import save_file

        checkpoint.mkdir(exist_ok=True)
        state = {k: v.detach().to("cpu").contiguous() for k, v in loaded.predictor.model.state_dict().items()}
        save_file(state, str(checkpoint / "best.safetensors"))
        shutil.copyfile(parent / "config.json", checkpoint / "config.json")
        for directory in ("backbone_config", "tokenizer"):
            shutil.copytree(parent / directory, checkpoint / directory, dirs_exist_ok=True)

    def _result(
        self,
        job: TrainJobSpec,
        base: ModelSnapshot,
        batch: TrainingBatch,
        run: Path,
        checkpoint: Path,
        loaded: _Loaded,
        packing: dict[str, int],
        *,
        history: list[dict[str, Any]],
        started: float,
        extra: dict[str, Any],
        metrics: dict[str, float],
    ) -> TrainResult:
        settings = self._settings
        checksum = weights_checksum(checkpoint)
        if checksum == base.checkpoint.checksum:
            raise ContractError("training did not change the weights")
        manifest = {
            "kind": "trained-nanojev",
            "repo": "local:honing-nanojev/" + run.name,
            "revision": checksum,
            "path": str(checkpoint.resolve()),
            "weightsSha256": checksum,
            "parent": {"repo": loaded.pin["repo"], "revision": loaded.pin["revision"]},
            "base": {"location": base.checkpoint.location, "checksum": base.checkpoint.checksum},
            "sourceRevision": loaded.revision,
            "batchId": batch.batch_id,
            "updatePolicy": job.update_policy.kind,
            "packing": packing,
            "history": history,
            **extra,
            "settings": {
                "epochs": job.steps,
                "batchQuestions": settings.batch_questions,
                "seed": settings.seed,
                "backboneLr": settings.backbone_lr,
                "headLr": settings.head_lr,
                "weightDecay": settings.weight_decay,
                "maxLength": settings.max_length,
                "validationGroups": settings.validation_groups,
                "clipRatio": settings.clip_ratio,
                "deterministic": settings.deterministic,
            },
            "elapsedSeconds": time.time() - started,
        }
        (run / "checkpoint.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return TrainResult(snapshot=nanojev_snapshot(checkpoint), metrics=metrics)


def release_path(pins: dict[str, Any], model: str = "nanojev") -> Path:
    """The pinned release's local directory in the Hugging Face cache."""
    from huggingface_hub import snapshot_download

    pin = pins["nanojevModels"][model]
    return Path(snapshot_download(pin["repo"], revision=pin["revision"], allow_patterns=DOWNLOAD_PATTERNS))
