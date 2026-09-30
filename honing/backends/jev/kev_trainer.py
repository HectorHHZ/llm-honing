"""Train a Kev decision model on typed decisions, on one local GPU.

Training continues a released Kev checkpoint the way Kev's own delta training
does (``kev.train --init_from``): the adapter and pointer head are loaded into a
fresh LoRA model with :meth:`kev.checkpoint.Checkpoint.warm_start`, weights stay
fp32 with a bf16 autocast forward, gradients are checkpointed and clipped at
1.0, and AdamW runs under a one-cycle schedule. The trained checkpoint is saved
in Kev's layout (adapter, ``head.pt``, tokenizer) with temperature 1.0, raw
probabilities, as ``kev.train`` saves one, so :class:`KevDecisionModel` serves it
directly.

The data is prepared as for Laya (:mod:`honing.backends.jev.trainer`): every
question of every decision is one example, packed exactly as inference packs it
(:func:`honing.backends.jev.kev.prepare_request`); identical inputs are
deduplicated and must agree on their label; examples from validation groups are
held out and must not also appear in training. ``sft`` keeps the checkpoint
with the best validation loss. ``grpo`` runs the same clipped policy gradient
as the Laya trainer (:mod:`honing.backends.jev.policy_gradient`) and keeps the
final weights.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import statistics
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from honing.backends.jev.kev import import_kev, kev_snapshot, prepare_request, weights_checksum
from honing.backends.jev.laya import gpu_lock
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
    name="local-kev",
    architectures="uniform",
    train_numerics=frozenset({"bf16"}),
    weight_formats=frozenset({"hf_safetensors"}),
    algorithms=(("sft", "native"), ("grpo", "native")),
    contracts=frozenset({"typed_decisions"}),
)


@dataclass(frozen=True)
class KevTrainerSettings:
    """Where and how the trainer runs.

    ``model`` names the pinned release in ``pins.json`` that every trained
    checkpoint descends from. ``lr`` is the adapter's learning rate (Kev's
    delta default, 2e-5) and ``head_lr`` the pointer head's (0 means the same).
    Decisions whose ``group`` matches ``validation_groups`` are held out; the
    default holds out seed 101, as the Laya trainer does.
    """

    pins_path: Path
    source_dir: Path
    work_dir: Path
    model: str = "kev-0.8b"
    gpu: str = "0"
    batch_size: int = 4
    seed: int = 7
    lr: float = 2e-5
    head_lr: float = 0.0
    weight_decay: float = 0.01
    validation_groups: str = r"^[^:]+:101:"
    min_validation: int = 50
    lock_dir: Path | None = None
    deterministic: bool = False
    clip_ratio: float = 0.2


@dataclass(frozen=True)
class _Loaded:
    """A warm-started trainable model and what training records about it."""

    torch: Any
    tok: Any
    model: Any
    meta: Any
    init_source: dict[str, Any]
    revision: str
    pin: dict[str, Any]


class KevTrainer:
    """Satisfies :class:`~honing.core.Trainer` for Kev decision models on ``typed_decisions`` data.

    ``job.steps`` is the number of passes (epochs) over the batch. ``lora``
    and ``full`` both train the adapter and the pointer head (a Kev checkpoint
    has no other trainable weights); ``param_subset`` trains only the
    parameters whose names match ``trainable_patterns``.
    """

    support = SUPPORT

    def __init__(self, settings: KevTrainerSettings) -> None:
        self._settings = settings

    def train(self, job: TrainJobSpec, base: ModelSnapshot, batch: TrainingBatch | None) -> TrainResult:
        if job.algorithm not in ("sft", "grpo") or job.contract != "typed_decisions":
            raise ContractError("the local Kev trainer runs sft or grpo on typed_decisions only")
        if job.update_policy.kind not in ("lora", "full", "param_subset"):
            raise ContractError(f"the local Kev trainer cannot apply a {job.update_policy.kind} update")
        if batch is None or batch.contract.kind != "typed_decisions":
            raise ContractError("the local Kev trainer needs a typed_decisions batch")
        run = self._settings.work_dir / f"train-{uuid.uuid4().hex[:12]}"
        run.mkdir(parents=True)
        try:
            with gpu_lock(self._settings.gpu, self._settings.lock_dir):
                return self._run(job, base, batch, run)
        except Exception as error:
            (run / "error.json").write_text(json.dumps({"error": str(error)}) + "\n")
            raise

    # -------------------------------------------------------------- setup

    def _load(self, base: ModelSnapshot) -> _Loaded:
        """The pinned Kev source, a trainable model warm-started from ``base``, and its tokenizer."""
        settings = self._settings
        os.environ.update(HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        if settings.deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        pins = json.loads(settings.pins_path.read_text())
        revision = import_kev(settings.source_dir, pins)
        import torch
        from kev.checkpoint import Checkpoint, Meta
        from kev.model import DecisionModel, load_tokenizer

        if settings.deterministic:
            torch.use_deterministic_algorithms(True)
        torch.manual_seed(settings.seed)
        random.seed(settings.seed)
        parent = Path(base.checkpoint.location)
        if weights_checksum(parent) != base.checkpoint.checksum:
            raise ContractError("the base checkpoint's weights differ from its checksum")
        checkpoint = Checkpoint(str(parent))
        meta = checkpoint.meta
        pin = pins["kevModels"][settings.model]
        if (meta.base, meta.base_revision) != (pin["base"], pin["baseRevision"]):
            raise ContractError("the base checkpoint's Qwen base differs from pins.json")
        tok = load_tokenizer(meta.base, revision=meta.base_revision)
        lora_targets = meta.extra.get("args", {}).get("lora_targets", "all")
        model = DecisionModel(
            meta.base,
            tok,
            "cuda",
            lora=meta.lora,
            revision=meta.base_revision,
            head_dim=meta.head_dim,
            lora_targets=lora_targets,
            option_isolation=meta.option_isolation,
            special_embeddings=meta.special_embeddings,
            dtype=torch.float32,
            lora_placement=meta.lora_placement,
        )
        model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.lm.config.use_cache = False
        ours = Meta(
            base=meta.base,
            base_revision=meta.base_revision,
            lora=meta.lora,
            head_dim=meta.head_dim,
            option_isolation=meta.option_isolation,
            special_embeddings=meta.special_embeddings,
            weights_dtype="fp32",
            lora_placement=meta.lora_placement,
        )
        init_source = checkpoint.warm_start(model, ours)
        model.head.temperature = 1.0
        return _Loaded(torch, tok, model, ours, init_source, revision, pin)

    def _examples(
        self, decisions: Sequence[DecisionSample], tok: Any, model: Any, advantages: Sequence[float] | None = None
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
        """Every question of every decision as one encoded example, packed as inference packs it."""
        from kev.api import SystemOneRequest, to_record
        from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE

        held_out = re.compile(self._settings.validation_groups)
        items: list[dict[str, Any]] = []
        validation: list[dict[str, Any]] = []
        packing = {"maxInputTokens": 0, "trimmedRequests": 0}
        for index, decision in enumerate(decisions):
            if advantages is not None and advantages[index] == 0.0:
                continue
            parts = {name: list(lines) for name, lines in decision.parts}
            questions = {key: laya_question(q) for key, q in decision.questions}
            state, audit, trimmed = prepare_request(tok, parts, questions, SERVE_MAX_STATE, SERVE_MAX_BRANCH)
            packing["maxInputTokens"] = max(packing["maxInputTokens"], max(v["inputTokens"] for v in audit.values()))
            packing["trimmedRequests"] += bool(trimmed)
            chosen = dict(decision.chosen)
            behavior = dict(decision.behavior_probs or ())
            target = (
                validation
                if advantages is None and decision.group is not None and held_out.search(decision.group)
                else items
            )
            for key, question in questions.items():
                record, _ = to_record(SystemOneRequest(state=state, questions={key: question}))
                keys = list(question["criteria"])
                label = keys.index(chosen[key])
                record["questions"][0]["label"] = label
                encoded = model.encode(
                    tok, record, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH, strict=True
                )
                item = {"enc": encoded, "ids": encoded["ids"], "markers": encoded["opt_idx"], "label": label}
                if advantages is not None:
                    item["advantage"] = advantages[index]
                    item["behavior"] = dict(behavior.get(key, ())).get(chosen[key])
                target.append(item)
        return items, validation, packing

    def _trainable(self, job: TrainJobSpec, model: Any) -> list[Any]:
        if job.update_policy.kind == "param_subset":
            patterns = [re.compile(p) for p in job.update_policy.trainable_patterns]
            for name, parameter in model.named_parameters():
                if parameter.requires_grad and not any(p.search(name) for p in patterns):
                    parameter.requires_grad_(False)
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise ContractError("no parameter matches the update policy")
        return trainable

    def _optimizer(self, torch: Any, model: Any, steps: int) -> tuple[Any, Any]:
        settings = self._settings
        head_ids = {id(p) for p in model.head.parameters()}
        adapter = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
        head = [p for p in model.head.parameters() if p.requires_grad]
        rates = [(adapter, settings.lr), (head, settings.head_lr or settings.lr)]
        groups = [{"params": params, "lr": lr} for params, lr in rates if params]
        optimizer = torch.optim.AdamW(groups, weight_decay=settings.weight_decay)
        schedule = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=[g["lr"] for g in groups], total_steps=max(steps, 1), pct_start=0.1
        )
        return optimizer, schedule

    # -------------------------------------------------------------- training

    def _run(self, job: TrainJobSpec, base: ModelSnapshot, batch: TrainingBatch, run: Path) -> TrainResult:
        loaded = self._load(base)
        if job.algorithm == "grpo":
            return self._policy_gradient(job, base, batch, run, loaded)
        return self._supervised(job, base, batch, run, loaded)

    def _logits(self, torch: Any, model: Any, items: Sequence[dict[str, Any]]) -> Any:
        """Option logits of each single-question example, padded into one [N, K] tensor (-1e4 marks no option)."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            nested = model.forward_batch([item["enc"] for item in items])
        width = max(len(logits[0]) for logits in nested)
        out = torch.full((len(items), width), -1e4, device="cuda")
        for row, logits in enumerate(nested):
            out[row, : len(logits[0])] = logits[0].float()
        return out

    def _supervised(
        self, job: TrainJobSpec, base: ModelSnapshot, batch: TrainingBatch, run: Path, loaded: _Loaded
    ) -> TrainResult:
        torch, tok, model = loaded.torch, loaded.tok, loaded.model
        settings = self._settings
        items, validation, packing = self._examples(batch.decisions, tok, model)
        items, validation, overlap = separate_unique(items, validation)
        if not items or len(validation) < settings.min_validation:
            raise ContractError(
                f"need at least {settings.min_validation} novel validation examples after deduplication, "
                f"got {len(validation)}"
            )
        self._trainable(job, model)
        steps = job.steps * math.ceil(len(items) / settings.batch_size)
        optimizer, schedule = self._optimizer(torch, model, steps)
        labels = lambda chunk: torch.tensor([x["label"] for x in chunk], device="cuda")  # noqa: E731

        def evaluate() -> tuple[float, float]:
            model.eval()
            losses: list[float] = []
            correct = 0
            with torch.no_grad():
                for start in range(0, len(validation), settings.batch_size):
                    chunk = validation[start : start + settings.batch_size]
                    logits = self._logits(torch, model, chunk)
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
            for start in range(0, len(items), settings.batch_size):
                chunk = items[start : start + settings.batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(self._logits(torch, model, chunk), labels(chunk))
                if not torch.isfinite(loss):
                    raise ContractError("nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                schedule.step()
                total += float(loss.detach()) * len(chunk)
                count += len(chunk)
            val, accuracy = evaluate()
            history.append(
                {"epoch": epoch + 1, "trainLoss": total / count, "validationLoss": val, "validationAccuracy": accuracy}
            )
            if val < best:
                best = val
                self._save(checkpoint, loaded)
        return self._result(
            job, base, batch, run, checkpoint, loaded.revision, loaded.pin, packing,
            history=history, started=started,
            extra={"dataOverlapAudit": overlap, "trainExamples": len(items), "validationExamples": len(validation),
                   "initialValidation": initial},
            metrics={
                "initial_validation_loss": initial[0],
                "validation_loss": best,
                "final_validation_accuracy": history[-1]["validationAccuracy"],
                "train_examples": float(len(items)),
                "validation_examples": float(len(validation)),
            },
        )  # fmt: skip

    def _policy_gradient(
        self, job: TrainJobSpec, base: ModelSnapshot, batch: TrainingBatch, run: Path, loaded: _Loaded
    ) -> TrainResult:
        torch, tok, model = loaded.torch, loaded.tok, loaded.model
        from honing.backends.jev.policy_gradient import group_advantages

        settings = self._settings
        advantages = group_advantages(batch.decisions)
        examples, _, packing = self._examples(batch.decisions, tok, model, advantages)
        if not examples:
            raise ContractError("every group's rewards are equal; there is nothing to learn from this batch")

        def chosen_log_probs(chunk: Sequence[dict[str, Any]]) -> Any:
            logits = self._logits(torch, model, chunk)
            chosen = torch.tensor([x["label"] for x in chunk], device="cuda")
            return torch.log_softmax(logits, dim=-1).gather(1, chosen[:, None]).squeeze(1)

        model.eval()
        with torch.no_grad():
            for start in range(0, len(examples), settings.batch_size):
                chunk = examples[start : start + settings.batch_size]
                for item, value in zip(chunk, chosen_log_probs(chunk).tolist(), strict=True):
                    item["old"] = value
        behavior_gap = [abs(math.exp(x["old"]) - x["behavior"]) for x in examples if x["behavior"] is not None]
        self._trainable(job, model)
        steps = job.steps * math.ceil(len(examples) / settings.batch_size)
        optimizer, schedule = self._optimizer(torch, model, steps)
        eps = settings.clip_ratio
        started, history = time.time(), []
        for epoch in range(job.steps):
            random.Random(settings.seed + epoch).shuffle(examples)
            model.train()
            totals = {"loss": 0.0, "clipped": 0.0, "kl": 0.0, "ratio": 0.0}
            count = 0
            for start in range(0, len(examples), settings.batch_size):
                chunk = examples[start : start + settings.batch_size]
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
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                schedule.step()
                n = len(chunk)
                totals["loss"] += float(loss.detach()) * n
                totals["clipped"] += float(((ratio - 1).abs() > eps).float().sum())
                totals["kl"] += float((old - log_probs.detach()).sum())
                totals["ratio"] += float(ratio.detach().sum())
                count += n
            history.append({"epoch": epoch + 1, **{k: v / count for k, v in totals.items()}})
        checkpoint = run / "checkpoint"
        self._save(checkpoint, loaded)
        skipped = sum(1 for a in advantages if a == 0.0)
        final = history[-1]
        return self._result(
            job, base, batch, run, checkpoint, loaded.revision, loaded.pin, packing,
            history=history, started=started,
            extra={"algorithm": "grpo", "sourceReleases": sorted(batch.source_releases),
                   "sourceHarnesses": sorted(batch.source_harnesses), "decisions": len(batch.decisions),
                   "decisionsWithoutSignal": skipped, "examples": len(examples),
                   "meanBehaviorGap": statistics.fmean(behavior_gap) if behavior_gap else None},
            metrics={"policy_loss": final["loss"], "approx_kl": final["kl"], "clipped_fraction": final["clipped"],
                     "examples": float(len(examples)), "decisions_without_signal": float(skipped)},
        )  # fmt: skip

    # -------------------------------------------------------------- output

    def _save(self, checkpoint: Path, loaded: _Loaded) -> None:
        """Kev's checkpoint layout: the adapter, ``head.pt`` at temperature 1.0, and the tokenizer."""
        from kev.checkpoint import write_meta

        model, tok, meta, init_source = loaded.model, loaded.tok, loaded.meta, loaded.init_source
        checkpoint.mkdir(exist_ok=True)
        model.lm.save_pretrained(str(checkpoint))
        meta.head = {k: v.detach().cpu() for k, v in model.head.state_dict().items()}
        meta.temperature = 1.0
        meta.extra = {"init_source": init_source, "trainer": "honing.backends.jev.kev_trainer"}
        write_meta(str(checkpoint), meta)
        tok.save_pretrained(str(checkpoint))

    def _result(
        self,
        job: TrainJobSpec,
        base: ModelSnapshot,
        batch: TrainingBatch,
        run: Path,
        checkpoint: Path,
        revision: str,
        pin: dict[str, Any],
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
            "kind": "trained-kev",
            "repo": "local:honing-kev/" + run.name,
            "revision": checksum,
            "path": str(checkpoint.resolve()),
            "weightsSha256": checksum,
            "parent": {"repo": pin["repo"], "revision": pin["revision"]},
            "base": {"location": base.checkpoint.location, "checksum": base.checkpoint.checksum},
            "sourceRevision": revision,
            "batchId": batch.batch_id,
            "updatePolicy": job.update_policy.kind,
            "packing": packing,
            "history": history,
            **extra,
            "settings": {
                "epochs": job.steps,
                "batchSize": settings.batch_size,
                "seed": settings.seed,
                "lr": settings.lr,
                "headLr": settings.head_lr,
                "weightDecay": settings.weight_decay,
                "validationGroups": settings.validation_groups,
                "clipRatio": settings.clip_ratio,
                "deterministic": settings.deterministic,
            },
            "elapsedSeconds": time.time() - started,
        }
        (run / "checkpoint.json").write_text(json.dumps(manifest, indent=2) + "\n")
        snapshot = kev_snapshot(checkpoint, base_config_of(base))
        return TrainResult(snapshot=snapshot, metrics=metrics)


def base_config_of(base: ModelSnapshot) -> Path:
    """The Qwen base's ``config.json`` in the local Hugging Face cache, for the snapshot's layer shape."""
    from kev.checkpoint import read_meta, resolve_run

    meta = read_meta(base.checkpoint.location)
    return Path(resolve_run(f"{meta.base}@{meta.base_revision}")) / "config.json"
