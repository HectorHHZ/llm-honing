"""Train a Kev decision model on typed decisions, on one local GPU.

Training continues a released Kev checkpoint the way Kev's own delta training
does (``kev.train --init_from``): the adapter and pointer head are loaded into a
fresh LoRA model with :meth:`kev.checkpoint.Checkpoint.warm_start`, weights stay
fp32 with a bf16 autocast forward, gradients are checkpointed and clipped at
1.0, and AdamW runs under a one-cycle schedule. The trained checkpoint is saved
in Kev's layout (adapter, ``head.pt``, tokenizer) with temperature 1.0, raw
probabilities, as ``kev.train`` saves one, so :class:`KevDecisionModel` serves it
directly.

The data handling, the losses and the loops are the ones every decision-model
trainer shares (:mod:`honing.backends.jev.decision_training`): every question
of every decision is one example, packed exactly as inference packs it
(:func:`honing.backends.jev.kev.prepare_request`); identical inputs are
deduplicated and must agree on their label; examples from validation groups are
held out and must not also appear in training. ``sft`` keeps the checkpoint
with the best validation loss. ``grpo`` runs the clipped policy gradient and
keeps the final weights.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
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
from honing.backends.jev.kev import import_kev, kev_snapshot, prepare_request, weights_checksum
from honing.backends.jev.laya import gpu_lock
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
    ) -> tuple[list[Example], list[Example], dict[str, int]]:
        """Every question of every decision as one encoded example, packed as inference packs it."""
        from kev.api import SystemOneRequest, to_record
        from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE

        def pack(parts: dict[str, list[str]], questions: dict[str, dict[str, Any]]) -> tuple[Any, Any, Any]:
            return prepare_request(tok, parts, questions, SERVE_MAX_STATE, SERVE_MAX_BRANCH)

        def encode(
            state: Any, questions: dict[str, dict[str, Any]], chosen: dict[str, str]
        ) -> Iterator[tuple[str, Example]]:
            for key, question in questions.items():
                record, _ = to_record(SystemOneRequest(state=state, questions={key: question}))
                keys = list(question["criteria"])
                label = keys.index(chosen[key])
                record["questions"][0]["label"] = label
                encoded = model.encode(
                    tok, record, max_state=SERVE_MAX_STATE, max_branch=SERVE_MAX_BRANCH, strict=True
                )
                yield key, {"enc": encoded, "ids": encoded["ids"], "markers": encoded["opt_idx"], "label": label}

        return decision_examples(
            decisions,
            pack=pack,
            encode=encode,
            validation_groups=self._settings.validation_groups,
            advantages=advantages,
        )

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

    @staticmethod
    def _forward(torch: Any, model: Any) -> Forward:
        """Option logits of each single-question example, padded into one [N, K] tensor (-1e4 marks no option),
        and the examples' labels."""

        def forward(items: Sequence[Example]) -> tuple[Any, Any, Any]:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                nested = model.forward_batch([item["enc"] for item in items])
            width = max(len(logits[0]) for logits in nested)
            out = torch.full((len(items), width), -1e4, device="cuda")
            for row, logits in enumerate(nested):
                out[row, : len(logits[0])] = logits[0].float()
            return out, torch.tensor([x["label"] for x in items], device="cuda"), None

        return forward

    def _supervised(
        self, job: TrainJobSpec, base: ModelSnapshot, batch: TrainingBatch, run: Path, loaded: _Loaded
    ) -> TrainResult:
        torch, tok, model = loaded.torch, loaded.tok, loaded.model
        settings = self._settings
        items, validation, packing = self._examples(batch.decisions, tok, model)
        items, validation, overlap = deduplicated_split(items, validation, settings.min_validation)
        self._trainable(job, model)
        steps = job.steps * math.ceil(len(items) / settings.batch_size)
        optimizer, schedule = self._optimizer(torch, model, steps)
        started = time.time()
        checkpoint = run / "checkpoint"
        result = supervised_epochs(
            torch,
            model,
            optimizer,
            items,
            validation,
            forward=self._forward(torch, model),
            epochs=job.steps,
            batch_size=settings.batch_size,
            seed=settings.seed,
            save=lambda: self._save(checkpoint, loaded),
            schedule=schedule,
        )
        fields, metrics = result.record(items, validation, overlap)
        return self._result(
            job, base, batch, run, checkpoint, loaded.revision, loaded.pin, packing,
            history=result.history, started=started, extra=fields, metrics=metrics,
        )  # fmt: skip

    def _policy_gradient(
        self, job: TrainJobSpec, base: ModelSnapshot, batch: TrainingBatch, run: Path, loaded: _Loaded
    ) -> TrainResult:
        torch, tok, model = loaded.torch, loaded.tok, loaded.model
        settings = self._settings
        advantages = group_advantages(batch.decisions)
        examples, _, packing = self._examples(batch.decisions, tok, model, advantages)
        forward = self._forward(torch, model)
        behavior_gap = record_start(torch, model, examples, forward=forward, batch_size=settings.batch_size)
        self._trainable(job, model)
        steps = job.steps * math.ceil(len(examples) / settings.batch_size)
        optimizer, schedule = self._optimizer(torch, model, steps)
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
            schedule=schedule,
        )
        checkpoint = run / "checkpoint"
        self._save(checkpoint, loaded)
        fields, metrics = policy_gradient_record(batch, advantages, examples, history, behavior_gap)
        return self._result(
            job, base, batch, run, checkpoint, loaded.revision, loaded.pin, packing,
            history=history, started=started, extra=fields, metrics=metrics,
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
