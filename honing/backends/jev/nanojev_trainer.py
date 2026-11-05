"""Train a NanoJev decision model on typed decisions, on one local GPU.

The model is loaded with NanoJev's own ``DecisionPredictor`` (fp32 parameters)
and trained the way NanoJev's released checkpoint was
(``scripts/train_unified_games.py``, per its ``config.json``): the backbone and
the heads as two AdamW groups (backbone learning rate 1e-5, heads 1e-4, weight
decay 0.01), a bf16 autocast forward, gradient checkpointing on the backbone,
and gradients clipped at 1.0. Examples are NanoJev's candidate paths from
``prepare_examples``, packed exactly as inference packs them
(:func:`honing.backends.jev.nanojev.pack_request`).

The data handling, the losses and the loops are the ones every decision-model
trainer shares (:mod:`honing.backends.jev.decision_training`): every question
of every decision is one example; identical inputs are deduplicated and must
agree on their label; examples from validation groups are held out and must
not also appear in training. ``sft`` keeps the checkpoint with the best
validation loss; ``grpo`` runs the clipped policy gradient and keeps the final
weights. The trained
checkpoint has NanoJev's layout (``best.safetensors``, ``config.json``,
``backbone_config/``, ``tokenizer/``), so ``DecisionPredictor`` and
:class:`~honing.backends.jev.nanojev.NanoJevDecisionModel` load it directly.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import random
import re
import shutil
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
    ) -> tuple[list[Example], list[Example], dict[str, int]]:
        """Every question of every decision as NanoJev's example (candidate paths), packed as inference packs it."""
        tokenizer, limit = loaded.predictor.tokenizer, loaded.predictor.limit

        def pack(parts: dict[str, list[str]], questions: dict[str, dict[str, Any]]) -> tuple[Any, Any, Any]:
            measured = functools.partial(measure, loaded.prepare_examples, tokenizer, questions=questions)
            return pack_request(parts, questions, measured, limit)

        def encode(
            state: Any, questions: dict[str, dict[str, Any]], chosen: dict[str, str]
        ) -> Iterator[tuple[str, Example]]:
            for example in loaded.prepare_examples(payload(state, questions), tokenizer, limit):
                key = example["qid"]
                label = example["candidate_ids"].index(chosen[key])
                yield key, {**example, "label": label, "ids": _identity(example), "markers": []}

        return decision_examples(
            decisions,
            pack=pack,
            encode=encode,
            validation_groups=self._settings.validation_groups,
            advantages=advantages,
        )

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

    def _forward(self, loaded: _Loaded) -> Forward:
        """A chunk's option logits in fp32 from a bf16 autocast forward, and its labels."""
        torch, predictor = loaded.torch, loaded.predictor

        def forward(chunk: Sequence[Example]) -> tuple[Any, Any, Any]:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = predictor.model(list(chunk), predictor.tokenizer.pad_token_id)
            return logits.float(), torch.tensor([x["label"] for x in chunk], device="cuda"), None

        return forward

    # -------------------------------------------------------------- training

    def _supervised(
        self, job: TrainJobSpec, base: ModelSnapshot, batch: TrainingBatch, run: Path, loaded: _Loaded
    ) -> TrainResult:
        settings, torch, model = self._settings, loaded.torch, loaded.predictor.model
        items, validation, packing = self._examples(batch.decisions, loaded)
        items, validation, overlap = deduplicated_split(items, validation, settings.min_validation)
        self._trainable(job, model)
        optimizer = self._optimizer(torch, model)
        started = time.time()
        checkpoint = run / "checkpoint"
        result = supervised_epochs(
            torch,
            model,
            optimizer,
            items,
            validation,
            forward=self._forward(loaded),
            epochs=job.steps,
            batch_size=settings.batch_questions,
            seed=settings.seed,
            save=lambda: self._save(checkpoint, Path(base.checkpoint.location), loaded),
        )
        fields, metrics = result.record(items, validation, overlap)
        return self._result(
            job,
            base,
            batch,
            run,
            checkpoint,
            loaded,
            packing,
            history=result.history,
            started=started,
            extra=fields,
            metrics=metrics,
        )

    def _policy_gradient(
        self, job: TrainJobSpec, base: ModelSnapshot, batch: TrainingBatch, run: Path, loaded: _Loaded
    ) -> TrainResult:
        settings, torch, model = self._settings, loaded.torch, loaded.predictor.model
        advantages = group_advantages(batch.decisions)
        examples, _, packing = self._examples(batch.decisions, loaded, advantages)
        forward = self._forward(loaded)
        behavior_gap = record_start(torch, model, examples, forward=forward, batch_size=settings.batch_questions)
        self._trainable(job, model)
        optimizer = self._optimizer(torch, model)
        started = time.time()
        history = policy_gradient_epochs(
            torch,
            model,
            optimizer,
            examples,
            forward=forward,
            epochs=job.steps,
            batch_size=settings.batch_questions,
            seed=settings.seed,
            clip_ratio=settings.clip_ratio,
        )
        checkpoint = run / "checkpoint"
        self._save(checkpoint, Path(base.checkpoint.location), loaded)
        fields, metrics = policy_gradient_record(batch, advantages, examples, history, behavior_gap)
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
            extra=fields,
            metrics=metrics,
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
