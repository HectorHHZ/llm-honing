"""Edit a NanoJev decision head along one direction read from the model itself, with no training.

This is the weight-space editing method from "Sparks of Weight-Space
Programming" (Takehi, Teoh, Zhang and Khattab, 2026), as a Honing
:class:`~honing.core.Trainer` for the ``direction_edit`` algorithm. The batch
is ``typed_decisions`` data, usually a teacher's rounds. Every question in it
is run once through the model, forward only, and the head's input for every
option is kept: the LayerNorm output ``h`` that NanoJev's scalar head reads
at the end of each option's candidate path. The direction is the mean, over
the questions the settings select, of ``h[target] - mean h[other options]``,
where the target is the recorded label or one named option. Its unit vector
times ``strength`` is added to ``scalar.weight``. Nothing else changes: the
set-attention head, the norm and the backbone keep their released values, and
the edit is 1,024 numbers.

The method exists because of a finding on Tank War: the released head is
orthogonal to the direction that separates the ``fire`` and ``hold`` options
(cosine -0.05), so it answers that question at random, and one unit of the
direction makes it take every shot the planner permits. The same reading
applies to any typed question whose options share a consistent direction in
the head's input; ``target="label"`` reads the direction toward whatever the
batch's labels chose.

There are no gradients, no optimizer and no iterative fit: ``job.steps`` is
ignored, and the edit is the same whichever order the decisions come in. The
checkpoint has NanoJev's layout plus a ``checkpoint.json`` manifest, so the
decision server loads it with ``--checkpoint-manifest`` and
:func:`~honing.backends.jev.nanojev_serving.nanojev_endpoints` starts it on
demand.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import math
import os
import shutil
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from honing.backends.jev.nanojev import (
    import_nanojev,
    measure,
    nanojev_snapshot,
    pack_request,
    payload,
    weights_checksum,
)
from honing.backends.jev.trainer import laya_question

if TYPE_CHECKING:
    import numpy as np

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
    name="nanojev-direction-edit",
    architectures="uniform",
    train_numerics=frozenset({"bf16"}),
    weight_formats=frozenset({"hf_safetensors"}),
    algorithms=(("direction_edit", "native"),),
    contracts=frozenset({"typed_decisions"}),
)
HEAD_WEIGHT = "scalar.weight"
LABEL = "label"

# The head's input for every option of every question (questions x options x hidden, as a numpy array), and
# what each question was. numpy and safetensors are imported where they are used: they exist only where a
# NanoJev checkpoint does.
HeadInputs = tuple[Any, list[dict[str, Any]]]
Reader = Callable[[Sequence[DecisionSample], Path], HeadInputs]


@dataclass(frozen=True)
class DirectionEditSettings:
    """Which questions the direction is read from, how strongly it is applied, and where the model runs.

    ``question`` selects the questions by id (``"weapon"``); ``None`` takes
    every question. ``target`` is ``"label"`` (the direction toward the
    batch's own labels) or an option id (``"fire"``: the direction toward that
    option, whatever the label); a question without that option is skipped.
    ``strength`` multiplies the unit direction; the metrics report the logit
    margin it adds on the batch. ``max_questions`` caps the forward pass.
    ``pins_path`` names the pinned NanoJev source and release, as the other
    NanoJev backends do; ``source_dir`` is the pinned checkout.
    """

    pins_path: Path
    source_dir: Path
    work_dir: Path
    model: str = "nanojev"
    gpu: str = "0"
    question: str | None = None
    target: str = LABEL
    strength: float = 1.0
    max_questions: int = 1024
    max_length: int = 4096
    lock_dir: Path | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.strength) or self.strength == 0:
            raise ValueError("strength must be a finite, nonzero number")
        if self.max_questions <= 0 or self.max_length <= 0:
            raise ValueError("max_questions and max_length must be positive")
        if not self.target:
            raise ValueError("target must be 'label' or an option id")


def direction(inputs: HeadInputs, question: str | None, target: str) -> tuple[Any, dict[str, float]]:
    """The unit direction toward ``target`` over the selected questions, and what it is made of.

    Returns the direction and ``{"questions": rows used, "margin_per_unit":
    mean logit gap one unit of it adds between the target and the other
    options, "target_argmax_before": share of rows whose target already had
    the largest projection on the direction}``.
    """
    import numpy as np

    h_in, meta = inputs
    diffs = []
    for i, m in enumerate(meta):
        if question is not None and m["question"] != question:
            continue
        options = list(m["options"])
        if target == LABEL:
            t = int(m["label"])
        elif target in options:
            t = options.index(target)
        else:
            continue
        k = len(options)
        if k < 2:
            continue
        others = [j for j in range(k) if j != t]
        diffs.append(h_in[i, t] - h_in[i, others].mean(axis=0))
    if not diffs:
        raise ContractError(
            f"no question in the batch offers a direction toward {target!r}"
            + (f" on question {question!r}" if question else "")
        )
    stacked = np.stack(diffs)
    mean = stacked.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm == 0:
        raise ContractError("the selected questions give no direction: the target and the other options coincide")
    unit = (mean / norm).astype(np.float32)
    return unit, {
        "questions": float(len(diffs)),
        "margin_per_unit": float((stacked @ unit).mean()),
        "direction_norm": norm,
    }


def edited_head(weight: np.ndarray, unit: np.ndarray, strength: float) -> np.ndarray:
    """``scalar.weight`` with ``strength`` units of the direction added, in the weight's shape and dtype."""
    import numpy as np

    flat = weight.reshape(-1).astype(np.float32) + np.float32(strength) * unit
    return flat.reshape(weight.shape).astype(weight.dtype)


def _sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def _target_share(inputs: HeadInputs, weight: np.ndarray, question: str | None, target: str) -> float:
    """The share of selected questions whose target option gets the largest scalar score under ``weight``."""
    import numpy as np

    h_in, meta = inputs
    w = weight.reshape(-1).astype(np.float32)
    hits = total = 0
    for i, m in enumerate(meta):
        if question is not None and m["question"] != question:
            continue
        options = list(m["options"])
        t = int(m["label"]) if target == LABEL else options.index(target) if target in options else None
        if t is None:
            continue
        total += 1
        hits += int(np.argmax(h_in[i, : len(options)] @ w)) == t
    return hits / total if total else float("nan")


class NanoJevDirectionEditTrainer:
    """Satisfies :class:`~honing.core.Trainer` for ``direction_edit`` on NanoJev checkpoints.

    ``reader`` gives the head's inputs for a batch's decisions on a base
    checkpoint; the default runs NanoJev on the GPU in ``settings``. Tests
    pass one that returns prepared arrays.
    """

    support = SUPPORT

    def __init__(self, settings: DirectionEditSettings, reader: Reader | None = None) -> None:
        self._settings = settings
        self._reader = reader or self._read

    def train(self, job: TrainJobSpec, base: ModelSnapshot, batch: TrainingBatch | None) -> TrainResult:
        settings = self._settings
        if job.algorithm != "direction_edit" or job.contract != "typed_decisions":
            raise ContractError("the direction-edit trainer runs direction_edit on typed_decisions only")
        if job.update_policy.kind not in ("full", "param_subset"):
            raise ContractError(f"the direction-edit trainer cannot apply a {job.update_policy.kind} update")
        if batch is None or batch.contract.kind != "typed_decisions" or not batch.decisions:
            raise ContractError("the direction-edit trainer needs a typed_decisions batch with decisions")
        import numpy as np
        from safetensors.numpy import load_file, save_file

        parent = Path(base.checkpoint.location)
        if weights_checksum(parent) != base.checkpoint.checksum:
            raise ContractError("the base checkpoint's weights differ from its checksum")
        pins = json.loads(settings.pins_path.read_text())
        pin = pins["nanojevModels"][settings.model]
        run = settings.work_dir / f"edit-{uuid.uuid4().hex[:12]}"
        run.mkdir(parents=True)
        started = time.time()
        try:
            decisions = batch.decisions[: settings.max_questions]
            inputs = self._reader(decisions, parent)
            unit, stats = direction(inputs, settings.question, settings.target)
            state = load_file(str(parent / "best.safetensors"))
            if HEAD_WEIGHT not in state:
                raise ContractError(f"the base checkpoint has no {HEAD_WEIGHT}")
            before = state[HEAD_WEIGHT]
            if before.reshape(-1).shape != unit.shape:
                raise ContractError(f"{HEAD_WEIGHT} has {before.size} entries, the head's input {unit.size}")
            after = edited_head(before, unit, settings.strength)
            state[HEAD_WEIGHT] = after
            w0, w1 = before.reshape(-1).astype(np.float32), after.reshape(-1).astype(np.float32)
            metrics = {
                "questions": stats["questions"],
                "margin_added": settings.strength * stats["margin_per_unit"],
                "head_cosine": float(w0 @ w1 / (np.linalg.norm(w0) * np.linalg.norm(w1))),
                "head_direction_cosine": float(w0 @ unit / np.linalg.norm(w0)),
                "target_argmax_before": _target_share(inputs, before, settings.question, settings.target),
                "target_argmax_after": _target_share(inputs, after, settings.question, settings.target),
                "entries_changed": float(int((w0 != w1).sum())),
            }
            checkpoint = run / "checkpoint"
            checkpoint.mkdir()
            save_file({k: np.ascontiguousarray(v) for k, v in state.items()}, str(checkpoint / "best.safetensors"))
            shutil.copyfile(parent / "config.json", checkpoint / "config.json")
            for directory in ("backbone_config", "tokenizer"):
                shutil.copytree(parent / directory, checkpoint / directory)
            checksum = _sha256(checkpoint / "best.safetensors")
            if checksum == base.checkpoint.checksum:
                raise ContractError("the edit did not change the weights")
            manifest = {
                "kind": "trained-nanojev",
                "repo": "local:honing-nanojev-direction-edit/" + run.name,
                "revision": checksum,
                "path": str(checkpoint.resolve()),
                "weightsSha256": checksum,
                "parent": {"repo": pin["repo"], "revision": pin["revision"]},
                "base": {"location": base.checkpoint.location, "checksum": base.checkpoint.checksum},
                "sourceRevision": pins["nanojevSource"]["revision"],
                "batchId": batch.batch_id,
                "algorithm": "direction_edit",
                "edit": {
                    "tensor": HEAD_WEIGHT,
                    "question": settings.question,
                    "target": settings.target,
                    "strength": settings.strength,
                    "direction_norm": stats["direction_norm"],
                    "direction": unit.tolist(),
                },
                "metrics": metrics,
                "settings": {
                    "model": settings.model,
                    "maxQuestions": settings.max_questions,
                    "maxLength": settings.max_length,
                },
                "elapsedSeconds": time.time() - started,
            }
            (run / "checkpoint.json").write_text(json.dumps(manifest, indent=2) + "\n")
        except Exception as error:
            (run / "error.json").write_text(json.dumps({"error": str(error)}) + "\n")
            raise
        return TrainResult(snapshot=nanojev_snapshot(checkpoint), metrics=metrics)

    # -------------------------------------------------------------- the forward pass

    def _read(self, decisions: Sequence[DecisionSample], parent: Path) -> HeadInputs:
        """Run every question once on the GPU and keep the head's input for each option."""
        from honing.backends.jev.laya import gpu_lock

        settings = self._settings
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        pins = json.loads(settings.pins_path.read_text())
        upstream, _ = import_nanojev(settings.source_dir, pins)
        with gpu_lock(settings.gpu, settings.lock_dir), contextlib.redirect_stdout(sys.stderr):
            import numpy as np
            import torch

            predictor = upstream.DecisionPredictor(
                str(parent), max_length=settings.max_length, device_name="cuda:0", precision="bf16"
            )
            model, tokenizer = predictor.model, predictor.tokenizer
            examples: list[dict[str, Any]] = []
            for decision in decisions:
                parts = {name: list(lines) for name, lines in decision.parts}
                questions = {key: laya_question(q) for key, q in decision.questions}
                state, _, _ = pack_request(
                    parts,
                    questions,
                    functools.partial(measure, upstream.prepare_examples, tokenizer, questions=questions),
                    predictor.limit,
                )
                chosen = dict(decision.chosen)
                for example in upstream.prepare_examples(payload(state, questions), tokenizer, predictor.limit):
                    example["_label"] = example["candidate_ids"].index(chosen[example["qid"]])
                    examples.append(example)
            hidden = int(model.backbone.config.hidden_size)
            kmax = max(len(e["candidate_ids"]) for e in examples)
            h_in = np.zeros((len(examples), kmax, hidden), dtype=np.float32)
            meta = []
            captured: dict[str, Any] = {}
            handle = model.norm.register_forward_hook(
                lambda m, i, o: captured.__setitem__("h", o.detach().float().cpu())
            )
            try:
                with torch.no_grad():
                    for start in range(0, len(examples), 8):
                        chunk = examples[start : start + 8]
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            model(chunk, tokenizer.pad_token_id)
                        h = captured["h"]
                        for i, example in enumerate(chunk):
                            k = len(example["candidate_ids"])
                            h_in[start + i, :k] = h[i, :k].numpy()
                            meta.append(
                                {
                                    "question": example["qid"],
                                    "options": list(example["candidate_ids"]),
                                    "label": example["_label"],
                                }
                            )
            finally:
                handle.remove()
        return h_in, meta


__all__ = [
    "HEAD_WEIGHT",
    "SUPPORT",
    "DirectionEditSettings",
    "HeadInputs",
    "NanoJevDirectionEditTrainer",
    "direction",
    "edited_head",
]
