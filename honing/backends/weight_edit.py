"""Weight changes through the Trainer contract, without an optimizer.

A modifier returns named tensor operations. The trainer checks the update
policy, writes a separate safetensors checkpoint, and returns a TrainResult.
Model-specific code only supplies the modifier and checkpoint layout.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from honing.backends.hf_checkpoint import checkpoint_checksum, weight_shards
from honing.campaign import rules
from honing.core import (
    BackendSupport,
    CheckpointRef,
    ContractError,
    DataContract,
    ModelSnapshot,
    TrainingBatch,
    TrainJobSpec,
    TrainResult,
)


@dataclass(frozen=True)
class TensorEdit:
    """One named tensor: add an exact-shape delta, scale by a scalar, or replace.

    ``value`` is a CPU array/tensor or JSON numbers. Broadcasting is not used.
    The result keeps the original shape and dtype. Integer buffers cannot be edited.
    """

    parameter: str
    operation: Literal["add", "scale", "replace"]
    value: Any


@dataclass(frozen=True)
class EditPlan:
    """A modifier's operations, finite numeric metrics, and JSON metadata."""

    edits: tuple[TensorEdit, ...]
    metrics: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


Modifier = Callable[[ModelSnapshot, TrainingBatch | None], EditPlan]


@dataclass(frozen=True)
class CheckpointLayout:
    """Storage adapters; the defaults are :mod:`honing.backends.hf_checkpoint`'s checkpoint identity.

    A model with another checksum convention supplies its existing checksum
    function and weight-file selector. ``manifest_fields`` adds serving metadata;
    the trainer owns identity, base, operations, metrics and update-policy fields.
    """

    checksum: Callable[[Path], str] = checkpoint_checksum
    files: Callable[[Path], Sequence[Path]] = weight_shards
    manifest_fields: Mapping[str, Any] = field(default_factory=dict)


def _apply(weight: Any, edit: TensorEdit) -> Any:
    import torch

    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ContractError(f"{edit.parameter}: only floating-point master weights can be edited")
    dtype = torch.float64 if weight.dtype == torch.float64 else torch.float32
    value = torch.as_tensor(edit.value, dtype=dtype, device="cpu")
    if not torch.isfinite(weight).all() or not torch.isfinite(value).all():
        raise ContractError(f"{edit.parameter}: weights and edit values must be finite")
    if edit.operation == "scale":
        if value.ndim != 0:
            raise ContractError(f"{edit.parameter}: scale needs one scalar")
        result = weight.to(dtype) * value
    elif edit.operation in ("add", "replace"):
        if tuple(value.shape) != tuple(weight.shape):
            raise ContractError(f"{edit.parameter}: edit shape {tuple(value.shape)} != {tuple(weight.shape)}")
        result = weight.to(dtype) + value if edit.operation == "add" else value
    else:
        raise ContractError(f"{edit.parameter}: unknown operation {edit.operation!r}")
    # A modifier can reuse one value tensor for several replacements. Give each
    # output its own storage: safetensors refuses tensors that share memory.
    result = result.to(weight.dtype).contiguous().clone()
    if not torch.isfinite(result).all():
        raise ContractError(f"{edit.parameter}: edit overflows the stored dtype")
    return result


class WeightEditTrainer:
    """A pluggable Trainer for static edits or data-derived direction edits.

    ``weight_edit`` takes no batch and runs once (steps=1). ``direction_edit``
    takes typed decisions; its modifier derives the operations from that batch.
    No operation can add parameters, change shapes, or bypass param_subset.
    """

    def __init__(
        self,
        work_dir: Path,
        modifier: Modifier,
        *,
        layout: CheckpointLayout | None = None,
        algorithm: Literal["weight_edit", "direction_edit"] = "weight_edit",
    ) -> None:
        if algorithm not in ("weight_edit", "direction_edit"):
            raise ContractError(f"unsupported weight-edit algorithm {algorithm!r}")
        self.work_dir = work_dir
        self.modifier = modifier
        self.layout = layout or CheckpointLayout()
        self.algorithm = algorithm
        self.support = BackendSupport(
            name="weight-edit",
            architectures="heterogeneous",
            train_numerics=frozenset({"bf16"}),
            weight_formats=frozenset({"hf_safetensors"}),
            algorithms=((algorithm, "native"),),
            contracts=frozenset({"typed_decisions" if algorithm == "direction_edit" else "token_sequences"}),
        )

    def train(self, job: TrainJobSpec, base: ModelSnapshot, batch: TrainingBatch | None) -> TrainResult:
        if job.algorithm != self.algorithm:
            raise ContractError(f"this trainer runs {self.algorithm}, not {job.algorithm}")
        rules.algorithm_requirements_met(job, self.support)
        rules.batch_meets_contract(job, batch)
        if job.algorithm == "weight_edit" and job.steps != 1:
            raise ContractError("weight_edit applies one plan; steps must be 1")
        if job.update_policy.kind not in ("full", "param_subset"):
            raise ContractError(f"weight edits cannot apply a {job.update_policy.kind} update")
        if base.checkpoint.weight_format != "hf_safetensors":
            raise ContractError("weight edits need hf_safetensors master weights")
        parent = Path(base.checkpoint.location).resolve()
        if self.work_dir.resolve().is_relative_to(parent):
            raise ContractError("the edit output directory must be outside the base checkpoint")
        if self.layout.checksum(parent) != base.checkpoint.checksum:
            raise ContractError("the base checkpoint's weights differ from its checksum")
        plan = self.modifier(base, batch)
        if not plan.edits or len({e.parameter for e in plan.edits}) != len(plan.edits):
            raise ContractError("an edit plan needs unique, nonempty parameter operations")
        if not all(math.isfinite(v) for v in plan.metrics.values()):
            raise ContractError("edit metrics must be finite")
        json.dumps(plan.metadata, allow_nan=False)
        for edit in plan.edits:
            if not edit.parameter:
                raise ContractError("an edit must name its parameter")
            if job.update_policy.kind == "param_subset" and not any(
                re.search(pattern, edit.parameter) for pattern in job.update_policy.trainable_patterns
            ):
                raise ContractError(f"{edit.parameter} is excluded by the update policy")
        run = self.work_dir / f"edit-{uuid.uuid4().hex}"
        run.mkdir(parents=True)
        checkpoint = run / "checkpoint"
        try:
            metrics, operations = self._write(parent, checkpoint, plan)
            checksum = self.layout.checksum(checkpoint)
            if checksum == base.checkpoint.checksum:
                raise ContractError("the edit did not change the weights")
            manifest = {
                **self.layout.manifest_fields,
                "kind": self.layout.manifest_fields.get("kind", "weight-edit"),
                "repo": "local:honing-weight-edit/" + run.name,
                "revision": checksum,
                "path": str(checkpoint.resolve()),
                "weightsSha256": checksum,
                "base": {"location": base.checkpoint.location, "checksum": base.checkpoint.checksum},
                "batchId": batch.batch_id if batch is not None else None,
                "algorithm": job.algorithm,
                "updatePolicy": {
                    "kind": job.update_policy.kind,
                    "trainablePatterns": list(job.update_policy.trainable_patterns),
                },
                "operations": operations,
                "editValues": "edit-values.safetensors",
                "edit": dict(plan.metadata),
                "metrics": metrics,
            }
            (run / "checkpoint.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        except Exception as error:
            shutil.rmtree(checkpoint, ignore_errors=True)
            (run / "error.json").write_text(json.dumps({"error": str(error)}) + "\n")
            raise
        snapshot = replace(
            base,
            checkpoint=CheckpointRef(str(checkpoint.resolve()), "hf_safetensors", checksum),
        )
        return TrainResult(snapshot=snapshot, metrics=metrics)

    def _write(self, parent: Path, checkpoint: Path, plan: EditPlan) -> tuple[dict[str, float], list[dict[str, Any]]]:
        import torch
        from safetensors import safe_open
        from safetensors.torch import load_file, save_file

        paths = list(self.layout.files(parent))
        if not paths or len(set(paths)) != len(paths):
            raise ContractError("checkpoint layout must select distinct weight files")
        locations: dict[str, Path] = {}
        for path in paths:
            # Resolve the lexical file name, not HF cache symlinks, which are copied below.
            if not path.absolute().is_relative_to(parent) or ".." in path.parts:
                raise ContractError("weight file paths must stay within the checkpoint")
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                for key in handle.keys():  # noqa: SIM118 - safe_open is not an iterable dictionary
                    if key in locations:
                        raise ContractError(f"duplicate tensor {key} across checkpoint shards")
                    locations[key] = path
        edits = {e.parameter: e for e in plan.edits}
        missing = set(edits) - locations.keys()
        if missing:
            raise ContractError(f"checkpoint has no parameters {sorted(missing)}")
        index = parent / "model.safetensors.index.json"
        if index.exists():
            mapping = json.loads(index.read_text())["weight_map"]
            actual = {name: str(path.relative_to(parent)) for name, path in locations.items()}
            if mapping != actual:
                raise ContractError("the safetensors index does not match the stored tensors")
        # Copy, never hard-link, so editing a candidate cannot mutate its parent.
        shutil.copytree(
            parent,
            checkpoint,
            ignore=lambda path, names: (
                set(names) & {"checkpoint.json", "SHA256_MANIFEST.json"} if Path(path) == parent else set()
            ),
        )
        changed = 0
        operations = []
        values = {}
        for path in sorted({locations[name] for name in edits}):
            state = load_file(str(path), device="cpu")
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                metadata = handle.metadata()
            for name, edit in edits.items():
                if locations[name] != path:
                    continue
                before = state[name]
                after = _apply(before, edit)
                value_dtype = torch.float64 if before.dtype == torch.float64 else torch.float32
                values[name] = torch.as_tensor(edit.value, dtype=value_dtype, device="cpu").contiguous().clone()
                entries = int(torch.count_nonzero(before != after))
                changed += entries
                state[name] = after
                operations.append(
                    {
                        "parameter": name,
                        "operation": edit.operation,
                        "valueTensor": name,
                        "shape": list(after.shape),
                        "dtype": str(after.dtype),
                        "entriesChanged": entries,
                        "resultSha256": hashlib.sha256(
                            after.reshape(-1).view(torch.uint8).numpy().tobytes()
                        ).hexdigest(),
                    }
                )
            save_file(state, str(checkpoint / path.relative_to(parent)), metadata=metadata)
        if not changed:
            raise ContractError("the edit did not change any stored weight values")
        save_file(values, str(checkpoint.parent / "edit-values.safetensors"))
        return {**plan.metrics, "entries_changed": float(changed), "tensors_edited": float(len(edits))}, operations


def weight_edit_trainer(
    context: Any, edits: Sequence[Mapping[str, Any]], *, layout: CheckpointLayout | None = None
) -> WeightEditTrainer:
    """Pipeline factory for JSON operations; paths resolve against the config root.

    Each entry has parameter, operation, and either value or value_file/value_tensor.
    A file value is one tensor from a safetensors file. It is loaded when train runs.
    """

    def modifier(base: ModelSnapshot, batch: TrainingBatch | None) -> EditPlan:
        from safetensors import safe_open

        operations = []
        for entry in edits:
            if set(entry) - {"parameter", "operation", "value", "value_file", "value_tensor"}:
                raise ContractError("unknown weight-edit configuration field")
            if ("value" in entry) == ("value_file" in entry) or (
                "value_tensor" in entry and "value_file" not in entry
            ):
                raise ContractError("give value or value_file, not both")
            if "value_file" in entry:
                path = Path(context.root) / Path(os.path.expandvars(entry["value_file"])).expanduser()
                with safe_open(str(path), framework="pt", device="cpu") as handle:
                    value = handle.get_tensor(entry.get("value_tensor", entry["parameter"]))
            else:
                value = entry["value"]
            operations.append(TensorEdit(entry["parameter"], entry["operation"], value))
        return EditPlan(tuple(operations))

    return WeightEditTrainer(Path(context.work_dir) / "weight-edits", modifier, layout=layout)


class NoTrainingData:
    """DataProvider for a campaign that only schedules batch-free edits."""

    def next_batch(self, contract: DataContract) -> TrainingBatch | None:
        raise ContractError("this campaign has no training data; configure a provider for data-consuming jobs")


def no_data(context: Any) -> NoTrainingData:
    """Pipeline data factory for batch-free editing."""
    return NoTrainingData()
