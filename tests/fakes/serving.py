"""A stand-in trainer whose checkpoints are real: for a campaign that serves every candidate through SGLang.

``NormScaleTrainer`` does not train. Job ``n`` writes a new Hugging Face
checkpoint that is its base with the final norm's weights scaled by
``factor ** n``, and links every other file to the base's, so two jobs on one
base still give two different checkpoints. The result is a
distinct, loadable model with its own checksum, whose answers move a little,
so a campaign's train moves reach the serving path the way a real trainer's
would: a checkpoint no server has loaded yet, started on demand by the
endpoints, gated on real episodes, then published or rolled back. It uses the
standard library only.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

from honing.backends.sglang.checkpoint import hf_snapshot, weight_shards
from honing.core import BackendSupport, ModelSnapshot, TrainingBatch, TrainJobSpec, TrainResult

SUPPORT = BackendSupport(
    name="norm-scale",
    architectures="uniform",
    train_numerics=frozenset({"bf16"}),
    weight_formats=frozenset({"hf_safetensors"}),
    algorithms=(("sft", "native"), ("grpo", "native")),
    contracts=frozenset({"typed_decisions"}),
)
NORMS = ("model.norm.weight", "model.language_model.norm.weight")


def _scaled(raw: bytes, dtype: str, factor: float) -> bytes:
    """A tensor's bytes with every element multiplied by ``factor`` (bf16 rounded to nearest even, or f32)."""
    if dtype == "F32":
        values = struct.unpack(f"<{len(raw) // 4}f", raw)
        return struct.pack(f"<{len(values)}f", *(v * factor for v in values))
    if dtype != "BF16":
        raise ValueError(f"cannot scale a {dtype} norm")
    out = bytearray()
    for (half,) in struct.iter_unpack("<H", raw):
        (value,) = struct.unpack("<f", struct.pack("<I", half << 16))
        (bits,) = struct.unpack("<I", struct.pack("<f", value * factor))
        out += struct.pack("<H", ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF)
    return bytes(out)


def scale_norm(base: Path, out: Path, factor: float) -> None:
    """Write ``out``: ``base`` with its final norm scaled by ``factor``; every other file is a link to base's."""
    out.mkdir(parents=True)
    shards = {path.name for path in weight_shards(base)}
    for path in base.iterdir():
        if path.name not in shards:
            (out / path.name).symlink_to(path.resolve())
    changed = False
    for name in sorted(shards):
        data = (base / name).read_bytes()
        length = int.from_bytes(data[:8], "little")
        header = json.loads(data[8 : 8 + length])
        norm = next((key for key in NORMS if key in header), None)
        if norm is None:
            (out / name).symlink_to((base / name).resolve())
            continue
        start, end = (8 + length + offset for offset in header[norm]["data_offsets"])
        scaled = _scaled(data[start:end], header[norm]["dtype"], factor)
        (out / name).write_bytes(data[:start] + scaled + data[end:])
        changed = True
    if not changed:
        raise ValueError(f"{base} has no final norm among {NORMS}")


class NormScaleTrainer:
    """Satisfies Trainer with real, servable checkpoints and no training; see the module docstring."""

    support = SUPPORT

    def __init__(self, work_dir: Path, factor: float) -> None:
        self.work_dir = work_dir
        self.factor = factor
        self.jobs = 0

    def train(self, job: TrainJobSpec, base: ModelSnapshot, batch: TrainingBatch | None) -> TrainResult:
        self.jobs += 1
        out = self.work_dir / f"{self.jobs:03d}-{job.algorithm}-from-{base.checkpoint.checksum[:8]}"
        scale_norm(Path(base.checkpoint.location), out, self.factor**self.jobs)
        return TrainResult(snapshot=hf_snapshot(out), metrics={"train_steps": float(job.steps)})


def trainer(context: Any, factor: float = 1.01) -> NormScaleTrainer:
    return NormScaleTrainer(Path(context.work_dir) / "trained", factor)
