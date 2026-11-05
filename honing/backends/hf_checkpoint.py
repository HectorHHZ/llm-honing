"""A Hugging Face checkpoint directory as Honing names it: its weight shards, their checksum, and its snapshot.

Every backend that reads or writes Hugging Face weights names a checkpoint
with these functions: the serving engine (:mod:`honing.backends.sglang`), the
weight editor (:mod:`honing.backends.weight_edit`), and the decision-model
loaders under :mod:`honing.backends.jev`. A checkpoint therefore keeps one
identity whichever backend wrote, edited, or serves it. The module imports
only the standard library and :mod:`honing.core`.

The checksum is the one StartLux-Decision's checkpoints already carry: the
SHA-256 of each shard's file name and SHA-256, in the order of the index. An
engine that serves a checkpoint reports this checksum, computed by Honing
from the files, so a rollout it produces names the release whose weights
produced it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from honing.core import ArchitectureSpec, CheckpointRef, LayerShape, ModelSnapshot


def weight_shards(checkpoint: Path) -> list[Path]:
    """The checkpoint's weight files: the shards its index lists, or its ``*.safetensors`` files."""
    index = checkpoint / "model.safetensors.index.json"
    if index.exists():
        names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        return [checkpoint / name for name in names]
    shards = sorted(checkpoint.glob("*.safetensors"))
    if not shards:
        raise ValueError(f"{checkpoint} holds no safetensors weights")
    return shards


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_CHECKSUMS: dict[tuple[tuple[str, int, int], ...], str] = {}


def checkpoint_checksum(checkpoint: Path) -> str:
    """One checksum over the checkpoint's weights.

    Hashing a large checkpoint reads every shard once (several minutes for hundreds of GB), and a checkpoint is
    named by its checksum (``hf_snapshot``) and then verified against it on its first ``resolve``. The result is
    kept per process, keyed by each shard's path, size and modification time, so unchanged weights are read once.
    """
    shards = weight_shards(checkpoint)
    key = tuple((str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns) for path in shards)
    if key not in _CHECKSUMS:
        lines = "".join(f"{path.name}:{_sha256(path)}\n" for path in shards)
        _CHECKSUMS[key] = hashlib.sha256(lines.encode()).hexdigest()
    return _CHECKSUMS[key]


def safetensors_parameters(path: Path) -> int:
    """Count the parameters in a safetensors file from its header alone."""
    with path.open("rb") as handle:
        length = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(length))
    total = 0
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        count = 1
        for dimension in tensor["shape"]:
            count *= int(dimension)
        total += count
    return total


def hf_snapshot(checkpoint: Path) -> ModelSnapshot:
    """The checkpoint as a ModelSnapshot: its text decoder's shape, parameter count and checksum."""
    config = json.loads((checkpoint / "config.json").read_text())
    text = config.get("text_config", config)
    experts = int(text.get("num_experts") or text.get("num_local_experts") or 0)
    ffn = int(text.get("intermediate_size") or text.get("moe_intermediate_size"))
    layer = LayerShape(attention_heads=int(text["num_attention_heads"]), ffn_size=ffn, num_experts=experts)
    return ModelSnapshot(
        checkpoint=CheckpointRef(
            location=str(checkpoint), weight_format="hf_safetensors", checksum=checkpoint_checksum(checkpoint)
        ),
        architecture=ArchitectureSpec(
            hidden_size=int(text["hidden_size"]), layers=(layer,) * int(text["num_hidden_layers"])
        ),
        parameters=sum(safetensors_parameters(path) for path in weight_shards(checkpoint)),
    )
