"""A Hugging Face checkpoint directory as Honing names it: its weight shards, their checksum, and its snapshot.

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

from honing.backends.jev.laya import safetensors_parameters
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


def checkpoint_checksum(checkpoint: Path) -> str:
    """One checksum over the checkpoint's weights."""
    lines = "".join(f"{path.name}:{_sha256(path)}\n" for path in weight_shards(checkpoint))
    return hashlib.sha256(lines.encode()).hexdigest()


def checkpoint_identity(checkpoint: Path, checksum: str) -> tuple[str, str]:
    """A name and revision for a checkpoint: the repository and commit of a Hugging Face cache snapshot, or the
    directory's name and the first 12 characters of the weights' checksum."""
    parts = checkpoint.resolve().parts
    if len(parts) >= 3 and parts[-2] == "snapshots" and parts[-3].startswith("models--"):
        return parts[-3].removeprefix("models--").replace("--", "/"), parts[-1]
    return checkpoint.name, checksum[:12]


def context_length(checkpoint: Path) -> int | None:
    """The longest sequence to serve the checkpoint for, from its config, or None when the config does not say.

    For LongRoPE (Phi-3.5) it is the original length, not the extended one. Past
    it, transformers switches the sequence to the long scaling factors, while
    SGLang 0.5.21 switches every request in a batch whenever any position in the
    batch passes it (``rope_variant.py``), and its radix cache can join keys
    computed under both. Serving no further than the original length keeps the
    short factors, and every answer, independent of what else is batched.
    """
    config = json.loads((checkpoint / "config.json").read_text())
    for section in (config.get("text_config") or {}, config):
        scaling = section.get("rope_scaling") or {}
        original = section.get("original_max_position_embeddings")
        if (scaling.get("rope_type") or scaling.get("type")) == "longrope" and isinstance(original, int):
            return original
    for section in (config.get("text_config") or {}, config):
        for key in ("max_position_embeddings", "seq_length", "max_sequence_length"):
            if isinstance(section.get(key), int) and section[key] > 0:
                return int(section[key])
    return None


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
