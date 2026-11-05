"""What serving needs from a checkpoint beyond its identity: the name it is reported under and how long to serve it.

The checkpoint's weight shards, checksum and snapshot are
:mod:`honing.backends.hf_checkpoint`'s, shared with every other backend.
"""

from __future__ import annotations

import json
from pathlib import Path


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
