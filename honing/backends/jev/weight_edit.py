"""NanoJev checkpoint layout for the general weight modification trainer."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from honing.backends.jev.nanojev import weights_checksum
from honing.backends.weight_edit import CheckpointLayout, WeightEditTrainer
from honing.backends.weight_edit import weight_edit_trainer as configured_trainer


def nanojev_layout(pins_path: Path, model: str = "nanojev") -> CheckpointLayout:
    """Keep NanoJev's checksum and the serving manifest used by its model loader."""
    pins = json.loads(pins_path.read_text())
    pin = pins["nanojevModels"][model]
    return CheckpointLayout(
        checksum=weights_checksum,
        files=lambda parent: [parent / "best.safetensors"],
        manifest_fields={
            "kind": "trained-nanojev",
            "parent": {"repo": pin["repo"], "revision": pin["revision"]},
            "sourceRevision": pins["nanojevSource"]["revision"],
        },
    )


def weight_edit_trainer(
    context: Any,
    edits: Sequence[Mapping[str, Any]],
    pins: str = "integrations/ra2/pins.json",
    model: str = "nanojev",
) -> WeightEditTrainer:
    """Pipeline factory: same operations as the general editor, with NanoJev storage."""
    layout = nanojev_layout(Path(context.root) / os.path.expandvars(pins), model)
    return configured_trainer(context, edits, layout=layout)
