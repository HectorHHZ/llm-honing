"""Config factories for a campaign on real NanoJev weights: the pinned release, servers on demand, and the
direction-edit trainer. ``configs/e2e/tank_direction_edit.json`` uses them.

The environment names the pinned NanoJev checkout and the GPUs:
``HONING_NANOJEV_SOURCE`` (the checkout at the revision in ``pins.json``),
``HONING_NANOJEV_GPUS`` (comma-separated devices for the decision servers;
a gate serves the head and the candidate at once, so give it two) and
``HONING_NANOJEV_EDIT_GPU`` (the device the trainer's forward pass uses).
``HF_HOME`` must hold the pinned release (``huggingface-cli download
C-Tianyu/NanoJev --revision <pins.json revision>``).
"""

from __future__ import annotations

import atexit
import os
from pathlib import Path
from typing import Any

from honing.backends.jev.direction_edit import DirectionEditSettings, NanoJevDirectionEditTrainer
from honing.backends.jev.nanojev_serving import NanoJevModelSource, NanoJevServing, nanojev_endpoints
from honing.backends.sglang.endpoints import SGLangPolicyEndpoints

DEFAULT_PINS = "integrations/ra2/pins.json"


def _pins(context: Any, pins: str) -> Path:
    return (Path(context.root) / os.path.expandvars(pins)).resolve()


def model_source(context: Any, pins: str = DEFAULT_PINS, model: str = "nanojev") -> NanoJevModelSource:
    """The pinned NanoJev release as the campaign's first model."""
    return NanoJevModelSource(_pins(context, pins), model)


def endpoints(
    context: Any,
    source: str = "$HONING_NANOJEV_SOURCE",
    gpus: str = "$HONING_NANOJEV_GPUS",
    pins: str = DEFAULT_PINS,
    model: str = "nanojev",
    max_length: int = 4096,
    inference: str = "tree",
) -> SGLangPolicyEndpoints:
    """A NanoJev decision server per checkpoint, started on demand; they stop when the process exits."""
    served = nanojev_endpoints(
        NanoJevServing(
            pins_path=_pins(context, pins),
            source_dir=Path(os.path.expandvars(source)),
            gpus=tuple(g.strip() for g in os.path.expandvars(gpus).split(",") if g.strip()),
            model=model,
            max_length=max_length,
            inference=inference,
            log_dir=Path(context.work_dir) / "serving",
        )
    )
    atexit.register(served.close)
    return served


def direction_edit_trainer(
    context: Any,
    source: str = "$HONING_NANOJEV_SOURCE",
    gpu: str = "$HONING_NANOJEV_EDIT_GPU",
    pins: str = DEFAULT_PINS,
    model: str = "nanojev",
    question: str | None = None,
    target: str = "label",
    strength: float = 1.0,
    max_questions: int = 1024,
    max_length: int = 4096,
) -> NanoJevDirectionEditTrainer:
    """The direction-edit trainer: ``question``, ``target`` and ``strength`` choose the edit."""
    return NanoJevDirectionEditTrainer(
        DirectionEditSettings(
            pins_path=_pins(context, pins),
            source_dir=Path(os.path.expandvars(source)),
            work_dir=Path(context.work_dir) / "edits",
            model=model,
            gpu=os.path.expandvars(gpu),
            question=question,
            target=target,
            strength=strength,
            max_questions=max_questions,
            max_length=max_length,
        )
    )


__all__ = ["direction_edit_trainer", "endpoints", "model_source"]
