"""Config factories for a campaign on real decision-model weights: the pinned NanoJev release, servers on
demand, and the trainers. ``configs/e2e/tank_direction_edit.json`` and ``configs/e2e/tank_nanojev_levers.json``
use them.

The environment names the pinned NanoJev checkout and the GPUs:
``HONING_NANOJEV_SOURCE`` (the checkout at the revision in ``pins.json``),
``HONING_NANOJEV_GPUS`` (comma-separated devices for the decision servers;
a gate serves the head and the candidate at once, so give it two),
``HONING_NANOJEV_EDIT_GPU`` (the device the direction-edit trainer's forward
pass uses) and ``HONING_NANOJEV_TRAIN_GPU`` (the device the NanoJev trainer
trains on). ``HF_HOME`` must hold the pinned release (``huggingface-cli
download C-Tianyu/NanoJev --revision <pins.json revision>``).

The Kev and Laya trainers follow the same pattern: ``HONING_KEV_SOURCE`` and
``HONING_KEV_TRAIN_GPU``, ``HONING_LAYA_SOURCE`` and ``HONING_LAYA_TRAIN_GPU``.
Each trainer factory takes the rest of its settings class's fields by name
(``batch_questions``, ``head_lr``, ``validation_groups`` and so on) and
refuses a name the class does not have.
"""

from __future__ import annotations

import atexit
import dataclasses
import os
from pathlib import Path
from typing import Any, TypeVar

from honing.backends.jev.direction_edit import DirectionEditSettings, NanoJevDirectionEditTrainer
from honing.backends.jev.kev_trainer import KevTrainer, KevTrainerSettings
from honing.backends.jev.nanojev_serving import NanoJevModelSource, NanoJevServing, nanojev_endpoints
from honing.backends.jev.nanojev_trainer import NanoJevTrainer, NanoJevTrainerSettings
from honing.backends.jev.trainer import JevTrainerSettings, LocalJevTrainer
from honing.backends.sglang.endpoints import SGLangPolicyEndpoints
from honing.core import ContractError

DEFAULT_PINS = "integrations/ra2/pins.json"
DEFAULT_LAYA_RECORD = "integrations/ra2/.cache/models/typed-decisions.json"

_Settings = TypeVar("_Settings")


def _pins(context: Any, pins: str) -> Path:
    return (Path(context.root) / os.path.expandvars(pins)).resolve()


def _settings(kind: type[_Settings], slot: str, placed: dict[str, Any], tuning: dict[str, Any]) -> _Settings:
    """``kind`` from the paths and devices a factory placed, plus the config's own tuning fields.

    A tuning field ``kind`` does not have, or one the factory places itself, is refused by name.
    """
    known = {field.name for field in dataclasses.fields(kind)}  # type: ignore[arg-type]
    unknown = sorted(set(tuning) - (known - set(placed)))
    if unknown:
        allowed = sorted(known - set(placed))
        raise ContractError(f"the {slot} factory has no setting {unknown}; it takes {allowed}")
    return kind(**placed, **tuning)


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


def nanojev_trainer(
    context: Any,
    source: str = "$HONING_NANOJEV_SOURCE",
    gpu: str = "$HONING_NANOJEV_TRAIN_GPU",
    pins: str = DEFAULT_PINS,
    model: str = "nanojev",
    **tuning: Any,
) -> NanoJevTrainer:
    """The NanoJev trainer (``sft`` and ``grpo`` on typed decisions), writing under ``training/nanojev``."""
    placed = {
        "pins_path": _pins(context, pins),
        "source_dir": Path(os.path.expandvars(source)),
        "work_dir": Path(context.work_dir) / "training" / "nanojev",
        "model": model,
        "gpu": os.path.expandvars(gpu),
    }
    return NanoJevTrainer(_settings(NanoJevTrainerSettings, "nanojev_trainer", placed, tuning))


def kev_trainer(
    context: Any,
    source: str = "$HONING_KEV_SOURCE",
    gpu: str = "$HONING_KEV_TRAIN_GPU",
    pins: str = DEFAULT_PINS,
    model: str = "kev-0.8b",
    **tuning: Any,
) -> KevTrainer:
    """The Kev trainer (``sft`` and ``grpo`` on typed decisions), writing under ``training/kev``."""
    placed = {
        "pins_path": _pins(context, pins),
        "source_dir": Path(os.path.expandvars(source)),
        "work_dir": Path(context.work_dir) / "training" / "kev",
        "model": model,
        "gpu": os.path.expandvars(gpu),
    }
    return KevTrainer(_settings(KevTrainerSettings, "kev_trainer", placed, tuning))


def laya_trainer(
    context: Any,
    source: str = "$HONING_LAYA_SOURCE",
    gpu: str = "$HONING_LAYA_TRAIN_GPU",
    pins: str = DEFAULT_PINS,
    model_record: str = DEFAULT_LAYA_RECORD,
    **tuning: Any,
) -> LocalJevTrainer:
    """The Laya trainer, :class:`~honing.backends.jev.trainer.LocalJevTrainer`, writing under ``training/laya``.

    ``model_record`` is the prepared record of the pinned base model, relative to the repository root.
    """
    placed = {
        "laya_source": Path(os.path.expandvars(source)),
        "pins_path": _pins(context, pins),
        "model_record": (Path(context.root) / os.path.expandvars(model_record)).resolve(),
        "work_dir": Path(context.work_dir) / "training" / "laya",
        "gpu": os.path.expandvars(gpu),
    }
    return LocalJevTrainer(_settings(JevTrainerSettings, "laya_trainer", placed, tuning))


__all__ = [
    "direction_edit_trainer",
    "endpoints",
    "kev_trainer",
    "laya_trainer",
    "model_source",
    "nanojev_trainer",
]
