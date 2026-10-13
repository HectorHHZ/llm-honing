"""Factories for e2e configs: a real Hugging Face checkpoint as the model source, and SGLang policy endpoints.

Paths and GPUs come from the config's args, with ``$VARIABLES`` expanded from
the environment: by default ``HONING_SGLANG_CHECKPOINT`` (a checkpoint
directory), ``SGLANG_PYTHON`` (the SGLang environment's interpreter) and
``HONING_SGLANG_GPUS`` (comma-separated GPU indices or UUIDs).
"""

from __future__ import annotations

import atexit
import os
from pathlib import Path
from typing import Any

from honing.backends.sglang.checkpoint import hf_snapshot
from honing.backends.sglang.endpoints import ServingSettings, SGLangPolicyEndpoints
from honing.core import EpisodeOutcome, Grade, ModelSnapshot, Task


class HFModelSource:
    """Satisfies ModelSource: a Hugging Face checkpoint directory, named by its own weights' checksum."""

    def __init__(self, checkpoint: Path) -> None:
        self.checkpoint = checkpoint

    def materialize(self, work_dir: Path) -> ModelSnapshot:
        return hf_snapshot(self.checkpoint)


def model_source(context: Any, checkpoint: str = "$HONING_SGLANG_CHECKPOINT") -> HFModelSource:
    return HFModelSource(Path(os.path.expandvars(checkpoint)))


def endpoints(
    context: Any,
    runtime: str = "jev_decision",
    python: str = "$SGLANG_PYTHON",
    gpus: str = "$HONING_SGLANG_GPUS",
    max_length: int = 8192,
    prefix_cache: bool = True,
) -> SGLangPolicyEndpoints:
    """Endpoints that start a server per checkpoint; they stop when the process exits."""
    served = SGLangPolicyEndpoints(
        ServingSettings(
            python=Path(os.path.expandvars(python)),
            gpus=tuple(g.strip() for g in os.path.expandvars(gpus).split(",") if g.strip()),
            runtime=runtime,  # type: ignore[arg-type]
            max_length=max_length,
            prefix_cache=prefix_cache,
            log_dir=Path(context.work_dir) / "serving",
        )
    )
    atexit.register(served.close)
    return served


class ReplyGrader:
    """Satisfies TaskGrader for a check of the serving path: 1.0 when the agent's final output is a nonempty reply
    from the policy, 0.0 with feedback otherwise. It says nothing about the reply's quality."""

    async def grade(self, task: Task, outcome: EpisodeOutcome) -> Grade:
        if outcome.final_output.strip():
            return Grade(score=1.0, feedback=None)
        return Grade(score=0.0, feedback="the policy returned an empty reply")


def reply_grader(context: Any) -> ReplyGrader:
    return ReplyGrader()
