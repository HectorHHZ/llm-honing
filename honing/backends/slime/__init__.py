"""Slime, the training engine for language models, behind Honing's Trainer interface.

Honing never imports Slime. :class:`SlimeTrainer` runs each job as a subprocess
in Slime's image and passes files: rollouts in Slime's layout
(:mod:`~honing.backends.slime.rollout_data`), Hugging Face checkpoints, and the
metrics Slime logged. :mod:`~honing.backends.slime.worker` is the part that runs
inside the image. :class:`LocalContainerSandbox` is slime's agent ``Sandbox`` as a
local Docker container; it satisfies slime's protocol without importing slime.
"""

from honing.backends.slime.arguments import (
    GrpoSettings,
    JobSettings,
    OpdSettings,
    SftSettings,
    SlimeSettings,
    megatron_model_args,
    train_args,
)
from honing.backends.slime.rollout_data import RolloutLayout, policy_sample, read_samples, slime_samples, write_samples
from honing.backends.slime.sandbox import ContainerLimits, LocalContainerSandbox, SandboxError, SandboxTimeoutError
from honing.backends.slime.trainer import (
    SUPPORT,
    HFModelSource,
    SlimeEnvironment,
    SlimeRunError,
    SlimeTrainer,
    SlimeTrainerSettings,
    hf_snapshot,
)

__all__ = [
    "SUPPORT",
    "ContainerLimits",
    "GrpoSettings",
    "HFModelSource",
    "JobSettings",
    "LocalContainerSandbox",
    "OpdSettings",
    "RolloutLayout",
    "SandboxError",
    "SandboxTimeoutError",
    "SftSettings",
    "SlimeEnvironment",
    "SlimeRunError",
    "SlimeSettings",
    "SlimeTrainer",
    "SlimeTrainerSettings",
    "hf_snapshot",
    "megatron_model_args",
    "policy_sample",
    "read_samples",
    "slime_samples",
    "train_args",
    "write_samples",
]
