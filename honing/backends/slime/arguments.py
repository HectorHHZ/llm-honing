"""The Slime command line for one training job.

Slime builds its Megatron model from command-line arguments, not from the
checkpoint, and checks them against the Hugging Face config when it starts
(``_hf_validate_args`` in ``slime/backends/megatron_utils/arguments.py``).
:func:`megatron_model_args` derives them from ``config.json`` for the dense
decoder families Slime loads straight from Hugging Face weights (``llama``,
``qwen2``, ``qwen3``), so the arguments always match the checkpoint being
trained. Other families are refused until their arguments are written out.

:func:`train_args` is one train-only job: Slime reads the samples Honing
wrote (``--load-debug-rollout-data``), never starts SGLang, loads the policy
from the base checkpoint, and saves the trained policy as a Hugging Face
directory (``--save-hf``). The algorithm adds its own arguments:

``grpo``
    The GRPO advantage over each group's rewards, with the KL to the base
    checkpoint computed and reported (``--use-kl-loss``).
``sft``
    Slime's ``sft_loss``: the negative log-likelihood of the trained tokens,
    with no advantages (``--disable-compute-advantages-and-returns``).
``opd``
    On-policy distillation: a teacher's log-probability of every token the
    student sampled, and ``opd_kl_coef`` times the per-token reverse KL
    subtracted from the advantage. Samples without a group or reward have no
    task advantage, so the update is pure distillation; grouped, rewarded
    samples add GRPO's advantage. The teacher's scores come one of two ways:

    - a checkpoint loaded in Megatron beside the student (``--opd-type megatron
      --opd-teacher-load``), which must share the student's architecture;
    - with no teacher checkpoint, the scores written into the rollout
      (``teacher_log_probs``) when it was collected (``--opd-type sglang``,
      which in train-only mode reads them from the samples). On-policy
      self-distillation takes this path: the teacher is the student itself,
      scored under privileged context.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from honing.core import ContractError

DENSE_FAMILIES = frozenset({"llama", "qwen2", "qwen3"})


def read_config(checkpoint: Path) -> dict[str, Any]:
    path = checkpoint / "config.json"
    if not path.is_file():
        raise ContractError(f"{checkpoint} is not a Hugging Face checkpoint: no config.json")
    config: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return config


def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def megatron_model_args(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Megatron model arguments for a dense decoder's Hugging Face config."""
    family = config.get("model_type")
    if family not in DENSE_FAMILIES:
        raise ContractError(f"no Megatron arguments for model_type {family!r}; supported: {sorted(DENSE_FAMILIES)}")
    heads = int(config["num_attention_heads"])
    groups = int(config.get("num_key_value_heads", heads))
    rope = config.get("rope_theta", (config.get("rope_parameters") or {}).get("rope_theta"))
    if rope is None:
        raise ContractError(f"{family} config has no rope_theta")
    args = [
        "--swiglu",
        "--num-layers", str(config["num_hidden_layers"]),
        "--hidden-size", str(config["hidden_size"]),
        "--ffn-hidden-size", str(config["intermediate_size"]),
        "--num-attention-heads", str(heads),
        "--use-rotary-position-embeddings",
        "--disable-bias-linear",
        "--normalization", "RMSNorm",
        "--norm-epsilon", repr(float(config["rms_norm_eps"])),
        "--rotary-base", _number(rope),
        "--vocab-size", str(config["vocab_size"]),
    ]  # fmt: skip
    if groups != heads:
        args += ["--group-query-attention", "--num-query-groups", str(groups)]
    if family == "qwen2" or config.get("attention_bias", False):
        args.append("--add-qkv-bias")
    if family == "qwen3":
        args += ["--qk-layernorm", "--kv-channels", str(config.get("head_dim", int(config["hidden_size"]) // heads))]
    if not config.get("tie_word_embeddings", False):
        args.append("--untie-embeddings-and-output-weights")
    return tuple(args)


@dataclass(frozen=True)
class JobSettings:
    """What every job sets. Anything else goes through ``extra_args``, appended last."""

    lr: float = 1e-6
    max_tokens_per_gpu: int = 9216
    seed: int = 1234
    extra_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class GrpoSettings(JobSettings):
    kl_loss_coef: float = 0.0
    eps_clip: float = 0.2
    eps_clip_high: float = 0.28


@dataclass(frozen=True)
class SftSettings(JobSettings):
    lr: float = 1e-5


@dataclass(frozen=True)
class OpdSettings(JobSettings):
    opd_kl_coef: float = 1.0
    eps_clip: float = 0.2
    eps_clip_high: float = 0.28


@dataclass(frozen=True)
class SlimeSettings:
    """The settings for each algorithm the Slime trainer runs."""

    grpo: GrpoSettings = field(default_factory=GrpoSettings)
    sft: SftSettings = field(default_factory=SftSettings)
    opd: OpdSettings = field(default_factory=OpdSettings)


def _algorithm_args(algorithm: str, settings: SlimeSettings, base: Path, teacher: Path | None) -> tuple[str, ...]:
    if algorithm == "grpo":
        grpo = settings.grpo
        return (
            "--advantage-estimator", "grpo",
            "--ref-load", str(base),
            "--use-kl-loss",
            "--kl-loss-coef", repr(grpo.kl_loss_coef),
            "--kl-loss-type", "low_var_kl",
            "--entropy-coef", "0.0",
            "--eps-clip", repr(grpo.eps_clip),
            "--eps-clip-high", repr(grpo.eps_clip_high),
        )  # fmt: skip
    if algorithm == "sft":
        return ("--loss-type", "sft_loss", "--disable-compute-advantages-and-returns")
    if algorithm == "opd":
        opd = settings.opd
        source = (
            ("--opd-type", "sglang")
            if teacher is None
            else ("--opd-type", "megatron", "--opd-teacher-load", str(teacher))
        )
        return (
            "--advantage-estimator", "grpo",
            "--use-opd",
            *source,
            "--opd-kl-coef", repr(opd.opd_kl_coef),
            "--entropy-coef", "0.0",
            "--eps-clip", repr(opd.eps_clip),
            "--eps-clip-high", repr(opd.eps_clip_high),
        )  # fmt: skip
    raise ContractError(f"the Slime trainer has no arguments for {algorithm}")


def train_args(
    *,
    algorithm: str,
    base: Path,
    rollout_template: str,
    save_hf: str,
    save: str,
    prompts: int,
    samples_per_prompt: int,
    steps: int,
    gpus: int,
    model_args: Sequence[str],
    settings: SlimeSettings,
    teacher: Path | None = None,
) -> tuple[str, ...]:
    """Arguments to ``train.py`` for ``steps`` optimizer steps of ``algorithm`` over one rollout.

    The rollout holds ``prompts`` groups of ``samples_per_prompt``. For ``opd``, ``teacher`` is the checkpoint
    loaded beside the student, or ``None`` when the samples carry the teacher's log-probabilities. Slime saves
    only at a save interval, and writes its Megatron checkpoint (``save``, without optimizer state) before the
    Hugging Face copy (``save_hf``) the campaign reads.
    """
    job: JobSettings = getattr(settings, algorithm, JobSettings())
    samples = prompts * samples_per_prompt
    if samples % steps:
        raise ContractError(f"{samples} samples do not split into {steps} equal optimizer steps")
    return (
        "--actor-num-nodes", "1",
        "--actor-num-gpus-per-node", str(gpus),
        "--colocate",
        "--calculate-per-token-loss",
        *model_args,
        "--hf-checkpoint", str(base),
        "--load", str(base),
        "--save", save,
        "--save-interval", "1",
        "--no-save-optim",
        "--save-hf", save_hf,
        "--load-debug-rollout-data", rollout_template,
        "--debug-train-only",
        "--num-rollout", "1",
        "--rollout-batch-size", str(prompts),
        "--n-samples-per-prompt", str(samples_per_prompt),
        "--global-batch-size", str(samples // steps),
        *_algorithm_args(algorithm, settings, base, teacher),
        "--optimizer", "adam",
        "--lr", repr(job.lr),
        "--lr-decay-style", "constant",
        "--weight-decay", "0.1",
        "--adam-beta1", "0.9",
        "--adam-beta2", "0.98",
        "--tensor-model-parallel-size", "1",
        "--pipeline-model-parallel-size", "1",
        "--use-dynamic-batch-size",
        "--max-tokens-per-gpu", str(job.max_tokens_per_gpu),
        "--attention-dropout", "0.0",
        "--hidden-dropout", "0.0",
        "--accumulate-allreduce-grads-in-fp32",
        "--attention-softmax-in-fp32",
        "--attention-backend", "flash",
        "--seed", str(job.seed),
        *job.extra_args,
    )  # fmt: skip
