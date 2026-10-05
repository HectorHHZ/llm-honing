"""The Slime command line for one training job.

Slime builds its Megatron model from command-line arguments, not from the
checkpoint, and checks them against the Hugging Face config when it starts
(``_hf_validate_args`` in ``slime/backends/megatron_utils/arguments.py``).
:func:`megatron_model_args` derives them from ``config.json`` for the dense
decoder families Slime loads straight from Hugging Face weights (``llama``,
``qwen2``, ``qwen3``), so the arguments always match the checkpoint being
trained. Other families are refused until their arguments are written out.

:func:`grpo_train_args` is the train-only GRPO job: Slime reads the rollout
Honing wrote (``--load-debug-rollout-data``), never starts SGLang, loads the
policy and the KL reference from the base checkpoint, and saves the trained
policy as a Hugging Face directory (``--save-hf``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
class GrpoSettings:
    """The GRPO hyperparameters Honing sets. Anything else goes through ``extra_args``."""

    lr: float = 1e-6
    kl_loss_coef: float = 0.0
    eps_clip: float = 0.2
    eps_clip_high: float = 0.28
    max_tokens_per_gpu: int = 9216
    seed: int = 1234
    extra_args: tuple[str, ...] = ()


def grpo_train_args(
    *,
    base: Path,
    rollout_template: str,
    save_hf: str,
    save: str,
    prompts: int,
    samples_per_prompt: int,
    steps: int,
    gpus: int,
    model_args: Sequence[str],
    settings: GrpoSettings,
) -> tuple[str, ...]:
    """Arguments to ``train.py`` for ``steps`` optimizer steps over one rollout of ``prompts`` groups.

    Slime saves only at a save interval, and writes its Megatron checkpoint (``save``, without optimizer state)
    before the Hugging Face copy (``save_hf``) the campaign reads.

    The KL loss is always computed (``--use-kl-loss``) so its value is reported, and it only changes the
    update when ``kl_loss_coef`` is non-zero.
    """
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
        "--ref-load", str(base),
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
        "--advantage-estimator", "grpo",
        "--use-kl-loss",
        "--kl-loss-coef", repr(settings.kl_loss_coef),
        "--kl-loss-type", "low_var_kl",
        "--entropy-coef", "0.0",
        "--eps-clip", repr(settings.eps_clip),
        "--eps-clip-high", repr(settings.eps_clip_high),
        "--optimizer", "adam",
        "--lr", repr(settings.lr),
        "--lr-decay-style", "constant",
        "--weight-decay", "0.1",
        "--adam-beta1", "0.9",
        "--adam-beta2", "0.98",
        "--tensor-model-parallel-size", "1",
        "--pipeline-model-parallel-size", "1",
        "--use-dynamic-batch-size",
        "--max-tokens-per-gpu", str(settings.max_tokens_per_gpu),
        "--attention-dropout", "0.0",
        "--hidden-dropout", "0.0",
        "--accumulate-allreduce-grads-in-fp32",
        "--attention-softmax-in-fp32",
        "--attention-backend", "flash",
        "--seed", str(settings.seed),
        *settings.extra_args,
    )  # fmt: skip
