"""Train a language model with Slime behind :class:`~honing.core.Trainer`.

Honing runs the loop and Slime runs one update. A rollout provider collects the
samples and the campaign checks them; :class:`SlimeTrainer` writes them in
Slime's rollout layout (:mod:`honing.backends.slime.rollout_data`) and runs
Slime train-only on them (``--load-debug-rollout-data --debug-train-only``), so
Slime never generates, and never publishes, weights the gate has not judged.

Slime runs in its own image with its own torch, CUDA build, and Megatron. Each
job is a subprocess in that environment (:class:`SlimeEnvironment`), and only
files cross: the rollout, the base checkpoint path, the trained Hugging Face
checkpoint, and the metrics Slime logged. The trained checkpoint lands in a
fresh run directory with a ``checkpoint.json`` manifest naming its parent, the
job, the batch, and the Slime revision, and its snapshot checksum is computed
here over the weights Slime wrote.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from honing.backends.slime.arguments import GrpoSettings, grpo_train_args, megatron_model_args, read_config
from honing.backends.slime.rollout_data import slime_samples, write_samples
from honing.campaign import rules
from honing.core import (
    LANGUAGE_MODEL_CONTRACTS,
    ArchitectureSpec,
    BackendSupport,
    CheckpointRef,
    ContractError,
    HoningError,
    ModelSnapshot,
    TrainingBatch,
    TrainJobSpec,
    TrainResult,
)

SUPPORT = BackendSupport(
    name="slime",
    architectures="uniform",
    train_numerics=frozenset({"bf16"}),
    weight_formats=frozenset({"hf_safetensors"}),
    algorithms=(("grpo", "native"),),
    contracts=LANGUAGE_MODEL_CONTRACTS,
)
SLIME_REVISION = "41014d1f29e201137fdffce737bb8bac65bc5219"


class SlimeRunError(HoningError):
    """A Slime subprocess failed. The message names the log to read."""


@dataclass(frozen=True)
class SlimeEnvironment:
    """How to run a command where Slime is installed.

    ``prefix`` enters the environment, for example
    ``("enroot", "start", "--rw", "--mount", "/project:/project", "slime")`` for the
    image imported with enroot, or ``()`` when Honing already runs inside it.
    ``honing_root`` is this repository's path as seen from inside, put on
    ``PYTHONPATH`` for the worker, ahead of ``python_path``: Megatron-LM's
    checkout, which Slime imports ``megatron.training`` from, and a Slime
    checkout when it should replace the image's own. Paths Honing passes
    (checkpoints, run directories) must be visible at the same path on both sides.
    """

    honing_root: Path
    prefix: tuple[str, ...] = ()
    python: str = "python3"
    slime_root: str = "/root/slime"
    python_path: tuple[str, ...] = ("/root/Megatron-LM",)
    env: Mapping[str, str] = field(default_factory=dict)

    def command(self, argv: Sequence[str], env: Mapping[str, str]) -> list[str]:
        path = ":".join((str(self.honing_root), *self.python_path))
        variables = {"PYTHONPATH": path, "PYTHONUNBUFFERED": "1", **self.env, **env}
        return [*self.prefix, "env", *(f"{key}={value}" for key, value in variables.items()), *argv]


@dataclass(frozen=True)
class SlimeTrainerSettings:
    """Where jobs run and on which GPUs.

    ``gpus`` are indices. With ``lock_gpus`` every one is locked for the whole job and must have no other compute
    process; without it they are passed through as ``CUDA_VISIBLE_DEVICES`` unchecked, for a machine whose GPUs a
    scheduler already isolates.
    """

    environment: SlimeEnvironment
    work_dir: Path
    gpus: tuple[str, ...] = ("0",)
    grpo: GrpoSettings = field(default_factory=GrpoSettings)
    lock_dir: Path | None = None
    lock_gpus: bool = True
    timeout_s: float = 6 * 3600


def _sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def _safetensors_parameters(path: Path) -> int:
    with path.open("rb") as handle:
        header = json.loads(handle.read(int.from_bytes(handle.read(8), "little")))
    total = 0
    for name, tensor in header.items():
        if name != "__metadata__":
            count = 1
            for dimension in tensor["shape"]:
                count *= int(dimension)
            total += count
    return total


def hf_snapshot(checkpoint: Path) -> ModelSnapshot:
    """A Hugging Face checkpoint directory as a ModelSnapshot: one checksum over its config and weight files."""
    config = read_config(checkpoint)
    weights = sorted(checkpoint.glob("*.safetensors"))
    if not weights:
        raise ContractError(f"{checkpoint} has no safetensors weights")
    parts = [f"config.json:{_sha256(checkpoint / 'config.json')}"] + [
        f"{path.name}:{_sha256(path)}" for path in weights
    ]
    text = config.get("text_config", config)
    return ModelSnapshot(
        checkpoint=CheckpointRef(
            location=str(checkpoint),
            weight_format="hf_safetensors",
            checksum=hashlib.sha256("\n".join(parts).encode()).hexdigest(),
        ),
        architecture=ArchitectureSpec.uniform_layers(
            num_layers=int(text["num_hidden_layers"]),
            hidden_size=int(text["hidden_size"]),
            attention_heads=int(text["num_attention_heads"]),
            ffn_size=int(text["intermediate_size"]),
        ),
        parameters=sum(_safetensors_parameters(path) for path in weights),
    )


class HFModelSource:
    """Satisfies ModelSource with a Hugging Face checkpoint directory that is used in place."""

    def __init__(self, checkpoint: Path) -> None:
        self._checkpoint = checkpoint

    def materialize(self, work_dir: Path) -> ModelSnapshot:
        return hf_snapshot(self._checkpoint)


def _gpu_uuids(indices: Sequence[str]) -> list[str]:
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    known = {row.split(",")[0].strip(): row.split(",")[1].strip() for row in rows.splitlines() if row.strip()}
    missing = [index for index in indices if index not in known]
    if missing:
        raise ContractError(f"GPU indices {missing} not found")
    return [known[index] for index in indices]


@contextlib.contextmanager
def gpu_locks(indices: Sequence[str], lock_dir: Path | None = None) -> Iterator[list[str]]:
    """Hold every GPU in ``indices``: an exclusive per-user lock each, and no other compute process on any.

    The lock files are the ones the Jev trainers take, so a Slime job and a Jev job never share a GPU.
    """
    uuids = _gpu_uuids(indices)
    directory = lock_dir or Path(tempfile.gettempdir()) / f"laya-{os.getuid()}"
    directory.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as stack:
        for gpu in uuids:
            handle = stack.enter_context((directory / f"{gpu}.lock").open("a"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        used = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True
        )
        busy = [gpu for gpu in uuids if gpu in used]
        if busy:
            raise ContractError(f"GPUs {busy} already have a compute process")
        yield uuids


class SlimeTrainer:
    """Satisfies Trainer: one Slime train-only job per call, on Honing's rollouts."""

    def __init__(self, settings: SlimeTrainerSettings, support: BackendSupport = SUPPORT) -> None:
        self._settings = settings
        self.support = support

    def train(self, job: TrainJobSpec, base: ModelSnapshot, batch: TrainingBatch | None) -> TrainResult:
        rules.algorithm_requirements_met(job, self.support)
        rules.batch_meets_contract(job, batch)
        if batch is None:  # batch_meets_contract refuses a missing batch; this narrows the type
            raise ContractError(f"{job.algorithm} needs a batch")
        if job.algorithm != "grpo":
            raise ContractError(f"the Slime trainer runs grpo only, not {job.algorithm}")
        if base.checkpoint.weight_format != "hf_safetensors":
            raise ContractError("the Slime trainer starts from a Hugging Face checkpoint")
        settings = self._settings
        base_dir = Path(base.checkpoint.location)
        run = settings.work_dir / f"{job.algorithm}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        run.mkdir(parents=True)

        rows, layout = slime_samples(batch)
        write_samples(run / "rollout" / "0.json", rows)
        args = grpo_train_args(
            base=base_dir,
            rollout_template=str(run / "rollout" / "{rollout_id}.pt"),
            save_hf=str(run / "hf" / "{rollout_id}"),
            save=str(run / "megatron"),
            prompts=layout.prompts,
            samples_per_prompt=layout.samples_per_prompt,
            steps=job.steps,
            gpus=len(settings.gpus),
            model_args=megatron_model_args(read_config(base_dir)),
            settings=settings.grpo,
        )
        (run / "job.json").write_text(
            json.dumps({"job": _jsonable(job), "batch_id": batch.batch_id, "train_args": list(args)}, indent=2) + "\n"
        )
        locks = gpu_locks(settings.gpus, settings.lock_dir) if settings.lock_gpus else contextlib.nullcontext()
        with locks:
            # Indices, not UUIDs: Slime's Ray placement parses CUDA_VISIBLE_DEVICES as integers.
            visible = {"CUDA_VISIBLE_DEVICES": ",".join(settings.gpus)}
            worker = [settings.environment.python, "-m", "honing.backends.slime.worker"]
            self._run(
                [*worker, "save-rollout", str(run / "rollout" / "0.json"), str(run / "rollout" / "0.pt")],
                run,
                "rollout",
                visible,
            )
            train = [
                settings.environment.python,
                f"{settings.environment.slime_root}/train.py",
                "--use-tensorboard",
                *args,
            ]
            self._run(train, run, "train", {**visible, "TENSORBOARD_DIR": str(run / "tensorboard")})
            self._run(
                [*worker, "metrics", str(run / "tensorboard"), str(run / "metrics.json")], run, "metrics", visible
            )

        trained = run / "hf" / "0"
        snapshot = hf_snapshot(trained)
        if snapshot.checkpoint.checksum == base.checkpoint.checksum:
            raise SlimeRunError(f"Slime returned the base weights unchanged; see {run / 'train.log'}")
        metrics = _train_metrics(json.loads((run / "metrics.json").read_text()), rows)
        manifest = {
            "parent": {"location": base.checkpoint.location, "checksum": base.checkpoint.checksum},
            "checksum": snapshot.checkpoint.checksum,
            "job": _jsonable(job),
            "batch_id": batch.batch_id,
            "samples": len(rows),
            "slime_revision": SLIME_REVISION,
            "metrics": metrics,
        }
        (run / "checkpoint.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return TrainResult(snapshot=snapshot, metrics=metrics)

    def _run(self, argv: Sequence[str], run: Path, name: str, env: Mapping[str, str]) -> None:
        log = run / f"{name}.log"
        command = self._settings.environment.command(argv, env)
        with log.open("w") as handle:
            handle.write(" ".join(command) + "\n")
            handle.flush()
            completed = subprocess.run(
                command, stdout=handle, stderr=subprocess.STDOUT, timeout=self._settings.timeout_s, check=False
            )
        if completed.returncode != 0:
            raise SlimeRunError(f"Slime step {name!r} exited with {completed.returncode}; see {log}")


def _jsonable(job: TrainJobSpec) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(job), default=str))


def _train_metrics(logged: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """The rollout's reward and Slime's training scalars, as finite floats."""
    rewards = [float(row["reward"]) for row in rows if row.get("reward") is not None]
    metrics: dict[str, float] = {"train_steps": float(logged.get("steps", 0))}
    if rewards:
        metrics["rollout_reward_mean"] = sum(rewards) / len(rewards)
    for tag, value in logged.get("scalars", {}).items():
        if tag.startswith("train/") and "@" not in tag and isinstance(value, (int, float)) and math.isfinite(value):
            metrics[f"slime_{tag.removeprefix('train/')}"] = float(value)
    return metrics
