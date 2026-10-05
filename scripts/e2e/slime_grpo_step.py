"""One GRPO step through SlimeTrainer on a small Qwen model, with the reward before and after.

The GPU check for the Slime trainer (#68). It runs outside Slime's image and
reaches it the way the trainer does, through a :class:`SlimeEnvironment`:

1. ``slime_rollouts.py`` samples grouped, rewarded answers from the base model
   with SGLang (the training rollout, seed ``--seed``), and from the same model
   on held-out questions (seed ``--seed + 1``), the reward before.
2. :class:`SlimeTrainer` trains on the training rollout and returns a
   ``TrainResult``: new Hugging Face weights, a manifest, and Slime's metrics
   (losses and KL).
3. The trained checkpoint answers the held-out questions again, the reward after.

    python scripts/e2e/slime_grpo_step.py --base /tmp/models/Qwen2.5-1.5B-Instruct \\
        --work-dir /tmp/honing-slime --enroot slime --gpus 0

``--enroot NAME`` enters the enroot container NAME (``envs/slime/import_image.sh``
creates one); without it the commands run in the current environment, which
must then be Slime's image.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from honing.backends.slime import (  # noqa: E402
    GrpoSettings,
    SlimeEnvironment,
    SlimeTrainer,
    SlimeTrainerSettings,
    hf_snapshot,
)
from honing.backends.slime.trainer import gpu_locks  # noqa: E402
from honing.core import DataContract, PolicySample, TrainingBatch, TrainJobSpec  # noqa: E402


def environment(args: argparse.Namespace) -> SlimeEnvironment:
    prefix: tuple[str, ...] = ()
    paths = [str(ROOT), str(args.work_dir), str(args.base.parent), *args.mount]
    if args.slime_source:
        paths.append(str(args.slime_source))
    if args.enroot:
        mounts = [f"{path}:{path}" for path in paths]
        prefix = ("enroot", "start", "--rw", *(item for mount in mounts for item in ("--mount", mount)), args.enroot)
    if args.slime_source is None:
        return SlimeEnvironment(honing_root=ROOT, prefix=prefix)
    # The pinned Slime checkout goes ahead of the image's own install on the import path.
    return SlimeEnvironment(
        honing_root=ROOT,
        prefix=prefix,
        slime_root=str(args.slime_source),
        python_path=(str(args.slime_source), "/root/Megatron-LM"),
    )


def sample(env: SlimeEnvironment, model: Path, out: Path, seed: int, args: argparse.Namespace, devices: str) -> dict:
    argv = [
        env.python, str(ROOT / "scripts" / "e2e" / "slime_rollouts.py"),
        "--model", str(model), "--out", str(out), "--seed", str(seed),
        "--prompts", str(args.prompts), "--samples-per-prompt", str(args.samples_per_prompt),
    ]  # fmt: skip
    log = out.with_suffix(".log")
    with log.open("w") as handle:
        completed = subprocess.run(
            env.command(argv, {"CUDA_VISIBLE_DEVICES": devices}), stdout=handle, stderr=subprocess.STDOUT, check=False
        )
    if completed.returncode != 0:
        raise SystemExit(f"sampling failed; see {log}")
    return json.loads(out.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True, help="Hugging Face checkpoint directory")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--enroot", default=None, help="enroot container with Slime's image")
    parser.add_argument("--slime-source", type=Path, default=None, help="Slime checkout at the pinned commit")
    parser.add_argument("--mount", action="append", default=[], help="extra path to mount at the same place")
    parser.add_argument("--gpus", default="0", help="comma-separated GPU indices")
    parser.add_argument("--prompts", type=int, default=16)
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    run = args.work_dir / time.strftime("grpo-step-%Y%m%d-%H%M%S")
    run.mkdir()
    env = environment(args)
    gpus = tuple(args.gpus.split(","))
    base = hf_snapshot(args.base)

    devices = ",".join(gpus)
    with gpu_locks(gpus):
        train_rollout = sample(env, args.base, run / "rollout-train.json", args.seed, args, devices)
        before = sample(env, args.base, run / "eval-before.json", args.seed + 1, args, devices)
    samples = tuple(
        PolicySample(
            **{key: tuple(row[key]) if key in ("tokens", "loss_mask", "logprobs") else row[key] for key in row}
        )
        for row in train_rollout["samples"]
    )
    batch = TrainingBatch(
        batch_id=f"{run.name}-train",
        contract=DataContract("grouped_on_policy_samples", requires_reward=True, requires_logprobs=True),
        samples=samples,
    )
    trainer = SlimeTrainer(
        SlimeTrainerSettings(environment=env, work_dir=run / "train", gpus=gpus, grpo=GrpoSettings(lr=args.lr))
    )
    result = trainer.train(TrainJobSpec(algorithm="grpo", rollout="direct", steps=args.steps), base, batch)
    trained = Path(result.snapshot.checkpoint.location)
    with gpu_locks(gpus):
        after = sample(env, trained, run / "eval-after.json", args.seed + 1, args, devices)

    report = {
        "base": {"location": str(args.base), "checksum": base.checkpoint.checksum},
        "trained": {"location": str(trained), "checksum": result.snapshot.checkpoint.checksum},
        "train_rollout_reward": train_rollout["reward_mean"],
        "heldout_reward_before": before["reward_mean"],
        "heldout_reward_after": after["reward_mean"],
        "metrics": dict(result.metrics),
    }
    (run / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        raise SystemExit(main())
