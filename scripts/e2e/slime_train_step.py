"""One SFT, OPD, OPSD, or GRPO job through SlimeTrainer on a small model, with the reward before and after.

The GPU check for the Slime trainer (#68). It runs outside Slime's image and
reaches it the way the trainer does, through a :class:`SlimeEnvironment`:

1. ``slime_rollouts.py`` samples grouped, rewarded answers from the base model
   with SGLang (seed ``--seed``), and from the same model on held-out questions
   (seed ``--seed + 1``), the reward before.
2. The samples become the batch the algorithm trains on, as a campaign would
   hand it over:

   ``grpo``  the grouped, rewarded rollout (``grouped_on_policy_samples``)
   ``sft``   the correct answers only, as demonstrations (``token_sequences``)
   ``opd``   the rollout without rewards (``on_policy_samples``), with ``--teacher``
             loaded beside the student
   ``opsd``  the same, with the base checkpoint itself as the teacher (a
             ``lineage`` teacher named ``parent``)

3. :class:`SlimeTrainer` trains and returns a ``TrainResult``: new Hugging Face
   weights, a manifest, and Slime's metrics. The trained checkpoint answers the
   held-out questions again (skipped with ``--no-eval``), the reward after.

    python scripts/e2e/slime_train_step.py --algorithm sft --base /tmp/models/Qwen2.5-1.5B-Instruct \\
        --work-dir /tmp/honing-slime --enroot slime --slime-source /path/to/slime --gpus 0

``--enroot NAME`` enters the enroot container NAME (``setup-slime-enroot.sh``
creates one); without it the commands run in the current environment, which
must then be Slime's image. With ``opsd`` the teacher equals the student, so the
distillation signal starts at zero: the run checks the path, not a gain.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from honing.backends.slime import (  # noqa: E402
    GrpoSettings,
    OpdSettings,
    SftSettings,
    SlimeEnvironment,
    SlimeSettings,
    SlimeTrainer,
    SlimeTrainerSettings,
    hf_snapshot,
)
from honing.backends.slime.trainer import gpu_locks  # noqa: E402
from honing.core import DataContract, PolicySample, TeacherRef, TrainingBatch, TrainJobSpec  # noqa: E402

ALGORITHMS = ("grpo", "sft", "opd", "opsd")
DEFAULT_LR = {"grpo": 1e-6, "sft": 1e-5, "opd": 1e-6, "opsd": 1e-6}


def environment(args: argparse.Namespace) -> SlimeEnvironment:
    prefix: tuple[str, ...] = ()
    paths = [str(ROOT), str(args.work_dir), str(args.base.parent), *args.mount]
    if args.teacher is not None:
        paths.append(str(args.teacher.parent))
    if args.slime_source:
        paths.append(str(args.slime_source))
    if args.enroot:
        mounts = [f"{path}:{path}" for path in dict.fromkeys(paths)]
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


def sample(env: SlimeEnvironment, model: Path, out: Path, seed: int, args: argparse.Namespace) -> dict[str, Any]:
    argv = [
        env.python, str(ROOT / "scripts" / "e2e" / "slime_rollouts.py"),
        "--model", str(model), "--out", str(out), "--seed", str(seed),
        "--prompts", str(args.prompts), "--samples-per-prompt", str(args.samples_per_prompt),
    ]  # fmt: skip
    log = out.with_suffix(".log")
    with log.open("w") as handle:
        completed = subprocess.run(
            env.command(argv, {"CUDA_VISIBLE_DEVICES": args.gpus}),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise SystemExit(f"sampling failed; see {log}")
    result: dict[str, Any] = json.loads(out.read_text())
    return result


def batch_for(algorithm: str, rows: list[dict[str, Any]], batch_id: str, steps: int) -> TrainingBatch:
    """The batch a campaign would hand the algorithm, built from one sampled rollout."""
    if algorithm == "grpo":
        samples = tuple(
            PolicySample(
                tokens=tuple(row["tokens"]),
                loss_mask=tuple(row["loss_mask"]),
                reward=row["reward"],
                logprobs=tuple(row["logprobs"]),
                group=row["group"],
            )
            for row in rows
        )
        contract = DataContract("grouped_on_policy_samples", requires_reward=True, requires_logprobs=True)
        return TrainingBatch(batch_id=batch_id, contract=contract, samples=samples)
    if algorithm == "sft":
        correct = [row for row in rows if row["reward"] == 1.0]
        correct = correct[: len(correct) - len(correct) % steps]
        if not correct:
            raise SystemExit("the base model answered nothing correctly; no demonstrations for SFT")
        samples = tuple(
            PolicySample(tokens=tuple(row["tokens"]), loss_mask=tuple(row["loss_mask"])) for row in correct
        )
        return TrainingBatch(batch_id=batch_id, contract=DataContract("token_sequences"), samples=samples)
    samples = tuple(
        PolicySample(tokens=tuple(row["tokens"]), loss_mask=tuple(row["loss_mask"]), logprobs=tuple(row["logprobs"]))
        for row in rows
    )
    contract = DataContract("on_policy_samples", requires_logprobs=True)
    return TrainingBatch(batch_id=batch_id, contract=contract, samples=samples)


def job_for(algorithm: str, steps: int) -> TrainJobSpec:
    if algorithm == "grpo":
        return TrainJobSpec(algorithm="grpo", rollout="direct", steps=steps)
    if algorithm == "sft":
        return TrainJobSpec(algorithm="sft", steps=steps)
    teacher = (
        TeacherRef(kind="release", ref="teacher", placement="in_process")
        if algorithm == "opd"
        else TeacherRef(kind="lineage", ref="parent", placement="in_process")
    )
    return TrainJobSpec(algorithm="opd", rollout="direct", teacher=teacher, steps=steps)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algorithm", choices=ALGORITHMS, default="grpo")
    parser.add_argument("--base", type=Path, required=True, help="Hugging Face checkpoint directory")
    parser.add_argument("--teacher", type=Path, default=None, help="teacher checkpoint for --algorithm opd")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--enroot", default=None, help="enroot container with Slime's image")
    parser.add_argument("--slime-source", type=Path, default=None, help="Slime checkout at the pinned commit")
    parser.add_argument("--mount", action="append", default=[], help="extra path to mount at the same place")
    parser.add_argument("--gpus", default="0", help="comma-separated GPU indices")
    parser.add_argument("--prompts", type=int, default=16)
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=None, help="default: 1e-5 for sft, 1e-6 otherwise")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-eval", action="store_true", help="skip the held-out reward before and after")
    args = parser.parse_args()
    if args.algorithm == "opd" and args.teacher is None:
        parser.error("--algorithm opd needs --teacher")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    run = args.work_dir / time.strftime(f"{args.algorithm}-step-%Y%m%d-%H%M%S")
    run.mkdir()
    env = environment(args)
    gpus = tuple(args.gpus.split(","))
    base = hf_snapshot(args.base)
    lr = DEFAULT_LR[args.algorithm] if args.lr is None else args.lr

    with gpu_locks(gpus):
        rollout = sample(env, args.base, run / "rollout-train.json", args.seed, args)
        before = None if args.no_eval else sample(env, args.base, run / "eval-before.json", args.seed + 1, args)
    batch = batch_for(args.algorithm, rollout["samples"], f"{run.name}-train", args.steps)
    trainer = SlimeTrainer(
        SlimeTrainerSettings(
            environment=env,
            work_dir=run / "train",
            gpus=gpus,
            algorithms=SlimeSettings(grpo=GrpoSettings(lr=lr), sft=SftSettings(lr=lr), opd=OpdSettings(lr=lr)),
            teachers={} if args.teacher is None else {"teacher": args.teacher},
        )
    )
    result = trainer.train(job_for(args.algorithm, args.steps), base, batch)
    trained = Path(result.snapshot.checkpoint.location)
    after = None
    if not args.no_eval:
        with gpu_locks(gpus):
            after = sample(env, trained, run / "eval-after.json", args.seed + 1, args)

    report = {
        "algorithm": args.algorithm,
        "base": {"location": str(args.base), "checksum": base.checkpoint.checksum},
        "teacher": str(args.base if args.algorithm == "opsd" else args.teacher) if "opd" in args.algorithm else None,
        "trained": {"location": str(trained), "checksum": result.snapshot.checkpoint.checksum},
        "batch": {"kind": batch.contract.kind, "samples": len(batch.samples)},
        "train_rollout_reward": rollout["reward_mean"],
        "heldout_reward_before": None if before is None else before["reward_mean"],
        "heldout_reward_after": None if after is None else after["reward_mean"],
        "metrics": dict(result.metrics),
    }
    (run / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        raise SystemExit(main())
