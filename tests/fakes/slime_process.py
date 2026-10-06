"""A stand-in for the Python interpreter in Slime's image, for testing SlimeTrainer on a CPU.

SlimeTrainer runs ``<python> -m honing.backends.slime.worker <command> ...`` and
``<python> <slime_root>/train.py <args>``. This script answers both the way the
image would, without torch: the rollout is copied as JSON, "training" copies the
``--load`` checkpoint to ``--save-hf`` with every weight byte changed, and the
metrics are a fixed set of Slime's training scalars. ``FAKE_SLIME_FAIL=train``
makes training exit non-zero; ``FAKE_SLIME_UNCHANGED=1`` saves the base weights unchanged.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

SCALARS = {"train/loss": 0.12, "train/pg_loss": 0.1, "train/kl_loss": 0.003, "train/ppo_kl": 0.001}


def _option(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def worker(argv: list[str]) -> int:
    command, source, target = argv
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if command in ("save-rollout", "load-rollout"):
        shutil.copyfile(source, target_path)
    elif command == "metrics":
        target_path.write_text(json.dumps({"steps": 1, "scalars": SCALARS}))
    else:
        return 2
    return 0


def train(argv: list[str]) -> int:
    if os.environ.get("FAKE_SLIME_FAIL") == "train":
        print("fake Slime failed", file=sys.stderr)
        return 1
    rollout = Path(_option(argv, "--load-debug-rollout-data").format(rollout_id=0))
    if not rollout.is_file():
        print(f"no rollout at {rollout}", file=sys.stderr)
        return 1
    if "--use-opd" in argv and not (Path(_option(argv, "--opd-teacher-load")) / "config.json").is_file():
        print("opd teacher is not a checkpoint", file=sys.stderr)
        return 1
    base = Path(_option(argv, "--load"))
    target = Path(_option(argv, "--save-hf").format(rollout_id=0))
    shutil.copytree(base, target)
    if os.environ.get("FAKE_SLIME_UNCHANGED") != "1":
        for weights in target.glob("*.safetensors"):
            data = bytearray(weights.read_bytes())
            length = int.from_bytes(data[:8], "little")
            for index in range(8 + length, len(data)):
                data[index] = (data[index] + 1) % 256
            weights.write_bytes(bytes(data))
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if argv[:2] == ["-m", "honing.backends.slime.worker"]:
        return worker(argv[2:])
    if argv and argv[0].endswith("train.py"):
        return train(argv[1:])
    print(f"unexpected command: {argv}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
