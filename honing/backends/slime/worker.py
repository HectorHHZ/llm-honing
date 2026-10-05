"""The file conversions that need Slime's environment, run there as ``python -m honing.backends.slime.worker``.

Honing's process never imports torch. This module runs inside Slime's image,
with only the Honing source on ``PYTHONPATH``, and does the two conversions
that need packages from that image:

``save-rollout IN.json OUT.pt``
    Save a rollout written by :func:`~honing.backends.slime.rollout_data.write_samples`
    in the ``torch.save`` layout ``--load-debug-rollout-data`` reads.
``load-rollout IN.pt OUT.json``
    The reverse, which checks the round trip through Slime's own ``Sample``.
``metrics TB_DIR OUT.json``
    Read the scalars Slime logged to TensorBoard (``train/*``, ``rollout/*``)
    and write the last value of each, plus the number of steps logged.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any


def save_rollout(source: Path, target: Path) -> None:
    torch = importlib.import_module("torch")
    payload = json.loads(source.read_text(encoding="utf-8"))
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"rollout_id": payload["rollout_id"], "samples": payload["samples"]}, target)


def load_rollout(source: Path, target: Path) -> None:
    torch = importlib.import_module("torch")
    sample_type = importlib.import_module("slime.utils.types").Sample
    payload = torch.load(source, weights_only=False)
    samples = [sample_type.from_dict(row) for row in payload["samples"]]
    rows = [
        {
            "index": sample.index,
            "group_index": sample.group_index,
            "tokens": list(sample.tokens),
            "response_length": sample.response_length,
            "loss_mask": None if sample.loss_mask is None else list(sample.loss_mask),
            "reward": sample.reward,
            "rollout_log_probs": None if sample.rollout_log_probs is None else list(sample.rollout_log_probs),
            "metadata": sample.metadata,
        }
        for sample in samples
    ]
    target.write_text(json.dumps({"rollout_id": payload["rollout_id"], "samples": rows}) + "\n", encoding="utf-8")


def read_metrics(tensorboard_dir: Path, target: Path) -> None:
    accumulator_type = importlib.import_module("tensorboard.backend.event_processing.event_accumulator")
    metrics: dict[str, Any] = {}
    steps = 0
    for run in sorted({path.parent for path in tensorboard_dir.rglob("events.out.tfevents.*")}):
        accumulator = accumulator_type.EventAccumulator(str(run), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags()["scalars"]:
            events = accumulator.Scalars(tag)
            metrics[tag] = events[-1].value
            metrics[f"{tag}@first"] = events[0].value
            if tag.startswith("train/"):
                steps = max(steps, len(events))
    target.write_text(json.dumps({"steps": steps, "scalars": metrics}, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="honing.backends.slime.worker")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("save-rollout", "load-rollout", "metrics"):
        command = commands.add_parser(name)
        command.add_argument("source", type=Path)
        command.add_argument("target", type=Path)
    args = parser.parse_args(argv)
    if args.command == "save-rollout":
        save_rollout(args.source, args.target)
    elif args.command == "load-rollout":
        load_rollout(args.source, args.target)
    else:
        read_metrics(args.source, args.target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
