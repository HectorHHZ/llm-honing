"""One bounded heads-only SFT candidate using Honing's existing NanoJevTrainer."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import stat
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent.parent))


def writable_checkpoint(checkpoint: Path, work_dir: Path) -> None:
    """Clear copied cache read-only bits only inside this run's checkpoint."""
    destination, work = checkpoint.resolve(), work_dir.resolve()
    destination.relative_to(work)
    if destination == work:
        raise ValueError("checkpoint must be a child of the training work directory")
    if not destination.exists():
        return
    for child in [destination, *destination.rglob("*")]:
        child.resolve().relative_to(work)
        child.chmod(child.stat().st_mode | stat.S_IWUSR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from integrations.overcooked.model_env import configure

    configure()
    from honing.backends.jev.nanojev import nanojev_snapshot
    from honing.backends.jev.nanojev_trainer import NanoJevTrainer, NanoJevTrainerSettings
    from honing.core import NumericsPolicy, TrainJobSpec, UpdatePolicy
    from integrations.overcooked.data import training_batch

    dataset = json.loads(args.dataset.read_text())
    if dataset["kind"] != "overcooked-train-only-sft-v1":
        raise ValueError("unrecognized audited dataset")
    discovery = json.loads((ROOT / ".cache/model-discovery.json").read_text())
    checkpoint = Path(discovery["checkpoint"]["path"])
    base = nanojev_snapshot(checkpoint)
    if base.checkpoint.checksum != discovery["checkpoint"]["weightsSha256"]:
        raise ValueError("base checkpoint changed")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    events = (out / "training-events.jsonl").open("x")
    selected: dict[str, Any] = {}

    class LoggedTrainer(NanoJevTrainer):
        def _save(self, checkpoint: Path, parent: Path, loaded: Any) -> None:
            # copytree preserves HF cache modes; NTFS otherwise rejects the next save.
            writable_checkpoint(checkpoint, self._settings.work_dir)
            super()._save(checkpoint, parent, loaded)

        def _trainable(self, job: Any, model: Any) -> None:
            super()._trainable(job, model)
            selected.update(
                {name: parameter.numel() for name, parameter in model.named_parameters() if parameter.requires_grad}
            )

        def _forward(self, loaded: Any) -> Any:
            forward = super()._forward(loaded)
            counter = 0

            def logged(chunk: Any) -> Any:
                nonlocal counter
                logits, labels, extra = forward(chunk)
                with loaded.torch.no_grad():
                    loss = loaded.torch.nn.functional.cross_entropy(logits, labels).item()
                events.write(
                    json.dumps(
                        {
                            "forward": counter,
                            "elapsed_seconds": time.perf_counter() - started,
                            "phase": "train" if loaded.predictor.model.training else "internal_validation",
                            "questions": len(chunk),
                            "cross_entropy": loss,
                        }
                    )
                    + "\n"
                )
                events.flush()
                counter += 1
                return logits, labels, extra

            return logged

    settings = NanoJevTrainerSettings(
        pins_path=ROOT.parent / "ra2/pins.json",
        source_dir=Path(discovery["source"]),
        work_dir=out,
        batch_questions=1,
        seed=7,
        backbone_lr=1e-5,
        head_lr=1e-4,
        max_length=2048,
        validation_groups=r"^internal-val:",
        min_validation=8,
    )
    job = TrainJobSpec(
        "sft",
        update_policy=UpdatePolicy("param_subset", trainable_patterns=(r"^(norm|scalar|set_)",)),
        numerics=NumericsPolicy(mode="bf16"),
        steps=2,
        contract="typed_decisions",
    )
    try:
        result = LoggedTrainer(settings).train(job, base, training_batch(dataset["rows"], "overcooked-pilot-sft-v1"))
        manifests = list(out.glob("train-*/checkpoint.json"))
        if len(manifests) != 1:
            raise ValueError("training did not produce exactly one candidate manifest")
        receipt = {
            "method": "Honing NanoJevTrainer SFT, heads only",
            "base": asdict(base),
            "snapshot": asdict(result.snapshot),
            "metrics": dict(result.metrics),
            "checkpoint_manifest": str(manifests[0]),
            "trainable_parameters": selected,
            "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
            "epochs": 2,
            "seed": 7,
            "batch_questions": 1,
            "head_lr": 1e-4,
            "elapsed_seconds": time.perf_counter() - started,
            "logging": {
                "local": "training-events.jsonl records every forward loss and phase",
                "wandb_installed": importlib.util.find_spec("wandb") is not None,
                "wandb_cloud": False,
                "reason": "No W&B run configured; local losses and epoch metrics retained.",
            },
        }
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt), flush=True)
    finally:
        events.close()


if __name__ == "__main__":
    main()
