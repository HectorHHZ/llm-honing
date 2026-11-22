"""Verify a downloaded agent checkpoint and write a local NanoJev serving manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = Path(__file__).resolve().parent
PINS = ROOT / "integrations/ra2/pins.json"
CHECKPOINT_PIN = PACKAGE / "checkpoint_pin.json"
DEFAULT_CHECKPOINT = ROOT / "work/rea_issac_agent/checkpoint"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_checkpoint(
    checkpoint_dir: Path = DEFAULT_CHECKPOINT,
    *,
    manifest_out: Path | None = None,
    checkpoint_pin: Path = CHECKPOINT_PIN,
    pins_path: Path = PINS,
) -> Path:
    """Reject wrong weights or base config before making a portable local manifest."""
    checkpoint_dir = checkpoint_dir.resolve(strict=True)
    pin: dict[str, Any] = json.loads(checkpoint_pin.read_text())
    pins = json.loads(pins_path.read_text())
    model = pins["nanojevModels"]["nanojev"]
    source = pins["nanojevSource"]
    if pin.get("format") != "rea-issac-agent-checkpoint-pin-v1":
        raise ValueError("unsupported agent checkpoint pin")
    required = pin.get("files")
    if (
        not isinstance(required, dict)
        or not checkpoint_dir.is_dir()
        or any(not (checkpoint_dir / name).is_file() for name in required)
    ):
        raise ValueError("checkpoint directory lacks required NanoJev files")
    if required.get("best.safetensors") != pin["weights_sha256"]:
        raise ValueError("checkpoint pin has inconsistent weight hashes")
    for name, expected in required.items():
        actual = _sha256(checkpoint_dir / name)
        if actual != expected:
            raise ValueError(f"agent checkpoint file SHA-256 differs: {name}: {actual}")
    checksum = pin["weights_sha256"]
    config = json.loads((checkpoint_dir / "config.json").read_text())
    if (config.get("model"), config.get("resolved_model_revision")) != (model["base"], model["baseRevision"]):
        raise ValueError("agent checkpoint base model differs from the pinned NanoJev release")
    manifest = {
        "kind": "trained-nanojev",
        "repo": "local:rea-issac-agent",
        "revision": checksum,
        "path": str(checkpoint_dir),
        "weightsSha256": checksum,
        "parent": {"repo": model["repo"], "revision": model["revision"]},
        "sourceRevision": source["revision"],
        "base": {"checksum": pin["parent_weights_sha256"]},
        "batchId": f"rea-issac-agent-{pin['training_dataset_sha256'][:16]}",
    }
    target = (manifest_out or checkpoint_dir.parent / "checkpoint.local.json").resolve()
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise FileExistsError(f"existing manifest differs: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest-out", type=Path)
    args = parser.parse_args()
    print(prepare_checkpoint(args.checkpoint_dir, manifest_out=args.manifest_out))


if __name__ == "__main__":
    main()
