"""Experiment-only pinned base-Laya preparation and local Honing HTTP serving."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
sys.path.insert(0, str(REPO))


def checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def configure() -> None:
    os.environ.update(
        HF_HOME=str(ROOT / ".cache/laya-hf"),
        TMPDIR=str(ROOT / ".cache/laya-tmp"),
        USE_TF="0",
        TOKENIZERS_PARALLELISM="false",
        PYTHONUNBUFFERED="1",
    )
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "serve", "inspect"))
    parser.add_argument("--port", type=int, default=8805)
    parser.add_argument("--max-len", type=int, default=2048)
    args = parser.parse_args()
    if sys.platform != "linux":
        raise RuntimeError("Run in WSL with the existing CUDA Python environment")
    configure()
    pins_path = REPO / "integrations/ra2/pins.json"
    pins = json.loads(pins_path.read_text())
    pin = pins["models"]["english"]
    source = ROOT / ".cache/laya-source"
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != pins["layaSource"]["revision"]:
        raise ValueError("Laya source does not match repository pins")
    records = ROOT / ".cache/laya-models"
    record_file = records / "english.json"
    if args.mode == "prepare":
        from huggingface_hub import HfApi, snapshot_download

        os.environ["HF_HUB_OFFLINE"] = "0"
        info = HfApi().model_info(pin["repo"], revision=pin["revision"], files_metadata=True, token=False)
        if info.sha != pin["revision"]:
            raise ValueError("Model revision differs from pins")
        expected = next(f.lfs.sha256 for f in info.siblings if f.rfilename == "model.safetensors")
        path = Path(
            snapshot_download(
                pin["repo"],
                revision=pin["revision"],
                token=False,
                allow_patterns=["model.safetensors", "rl_agent_config.json", "encoder/*", "tokenizer/*"],
            )
        )
        actual = checksum(path / "model.safetensors")
        if actual != expected:
            raise ValueError("Downloaded weights do not match the published LFS SHA-256")
        records.mkdir(exist_ok=True)
        record = {**pin, "path": str(path), "weightsSha256": actual, "sourceCommit": revision}
        record_file.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    record = json.loads(record_file.read_text())
    if record["repo"] != pin["repo"] or record["revision"] != pin["revision"]:
        raise ValueError("The prepared model is not the pinned English base checkpoint")
    if checksum(Path(record["path"]) / "model.safetensors") != record["weightsSha256"]:
        raise ValueError("Prepared Laya weight hash mismatch")
    if args.mode == "inspect":
        from honing.backends.jev.laya import laya_snapshot

        snapshot = laya_snapshot(Path(record["path"]))
        report = {
            **record,
            "snapshot": asdict(snapshot),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "huggingface_hub", "numpy", "safetensors")
            },
            "wandb_sdk_available": importlib.util.find_spec("wandb") is not None,
            "wandb_key_configured": bool(os.environ.get("WANDB_API_KEY")),
        }
        (ROOT / ".cache/laya-model-discovery.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
        return
    from honing.backends.jev.server import main as serve

    serve(
        [
            "--family",
            "laya",
            "--pins",
            str(pins_path),
            "--model",
            "english",
            "--model-config-dir",
            str(records),
            "--laya-source",
            str(source),
            "--gpu",
            "0",
            "--max-len",
            str(args.max_len),
            "--port",
            str(args.port),
        ]
    )


if __name__ == "__main__":
    main()
