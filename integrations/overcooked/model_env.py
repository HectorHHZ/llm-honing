"""Inspect/reuse the existing Linux NanoJev runtime without changing shared loaders."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
sys.path.insert(0, str(REPO))


def configure() -> None:
    if sys.platform != "linux":
        raise RuntimeError("Run this helper in Ubuntu/WSL with the existing NanoJev Python")
    os.environ.setdefault("HF_HOME", "/opt/honing/cache/huggingface")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TMPDIR", "/opt/honing/tmp")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("USE_TF", "0")


def owned_servers(port: int) -> list[int]:
    """Find only this integration's exact launcher and port in Linux procfs."""
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = (proc / "cmdline").read_bytes().decode().strip("\0").split("\0")
            if str(Path(__file__).resolve()) not in argv or "serve" not in argv:
                continue
            index = argv.index("--port")
            if argv[index + 1] == str(port):
                found.append(int(proc.name))
        except (OSError, ValueError, IndexError):
            continue
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inspect", "serve", "status", "stop"))
    parser.add_argument("--source", type=Path, default=Path("/opt/honing/deps/nanojev"))
    parser.add_argument("--port", type=int, default=8803)
    parser.add_argument("--inference", choices=("reference", "tree", "tree-compiled"), default="reference")
    parser.add_argument("--checkpoint-manifest", type=Path)
    args = parser.parse_args()
    configure()
    if args.mode in ("status", "stop"):
        pids = owned_servers(args.port)
        if len(pids) > 1:
            raise RuntimeError("multiple matching integration servers; refusing to stop ambiguously")
        if args.mode == "stop" and pids:
            os.kill(pids[0], signal.SIGTERM)
        print(json.dumps({"port": args.port, "pids": pids, "signal_sent": args.mode == "stop" and bool(pids)}))
        return
    if args.mode == "serve":
        from honing.backends.jev.server import main as serve

        if owned_servers(args.port):
            raise RuntimeError("an Overcooked model server already owns this port")
        argv = [
            "--family",
            "nanojev",
            "--pins",
            str(REPO / "integrations/ra2/pins.json"),
            "--nanojev-source",
            str(args.source),
            "--nanojev-inference",
            args.inference,
            "--gpu",
            "0",
            "--max-len",
            "8192",
            "--port",
            str(args.port),
        ]
        if args.checkpoint_manifest:
            argv.extend(["--checkpoint-manifest", str(args.checkpoint_manifest.resolve())])
        serve(argv)
        return
    import torch

    from honing.backends.jev.nanojev import NanoJevSettings, import_nanojev, nanojev_snapshot, resolve_checkpoint

    pins_path = REPO / "integrations/ra2/pins.json"
    pins = json.loads(pins_path.read_text())
    _module, revision = import_nanojev(args.source, pins)
    settings = NanoJevSettings(pins_path, args.source, max_length=8192)
    checkpoint = resolve_checkpoint(settings, pins, revision)
    snapshot = nanojev_snapshot(Path(checkpoint["path"]))
    report = {
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "source": str(args.source),
        "source_revision": revision,
        "checkpoint": checkpoint,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cached_only": True,
        "architecture": asdict(snapshot.architecture),
        "parameters": snapshot.parameters,
    }
    destination = ROOT / ".cache/model-discovery.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
