"""Serve pinned Honing decision models for tank duels on an explicitly shared GPU.

Uses Honing's request packing, probability output and native model implementations.
Sharing is opt-in, respects the same GPU lock as the standard Honing loaders, checks
free memory before loading, and limits this process's PyTorch allocator. No process
is evicted. Launch one process per endpoint, never share model state across threads.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--model", choices=["english", "typed-decisions", "kev-0.8b", "kev-4b"], required=True)
    p.add_argument("--gpu", required=True)
    p.add_argument("--memory-gib", type=float, required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--allow-shared-gpu", action="store_true", required=True)
    args = p.parse_args()
    if args.memory_gib <= 0:
        p.error("memory-gib must be positive")
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.total", "--format=csv,noheader,nounits"], text=True
    )
    gpu = next(row.split(",") for row in rows.splitlines() if row.split(",")[0].strip() == args.gpu)
    uuid, free_mb, total_mb = gpu[1].strip(), float(gpu[2]), float(gpu[3])
    if free_mb < args.memory_gib * 1024 + 1024:
        raise RuntimeError("Not enough free GPU memory for the requested cap plus 1 GiB margin")
    lock_dir = Path(tempfile.gettempdir()) / f"laya-{os.getuid()}"
    lock_dir.mkdir(exist_ok=True)
    with (lock_dir / f"{uuid}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        os.environ.update(CUDA_VISIBLE_DEVICES=uuid, USE_TF="0", TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1")
        sys.path.insert(0, str(args.root))
        import torch

        torch.set_num_threads(2)
        torch.manual_seed(1)
        torch.cuda.set_per_process_memory_fraction(args.memory_gib * 1024 / total_mb)
        from honing.backends.jev.server import DecisionServer

        pins_path = args.root / "integrations/ra2/pins.json"
        pins = json.loads(pins_path.read_text())
        family = "kev" if args.model.startswith("kev") else "laya"
        source = args.root / "vendor" / family
        revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if revision != pins[family + "Source"]["revision"]:
            raise ValueError("Source differs from pins.json")
        sys.path.insert(0, str(source))
        meta = {
            "family": family,
            "model": args.model,
            "sourceCommit": revision,
            "gpu": args.gpu,
            "gpuUuid": uuid,
            "sharedGpu": True,
            "gpuMemoryLimitGiB": args.memory_gib,
            "torch": torch.__version__,
        }
        if family == "kev":
            from kev.checkpoint import Checkpoint, LoadOptions

            from honing.backends.jev.kev import KevDecisionModel, KevSettings, resolve_checkpoint

            settings = KevSettings(pins_path=pins_path, source_dir=source, model=args.model, gpu=args.gpu)
            record = resolve_checkpoint(settings, pins, revision)
            with contextlib.redirect_stdout(sys.stderr):
                tok, native = Checkpoint(record["path"]).load(
                    "cuda", LoadOptions(dtype=torch.bfloat16, backend="torch", cuda_graphs=False)
                )
            meta.update(record, maxState=8192, maxBranch=8192, dtype=native.dtype)
            model = KevDecisionModel(tok, native, meta, 8192, 8192)
        else:
            import laya

            from honing.backends.jev.laya import LayaDecisionModel
            from honing.backends.jev.token_budget import prepare_request

            record = json.loads((args.root / "models" / f"{args.model}.json").read_text())
            if any(record[k] != pins["models"][args.model][k] for k in ["repo", "revision"]):
                raise ValueError("Model record differs from pins.json")
            weights = Path(record["path"]) / "model.safetensors"
            digest = hashlib.sha256()
            with weights.open("rb") as f:
                for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            with contextlib.redirect_stdout(sys.stderr):
                native = laya.load(record["path"], device="cuda")
            meta.update(
                record, weightsSha256=digest.hexdigest(), maxLen=8192, headMaxLen=2048, dtype=str(native.dtype)
            )

            class TankLaya(LayaDecisionModel):
                def predict(self, parts, questions):
                    started = time.perf_counter()
                    state, audit, trimmed = prepare_request(
                        self._agent, parts, questions, self._max_len, 2048, self._build_sequence, self._render_options
                    )
                    with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
                        result = self._agent.predict(state, questions, max_len=self._max_len, head_max_len=2048)
                    return {
                        "answers": result["answers"],
                        "modelResult": result,
                        "stateText": state,
                        "tokenAudit": audit,
                        "trimmed": trimmed,
                        "latencyMs": (time.perf_counter() - started) * 1000,
                    }

            model = TankLaya(native, meta, 8192)
        server = DecisionServer(("127.0.0.1", args.port), model)
        print(json.dumps({"ready": True, "url": server.url, "metadata": meta}), flush=True)
        try:
            server.serve()
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
