"""Pinned local Laya inference over JSON lines; model failure stops the match."""

import contextlib
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

from token_budget import prepare_request

ROOT = Path(__file__).resolve().parents[2]


def emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def main():
    os.environ.update(USE_TF="0", TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1")
    pins = json.loads((ROOT / "pins.json").read_text())
    source = Path(os.environ.get("LAYA_SOURCE_DIR", ROOT / "deps/laya"))
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != pins["layaSource"]["revision"]:
        raise ValueError("LAYA_SOURCE_DIR must use the revision in pins.json")
    sys.path.insert(0, str(source))
    gpu = os.environ.get("LAYA_GPU", "0")
    devices = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    matches = [row.split(",")[1].strip() for row in devices.splitlines() if row.split(",")[0].strip() == gpu]
    if not matches:
        raise ValueError(f"GPU index {gpu} not found")
    uuid = matches[0]
    lock_dir = Path(os.environ.get("LAYA_LOCK_DIR", Path(tempfile.gettempdir()) / f"laya-{os.getuid()}"))
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / f"{uuid}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        used = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True
        )
        if uuid in used:
            raise RuntimeError(f"GPU {gpu} already has a compute process")
        os.environ["CUDA_VISIBLE_DEVICES"] = uuid
        model_type = os.environ.get("LAYA_MODEL_TYPE", "english")
        if model_type not in pins["models"]:
            raise ValueError("Unknown Laya model type")
        config_dir = Path(os.environ.get("LAYA_MODEL_CONFIG_DIR", ROOT / ".cache/models"))
        model = json.loads((config_dir / f"{model_type}.json").read_text())
        for key in ("repo", "revision"):
            if model[key] != pins["models"][model_type][key]:
                raise ValueError("Model config differs from pins.json")
        checkpoint_manifest = os.environ.get("LAYA_CHECKPOINT_MANIFEST")
        if checkpoint_manifest:
            trained = json.loads(Path(checkpoint_manifest).read_text())
            if trained.get("kind") != "trained-laya" or trained.get("sourceRevision") != revision:
                raise ValueError("Unrecognized trained checkpoint provenance")
            if trained.get("parent", {}).get("revision") != pins["models"][model_type]["revision"]:
                raise ValueError("Trained checkpoint parent does not match selected model")
            weight_file = Path(trained["path"]) / "model.safetensors"
            checksum = hashlib.sha256()
            with weight_file.open("rb") as weights:
                for block in iter(lambda: weights.read(8 * 1024 * 1024), b""):
                    checksum.update(block)
            if checksum.hexdigest() != trained["weightsSha256"]:
                raise ValueError("Trained checkpoint weight hash mismatch")
            model = {key: trained[key] for key in ("repo", "revision", "path", "weightsSha256", "parent")}
            model["trainingManifestSha256"] = hashlib.sha256(Path(checkpoint_manifest).read_bytes()).hexdigest()
        with contextlib.redirect_stdout(sys.stderr):
            import laya
            import torch
            import transformers
            from laya.common import build_sequence, render_options

            torch.set_num_threads(4)
            torch.manual_seed(1)
            agent = laya.load(model["path"], device="cuda")
        max_len = int(os.environ.get("LAYA_MAX_LEN", "2048"))
        head_max_len = 512
        if max_len <= 0:
            raise ValueError("LAYA_MAX_LEN must be positive")
        metadata = {
            **model,
            "sourceCommit": revision,
            "layaVersion": laya.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gpu": gpu,
            "gpuUuid": uuid,
            "dtype": str(agent.dtype),
            "maxLen": max_len,
            "headMaxLen": head_max_len,
            "checkpointMaxLen": agent.cfg.get("max_len"),
            "temperatureByOptions": agent.temperature_by_options,
            "python": sys.version.split()[0],
        }
        emit({"ready": True, "metadata": metadata})
        for line in sys.stdin:
            request = json.loads(line)
            started = time.perf_counter()
            state, audit, trimmed = prepare_request(
                agent, request["parts"], request["questions"], max_len, head_max_len, build_sequence, render_options
            )
            with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
                result = agent.predict(state, request["questions"], max_len=max_len, head_max_len=head_max_len)
            emit(
                {
                    "id": request["id"],
                    "answers": result["answers"],
                    "modelResult": result,
                    "stateText": state,
                    "tokenAudit": audit,
                    "trimmed": trimmed,
                    "latencyMs": (time.perf_counter() - started) * 1000,
                }
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        emit({"error": str(error)})
        sys.exit(1)
