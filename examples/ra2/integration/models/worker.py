"""Experimental pinned native JSONL worker; not a game or promotion entry point."""

import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

from .upstream_adapters import JevOmniChoices, NanoJevChoices

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def read_json(text):
    def nonfinite(_):
        raise ValueError("Nonfinite JSON value")

    return json.loads(text, object_pairs_hook=unique_object, parse_constant=nonfinite)


def checked_relative(name):
    path = Path(name)
    if not isinstance(name, str) or not name or path.is_absolute() or ".." in path.parts:
        raise ValueError("Prepared model filenames must be relative paths inside its directory")
    if str(path) != name:
        raise ValueError("Prepared model filenames must be canonical")
    return path


def verify_prepared(path, family, pins):
    """Verify required native inputs and every registered file before importing code."""
    if family not in {"nanojev", "jev-omni"}:
        raise ValueError("Unsupported native model family")
    manifest = read_json(path.read_text())
    if manifest.get("kind") != "prepared-upstream-model":
        raise ValueError("Expected prepared upstream provenance, not a trained-checkpoint claim")
    for key in ("source", "sourceRevision", "model", "modelRevision"):
        if key in pins[family] and manifest.get(key) != pins[family][key]:
            raise ValueError(f"Native source/model pin mismatch: {key}")
    directory = Path(manifest["path"]).resolve(strict=True)
    files = manifest.get("filesSha256")
    if not isinstance(files, dict) or not files:
        raise ValueError("Prepared file hashes are required")
    for name, expected in files.items():
        relative = checked_relative(name)
        if not isinstance(expected, str) or not re.fullmatch("[0-9a-f]{64}", expected):
            raise ValueError("Invalid registered file hash")
        target = (directory / relative).resolve(strict=True)
        if not target.is_relative_to(directory) or not target.is_file():
            raise ValueError("Prepared file escapes model directory")
        if digest(target) != expected:
            raise ValueError(f"Native file hash mismatch: {name}")
    required = (
        {
            "best.safetensors",
            "config.json",
            "backbone_config/config.json",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
        }
        if family == "nanojev"
        else {
            "backbone/config.json",
            "backbone/model.safetensors.index.json",
            "decision_config.json",
            "head.pt",
            "runtime_buffers.pt",
            "load_model.py",
            "tokenizer.json",
            "tokenizer_config.json",
        }
    )
    if not required <= files.keys():
        raise ValueError("Prepared manifest omits a required native input")
    if family == "jev-omni":
        index = read_json((directory / "backbone/model.safetensors.index.json").read_text())
        shards = set(index["weight_map"].values())
        if not shards or any("backbone/" + str(checked_relative(s)) not in files for s in shards):
            raise ValueError("Model index references an unverified weight shard")
    weights = {k: v for k, v in files.items() if k.endswith((".safetensors", ".pt"))}
    weight_set = hashlib.sha256(json.dumps(weights, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return manifest, {
        "family": family,
        "repo": manifest["model"],
        "revision": manifest["modelRevision"],
        "path": str(directory),
        "preparedManifestSha256": digest(path),
        "preparedManifestPath": str(path.resolve(strict=True)),
        "verifiedFilesSha256": files,
        "weightSetSha256": weight_set,
        "weightDigestDefinition": "sha256 of sorted compact JSON mapping native weight filenames to SHA256",
        "ephemeralStorage": manifest.get("ephemeralStorage", False),
    }


def verify_nano_source(source, pins):
    source = source.resolve(strict=True)
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    if revision != pins["nanojev"]["sourceRevision"] or dirty:
        raise ValueError("NanoJev requires the pinned unmodified tracked source checkout")
    return source, revision


@contextlib.contextmanager
def gpu_lock(gpu):
    """Use the same per-device worker lock as the Laya campaign dispatcher."""
    devices = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    matches = [row.split(",")[1].strip() for row in devices.splitlines() if row.split(",")[0].strip() == gpu]
    if len(matches) != 1:
        raise ValueError("GPU index not found or ambiguous")
    uuid = matches[0]
    directory = Path(os.environ.get("LAYA_LOCK_DIR", Path(tempfile.gettempdir()) / f"laya-{os.getuid()}"))
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{uuid}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        used = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True
        )
        if uuid in used.splitlines():
            raise RuntimeError("Selected GPU already has a compute process")
        previous = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = uuid
        try:
            yield uuid
        finally:
            if previous is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_native(family, manifest, source, max_length):
    if family == "nanojev":
        sys.path.insert(0, str(source))
        upstream = load_module(source / "scripts/predict_toy_decisions.py", "ra2_nanojev_predict")
        predictor = upstream.DecisionPredictor(
            manifest["path"], max_length=max_length, device_name="cuda:0", precision="bf16"
        )
        return NanoJevChoices(predictor, upstream.prepare_examples)
    upstream = load_module(Path(manifest["path"]) / "load_model.py", "ra2_jev_omni_loader")
    model, head, tokenizer = upstream.load_model(manifest["path"], device="cuda:0")
    return JevOmniChoices(model, head, tokenizer, upstream.predict, max_length=max_length)


def requests(adapter, stream, emit):
    """Validate monotonically increasing IDs; errors stop without a substitute choice."""
    last = 0
    for line in stream:
        value = read_json(line)
        if not isinstance(value, dict) or set(value) != {"id", "parts", "questions"}:
            raise ValueError("Expected id, parts and questions only")
        request_id = value["id"]
        if type(request_id) is not int or request_id <= last:
            raise ValueError("Request IDs must be positive and strictly increasing")
        with contextlib.redirect_stdout(sys.stderr):
            response = adapter.predict(value["parts"], value["questions"])
        emit({**response, "id": request_id})
        last = request_id


def emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["nanojev", "jev-omni"], required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--head-manifest", type=Path)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()
    if args.max_length <= 0:
        parser.error("max-length must be positive")
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        USE_TF="0",
        TOKENIZERS_PARALLELISM="false",
    )
    pins = read_json((ROOT / "league/model-sources.json").read_text())
    manifest, metadata = verify_prepared(args.prepared_manifest, args.family, pins)
    parent_metadata = dict(metadata)
    if args.head_manifest is not None:
        from .head_checkpoint import trained_runtime

        metadata = trained_runtime(parent_metadata, args.head_manifest)
    source = None
    if args.family == "nanojev":
        if args.source_root is None:
            parser.error("NanoJev needs its pinned --source-root")
        source, metadata["sourceCommit"] = verify_nano_source(args.source_root, pins)
        metadata["sourceRoot"] = str(source)
    with gpu_lock(args.gpu) as uuid:
        with contextlib.redirect_stdout(sys.stderr):
            import torch
            import transformers

            torch.set_num_threads(4)
            torch.manual_seed(1)
            adapter = load_native(args.family, manifest, source, args.max_length)
            if args.head_manifest is not None:
                from .head_checkpoint import load_head_artifact, trained_runtime

                load_head_artifact(adapter, args.family, args.head_manifest, parent_metadata)
                # Recheck the artifact after loading so ready cannot describe changed files.
                if trained_runtime(parent_metadata, args.head_manifest)["learnedHead"] != metadata["learnedHead"]:
                    raise ValueError("Learned native head changed while loading")
        metadata.update(
            protocol="native-choice-jsonl-v1",
            gpu=args.gpu,
            gpuUuid=uuid,
            maxLen=args.max_length,
            parameterStorage="float32",
            forwardAutocast="bfloat16",
            torch=torch.__version__,
            transformers=transformers.__version__,
            python=sys.version.split()[0],
            modelWeightsLoaded=True,
        )
        emit({"ready": True, "metadata": metadata})
        requests(adapter, sys.stdin, emit)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        emit({"error": str(exc)})
        sys.exit(1)
