"""Local supervised bootstrap from normal-view, train-split decision traces."""

import argparse
import fcntl
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from dataset import separate_unique

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def collate(items, pad, torch, device):
    n, length, width = len(items), max(len(x["ids"]) for x in items), max(len(x["markers"]) for x in items)
    ids = torch.full((n, length), pad, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    positions = torch.zeros((n, width), dtype=torch.long, device=device)
    mask = torch.zeros((n, width), dtype=torch.bool, device=device)
    for i, x in enumerate(items):
        ids[i, : len(x["ids"])] = torch.tensor(x["ids"], device=device)
        attention[i, : len(x["ids"])] = 1
        positions[i, : len(x["markers"])] = torch.tensor(x["markers"], device=device)
        mask[i, : len(x["markers"])] = True
    return (
        ids,
        attention,
        positions,
        mask,
        torch.zeros(n, dtype=torch.long, device=device),
        torch.tensor([x["label"] for x in items], device=device),
    )


def train(args):
    os.environ.update(HF_HUB_OFFLINE="1", USE_TF="0", TOKENIZERS_PARALLELISM="false")
    source = Path(os.environ["LAYA_SOURCE_DIR"])
    pins = json.loads((ROOT / "pins.json").read_text())
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != pins["layaSource"]["revision"]:
        raise ValueError("Laya source differs from pin")
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(ROOT / "integration/laya"))
    import torch
    from laya.agent import Agent
    from laya.common import QTYPES, build_model, build_sequence, render_options
    from safetensors.torch import load_file, save_file
    from token_budget import prepare_request
    from transformers import AutoTokenizer

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    config = json.loads((ROOT / ".cache/models/typed-decisions.json").read_text())
    parent = Path(config["path"])
    initialization = config
    if args.parent_manifest:
        initialization = json.loads(args.parent_manifest.read_text())
        if (
            initialization.get("kind") != "trained-laya"
            or initialization.get("sourceRevision") != revision
            or initialization.get("parent", {}).get("revision") != config["revision"]
        ):
            raise ValueError("Unrecognized parent lineage")
        parent = Path(initialization["path"])
        if digest(parent / "model.safetensors") != initialization["weightsSha256"]:
            raise ValueError("Parent weights differ from manifest")
    cfg = json.loads((parent / "rl_agent_config.json").read_text())
    cfg.update(max_len=2048, head_max_len=512)
    tok = AutoTokenizer.from_pretrained(parent / "tokenizer", local_files_only=True)
    formatter = SimpleNamespace(tok=tok, _to_internal=Agent._to_internal)
    protocol = json.loads((ROOT / "league/protocol.json").read_text())
    items, validation, inputs = [], [], []
    packing = {"maxInputTokens": 0, "trimmedRequests": 0}
    for directory in args.matches:
        manifest = json.loads((directory / "manifest.json").read_text())
        result = json.loads((directory / "result.json").read_text())
        if (
            manifest["split"] != "train"
            or manifest["seed"] not in protocol["train"]["seeds"]
            or manifest["map"] not in protocol["train"]["maps"]
        ):
            raise ValueError("Refusing non-training episode")
        if result.get("notModelEvaluation") is not True or result["kind"] not in {
            "teacher-demonstration",
            "dagger-labels",
        }:
            raise ValueError("Expected explicitly labeled training demonstrations or DAgger corrections")
        for rel, h in manifest["sourceSha256"].items():
            if digest(directory / "source" / rel) != h:
                raise ValueError("Demonstration source mismatch")
        path = directory / "samples.jsonl"
        inputs.append(
            {
                "episode": directory.name,
                "seed": manifest["seed"],
                "seat": manifest["seat"],
                "map": manifest["map"],
                "kind": manifest["kind"],
                "manifestSha256": digest(directory / "manifest.json"),
                "samplesSha256": digest(path),
            }
        )
        for line in path.open():
            row = json.loads(line)
            if (
                row["split"] != "train"
                or row["seed"] != manifest["seed"]
                or row["seat"] != manifest["seat"]
                or row["map"] != manifest["map"]
            ):
                raise ValueError("Row provenance mismatch")
            state, audit, trimmed = prepare_request(
                formatter, row["parts"], row["questions"], 2048, 512, build_sequence, render_options
            )
            packing["maxInputTokens"] = max(packing["maxInputTokens"], max(v["inputTokens"] for v in audit.values()))
            packing["trimmedRequests"] += bool(trimmed)
            for key, q in row["questions"].items():
                if q["type"] != "choice":
                    raise ValueError("Unexpected question type")
                internal = Agent._to_internal(q)
                ids, markers = build_sequence(tok, state, internal, 2048, 512)
                keys = list(q["criteria"])
                assert len(markers) == len(keys)
                label = keys.index(row["answers"][key]["choice"])
                item = {"ids": ids, "markers": markers, "label": label, "qtype": QTYPES["choice"]}
                (validation if manifest["seed"] == args.validation_seed else items).append(item)
    items, validation, overlap_audit = separate_unique(items, validation)
    if not items or len(validation) < 50:
        raise ValueError("Need at least 50 novel validation examples after input deduplication")
    device = "cuda"
    model = build_model(cfg, encoder_dir=str(parent / "encoder"))
    model.load_state_dict(load_file(parent / "model.safetensors"), strict=True)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.to(device)
    encoder = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [{"params": encoder, "lr": args.encoder_lr}, {"params": head, "lr": args.head_lr}], weight_decay=0.01
    )
    best = float("inf")
    started = time.time()
    history = []
    initial_validation = None

    def evaluate():
        model.eval()
        losses = []
        correct = 0
        with torch.no_grad():
            for start in range(0, len(validation), args.batch):
                batch = collate(validation[start : start + args.batch], tok.pad_token_id, torch, device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, _ = model(*batch[:5])
                logits = logits.float().masked_fill(~batch[3], -1e4)
                losses.extend(torch.nn.functional.cross_entropy(logits, batch[5], reduction="none").tolist())
                correct += int((logits.argmax(-1) == batch[5]).sum())
        return sum(losses) / len(losses), correct / len(validation)

    initial_validation = evaluate()
    print(
        json.dumps(
            {
                "event": "prepared",
                "train": len(items),
                "validation": len(validation),
                "initialValidation": initial_validation,
            }
        ),
        flush=True,
    )
    for epoch in range(args.epochs):
        random.Random(args.seed + epoch).shuffle(items)
        model.train()
        total = 0
        steps = 0
        for start in range(0, len(items), args.batch):
            batch = collate(items[start : start + args.batch], tok.pad_token_id, torch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, aux = model(*batch[:5])
            loss = (
                torch.nn.functional.cross_entropy(logits.float().masked_fill(~batch[3], -1e4), batch[5])
                + 0 * aux.float().sum()
            )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
            steps += 1
            if steps % 20 == 0:
                print(
                    json.dumps(
                        {"event": "train-step", "epoch": epoch + 1, "step": steps, "loss": float(loss.detach())}
                    ),
                    flush=True,
                )
        val, accuracy = evaluate()
        row = {
            "epoch": epoch + 1,
            "trainLoss": total / max(1, steps),
            "validationLoss": val,
            "validationAccuracy": accuracy,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if val < best:
            best = val
            checkpoint = args.out / "checkpoint"
            checkpoint.mkdir(exist_ok=True)
            save_file(
                {k: v.detach().to(torch.bfloat16).contiguous().cpu() for k, v in model.state_dict().items()},
                checkpoint / "model.safetensors",
            )
            model.encoder.config.save_pretrained(checkpoint / "encoder")
            tok.save_pretrained(checkpoint / "tokenizer")
            cfg["temperature"] = [1.0, 1.0, 1.0]
            cfg.pop("temperature_by_options", None)
            (checkpoint / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    weight_hash = digest(args.out / "checkpoint/model.safetensors")
    manifest = {
        "kind": "trained-laya",
        "repo": "local:ra2-laya/" + args.out.name,
        "revision": weight_hash,
        "path": str((args.out / "checkpoint").resolve()),
        "weightsSha256": weight_hash,
        "parent": config,
        "initialization": {key: initialization[key] for key in ("repo", "revision", "path")},
        "sourceRevision": revision,
        "trainingScriptSha256": digest(Path(__file__)),
        "data": inputs,
        "packing": packing,
        "dataOverlapAudit": overlap_audit,
        "datasetScriptSha256": digest(Path(__file__).with_name("dataset.py")),
        "validationSeed": args.validation_seed,
        "trainExamples": len(items),
        "validationExamples": len(validation),
        "algorithm": "supervised-cross-entropy-data-aggregation",
        "initialValidation": initial_validation,
        "history": history,
        "confidenceCalibrated": False,
        "elapsedSeconds": time.time() - started,
        "settings": vars(args)
        | {
            "out": str(args.out),
            "matches": [str(x) for x in args.matches],
            "parent_manifest": str(args.parent_manifest) if args.parent_manifest else None,
        },
    }
    assert weight_hash != digest(parent / "model.safetensors"), "Weights did not change"
    (args.out / "checkpoint.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out / "result.json").write_text(
        json.dumps(
            {
                "complete": True,
                "checkpoint": str(args.out / "checkpoint.json"),
                "weightsSha256": weight_hash,
                "history": history,
            },
            indent=2,
        )
        + "\n"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matches", nargs="+", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gpu", default="1")
    p.add_argument("--parent-manifest", type=Path)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--validation-seed", type=int, default=101)
    p.add_argument("--encoder-lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=1e-4)
    args = p.parse_args()
    args.out = args.out.resolve()
    args.matches = [x.resolve() for x in args.matches]
    if args.out.exists():
        raise ValueError("Training output already exists")
    devices = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    uuid = next(x.split(",")[1].strip() for x in devices.splitlines() if x.split(",")[0].strip() == args.gpu)
    lockdir = Path(tempfile.gettempdir()) / f"laya-{os.getuid()}"
    lockdir.mkdir(exist_ok=True)
    with (lockdir / f"{uuid}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if uuid in subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True
        ):
            raise RuntimeError("Selected GPU is busy")
        os.environ["CUDA_VISIBLE_DEVICES"] = uuid
        args.out.mkdir(parents=True)
        snapshot = args.out / "source"
        snapshot.mkdir()
        for filename in ("train_laya.py", "dataset.py"):
            (snapshot / filename).write_bytes(Path(__file__).with_name(filename).read_bytes())
        try:
            train(args)
        except Exception as e:
            (args.out / "error.json").write_text(json.dumps({"error": str(e)}) + "\n")
            raise


if __name__ == "__main__":
    main()
