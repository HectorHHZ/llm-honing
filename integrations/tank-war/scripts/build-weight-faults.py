"""Rebuild Tank War's real NanoJev scalar edits from pinned checkpoints.

The source checkpoints must have identical safetensors layouts and differ only
in ``scalar.weight``. This is the portable form of the audited Exp 0 builder;
it does not train a model or substitute a logits wrapper for changed weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import struct
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint(manifest_path: Path) -> tuple[dict[str, Any], Path]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    location = Path(manifest["path"])
    if not location.is_absolute():
        location = manifest_path.parent / location
    weights = location / "best.safetensors"
    if not weights.is_file() or sha256(weights) != manifest.get("weightsSha256"):
        raise ValueError(f"checkpoint digest mismatch: {manifest_path}")
    return manifest, weights


def tensor_span(weights: Path) -> tuple[int, int, bytes, int]:
    with weights.open("rb") as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise ValueError("incomplete safetensors header")
        length = struct.unpack("<Q", length_bytes)[0]
        raw = stream.read(length)
        if len(raw) != length:
            raise ValueError("incomplete safetensors header")
    header = json.loads(raw)
    tensor = header["scalar.weight"]
    if tensor["dtype"] != "F32" or tensor["shape"] != [1, 1024]:
        raise ValueError("unexpected scalar.weight tensor layout")
    start, end = (8 + length + offset for offset in tensor["data_offsets"])
    if end - start != 1024 * 4 or end > weights.stat().st_size:
        raise ValueError("invalid scalar.weight byte span")
    return start, end, raw, weights.stat().st_size


def same_except_scalar(left: Path, right: Path, start: int, end: int, size: int) -> bool:
    if left.stat().st_size != size or right.stat().st_size != size:
        return False
    with left.open("rb") as a, right.open("rb") as b:
        for lo, hi in ((0, start), (end, size)):
            a.seek(lo)
            b.seek(lo)
            remaining = hi - lo
            while remaining:
                count = min(8 * 1024 * 1024, remaining)
                if a.read(count) != b.read(count):
                    return False
                remaining -= count
    return True


def scalar_values(weights: Path, start: int) -> tuple[float, ...]:
    with weights.open("rb") as stream:
        stream.seek(start)
        return struct.unpack("<1024f", stream.read(1024 * 4))


def build(
    prior_manifest_path: Path,
    reverse_manifest_path: Path,
    out: Path,
    strengths: tuple[float, ...],
    clean_weights_sha256: str,
    clean_manifest_path: Path | None = None,
) -> list[dict[str, Any]]:
    if (
        not strengths
        or len(strengths) != len(set(strengths))
        or any(not math.isfinite(s) or s >= 0 for s in strengths)
    ):
        raise ValueError("strengths must be distinct finite negative numbers")
    if re.fullmatch(r"[0-9a-f]{64}", clean_weights_sha256) is None:
        raise ValueError("clean weights SHA-256 must be 64 lowercase hex characters")
    if out.exists():
        raise FileExistsError(out)
    prior_manifest, prior = checkpoint(prior_manifest_path)
    reverse_manifest, reverse = checkpoint(reverse_manifest_path)
    start, end, header, size = tensor_span(prior)
    if tensor_span(reverse) != (start, end, header, size):
        raise ValueError("source checkpoints have different tensor layouts")
    if not same_except_scalar(prior, reverse, start, end, size):
        raise ValueError("source checkpoints differ outside scalar.weight")
    prior_values = scalar_values(prior, start)
    reverse_values = scalar_values(reverse, start)
    if clean_manifest_path is not None:
        clean_manifest, clean = checkpoint(clean_manifest_path)
        if clean_manifest["weightsSha256"] != clean_weights_sha256:
            raise ValueError("clean checkpoint does not match declared digest")
        if tensor_span(clean) != (start, end, header, size) or not same_except_scalar(prior, clean, start, end, size):
            raise ValueError("clean checkpoint differs outside scalar.weight")
        clean_values = scalar_values(clean, start)
        symmetric_values = zip(clean_values, prior_values, reverse_values, strict=True)
        if any(abs(c - (p + r) / 2) > 1e-5 for c, p, r in symmetric_values):
            raise ValueError("source edits are not symmetric around the clean release")
    direction_norm = math.sqrt(sum(((p - r) / 2) ** 2 for p, r in zip(prior_values, reverse_values, strict=True)))
    if not 0.99 < direction_norm < 1.01:
        raise ValueError(f"unexpected direction norm: {direction_norm}")

    out.mkdir(parents=True)
    records = []
    for strength in strengths:
        name = f"reverse_strength_{str(abs(strength)).replace('.', 'p')}"
        destination = out / name
        shutil.copytree(reverse.parent, destination / "checkpoint", symlinks=True)
        edited = destination / "checkpoint" / "best.safetensors"
        values = [
            ((1 + strength) / 2) * p + ((1 - strength) / 2) * r
            for p, r in zip(prior_values, reverse_values, strict=True)
        ]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("nonfinite scalar edit")
        with edited.open("r+b") as stream:
            stream.seek(start)
            stream.write(struct.pack("<1024f", *values))
        digest = sha256(edited)
        manifest = {
            "kind": "trained-nanojev",
            "repo": f"local:tank-war-exp0-{name}",
            "revision": digest,
            "path": str((destination / "checkpoint").resolve()),
            "weightsSha256": digest,
            "parent": reverse_manifest.get("parent"),
            "sourceRevision": reverse_manifest.get("sourceRevision"),
            "changed": {"scalar.weight": 1024},
            "edit": {
                "formula": "release + strength * (prior_edit - reverse_edit) / 2",
                "strength": strength,
                "clean_weights_sha256": clean_weights_sha256,
                "prior_weights_sha256": prior_manifest["weightsSha256"],
                "reverse_weights_sha256": reverse_manifest["weightsSha256"],
                "direction_l2": direction_norm,
            },
        }
        (destination / "checkpoint.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        fault = {
            "layer": "weights",
            "family": "reverse_prior_scalar_edit",
            "instance": f"{name}-v1",
            "validation_status": "pending",
            "changed_tensor": "scalar.weight",
            "weights_sha256": digest,
            "clean_weights_sha256": clean_weights_sha256,
            "checkpoint_manifest": str(destination / "checkpoint.json"),
        }
        (destination / "fault.json").write_text(json.dumps(fault, indent=2) + "\n", encoding="utf-8")
        records.append({"name": name, "strength": strength, "weights_sha256": digest})
    (out / "catalog.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-manifest", type=Path)
    parser.add_argument("--clean-weights-sha256", required=True)
    parser.add_argument("--prior-manifest", required=True, type=Path)
    parser.add_argument("--reverse-manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--strength", required=True, type=float, action="append")
    args = parser.parse_args()
    records = build(
        args.prior_manifest,
        args.reverse_manifest,
        args.out,
        tuple(args.strength),
        args.clean_weights_sha256,
        args.clean_manifest,
    )
    print(json.dumps({"candidates": len(records), "out": str(args.out)}))


if __name__ == "__main__":
    main()
