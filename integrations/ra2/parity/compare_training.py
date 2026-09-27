"""Compare trained checkpoints against #49's own run-to-run variation.

Bitwise equality is impossible: #49's training is not deterministic on CUDA. So each run's
weight update (trained minus parent) is compared with #49 run A's update, and the agreement
between #49 and Honing must match the agreement between two #49 runs.
"""

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


def manifest(run):
    run = Path(run)
    return json.loads((run / "checkpoint.json").read_text())


def flat(weights, keys):
    return torch.cat([weights[k].double().flatten() for k in keys])


parent_path, a, b, ours = sys.argv[1:5]
parent = load_file(Path(parent_path) / "model.safetensors")
runs = {name: load_file(Path(p) / "checkpoint/model.safetensors") for name, p in (("A", a), ("B", b), ("ours", ours))}
keys = sorted(parent)
assert all(sorted(w) == keys for w in runs.values())
base = flat(parent, keys)
delta = {name: flat(w, keys) - base for name, w in runs.items()}
report = {}
for name in ("B", "ours"):
    d = delta[name]
    cos = float(torch.nn.functional.cosine_similarity(delta["A"], d, dim=0))
    rel = float((delta["A"] - d).norm() / delta["A"].norm())
    identical = sum(torch.equal(runs["A"][k], runs[name][k]) for k in keys)
    report[name] = {
        "cosine_with_A": cos,
        "relative_distance_to_A": rel,
        "bit_identical_tensors": f"{identical}/{len(keys)}",
        "update_norm": float(d.norm()),
    }
report["A"] = {"update_norm": float(delta["A"].norm())}
histories = {name: manifest(p) for name, p in (("A", a), ("B", b), ("ours", ours))}
for name, m in histories.items():
    report[name]["initial_validation"] = m["initialValidation"]
    report[name]["history"] = [
        {k: h[k] for k in ("trainLoss", "validationLoss", "validationAccuracy")} for h in m["history"]
    ]
    report[name]["examples"] = [m["trainExamples"], m["validationExamples"]]
print(json.dumps(report, indent=2))
same_data = (
    histories["A"]["initialValidation"] == histories["ours"]["initialValidation"]
    and report["A"]["examples"] == report["ours"]["examples"]
)
within = (
    report["ours"]["cosine_with_A"] >= report["B"]["cosine_with_A"] - 0.02
    and report["ours"]["relative_distance_to_A"] <= report["B"]["relative_distance_to_A"] * 1.1 + 0.02
)
print("same examples and identical initial validation:", same_data)
print("Honing within #49's run-to-run variation:", within)
print("PARITY" if same_data and within else "NO PARITY")
