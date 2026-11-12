"""Audit two independent clean-replay collectors against the frozen 64-seed plan.

Rows are accepted only after both recorded runs are present and their saved
states, requests, probabilities, and outcomes agree. The two collectors may
use different GPUs but must use the same weights and game build.
"""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path
from typing import Any

signature = runpy.run_path(str(Path(__file__).with_name("reproducibility-gate.py")))["signature"]


def _manifest(run: Path) -> dict[str, Any]:
    return json.loads((run / "manifest.json").read_text(encoding="utf-8"))


def _frozen_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    model = manifest["model"]
    return {
        "code_commit": manifest["code_commit"],
        "sources": manifest["sources"],
        "harness_sha256": manifest["harness_sha256"],
        "controller_sha256": manifest["controller_sha256"],
        "agent_seat": manifest["agent_seat"],
        "opponent_seat": manifest["opponent_seat"],
        "model_revision": model["revision"],
        "model_source_commit": model["source_commit"],
        "weights_sha256": model["weights_sha256"],
        "seconds": manifest["seconds"],
        "interval_ms": manifest["interval_ms"],
        "opponent_interval_ms": manifest["opponent_interval_ms"],
        "opponent": manifest["opponent"],
        "sample_seed": manifest["sample_seed"],
    }


def summarize(plan: dict[str, Any], roots: list[Path], code_commit: str) -> dict[str, Any]:
    seeds = plan["groups"]["identity"]
    if len(seeds) != 64 or len(set(seeds)) != 64:
        raise ValueError("identity plan must contain 64 distinct seeds")
    expected = {(seed, seat) for seed in seeds for seat in ("blue", "red")}
    found: dict[tuple[int, str], list[dict[str, Any]]] = {}
    identity: dict[str, Any] | None = None
    excluded: list[dict[str, Any]] = []
    for root in roots:
        log = root / "gate.jsonl"
        if not log.exists():
            continue
        for line in log.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = row.get("seed"), row.get("seat")
            if key not in expected or row.get("passed") is not True:
                raise ValueError(f"unexpected or failed replay gate row: {key}")
            run_dirs = [Path(path) for path in row["run_dirs"]]
            if len(run_dirs) != 2 or any(run.parent != root for run in run_dirs):
                raise ValueError(f"invalid replay paths for {key}")
            first, second = (signature(run) for run in run_dirs)
            if first != second or len(first["state_hashes"]) != row["decisions"]:
                raise ValueError(f"saved replay differs for {key}")
            manifests = [_manifest(run) for run in run_dirs]
            if any(manifest["code_commit"] != code_commit for manifest in manifests):
                excluded.append(
                    {
                        "seed": key[0],
                        "seat": key[1],
                        "reason": "code_commit_mismatch",
                        "runs": [str(run) for run in run_dirs],
                    }
                )
                continue
            for manifest in manifests:
                if manifest["seed"] != key[0] or manifest["seat"] != key[1] or manifest["kind"] != "evaluation":
                    raise ValueError(f"run identity differs for {key}")
                current = _frozen_identity(manifest)
                if identity is not None and current != identity:
                    raise ValueError(f"frozen model or build differs for {key}")
                identity = current
            found.setdefault(key, []).append(first)
    for key, repeats in found.items():
        if any(repeat != repeats[0] for repeat in repeats[1:]):
            raise ValueError(f"parallel collectors disagree for {key}")
    scores = []
    decisions = 0
    for key, repeats in found.items():
        recorded = repeats[0]
        outcome = recorded["result"]["outcome"][key[1]]
        scores.append((outcome + 1) / 2)
        decisions += len(recorded["state_hashes"])
    return {
        "expected_seat_pairs": 128,
        "verified_seat_pairs": len(found),
        "missing_seat_pairs": len(expected - found.keys()),
        "complete": set(found) == expected,
        "verified_decisions": decisions,
        "mean_clean_score": sum(scores) / len(scores) if scores else None,
        "frozen_identity": identity,
        "required_code_commit": code_commit,
        "excluded": excluded,
        "collector_roots": [str(root) for root in roots],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-plan", type=Path, required=True)
    parser.add_argument("--collector", type=Path, action="append", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads(args.seed_plan.read_text(encoding="utf-8")), args.collector, args.code_commit)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("verified_seat_pairs", "missing_seat_pairs", "complete")}))


if __name__ == "__main__":
    main()
