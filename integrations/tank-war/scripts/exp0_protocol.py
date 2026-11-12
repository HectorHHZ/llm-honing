"""Deterministic seed allocation and paired fault screening for Tank War Exp 0.

The input layout is one match directory per ``s<seed>-<seat>`` beneath each
arm's root. Each directory is an unmodified ``match.mjs`` output. This module
never drops an incomplete pair: an absent/error run invalidates validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from honing.backends.tank.grader import DRAW, LOSS, WIN

SEATS = ("blue", "red")
SEED_GROUPS = {
    "identity": 64,
    "fault_validation": 128,
    "pilot": 1000,
    "development": 2000,
    "final_test": 20000,
    "natural": 1000,
}
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 183
MIN_MEAN_DEGRADATION = 0.05


def seed_plan(version: str = "v1") -> dict[str, Any]:
    """Allocate disjoint, reproducible reset groups without touching gate seeds."""
    if not version or "/" in version or ".." in version:
        raise ValueError("version must be one path-safe component")
    used = set(range(1001, 1017)) | set(range(2001, 2009)) | set(range(3001, 3009))
    groups: dict[str, list[int]] = {}
    for stage, count in SEED_GROUPS.items():
        seeds: list[int] = []
        index = 0
        while len(seeds) < count:
            digest = hashlib.sha256(f"tank-war-exp0:{version}:{stage}:{index}".encode()).digest()
            seed = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
            index += 1
            if seed in used:
                continue
            used.add(seed)
            seeds.append(seed)
        groups[stage] = seeds
    return {"version": version, "algorithm": "sha256-first-31-bits", "seats": list(SEATS), "groups": groups}


def _score(result: dict[str, Any], seat: str) -> float:
    outcome = result.get("outcome", {}).get(seat)
    if outcome == 1:
        return WIN
    if outcome == 0:
        return DRAW
    if outcome == -1:
        return LOSS
    raise ValueError(f"invalid outcome for {seat}: {outcome!r}")


def _read_match(root: Path, seed: int, seat: str) -> tuple[dict[str, Any], dict[str, Any]]:
    run = root / f"s{seed}-{seat}"
    if (run / "error.json").exists():
        raise ValueError(f"match failed: {run / 'error.json'}")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    expected = {
        "seed": seed,
        "seat": seat,
        "opponent": "planner",
        "seconds": 30,
        "interval_ms": 200,
        "opponent_interval_ms": 300,
    }
    actual = {key: manifest.get(key) for key in expected}
    if actual != expected or manifest.get("kind") != "evaluation":
        raise ValueError(f"run identity mismatch in {run}: {actual}")
    if result.get("seed") != seed or result.get("seat") != seat:
        raise ValueError(f"result identity mismatch in {run}")
    return manifest, result


def _percentile(sorted_values: list[float], fraction: float) -> float:
    position = (len(sorted_values) - 1) * fraction
    left = int(position)
    right = min(left + 1, len(sorted_values) - 1)
    return sorted_values[left] + (sorted_values[right] - sorted_values[left]) * (position - left)


def paired_interval(block_effects: list[float]) -> tuple[float, float]:
    if len(block_effects) < 64:
        raise ValueError("at least 64 independent seed blocks are required")
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(block_effects)
    means = [sum(block_effects[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOTSTRAP_SAMPLES)]
    means.sort()
    return _percentile(means, 0.025), _percentile(means, 0.975)


def validate_fault(clean_root: Path, fault_root: Path, seeds: list[int]) -> dict[str, Any]:
    """Validate one fixed candidate without optional stopping or silent exclusions."""
    if len(seeds) != 128 or len(set(seeds)) != 128:
        raise ValueError("fault validation requires exactly 128 distinct seed blocks")
    effects: list[float] = []
    observations: list[dict[str, Any]] = []
    for seed in seeds:
        seat_effects = []
        for seat in SEATS:
            clean_manifest, clean_result = _read_match(clean_root, seed, seat)
            fault_manifest, fault_result = _read_match(fault_root, seed, seat)
            for key in ("opponent", "seconds", "interval_ms", "opponent_interval_ms", "opponent_seat"):
                if clean_manifest.get(key) != fault_manifest.get(key):
                    raise ValueError(f"unpaired {key} for seed {seed}, seat {seat}")
            effect = _score(fault_result, seat) - _score(clean_result, seat)
            seat_effects.append(effect)
            observations.append(
                {
                    "seed": seed,
                    "seat": seat,
                    "clean": _score(clean_result, seat),
                    "fault": _score(fault_result, seat),
                    "effect": effect,
                }
            )
        effects.append(sum(seat_effects) / len(SEATS))
    mean = sum(effects) / len(effects)
    lower, upper = paired_interval(effects)
    return {
        "seed_blocks": len(seeds),
        "rounds_per_arm": len(observations),
        "mean_fault_minus_clean": mean,
        "ci95": [lower, upper],
        "meaningful_degradation": mean <= -MIN_MEAN_DEGRADATION,
        "interval_shows_degradation": upper < 0,
        "retained": mean <= -MIN_MEAN_DEGRADATION and upper < 0,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "observations": observations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    seeds_parser = sub.add_parser("seed-plan")
    seeds_parser.add_argument("--version", default="v1")
    seeds_parser.add_argument("--out", type=Path, required=True)
    fault_parser = sub.add_parser("validate-fault")
    fault_parser.add_argument("--seed-plan", type=Path, required=True)
    fault_parser.add_argument("--clean-root", type=Path, required=True)
    fault_parser.add_argument("--fault-root", type=Path, required=True)
    fault_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "seed-plan":
        result = seed_plan(args.version)
    else:
        plan = json.loads(args.seed_plan.read_text(encoding="utf-8"))
        result = validate_fault(args.clean_root, args.fault_root, plan["groups"]["fault_validation"])
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "retained": result.get("retained")}))


if __name__ == "__main__":
    main()
