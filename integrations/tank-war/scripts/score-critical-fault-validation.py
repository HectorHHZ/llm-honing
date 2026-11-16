"""Score a fixed, two-seat Tank War fault screen without selecting on outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

LAYERS = ("harness", "weights", "controller")
SEATS = ("blue", "red")
OUTCOME_SCORE = {-1: 0.0, 0: 0.5, 1: 1.0}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(values: list[float], *, resamples: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    draws = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(resamples))
    return draws[int(0.025 * resamples) - 1], draws[int(0.975 * resamples) - 1]


def score_layer(root: Path, layer: str, seeds: list[int], protocol: dict) -> dict:
    frozen = read(root / layer / "freeze.json")
    expected = {(seed, seat) for seed in seeds for seat in SEATS}
    actual = {(entry["seed"], entry["seat"]) for entry in frozen["entries"]}
    if actual != expected or len(frozen["entries"]) != len(expected):
        raise ValueError(f"{layer}: collector freeze differs from fixed 64-block plan")
    if any(entry["layer"] != layer or entry["split"] != "fault_validation" for entry in frozen["entries"]):
        raise ValueError(f"{layer}: wrong layer or split in collector freeze")

    deltas: dict[int, dict[str, float]] = {}
    missing: list[dict] = []
    invalid: list[dict] = []
    attempted = 0
    for entry in frozen["entries"]:
        seed, seat = entry["seed"], entry["seat"]
        folder = root / layer / layer / f"s{seed}-{seat}"
        config_name = f"{layer}-{seed}-{seat}.json" if layer == "harness" else f"{layer}-{entry['trigger']}.json"
        config_path = root / layer / "configs" / config_name
        evidence_path = folder / "pair_evidence.json"
        if (folder / "clean").exists() or (folder / "fault").exists():
            attempted += 1
        if not evidence_path.exists():
            missing.append({"seed": seed, "seat": seat})
            continue
        evidence = read(evidence_path)
        source_key = "harness" if layer == "harness" else "weights_and_controller"
        expected_source_sha = protocol["match_sha256"][source_key]
        source_valid = all(
            read(folder / arm / "manifest.json")["sources"]["scripts/match.mjs"] == expected_source_sha
            for arm in ("clean", "fault")
        )
        if (
            read(config_path)["instance"] != protocol["fault_instances"][layer]
            or not source_valid
            or (evidence["seed"], evidence["seat"], evidence["layer"]) != (seed, seat, layer)
            or evidence["clean_result_sha256"] != sha(folder / "clean" / "result.json")
            or evidence["fault_result_sha256"] != sha(folder / "fault" / "result.json")
            or (evidence["triggered"] and evidence["instance"] != protocol["fault_instances"][layer])
            or (evidence["triggered"] and not evidence.get("layer_isolated"))
            or (not evidence["triggered"] and not evidence.get("no_trigger_equal"))
        ):
            invalid.append({"seed": seed, "seat": seat, "reason": "pair identity or layer isolation"})
            continue
        clean, fault = evidence["clean_outcome"], evidence["fault_outcome"]
        if clean not in OUTCOME_SCORE or fault not in OUTCOME_SCORE:
            invalid.append({"seed": seed, "seat": seat, "reason": "unsupported outcome"})
            continue
        deltas.setdefault(seed, {})[seat] = OUTCOME_SCORE[fault] - OUTCOME_SCORE[clean]

    blocks = [
        (deltas[seed]["blue"] + deltas[seed]["red"]) / 2 for seed in seeds if set(deltas.get(seed, {})) == set(SEATS)
    ]
    complete = len(blocks) == len(seeds) and not missing and not invalid
    result = {
        "instance": protocol["fault_instances"][layer],
        "planned_seed_blocks": len(seeds),
        "planned_pairs": len(expected),
        "attempted_pairs": attempted,
        "complete_seed_blocks": len(blocks),
        "missing": missing,
        "invalid": invalid,
        "status": "complete" if complete else "incomplete",
    }
    if blocks:
        mean = sum(blocks) / len(blocks)
        lower, upper = interval(
            blocks,
            resamples=protocol["bootstrap"]["resamples"],
            seed=protocol["bootstrap"]["rng_seed"],
        )
        result.update(mean_fault_minus_clean=mean, paired_bootstrap_95pct=[lower, upper])
        if complete:
            result["validated_score_drop"] = bool(
                mean <= protocol["retain_if"]["mean_lte"] and upper < protocol["retain_if"]["bootstrap_upper_lt"]
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol = read(args.root / "issue183-fault-validation-protocol.json")
    seed_plan = Path(protocol["seed_plan"])
    if sha(seed_plan) != protocol["seed_plan_sha256"]:
        raise ValueError("seed plan changed")
    if protocol["layers"] != list(LAYERS) or protocol["blocks_per_layer"] != 64:
        raise ValueError("unexpected validation protocol")
    seeds = read(seed_plan)["groups"][protocol["seed_group"]][: protocol["blocks_per_layer"]]
    if len(seeds) != 64 or len(set(seeds)) != 64:
        raise ValueError("validation seed group incomplete")
    layers = {layer: score_layer(args.root, layer, seeds, protocol) for layer in LAYERS}
    result = {
        "protocol_sha256": sha(args.root / "issue183-fault-validation-protocol.json"),
        "seed_group": protocol["seed_group"],
        "seed_indices": protocol["seed_indices"],
        "status": "complete" if all(row["status"] == "complete" for row in layers.values()) else "incomplete",
        "layers": layers,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "layers": layers}))


if __name__ == "__main__":
    main()
