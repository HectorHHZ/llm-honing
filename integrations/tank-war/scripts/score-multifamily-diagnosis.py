"""Score sealed Tank War whole-game causes across instances and reset seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

LAYERS = ("harness", "weights", "controller")
RUNS = {
    "sol": {"read": "sol-bare", "probe": "sol-bare", "read+fewshot": "sol-fewshot", "probe+fewshot": "sol-fewshot"},
    "astra": {
        "read": "astra-bare",
        "probe": "astra-bare",
        "read+fewshot": "astra-fewshot",
        "probe+fewshot": "astra-fewshot",
    },
}


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def macro_f1(rows: list[dict[str, Any]], key: str) -> float:
    scores = []
    for layer in LAYERS:
        true_positive = sum(row["label"] == layer and row[key]["cause"] == layer for row in rows)
        false_positive = sum(row["label"] != layer and row[key]["cause"] == layer for row in rows)
        false_negative = sum(row["label"] == layer and row[key]["cause"] != layer for row in rows)
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return sum(scores) / len(scores)


def calibration_error(rows: list[dict[str, Any]], key: str, bins: int = 5) -> float:
    error = 0.0
    for bin_index in range(bins):
        members = [row for row in rows if min(int(row[key]["confidence"] * bins), bins - 1) == bin_index]
        if members:
            accuracy = sum(row[key]["cause"] == row["label"] for row in members) / len(members)
            confidence = sum(row[key]["confidence"] for row in members) / len(members)
            error += len(members) / len(rows) * abs(accuracy - confidence)
    return error


def paired_interval(rows: list[dict[str, Any]], left: str, right: str, draws: int = 10_000) -> dict[str, Any]:
    faults = [row for row in rows if row["label"] in LAYERS]
    groups: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in faults:
        groups[row["label"]][row["instance"]].append(row)
    if any(len(groups[layer]) < 1 for layer in LAYERS):
        raise ValueError("a fault class has no eligible instance")

    def delta(row: dict[str, Any]) -> int:
        return int(row[right]["cause"] == row["label"]) - int(row[left]["cause"] == row["label"])

    point = (
        sum(
            sum(delta(row) for row in faults if row["label"] == layer) / sum(row["label"] == layer for row in faults)
            for layer in LAYERS
        )
        / 3
    )
    seed_ids = sorted({row["seed"] for row in faults})
    rng = random.Random(183)
    samples = []
    attempts = 0
    while len(samples) < draws and attempts < draws * 20:
        attempts += 1
        seed_weights = Counter(rng.choices(seed_ids, k=len(seed_ids)))
        layer_means = []
        for layer in LAYERS:
            instances = sorted(groups[layer])
            sampled_instances = rng.choices(instances, k=len(instances))
            values = [
                delta(row)
                for instance in sampled_instances
                for row in groups[layer][instance]
                for _ in range(seed_weights[row["seed"]])
            ]
            if not values:
                break
            layer_means.append(sum(values) / len(values))
        if len(layer_means) == 3:
            samples.append(sum(layer_means) / 3)
    if len(samples) != draws:
        raise ValueError("too few nonempty grouped bootstrap draws")
    samples.sort()
    return {
        "point": point,
        "instance_and_reset_group_bootstrap_95pct": [
            samples[int(0.025 * draws) - 1],
            samples[int(0.975 * draws) - 1],
        ],
        "resamples": draws,
        "rng_seed": 183,
        "independent_seed_blocks": len(seed_ids),
        "discarded_empty_draws": attempts - draws,
    }


def load_answers(source: Path, diagnosis: Path) -> list[dict[str, Any]]:
    sample = read(source / "sample-freeze.json")
    manifest = read(diagnosis / "manifest.json")
    if manifest["status"] != "frozen_before_any_new_answers" or manifest["sample_freeze_sha256"] != sha(
        source / "sample-freeze.json"
    ):
        raise ValueError("sample or diagnosis freeze changed")
    if len({case["episode_id"] for case in sample["cases"]}) != len(sample["cases"]):
        raise ValueError("duplicate frozen episode")
    rows = []
    for case in sample["cases"]:
        if sha(source / "inputs" / f"{case['episode_id']}.json") != case["input_sha256"]:
            raise ValueError(f"anonymous input changed: {case['episode_id']}")
        row: dict[str, Any] = {
            "episode_id": case["episode_id"],
            "seed": case["seed"],
            "instance": case["fault_instance"],
            "label": case["technical_label"],
        }
        for model, modes in RUNS.items():
            for mode, run_name in modes.items():
                run = diagnosis / run_name
                freeze = read(run / "freeze.json")
                if sha(run / "freeze.json") != manifest["runs"][run_name]["freeze_sha256"]:
                    raise ValueError(f"diagnosis run changed: {run_name}")
                if mode not in freeze["modes"]:
                    raise ValueError(f"mode not frozen: {run_name}/{mode}")
                target = run / mode / case["episode_id"]
                attempt, answer = read(target / "attempt.json"), read(target / "answer.json")
                checks = (
                    (source / "probes" / case["episode_id"] / "probes.json").read_text()
                    if mode.startswith("probe")
                    else "null"
                )
                if mode.startswith("probe") and sha(source / "probes" / case["episode_id"] / "probes.json") != next(
                    item["probes_sha256"] for item in freeze["cases"] if item["episode_id"] == case["episode_id"]
                ):
                    raise ValueError(f"controlled checks changed: {case['episode_id']}")
                if (
                    attempt.get("exit_code") != 0
                    or attempt.get("tool_call")
                    or attempt.get("output_validation_error")
                    or attempt.get("usage") is None
                    or attempt.get("input_sha256") != case["input_sha256"]
                    or attempt.get("checks_sha256") != hashlib.sha256(checks.encode()).hexdigest()
                    or attempt.get("prompt_sha256") != freeze["prompt_sha256"]
                    or attempt.get("schema_sha256") != freeze["schema_sha256"]
                    or attempt.get("model") != freeze["model"]
                    or attempt.get("effort") != freeze["effort"]
                    or answer.get("cause") not in (*LAYERS, "uncertain")
                ):
                    raise ValueError(f"invalid diagnosis: {run_name}/{mode}/{case['episode_id']}")
                row[f"{model}:{mode}"] = answer
        rows.append(row)
    for run_name in manifest["runs"]:
        for mode in read(diagnosis / run_name / "freeze.json")["modes"]:
            summary = read(diagnosis / run_name / mode / "summary.json")
            if (
                summary["status"] != "complete"
                or summary["completed"] != len(rows)
                or summary["expected"] != len(rows)
            ):
                raise ValueError(f"mode incomplete: {run_name}/{mode}")
    return rows


def summarize(rows: list[dict[str, Any]], source: Path, diagnosis: Path) -> dict[str, Any]:
    faults = [row for row in rows if row["label"] in LAYERS]
    controls = [row for row in rows if row["label"] == "no_injection"]
    if not faults or not controls:
        raise ValueError("faults or no-injection controls missing")
    mode_results = {}
    for model, modes in RUNS.items():
        for mode in modes:
            key = f"{model}:{mode}"
            mode_results[key] = {
                "correct": sum(row[key]["cause"] == row["label"] for row in faults),
                "fault_total": len(faults),
                "macro_f1": macro_f1(faults, key),
                "calibration_ece_5_bins": calibration_error(faults, key),
                "by_layer": {
                    layer: {
                        "correct": sum(row[key]["cause"] == layer for row in faults if row["label"] == layer),
                        "total": sum(row["label"] == layer for row in faults),
                    }
                    for layer in LAYERS
                },
                "abstentions": sum(row[key]["cause"] == "uncertain" for row in faults),
                "harness_vs_weights_correct": {
                    "correct": sum(
                        row[key]["cause"] == row["label"] for row in faults if row["label"] in ("harness", "weights")
                    ),
                    "total": sum(row["label"] in ("harness", "weights") for row in faults),
                },
                "by_instance": {
                    instance: {
                        "correct": sum(
                            row[key]["cause"] == row["label"] for row in faults if row["instance"] == instance
                        ),
                        "total": sum(row["instance"] == instance for row in faults),
                    }
                    for instance in sorted({row["instance"] for row in faults})
                },
                "control_unsupported_harness_or_controller": sum(
                    row[key]["cause"] in ("harness", "controller") for row in controls
                ),
                "control_called_weights": sum(row[key]["cause"] == "weights" for row in controls),
                "confusion": {
                    layer: dict(Counter(row[key]["cause"] for row in rows if row["label"] == layer))
                    for layer in (*LAYERS, "no_injection")
                },
                "usage": read(diagnosis / modes[mode] / mode / "summary.json"),
            }
    comparisons = {}
    for model in RUNS:
        for name, left, right in (
            ("probe_minus_read", "read", "probe"),
            ("probe_fewshot_minus_read_fewshot", "read+fewshot", "probe+fewshot"),
        ):
            try:
                comparisons[f"{model}:{name}"] = paired_interval(rows, f"{model}:{left}", f"{model}:{right}")
            except ValueError as error:  # noqa: PERF203
                comparisons[f"{model}:{name}"] = {"status": "unavailable", "reason": str(error)}
    primary = comparisons["sol:probe_minus_read"]
    majority = max(sum(row["label"] == layer for row in faults) for layer in LAYERS)
    return {
        "version": "tank183-v4-multifamily-diagnosis-score-v1",
        "sample_freeze_sha256": sha(source / "sample-freeze.json"),
        "diagnosis_manifest_sha256": sha(diagnosis / "manifest.json"),
        "faults": len(faults),
        "controls": len(controls),
        "independent_seed_blocks": len({row["seed"] for row in rows}),
        "fault_instances": dict(Counter(row["instance"] for row in faults)),
        "majority_baseline_correct": majority,
        "mode_results": mode_results,
        "paired_comparisons": comparisons,
        "technical_component_go_pending_human_review": (
            primary.get("instance_and_reset_group_bootstrap_95pct", [0])[0] > 0
            and mode_results["sol:probe"]["correct"] > majority
        ),
        "full_issue_go_no_go": "pending blinded human decision review",
        "limits": [
            "A same-state changed layer is not proof that the change caused the final loss.",
            "Only the sampled fault instances and 20 unexposed reset blocks support this comparison.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-blind", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--out-private", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    args = parser.parse_args()
    if args.out_private.exists() or args.out_summary.exists():
        raise FileExistsError("score output already exists")
    rows = load_answers(args.source_blind, args.diagnosis)
    result = summarize(rows, args.source_blind, args.diagnosis)
    args.out_private.write_text(json.dumps({"summary": result, "per_case": rows}, indent=2) + "\n")
    args.out_summary.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("faults", "controls", "paired_comparisons", "technical_component_go_pending_human_review")
            }
        )
    )


if __name__ == "__main__":
    main()
