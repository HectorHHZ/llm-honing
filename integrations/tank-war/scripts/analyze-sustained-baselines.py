"""Report cause baselines and calibration for a sealed Tank War blind run."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

LAYERS = ("harness", "weights", "controller")
MODES = ("read", "probe")


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def macro_f1(rows: list[dict[str, Any]], mode: str) -> float:
    scores = []
    for layer in LAYERS:
        true_positive = sum(row["label"] == layer and row[mode] == layer for row in rows)
        false_positive = sum(row["label"] != layer and row[mode] == layer for row in rows)
        false_negative = sum(row["label"] == layer and row[mode] != layer for row in rows)
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return sum(scores) / len(scores)


def analyze(source: Path, diagnosis: Path) -> dict[str, Any]:
    freeze = read(source / "sample-freeze.json")
    run = read(diagnosis / "freeze.json")
    if freeze["status"] != "frozen_before_model_answers" or run["source_sample_freeze_sha256"] != sha(
        source / "sample-freeze.json"
    ):
        raise ValueError("blind and diagnosis freezes differ")
    rows = []
    for case in freeze["cases"]:
        row: dict[str, Any] = {"label": case["technical_label"]}
        for mode in MODES:
            path = diagnosis / mode / case["episode_id"]
            attempt, answer = read(path / "attempt.json"), read(path / "answer.json")
            if (
                attempt.get("exit_code") != 0
                or attempt.get("tool_call")
                or attempt.get("output_validation_error")
                or attempt.get("usage") is None
                or attempt.get("input_sha256") != case["input_sha256"]
                or answer.get("cause") not in (*LAYERS, "uncertain")
            ):
                raise ValueError(f"invalid diagnosis: {mode}/{case['episode_id']}")
            confidence = answer.get("confidence")
            if not isinstance(confidence, (float, int)) or not 0 <= confidence <= 1:
                raise ValueError(f"invalid confidence: {mode}/{case['episode_id']}")
            row[mode] = answer["cause"]
            row[f"{mode}_confidence"] = float(confidence)
        rows.append(row)
    faults = [row for row in rows if row["label"] in LAYERS]
    controls = [row for row in rows if row["label"] == "no_injection"]
    class_counts = Counter(row["label"] for row in faults)
    if len(controls) != 10 or class_counts != Counter(dict.fromkeys(LAYERS, 10)):
        raise ValueError("unexpected sealed class balance")
    for mode in MODES:
        usage = read(diagnosis / mode / "summary.json")
        if usage["status"] != "complete" or usage["completed"] != len(rows):
            raise ValueError(f"incomplete mode: {mode}")
    modes = {}
    for mode in MODES:
        brier = sum((row[f"{mode}_confidence"] - int(row[mode] == row["label"])) ** 2 for row in faults) / len(faults)
        bins = []
        for low in (0.0, 0.2, 0.4, 0.6, 0.8):
            group = [
                row
                for row in faults
                if (low <= row[f"{mode}_confidence"] < low + 0.2) or (low == 0.8 and row[f"{mode}_confidence"] == 1)
            ]
            if group:
                bins.append(
                    {
                        "range": [low, low + 0.2],
                        "count": len(group),
                        "mean_confidence": sum(row[f"{mode}_confidence"] for row in group) / len(group),
                        "accuracy": sum(row[mode] == row["label"] for row in group) / len(group),
                    }
                )
        ece = sum(item["count"] * abs(item["mean_confidence"] - item["accuracy"]) for item in bins) / len(faults)
        modes[mode] = {
            "correct": sum(row[mode] == row["label"] for row in faults),
            "fault_total": len(faults),
            "macro_f1": macro_f1(faults, mode),
            "harness_vs_weights_correct": sum(
                row[mode] == row["label"] for row in faults if row["label"] in ("harness", "weights")
            ),
            "harness_vs_weights_total": 20,
            "abstentions": sum(row[mode] == "uncertain" for row in faults),
            "brier_for_stated_cause_confidence": brier,
            "expected_calibration_error_5_bins": ece,
            "calibration_bins": bins,
            "no_injection_unsupported_harness_or_controller": sum(
                row[mode] in ("harness", "controller") for row in controls
            ),
        }
    return {
        "version": "tank183-sustained-baselines-v1",
        "sample_freeze_sha256": sha(source / "sample-freeze.json"),
        "diagnosis_freeze_sha256": sha(diagnosis / "freeze.json"),
        "class_counts": dict(class_counts),
        "baselines": {
            "uniform_random_expected_correct": 10,
            "majority_correct": 10,
            **{f"always_{layer}_correct": 10 for layer in LAYERS},
            "fault_total": 30,
            "blind_human_cause": "unavailable: reviewer marked decisions, not cause",
        },
        "modes": modes,
        "scope": "one fault family per layer; controls have no constructed-fault gold cause",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-blind", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.source_blind, args.diagnosis)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"baselines": result["baselines"], "modes": result["modes"]}))


if __name__ == "__main__":
    main()
