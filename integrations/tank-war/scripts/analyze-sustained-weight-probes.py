"""Summarize the weight-side signal in one sealed Tank War blind run."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(source: Path, diagnosis: Path) -> dict[str, Any]:
    freeze = read(source / "sample-freeze.json")
    diagnosis_freeze = read(diagnosis / "freeze.json")
    probe_summary = read(source / "probes/summary.json")
    probe_audit = read(source / "probe-audit.json")
    if (
        freeze["status"] != "frozen_before_model_answers"
        or diagnosis_freeze["source_sample_freeze_sha256"] != sha(source / "sample-freeze.json")
        or probe_summary["status"] != "complete"
        or probe_summary["cases"] != len(freeze["cases"])
        or probe_audit["valid"] != len(freeze["cases"])
    ):
        raise ValueError("blind set, probes, or diagnosis does not match the sealed run")
    for mode in ("read", "probe"):
        summary = read(diagnosis / mode / "summary.json")
        if summary["status"] != "complete" or summary["completed"] != len(freeze["cases"]):
            raise ValueError(f"diagnosis mode incomplete: {mode}")
    weights = [case for case in freeze["cases"] if case["technical_label"] == "weights"]
    controls = [case for case in freeze["cases"] if case["technical_label"] == "no_injection"]
    if not weights or not controls:
        raise ValueError("weight cases or controls missing")
    reference_rows = []
    all_weight_rows = []
    control_rows = []
    answer_counts = {mode: {"weights_correct": 0, "controls_called_weights": 0} for mode in ("read", "probe")}
    for case in (*weights, *controls):
        probe = read(source / "probes" / case["episode_id"] / "probes.json")
        rows = probe["indices"]
        if [row["decision_index"] for row in rows] != list(range(case["decisions"])):
            raise ValueError("probe decision coverage differs from frozen game")
        if case["technical_label"] == "weights":
            reference = case["technical_reference_index"]
            if not isinstance(reference, int) or not 0 <= reference < len(rows):
                raise ValueError("weight reference index out of range")
            reference_rows.append(rows[reference])
            all_weight_rows.extend(rows)
        else:
            control_rows.extend(rows)
        for mode in ("read", "probe"):
            target = diagnosis / mode / case["episode_id"]
            attempt, answer = read(target / "attempt.json"), read(target / "answer.json")
            if (
                attempt.get("exit_code") != 0
                or attempt.get("tool_call")
                or attempt.get("output_validation_error")
                or attempt.get("usage") is None
                or attempt.get("input_sha256") != case["input_sha256"]
            ):
                raise ValueError(f"invalid diagnosis: {mode}/{case['episode_id']}")
            answer_counts[mode]["weights_correct"] += int(
                case["technical_label"] == "weights" and answer["cause"] == "weights"
            )
            answer_counts[mode]["controls_called_weights"] += int(
                case["technical_label"] == "no_injection" and answer["cause"] == "weights"
            )
    if not all(row["same_model_input"] and row["selected_action_changed"] for row in reference_rows):
        raise ValueError("a weight reference lacks the expected same-input choice change")
    deltas = [row["max_abs_probability_change"] for row in reference_rows]
    if any(delta is None for delta in deltas):
        raise ValueError("missing weight probability comparison")
    return {
        "version": "tank183-weight-probe-analysis-v1",
        "sample_freeze_sha256": sha(source / "sample-freeze.json"),
        "probe_summary_sha256": sha(source / "probes/summary.json"),
        "diagnosis_freeze_sha256": sha(diagnosis / "freeze.json"),
        "weight_cases": len(weights),
        "no_injection_controls": len(controls),
        "weight_reference": {
            "same_model_input": sum(row["same_model_input"] for row in reference_rows),
            "selected_action_changed": sum(row["selected_action_changed"] for row in reference_rows),
            "probability_delta_gt_0p01": sum(delta > 0.01 for delta in deltas),
            "probability_delta_min": min(deltas),
            "probability_delta_median": statistics.median(deltas),
            "probability_delta_max": max(deltas),
        },
        "all_weight_decisions": {
            "count": len(all_weight_rows),
            "same_input_and_delta_gt_0p01": sum(
                row["same_model_input"] and (row["max_abs_probability_change"] or 0) > 0.01 for row in all_weight_rows
            ),
            "selected_action_changed": sum(row["selected_action_changed"] for row in all_weight_rows),
        },
        "control_decisions": {
            "count": len(control_rows),
            "delta_gt_0p01": sum((row["max_abs_probability_change"] or 0) > 0.01 for row in control_rows),
            "selected_action_changed": sum(row["selected_action_changed"] for row in control_rows),
        },
        "diagnosis": answer_counts,
        "scope": "one reverse-direction checkpoint family; same-state signal, not proof of loss causation",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-blind", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.source_blind, args.diagnosis)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
