"""Score frozen System 0 cause and decision-window answers without relabeling losses."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def ratio(good: int, total: int) -> str:
    return f"{good}/{total}" if total else "0/0"


def paired_interval(records: list[dict], key: str) -> tuple[float, float] | None:
    strata = [[row for row in records if row["layer"] == layer] for layer in ("harness", "weights", "controller")]
    if any(not group for group in strata):
        return None
    rng = random.Random(183)
    samples = []
    for _ in range(10_000):
        values = [
            int(pick["probe"].get(key, False)) - int(pick["read"].get(key, False))
            for group in strata
            for pick in rng.choices(group, k=len(group))
        ]
        samples.append(sum(values) / len(values))
    samples.sort()
    return samples[249], samples[9749]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    freeze = read(out / "sample-freeze.json")
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != freeze["scoring_script_sha256"]:
        raise ValueError("scoring script changed after blind freeze")
    cases = freeze["cases"]
    records = []
    for case in cases:
        row = {
            key: case.get(key)
            for key in (
                "episode_id",
                "layer",
                "seed",
                "seat",
                "technical_reference_index",
                "outcome_improved_without_fault",
            )
        }
        for mode in ("read", "probe"):
            target = out / mode / case["episode_id"]
            attempt_path, answer_path = target / "attempt.json", target / "answer.json"
            if not attempt_path.exists() or not answer_path.exists():
                row[mode] = {"valid": False, "reason": "missing"}
                continue
            attempt, answer = read(attempt_path), read(answer_path)
            valid = (
                attempt.get("exit_code") == 0
                and not attempt.get("tool_call")
                and not attempt.get("output_validation_error")
                and attempt.get("usage") is not None
            )
            center, reference = answer["decision_index"], case["technical_reference_index"]
            result = {
                "valid": valid,
                "answer": answer,
                "seconds": attempt["elapsed_seconds"],
                "usage": attempt.get("usage"),
                "technical_window_hit": bool(
                    valid and reference is not None and center is not None and abs(center - reference) <= 2
                ),
                "technical_layer_hit": bool(valid and case["layer"] != "control" and answer["cause"] == case["layer"]),
                "control_false_alarm": bool(
                    valid and case["layer"] == "control" and answer["cause"] in {"harness", "controller"}
                ),
            }
            result["joint_hit"] = result["technical_window_hit"] and result["technical_layer_hit"]
            row[mode] = result
        records.append(row)
    by_layer = {}
    for layer in ("harness", "weights", "controller", "control"):
        subset = [record for record in records if record["layer"] == layer]
        by_layer[layer] = {"n": len(subset)}
        for mode in ("read", "probe"):
            by_layer[layer][mode] = {
                key: ratio(sum(bool(row[mode].get(key)) for row in subset), len(subset))
                for key in ("technical_window_hit", "technical_layer_hit", "joint_hit", "control_false_alarm")
            }
            by_layer[layer][mode]["valid_attempts"] = ratio(
                sum(row[mode].get("valid") is True for row in subset), len(subset)
            )
    faults = [record for record in records if record["layer"] != "control"]
    causal = [record for record in faults if record["outcome_improved_without_fault"] is True]
    paired = {
        key: {
            "read": sum(bool(row["read"].get(key)) for row in faults),
            "probe": sum(bool(row["probe"].get(key)) for row in faults),
            "probe_minus_read": sum(bool(row["probe"].get(key)) - bool(row["read"].get(key)) for row in faults),
            "denominator": len(faults),
            "paired_difference_95pct_stratified_bootstrap": paired_interval(faults, key),
        }
        for key in ("technical_window_hit", "technical_layer_hit", "joint_hit")
    }
    causal_summary = {
        mode: {
            "joint_hit": ratio(sum(bool(row[mode].get("joint_hit")) for row in causal), len(causal)),
            "window_hit": ratio(sum(bool(row[mode].get("technical_window_hit")) for row in causal), len(causal)),
            "layer_hit": ratio(sum(bool(row[mode].get("technical_layer_hit")) for row in causal), len(causal)),
        }
        for mode in ("read", "probe")
    }
    errors = defaultdict(list)
    for row in faults:
        if not row["probe"].get("joint_hit"):
            errors[row["layer"]].append(
                {
                    "episode_id": row["episode_id"],
                    "reference_index": row["technical_reference_index"],
                    "reported_index": row["probe"].get("answer", {}).get("decision_index"),
                    "reported_cause": row["probe"].get("answer", {}).get("cause"),
                    "outcome_improved_without_fault": row["outcome_improved_without_fault"],
                }
            )
    report = {
        "cases": len(records),
        "fault_cases": len(faults),
        "causal_subset": len(causal),
        "by_layer": by_layer,
        "paired": paired,
        "causal": causal_summary,
        "probe_errors": dict(errors),
        "records": records,
    }
    (out / "score.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {key: report[key] for key in ("cases", "fault_cases", "causal_subset", "by_layer", "paired", "causal")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
