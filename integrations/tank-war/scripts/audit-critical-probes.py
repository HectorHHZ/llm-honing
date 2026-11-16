"""Audit neutral repeat signatures against the frozen private construction record."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def changed(row: dict) -> bool:
    probability = row["max_abs_probability_change"]
    return bool(
        not row["same_model_input"]
        or row["option_set_changed"]
        or (probability is not None and probability > 0.01)
        or row["selected_action_changed"]
        or row["command_changed"]
        or row["execution_changed"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    freeze = read(args.out / "sample-freeze.json")
    if read(args.out / "probes" / "summary.json")["cases"] != len(freeze["cases"]):
        raise ValueError("probe set incomplete")
    records = []
    for case in freeze["cases"]:
        data = read(args.out / "probes" / case["episode_id"] / "probes.json")
        indices = data["indices"]
        index = case["technical_reference_index"]
        target = indices[index] if index is not None else None
        off_target = [row["decision_index"] for row in indices if row["decision_index"] != index and changed(row)]
        layer = case["layer"]
        if layer == "harness":
            expected = bool(
                target
                and not target["same_model_input"]
                and not target["same_question_text"]
                and target["selected_action_changed"]
            )
        elif layer == "weights":
            expected = bool(
                target
                and target["same_model_input"]
                and target["selected_action_changed"]
                and (target["choice_changed"] or (target["max_abs_probability_change"] or 0) > 0.01)
            )
        elif layer == "controller":
            expected = bool(
                target
                and target["same_model_input"]
                and not target["selected_action_changed"]
                and target["command_changed"]
                and not target["option_set_changed"]
                and (target["max_abs_probability_change"] or 0) <= 0.01
            )
        else:
            expected = not any(changed(row) for row in indices)
        records.append(
            {
                "episode_id": case["episode_id"],
                "layer": layer,
                "reference_index": index,
                "expected_signature": expected,
                "off_target_differences": off_target,
            }
        )
    summary = {
        "cases": len(records),
        "signature_pass_by_layer": {
            layer: {
                "pass": sum(row["expected_signature"] for row in records if row["layer"] == layer),
                "total": sum(row["layer"] == layer for row in records),
            }
            for layer in ("harness", "weights", "controller", "control")
        },
        "off_target_cases": sum(bool(row["off_target_differences"]) for row in records),
        "off_target_by_layer": dict(Counter(row["layer"] for row in records if row["off_target_differences"])),
        "records": records,
    }
    (args.out / "probe-audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: summary[key]
                for key in ("cases", "signature_pass_by_layer", "off_target_cases", "off_target_by_layer")
            }
        )
    )


if __name__ == "__main__":
    main()
