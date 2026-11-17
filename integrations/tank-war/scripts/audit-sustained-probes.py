"""Check sustained-fault probe signatures before blinded diagnosis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    freeze = read(args.out / "sample-freeze.json")
    if read(args.out / "probes" / "summary.json")["cases"] != len(freeze["cases"]):
        raise ValueError("probe coverage is incomplete")
    records = []
    for case in freeze["cases"]:
        probe = read(args.out / "probes" / case["episode_id"] / "probes.json")
        indices = probe["indices"]
        if [row["decision_index"] for row in indices] != list(range(case["decisions"])):
            raise ValueError(f"decision coverage differs: {case['episode_id']}")
        layer = case["technical_label"]
        if layer == "no_injection":
            valid = all(
                row["same_model_input"]
                and not row["option_set_changed"]
                and (row["max_abs_probability_change"] or 0) <= 0.01
                and not row["selected_action_changed"]
                and not row["command_changed"]
                and not row["execution_changed"]
                for row in indices
            )
        else:
            row = indices[case["technical_reference_index"]]
            if layer == "harness":
                valid = not row["same_model_input"] and row["selected_action_changed"]
            elif layer == "weights":
                valid = (
                    row["same_model_input"]
                    and row["selected_action_changed"]
                    and (row["choice_changed"] or (row["max_abs_probability_change"] or 0) > 0.01)
                )
            else:
                valid = (
                    row["same_model_input"]
                    and not row["selected_action_changed"]
                    and row["command_changed"]
                    and not row["option_set_changed"]
                    and (row["max_abs_probability_change"] or 0) <= 0.01
                )
        records.append({"episode_id": case["episode_id"], "layer": layer, "valid": bool(valid)})
    summary = {
        "cases": len(records),
        "valid": sum(row["valid"] for row in records),
        "invalid": [row["episode_id"] for row in records if not row["valid"]],
        "by_layer": {
            layer: {
                "valid": sum(row["valid"] for row in records if row["layer"] == layer),
                "total": sum(row["layer"] == layer for row in records),
            }
            for layer in ("harness", "weights", "controller", "no_injection")
        },
        "records": records,
    }
    (args.out / "probe-audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("cases", "valid", "invalid", "by_layer")}))
    if summary["invalid"]:
        raise ValueError("probe signature audit failed")


if __name__ == "__main__":
    main()
