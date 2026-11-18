"""Audit probability and action changes for each edited Tank War checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(source: Path) -> dict[str, Any]:
    freeze = read(source / "sample-freeze.json")
    summary = read(source / "probes/summary.json")
    audit = read(source / "probe-audit.json")
    cases = freeze["cases"]
    if (
        freeze["status"] != "frozen_before_model_answers"
        or summary["status"] != "complete"
        or summary["cases"] != len(cases)
        or summary["sample_freeze_sha256"] != sha(source / "sample-freeze.json")
        or audit["valid"] != len(cases)
    ):
        raise ValueError("frozen cases, probe completion, or signature audit differs")
    hashes = {item["episode_id"]: item["probe_sha256"] for item in summary["records"]}
    if set(hashes) != {case["episode_id"] for case in cases}:
        raise ValueError("probe IDs differ from frozen cases")
    by_instance: dict[str, list[tuple[dict[str, Any], list[dict[str, Any]]]]] = defaultdict(list)
    controls: list[dict[str, Any]] = []
    for case in cases:
        path = source / "probes" / case["episode_id"] / "probes.json"
        if sha(path) != hashes[case["episode_id"]]:
            raise ValueError(f"probe changed: {case['episode_id']}")
        indices = read(path)["indices"]
        if [row["decision_index"] for row in indices] != list(range(case["decisions"])):
            raise ValueError(f"incomplete decision coverage: {case['episode_id']}")
        if case["technical_label"] == "weights":
            reference = case["technical_reference_index"]
            if not isinstance(reference, int) or not 0 <= reference < len(indices):
                raise ValueError(f"invalid weight reference: {case['episode_id']}")
            row = indices[reference]
            if not row["same_model_input"] or not row["selected_action_changed"]:
                raise ValueError(f"weight reference lacks same-input action change: {case['episode_id']}")
            by_instance[case["fault_instance"]].append((row, indices))
        elif case["technical_label"] == "no_injection":
            controls.extend(indices)
    result = {}
    for instance, games in sorted(by_instance.items()):
        references = [row for row, _ in games]
        all_rows = [row for _, indices in games for row in indices]
        deltas = [row["max_abs_probability_change"] for row in references]
        if any(delta is None for delta in deltas):
            raise ValueError(f"missing probability delta for {instance}")
        result[instance] = {
            "games": len(games),
            "reference_same_input": sum(row["same_model_input"] for row in references),
            "reference_selected_action_changed": sum(row["selected_action_changed"] for row in references),
            "reference_delta_gt_0p01": sum(delta > 0.01 for delta in deltas),
            "reference_probability_delta": {
                "min": min(deltas),
                "median": statistics.median(deltas),
                "max": max(deltas),
            },
            "all_decisions": len(all_rows),
            "all_decisions_delta_gt_0p01": sum((row["max_abs_probability_change"] or 0) > 0.01 for row in all_rows),
            "all_decisions_selected_action_changed": sum(row["selected_action_changed"] for row in all_rows),
        }
    return {
        "version": "tank183-multifamily-weight-probe-analysis-v1",
        "sample_freeze_sha256": sha(source / "sample-freeze.json"),
        "probe_summary_sha256": sha(source / "probes/summary.json"),
        "probe_audit_sha256": sha(source / "probe-audit.json"),
        "edited_checkpoint_instances": result,
        "control_decisions": {
            "count": len(controls),
            "delta_gt_0p01": sum((row["max_abs_probability_change"] or 0) > 0.01 for row in controls),
            "selected_action_changed": sum(row["selected_action_changed"] for row in controls),
        },
        "scope": "Same-state checkpoint signal; not proof that the edit caused each final loss",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-blind", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = analyze(args.source_blind)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
