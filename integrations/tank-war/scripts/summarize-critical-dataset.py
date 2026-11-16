"""Write a machine-readable #183 data card from immutable match and blind logs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--blind", type=Path, required=True)
    args = parser.parse_args()
    summary = read(args.inventory / "summary.json")
    inventory = rows(args.inventory / "attempts.jsonl")
    retained = {row["run"] for row in rows(args.inventory / "losses.jsonl")}
    roots = (
        "fresh-pairs-v1",
        "controller-extra-v1",
        "weights-extra-v1",
        "harness-selected-fire-v3",
        "harness-selected-fire-extra-v4",
        "harness-selected-fire-supplement-v5",
        "weights-supplement-v2",
        "controller-supplement-v2",
        "weights-supplement-extra-v3",
    )
    per_root = {}
    all_pairs = []
    for name in roots:
        root = args.root / name
        evidence = [read(path) for path in root.rglob("pair_evidence.json")] if root.exists() else []
        journal = rows(root / "attempts.jsonl") if (root / "attempts.jsonl").exists() else []
        per_root[name] = {
            "planned_pairs": len(read(root / "freeze.json")["entries"]) if (root / "freeze.json").exists() else 0,
            "attempts_journaled": len(journal),
            "completed_pairs": len(evidence),
            "attempt_errors": sum(
                "error" in item or any(key.endswith("_returncode") and value for key, value in item.items())
                for item in journal
            ),
        }
        all_pairs.extend(evidence)
    per_layer = {}
    for layer in ("harness", "weights", "controller"):
        cases = [case for case in all_pairs if case["layer"] == layer]
        losses = [case for case in cases if case["fault_loss"]]
        eligible = [
            case
            for case in losses
            if case["triggered"]
            and case.get("same_prefix")
            and case.get("layer_isolated")
            and case.get("local_action_changed")
            and (layer != "harness" or case.get("selected_action_changed"))
        ]
        per_layer[layer] = {
            "completed_pairs": len(cases),
            "fault_losses": len(losses),
            "retained_fault_losses": sum(case["fault_run"] in retained for case in losses),
            "triggered": sum(case["triggered"] for case in cases),
            "local_action_changed": sum(case.get("local_action_changed", False) for case in cases),
            "technically_eligible_losses": len(eligible),
            "paired_clean_outcome_better": sum(case.get("outcome_improved_without_fault", False) for case in eligible),
        }
    old = read(args.root / "old-harness-verification-v1" / "summary.json")
    old_cases = [read(path) for path in (args.root / "old-harness-verification-v1").glob("s*/summary.json")]
    old_controls = args.root.parent / "v0-pilot" / "formal-controls-final-v1" / "games"
    old_better = 0
    for case in old_cases:
        if case["status"] != "verified":
            continue
        control = old_controls / f"s{case['seed']}-{case['seat']}" / "result.json"
        old_better += control.exists() and read(control)["outcome"][case["seat"]] > -1
    old_harness = {
        "complete_archived_losses": old["frozen_entries"],
        "same_state_checked": old["completed_checks"],
        "verified_selected_action_difference": old["verified"],
        "verified_unique_seeds": old["verified_unique_seeds"],
        "paired_clean_outcome_better": old_better,
    }
    recovery_root = args.root / "weight3-recovery-v1"
    recovery_results = [read(path) for path in recovery_root.rglob("result.json")]
    recovery = {
        "planned_missing_games": len(read(recovery_root / "freeze.json")["pending"]),
        "completed_games": len(recovery_results),
        "losses": sum(result["outcome"][result["seat"]] == -1 for result in recovery_results),
        "journal_entries": len(rows(recovery_root / "attempts.jsonl")),
    }
    blind_freeze = read(args.blind / "sample-freeze.json")
    replay = (
        read(args.blind / "replays" / "summary.json") if (args.blind / "replays" / "summary.json").exists() else None
    )
    report = {
        "dataset_version": args.root.name,
        "utilization": {
            "denominator_complete_unique_losses": summary["complete_unique_losses"],
            "retained_complete_unique_losses": summary["retained_complete_losses"],
            "ratio": summary["utilization_of_complete_losses"],
            "aligned_trace_losses": summary["aligned_trace_losses"],
            "state_hash_verified_losses": summary["state_hash_verified_losses"],
            "full_replay_gate": replay,
            "limitation": (
                "Full deterministic replay was required for blind cases; the complete-loss inventory was checked "
                "for file completeness, alignment, and every state hash, "
                "not full-engine replay of every archived match."
            ),
        },
        "per_root": per_root,
        "new_pairs_by_layer": per_layer,
        "archived_harness": old_harness,
        "edited_checkpoint_recovery": recovery,
        "unusable_loss_records": [
            {key: row.get(key) for key in ("run", "error", "missing", "trace_aligned", "state_hashes_valid")}
            for row in inventory
            if row.get("outcome") == -1
            and not row.get("excluded")
            and (
                row.get("error")
                or row.get("missing")
                or not row.get("trace_aligned")
                or not row.get("state_hashes_valid")
            )
        ],
        "blind_sample": {
            "cases": len(blind_freeze["cases"]),
            "by_layer": dict(Counter(case["layer"] for case in blind_freeze["cases"])),
            "selection_counts": blind_freeze["selection_counts"],
        },
        "neutral_probe_audit": (
            read(args.blind / "probe-audit.json") if (args.blind / "probe-audit.json").exists() else None
        ),
        "diagnosis_score": read(args.blind / "score.json") if (args.blind / "score.json").exists() else None,
    }
    (args.root / "DATA_CARD.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "utilization": report["utilization"],
                "new_pairs_by_layer": per_layer,
                "archived_harness": old_harness,
                "blind_sample": report["blind_sample"],
            }
        )
    )


if __name__ == "__main__":
    main()
