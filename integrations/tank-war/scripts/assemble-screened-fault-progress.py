"""Index every completed final game from screened Tank War fault instances.

This is an evaluation-only progress inventory. A missing same-state clean pair
keeps the game in the index but cannot yield a verified cause label.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any

SOURCE = Path(__file__).with_name("assemble-sustained-fault-dataset.py")
SPEC = importlib.util.spec_from_file_location("tank_sustained_assembler", SOURCE)
assert SPEC and SPEC.loader
common = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(common)


def clean_index(roots: list[Path]) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    for root in roots:
        for result in root.glob("s*-*/result.json"):
            run = result.parent
            prior = runs.get(run.name)
            if prior is not None and common.digest(prior / "result.json") != common.digest(result):
                raise ValueError(f"conflicting clean result for {run.name}")
            runs[run.name] = run
    return runs


def unpaired_record(fault: Path, layer: str, instance: str) -> dict[str, Any]:
    result = common.read(fault / "result.json")
    seat = result["seat"]
    return {
        "sample_id": hashlib.sha256(f"{instance}:{result['seed']}:{seat}".encode()).hexdigest()[:24],
        "seed": result["seed"],
        "seat": seat,
        "layer": layer,
        "instance": instance,
        "fault_run": str(fault),
        "clean_run": None,
        "source_hashes": {
            "fault_result": common.digest(fault / "result.json"),
            "fault_manifest": common.digest(fault / "manifest.json"),
        },
        "fault_outcome": result["outcome"][seat],
        "clean_outcome": None,
        "fault_score_minus_clean": None,
        "fault_loss": result["outcome"][seat] == -1,
        "verified_local_effect": False,
        "technical_reference": {"verified": False, "reason": "missing same-build clean counterpart"},
        "cause_label": None,
        "annotation_status": "uncertain",
    }


def index_instance(setting: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    layer, instance = setting["layer"], setting["instance"]
    screen, screen_seeds = common.screen_check(Path(setting["screen"]), setting["screen_sha256"])
    if screen["candidate"]["layer"] != layer or screen["candidate"]["instance"] != instance:
        raise ValueError(f"screen names a different instance: {instance}")
    screen_manifest = common.read(next(Path(setting["screen_fault_games"]).glob("s*-*/manifest.json")))
    cleans = clean_index([Path(root) for root in setting["clean_games"]])
    faults = sorted(Path(setting["fault_games"]).glob("s*-*/result.json"))
    if not faults:
        raise ValueError(f"no final games for {instance}")
    records = []
    for result_path in faults:
        fault = result_path.parent
        result = common.read(result_path)
        if result["seed"] in screen_seeds:
            raise ValueError(f"final game reuses a validation seed: {fault}")
        clean = cleans.get(fault.name)
        record = (
            common.label_pair(clean, fault, layer, screen_manifest)
            if clean is not None
            else unpaired_record(fault, layer, instance)
        )
        record["instance"] = instance
        record["sample_id"] = hashlib.sha256(f"{instance}:{result['seed']}:{result['seat']}".encode()).hexdigest()[:24]
        records.append(record)
    counts = Counter()
    for record in records:
        counts["fault_losses"] += record["fault_loss"]
        counts["verified_fault_losses"] += record["cause_label"] == layer
        counts["unpaired_games"] += record["clean_run"] is None
        counts["verified_local_effects"] += record["verified_local_effect"]
    return records, {
        "layer": layer,
        "instance": instance,
        "screen_sha256": setting["screen_sha256"],
        "screen_mean_fault_minus_clean": screen["mean_fault_minus_clean"],
        "screen_ci95": screen["ci95"],
        "completed_final_games": len(records),
        **dict(counts),
        "uncertain_reasons": dict(Counter(
            record["technical_reference"]["reason"]
            for record in records
            if not record["verified_local_effect"]
        )),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    protocol = common.read(args.protocol)
    records: list[dict[str, Any]] = []
    reports = []
    seen = set()
    for setting in protocol["instances"]:
        if setting["instance"] in seen:
            raise ValueError(f"duplicate instance: {setting['instance']}")
        seen.add(setting["instance"])
        rows, report = index_instance(setting)
        records.extend(rows)
        reports.append(report)
    if len({record["sample_id"] for record in records}) != len(records):
        raise ValueError("duplicate fault sample ID")
    controls = clean_index([Path(protocol["clean_controls"])])
    control_rows = []
    for name, run in sorted(controls.items()):
        result = common.read(run / "result.json")
        control_rows.append({
            "sample_id": hashlib.sha256(f"control:{result['seed']}:{result['seat']}".encode()).hexdigest()[:24],
            "seed": result["seed"],
            "seat": result["seat"],
            "source_run": str(run),
            "outcome": result["outcome"][result["seat"]],
            "condition": "no_injection",
            "cause_label": None,
            "result_sha256": common.digest(run / "result.json"),
        })
    args.out.mkdir(parents=True)
    (args.out / "protocol.json").write_bytes(args.protocol.read_bytes())
    (args.out / "fault-games.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    (args.out / "controls.jsonl").write_text("".join(json.dumps(row) + "\n" for row in control_rows))
    summary = {
        "version": protocol["version"],
        "status": "progress_inventory_not_blind_test",
        "protocol_sha256": common.digest(args.out / "protocol.json"),
        "instances": reports,
        "fault_games": len(records),
        "fault_losses_retained": sum(record["fault_loss"] for record in records),
        "verified_fault_losses": sum(record["cause_label"] is not None for record in records),
        "unpaired_fault_games": sum(record["clean_run"] is None for record in records),
        "controls": len(control_rows),
        "fault_games_sha256": common.digest(args.out / "fault-games.jsonl"),
        "controls_sha256": common.digest(args.out / "controls.jsonl"),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
