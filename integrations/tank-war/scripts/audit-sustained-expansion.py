"""Index existing Tank War fault instances without dropping incomplete attempts.

This is a private, read-only audit of game files. It does not promote an
instance into a final test set or count a screen game as a collected sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from collections import Counter
from pathlib import Path
from typing import Any

BASE = runpy.run_path(str(Path(__file__).with_name("assemble-sustained-fault-dataset.py")))
screen_check = BASE["screen_check"]
label_pair = BASE["label_pair"]
digest = BASE["digest"]
SOURCE_FILES = (
    "manifest.json",
    "result.json",
    "states.jsonl",
    "decisions.jsonl",
    "execution.jsonl",
    "events.jsonl",
    "model_calls.jsonl",
)


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def clean_candidates(name: str, roots: list[Path]) -> list[Path]:
    return [root / name for root in roots if (root / name / "result.json").is_file()]


def audit(spec: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    seen: set[tuple[str, int, str]] = set()
    for item in spec["instances"]:
        layer, instance = item["layer"], item["instance"]
        screen_path = Path(item["screen"])
        screen, validation_seeds = screen_check(screen_path, item["screen_sha256"])
        if screen["candidate"]["instance"] != instance or screen["layer"] != layer:
            raise ValueError(f"screen identity differs for {instance}")
        screen_manifest = read(next(Path(item["screen_fault_games"]).glob("s*-*/manifest.json")))
        clean_roots = [Path(value) for value in item["clean_roots"]]
        instance_rows = []
        for fault_root_name in item["fault_roots"]:
            fault_root = Path(fault_root_name)
            for fault in sorted(fault_root.glob("s*-*")):
                if not fault.is_dir():
                    continue
                attempt = {"instance": instance, "layer": layer, "fault_run": str(fault)}
                if (fault / "error.json").exists():
                    attempt["status"] = "error"
                elif not (fault / "result.json").is_file() or not (fault / "manifest.json").is_file():
                    attempt["status"] = "incomplete"
                else:
                    try:
                        result = read(fault / "result.json")
                        attempt["fault_outcome"] = result["outcome"][result["seat"]]
                        attempt["fault_source_hashes"] = {name: digest(fault / name) for name in SOURCE_FILES}
                    except (KeyError, ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
                        attempt.update(status="incomplete_logs", error=repr(exc))
                        attempts.append(attempt)
                        continue
                    clean_options = clean_candidates(fault.name, clean_roots)
                    valid = []
                    errors = []
                    for clean in clean_options:
                        try:
                            row = label_pair(clean, fault, layer, screen_manifest)
                            if row["seed"] in validation_seeds:
                                raise ValueError("collection reused a validation seed")
                            row["source_hashes"] = {
                                arm: {name: digest(root / name) for name in SOURCE_FILES}
                                for arm, root in (("fault", fault), ("clean", clean))
                            }
                            valid.append((clean, row))
                        except (KeyError, ValueError, FileNotFoundError, json.JSONDecodeError) as exc:  # noqa: PERF203
                            errors.append(f"{clean}: {exc}")
                    if not valid:
                        attempt.update(status="unmatched", clean_candidates=len(clean_options), errors=errors)
                    elif len({digest(clean / "result.json") for clean, _ in valid}) > 1:
                        attempt.update(status="ambiguous_clean", clean_candidates=len(valid))
                    else:
                        row = valid[0][1]
                        key = instance, row["seed"], row["seat"]
                        if key in seen:
                            attempt["status"] = "duplicate"
                        else:
                            seen.add(key)
                            row["sample_id"] = hashlib.sha256(
                                f"{instance}:{row['seed']}:{row['seat']}".encode()
                            ).hexdigest()[:24]
                            row["fault_instance"] = instance
                            row["fault_family"] = item["family"]
                            labels.append(row)
                            instance_rows.append(row)
                            attempt["status"] = "labeled"
                            attempt["sample_id"] = row["sample_id"]
                attempts.append(attempt)
        outcomes = Counter(row["status"] for row in attempts if row["instance"] == instance)
        summaries[instance] = {
            "layer": layer,
            "family": item["family"],
            "screen_mean_fault_minus_clean": screen["mean_fault_minus_clean"],
            "screen_ci95": screen["ci95"],
            "attempts": dict(outcomes),
            "fault_losses": sum(row["fault_loss"] for row in instance_rows),
            "verified_fault_losses": sum(row["cause_label"] == layer for row in instance_rows),
            "uncertain_losses": sum(row["fault_loss"] and row["cause_label"] is None for row in instance_rows),
            "complete_fault_loss_records": sum(
                row.get("fault_outcome") == -1 for row in attempts if row["instance"] == instance
            ),
        }
    summary = {
        "version": spec["version"],
        "instances": summaries,
        "labeled_games": len(labels),
        "attempts": len(attempts),
    }
    return labels, attempts, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    spec = read(args.spec)
    labels, attempts, summary = audit(spec)
    args.out.mkdir(parents=True)
    (args.out / "spec.json").write_bytes(args.spec.read_bytes())
    (args.out / "labels.jsonl").write_text("".join(json.dumps(row) + "\n" for row in labels))
    (args.out / "attempts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in attempts))
    (args.out / "unverified-games.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in attempts if row["status"] != "labeled" and "fault_outcome" in row)
    )
    summary["spec_sha256"] = digest(args.out / "spec.json")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
