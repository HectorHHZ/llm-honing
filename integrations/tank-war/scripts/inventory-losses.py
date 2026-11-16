"""Inventory every Tank War match and retain complete losses without cherry-picking.

This is game-local private experiment bookkeeping. It does not label a loss as
caused by an injection; a later same-state check supplies that evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

REQUIRED = ("manifest.json", "result.json", "states.jsonl", "model_calls.jsonl", "execution.jsonl", "events.jsonl")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def layer_from_manifest(manifest: dict) -> str:
    if manifest.get("critical_layer"):
        return manifest["critical_layer"]
    if manifest.get("replay_source"):
        return "replay"
    if manifest.get("controller_sha256") and manifest.get("harness_sha256"):
        return "unknown_pending_pair_check"
    return "unknown"


def inventory(root: Path, names: list[str]) -> tuple[list[dict], dict]:
    records = []
    counts: dict[str, Counter] = defaultdict(Counter)
    seen: dict[tuple, dict] = {}
    for name in names:
        directory = root / name
        if not directory.is_dir():
            counts[name]["missing_root"] += 1
            continue
        for result_file in sorted(directory.rglob("result.json")):
            run = result_file.parent
            row = {"run": str(run), "source_root": name}
            counts[name]["results"] += 1
            try:
                result = json.loads(result_file.read_text())
                manifest = json.loads((run / "manifest.json").read_text())
                seat, seed = result["seat"], result["seed"]
                if manifest.get("seat") != seat or manifest.get("seed") != seed:
                    raise ValueError("manifest/result identity differs")
                if manifest.get("replay_source"):
                    row.update(
                        seed=seed, seat=seat, outcome=result["outcome"][seat], excluded="derived_deterministic_replay"
                    )
                    counts[name]["derived_replays"] += 1
                    records.append(row)
                    continue
                row.update(
                    seed=seed,
                    seat=seat,
                    outcome=result["outcome"][seat],
                    code_commit=manifest.get("code_commit"),
                    checkpoint_sha256=manifest.get("model", {}).get("weights_sha256"),
                    harness_sha256=manifest.get("harness_sha256"),
                    controller_sha256=manifest.get("controller_sha256"),
                    critical_fault_sha256=manifest.get("critical_fault_sha256"),
                    decision_count=result.get("decisions", {}).get(seat),
                    layer_status=layer_from_manifest(manifest),
                    missing=[part for part in REQUIRED if not (run / part).is_file()],
                )
                if row["outcome"] not in (-1, 0, 1):
                    raise ValueError("invalid outcome")
                if row["outcome"] == -1:
                    counts[name]["losses"] += 1
                    if not row["missing"]:
                        counts[name]["complete_losses"] += 1
                        states = [json.loads(line) for line in (run / "states.jsonl").read_text().splitlines() if line]
                        calls = sum(1 for line in (run / "model_calls.jsonl").open() if line.strip())
                        executions = sum(1 for line in (run / "execution.jsonl").open() if line.strip())
                        row["trace_aligned"] = len(states) == calls == executions == row["decision_count"]
                        row["state_hashes_valid"] = all(
                            hashlib.sha256(
                                json.dumps(state["raw_state"], separators=(",", ":"), ensure_ascii=False).encode()
                            ).hexdigest()
                            == state["state_hash"]
                            for state in states
                        )
                        row["result_sha256"] = digest(result_file)
                        row["states_sha256"] = digest(run / "states.jsonl")
                key = (
                    row["code_commit"],
                    seed,
                    seat,
                    row["checkpoint_sha256"],
                    row["harness_sha256"],
                    row["controller_sha256"],
                    row["critical_fault_sha256"],
                )
                if key in seen:
                    previous = seen[key]
                    row["duplicate_of"] = previous["run"]
                    counts[name]["duplicates"] += 1
                    if row["outcome"] != previous["outcome"]:
                        row["error"] = "duplicate identity has different outcome"
                else:
                    seen[key] = row
            except (OSError, ValueError, KeyError, TypeError) as error:
                row["error"] = str(error)
                counts[name]["errors"] += 1
            records.append(row)
    unique_complete_losses = [
        row
        for row in records
        if row.get("outcome") == -1
        and not row.get("missing")
        and "error" not in row
        and "duplicate_of" not in row
        and not row.get("excluded")
    ]
    all_unique_losses = [
        row for row in records if row.get("outcome") == -1 and "duplicate_of" not in row and not row.get("excluded")
    ]
    summary = {
        "roots": {name: dict(value) for name, value in counts.items()},
        "unique_losses": len(all_unique_losses),
        "complete_unique_losses": len(unique_complete_losses),
        "retained_complete_losses": len(unique_complete_losses),
        "aligned_trace_losses": sum(row.get("trace_aligned") is True for row in unique_complete_losses),
        "state_hash_verified_losses": sum(row.get("state_hashes_valid") is True for row in unique_complete_losses),
        "utilization_of_complete_losses": 1.0 if unique_complete_losses else None,
        "note": "All complete losses are retained, including cases with no proven injection effect.",
    }
    return records, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    records, summary = inventory(args.root, args.source)
    (args.out / "attempts.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
    (args.out / "losses.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in records
            if row.get("outcome") == -1
            and not row.get("missing")
            and "error" not in row
            and "duplicate_of" not in row
            and not row.get("excluded")
        )
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
