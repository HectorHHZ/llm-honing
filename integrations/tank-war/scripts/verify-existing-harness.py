"""Check old Tank War harness losses against the clean model at identical states.

The seed order, seat order, and first-difference rule are written before any
counterfactual is run. Every attempted decision and failure remains on disk.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import threading
import time
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-games", type=Path, required=True)
    parser.add_argument("--old-config", type=Path, required=True)
    parser.add_argument("--old-controls", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--node", type=Path, required=True)
    parser.add_argument("--clean-url", action="append", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--deadline-unix", type=float, required=True)
    args = parser.parse_args()
    if len(args.clean_url) != 2:
        parser.error("two clean model servers are required")
    args.out.mkdir(parents=True, exist_ok=True)
    config = load(args.old_config)
    entries = []
    for seed in config["seeds"]:
        for seat in ("blue", "red"):
            run = args.old_games / f"s{seed}-{seat}"
            if not (run / "result.json").exists():
                continue
            result = load(run / "result.json")
            if result["outcome"][seat] != -1:
                continue
            entries.append({"seed": seed, "seat": seat, "run": str(run), "result_sha256": sha(run / "result.json")})
    frozen = args.out / "freeze.json"
    protocol = {
        "version": "old-harness-first-difference-v1",
        "entries": entries,
        "order": "old formal config seed order; blue before red",
        "decision_order": "zero-based chronological; first selected-action change",
        "verification": "same raw-state hash, clean published checkpoint, different request, selected action differs",
        "loss_policy": "all complete losses retained; failures and no-difference kept with reasons",
    }
    if frozen.exists():
        if load(frozen) != protocol:
            raise ValueError("frozen old-harness protocol changed")
    else:
        frozen.write_text(json.dumps(protocol, indent=2) + "\n")
    lock = threading.Lock()

    def inspect(entry: dict) -> dict:
        run = Path(entry["run"])
        source = args.out / f"s{entry['seed']}-{entry['seat']}"
        source.mkdir(exist_ok=True)
        summary_path = source / "summary.json"
        if summary_path.exists():
            return load(summary_path)
        states = lines(run / "states.jsonl")
        calls = lines(run / "model_calls.jsonl")
        manifest = load(run / "manifest.json")
        if len(states) != len(calls):
            summary = {**entry, "status": "incomplete_trace", "decisions": len(states)}
        else:
            summary = {
                **entry,
                "status": "no_local_difference",
                "decisions": len(states),
                "attempted_decisions": 0,
                "technical_reference_index": None,
                "same_state": None,
                "request_changed": None,
                "checkpoint_unchanged": None,
                "selected_action_changed": False,
                "paired_clean_outcome": "not_available",
            }
            if args.old_controls:
                control_result = args.old_controls / f"s{entry['seed']}-{entry['seat']}" / "result.json"
                if control_result.exists():
                    summary["paired_clean_outcome"] = load(control_result)["outcome"][entry["seat"]]
            for index in range(len(states)):
                if time.time() >= args.deadline_unix:
                    summary["status"] = "deadline_before_all_decisions"
                    break
                comparison_path = source / f"decision-{index:03}.json"
                if not comparison_path.exists():
                    command = [
                        str(args.node),
                        str(args.repo / "integrations/tank-war/scripts/clean-counterfactual.mjs"),
                        f"--run={run}",
                        f"--decision-index={index}",
                        f"--clean-decision-server={args.clean_url[entry['seed'] % 2]}",
                        f"--out={comparison_path}",
                    ]
                    started = time.monotonic()
                    process = subprocess.run(command, cwd=args.repo, text=True, capture_output=True, timeout=330)
                    with lock, (args.out / "attempts.jsonl").open("a") as output:
                        output.write(
                            json.dumps(
                                {
                                    "seed": entry["seed"],
                                    "seat": entry["seat"],
                                    "decision_index": index,
                                    "returncode": process.returncode,
                                    "seconds": round(time.monotonic() - started, 2),
                                    "stderr": process.stderr[-1200:] if process.returncode else None,
                                }
                            )
                            + "\n"
                        )
                    if process.returncode:
                        summary["status"] = "comparison_error"
                        summary["error_decision"] = index
                        break
                comparison = load(comparison_path)
                summary["attempted_decisions"] = index + 1
                if not comparison["local_decision_changed"]:
                    continue
                reference = comparison["reference_model_calls"]
                if len(reference) != 1:
                    summary["status"] = "reference_call_count_error"
                    break
                same_state = comparison["source_state_hash"] == states[index]["state_hash"]
                request_changed = reference[0]["request"] != calls[index]["request"]
                checkpoint_unchanged = (
                    comparison["clean_model"]["weights_sha256"] == manifest["model"]["weights_sha256"]
                )
                summary.update(
                    status=(
                        "verified" if same_state and request_changed and checkpoint_unchanged else "isolation_failed"
                    ),
                    technical_reference_index=index,
                    same_state=same_state,
                    request_changed=request_changed,
                    checkpoint_unchanged=checkpoint_unchanged,
                    selected_action_changed=True,
                    command_changed=comparison["local_command_changed"],
                    execution_changed=comparison["local_execution_changed"],
                    comparison_sha256=sha(comparison_path),
                )
                break
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        with lock:
            print(
                json.dumps(
                    {
                        "seed": entry["seed"],
                        "seat": entry["seat"],
                        "status": summary["status"],
                        "index": summary.get("technical_reference_index"),
                    }
                ),
                flush=True,
            )
        return summary

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(inspect, entries))
    results = [load(args.out / f"s{x['seed']}-{x['seat']}" / "summary.json") for x in entries]
    (args.out / "summary.json").write_text(
        json.dumps(
            {
                "losses": len(entries),
                "verified": sum(x["status"] == "verified" for x in results),
                "status_counts": {
                    status: sum(x["status"] == status for x in results)
                    for status in sorted({x["status"] for x in results})
                },
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
