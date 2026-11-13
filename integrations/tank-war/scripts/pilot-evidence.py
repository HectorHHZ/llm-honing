"""Prepare private, replay-checked Tank War pilot evidence for human review."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import subprocess
from pathlib import Path
from typing import Any


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _run(command: list[str], cwd: Path, log: Path, timeout_seconds: int = 300) -> bool:
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout_seconds, check=False)
        attempt = {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        attempt = {"command": command, "error": repr(error)}
    with log.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(attempt) + "\n")
    return attempt.get("returncode") == 0


def _single_layer(clean: dict[str, Any], fault: dict[str, Any], layer: str) -> bool:
    if clean["code_commit"] != fault["code_commit"] or clean["sources"] != fault["sources"]:
        return False
    changed = {
        "harness": clean["harness_sha256"] != fault["harness_sha256"],
        "weights": clean["model"]["weights_sha256"] != fault["model"]["weights_sha256"],
        "controller": clean["controller_sha256"] != fault["controller_sha256"],
    }
    if layer == "harness":
        return changed["harness"] and not changed["weights"]
    return changed[layer] and all(not value for name, value in changed.items() if name != layer)


def _stratum(comparison: dict[str, Any]) -> str:
    faulty = comparison["faulty"]["selected_action"]
    clean = comparison["clean"]["selected_action"]
    if faulty["weapon"] != clean["weapon"]:
        return "aim_fire"
    if faulty["motion"] != clean["motion"]:
        return "movement_avoidance"
    before = comparison["faulty"]["acknowledged_command"]["controls"]
    after = comparison["clean"]["acknowledged_command"]["controls"]
    return "aim_fire" if before["fire"] != after["fire"] else "movement_avoidance"


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    plan = _json(args.seed_plan)
    seeds = plan["groups"]["pilot"][args.start_index : args.start_index + args.seed_count]
    if len(seeds) != args.seed_count:
        raise ValueError("pilot seed slice is incomplete")
    fault = _json(args.fault_manifest)
    if fault["layer"] != args.layer:
        raise ValueError("fault manifest layer differs")
    args.out.mkdir(parents=True, exist_ok=False)
    signature = runpy.run_path(str(Path(__file__).with_name("reproducibility-gate.py")))["signature"]
    attempts = args.out / "attempts.jsonl"
    selected = args.out / "samples.jsonl"
    outcomes = args.out / "outcomes.jsonl"
    accepted = 0
    for seed in seeds:
        for seat in ("blue", "red"):
            source = args.fault_root / f"s{seed}-{seat}"
            clean_run = args.clean_root / f"s{seed}-{seat}"
            result = _json(source / "result.json")
            outcome = result["outcome"][seat]
            with outcomes.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"seed": seed, "seat": seat, "outcome": outcome, "run": str(source)}) + "\n")
            if outcome != -1 or accepted >= args.target:
                continue
            clean_manifest = _json(clean_run / "manifest.json")
            fault_manifest = _json(source / "manifest.json")
            if not _single_layer(clean_manifest, fault_manifest, args.layer):
                raise ValueError(f"more than one layer changed in {source}")
            sample_id = hashlib.sha256(
                f"{fault_manifest['code_commit']}:{fault['instance']}:{seed}:{seat}".encode()
            ).hexdigest()[:16]
            directory = args.out / sample_id
            directory.mkdir()
            scan = directory / "same-state-comparisons.jsonl"
            command = [
                os.fspath(args.node),
                os.fspath(Path(__file__).with_name("scan-counterfactual.mjs")),
                f"--run={source}",
                f"--out={scan}",
            ]
            if args.layer in ("harness", "weights"):
                command.append(f"--clean-decision-server={args.clean_url}")
            if args.require_effective:
                command.append("--require-effective=true")
            if args.require_execution:
                command.append("--require-execution=true")
            if not _run(command, args.repo, attempts):
                continue
            comparisons = _rows(scan)
            if not comparisons or not comparisons[-1]["locally_verified"]:
                continue
            comparison = comparisons[-1]
            if args.require_effective and not (
                comparison["local_command_changed"] or comparison["local_execution_changed"]
            ):
                continue
            if args.require_execution and not comparison["local_execution_changed"]:
                continue
            replay = directory / "replayed"
            manifest = fault_manifest
            replay_command = [
                os.fspath(args.node),
                os.fspath(Path(__file__).with_name("match.mjs")),
                f"--seed={seed}",
                f"--seat={seat}",
                f"--replay-dir={source}",
                f"--out={replay}",
                f"--seconds={manifest['seconds']}",
                f"--interval-ms={manifest['interval_ms']}",
                f"--opponent-interval-ms={manifest['opponent_interval_ms']}",
            ]
            if args.harness_dir:
                replay_command.append(f"--harness-dir={args.harness_dir}")
            if args.controller_fault_file:
                replay_command.append(f"--controller-fault-file={args.controller_fault_file}")
            if not _run(replay_command, args.repo, attempts):
                continue
            if signature(source) != signature(replay) or any(
                (source / name).read_bytes() != (replay / name).read_bytes()
                for name in ("execution.jsonl", "events.jsonl")
            ):
                raise ValueError(f"replay differs for {source}")
            labels = directory / "planner-reference-all-states.jsonl"
            label_command = [
                os.fspath(args.node),
                os.fspath(Path(__file__).with_name("label-recorded-states.mjs")),
                f"--run={source}",
                f"--out={labels}",
            ]
            if not _run(label_command, args.repo, attempts):
                continue
            index = comparison["decision_index"]
            reference = next((row for row in _rows(labels) if row["decision_index"] == index), None)
            if reference is None:
                raise ValueError(f"missing planner reference at decision {index}")
            (directory / "reference-action.json").write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")
            packet = {
                "sample_id": sample_id,
                "split": "pilot",
                "seed": seed,
                "seat": seat,
                "source_run": str(source),
                "clean_run": str(clean_run),
                "fault_instance": fault["instance"],
                "injected_layer": args.layer,
                "outcome": outcome,
                "decision_index": index,
                "window": [max(0, index - 5), min(index + 5, result["decisions"][seat] - 1)],
                "state_hash": comparison["source_state_hash"],
                "task_stratum": _stratum(comparison),
                "local_decision_changed": comparison["local_decision_changed"],
                "local_command_changed": comparison["local_command_changed"],
                "local_execution_changed": comparison["local_execution_changed"],
                "same_state_evidence": str(scan),
                "replay": str(replay),
                "reference_action": str(directory / "reference-action.json"),
                "review_status": "pending_zhuohang_and_nick",
            }
            (directory / "private-review-packet.json").write_text(
                json.dumps(packet, indent=2) + "\n", encoding="utf-8"
            )
            with selected.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(packet) + "\n")
            accepted += 1
    summary = {
        "layer": args.layer,
        "instance": fault["instance"],
        "machine_supported": accepted,
        "target": args.target,
        "require_effective": args.require_effective,
        "require_execution": args.require_execution,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-plan", type=Path, required=True)
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--target", type=int, default=5)
    parser.add_argument("--require-effective", action="store_true")
    parser.add_argument("--require-execution", action="store_true")
    parser.add_argument("--layer", choices=("harness", "weights", "controller"), required=True)
    parser.add_argument("--fault-manifest", type=Path, required=True)
    parser.add_argument("--fault-root", type=Path, required=True)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--clean-url")
    parser.add_argument("--harness-dir", type=Path)
    parser.add_argument("--controller-fault-file", type=Path)
    parser.add_argument("--node", type=Path, default=Path("node"))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.target < 1 or args.seed_count < 1 or args.start_index < 0:
        parser.error("target and seed count must be positive")
    if args.layer in ("harness", "weights") and not args.clean_url:
        parser.error("harness and weights checks require the clean decision server")
    print(json.dumps(prepare(args)))


if __name__ == "__main__":
    main()
