"""Screen one real Tank War checkpoint against the frozen clean 128-seed arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

from exp0_protocol import validate_fault


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def screen(args: argparse.Namespace) -> dict[str, Any]:
    from importlib.machinery import SourceFileLoader

    gate = SourceFileLoader("tank_screen_fault", str(Path(__file__).with_name("screen-fault.py"))).load_module()
    plan = _json(args.seed_plan)
    seeds = plan["groups"]["fault_validation"]
    if len(seeds) != 128 or len(set(seeds)) != 128:
        raise ValueError("weight screen requires 128 distinct seed blocks")
    gate.require_gate(args.gate_log, plan["groups"]["identity"])
    candidate = _json(args.candidate)
    manifest_path = Path(candidate.get("checkpoint_manifest", ""))
    if (
        candidate.get("layer") != "weights"
        or not candidate.get("instance")
        or not manifest_path.is_file()
        or _json(manifest_path).get("weightsSha256") != candidate.get("weights_sha256")
    ):
        raise ValueError("candidate is not a pinned real weight-fault checkpoint")
    with urllib.request.urlopen(args.url + "/v1/health", timeout=30) as response:
        health = json.load(response)
    metadata = health["metadata"]
    if not health["ready"] or metadata["weightsSha256"] != candidate["weights_sha256"] or metadata["maxLen"] != 8192:
        raise ValueError("decision server does not serve the candidate's pinned 8192-token checkpoint")
    record = {
        "candidate": candidate,
        "candidate_sha256": _sha(args.candidate),
        "seed_plan_sha256": _sha(args.seed_plan),
        "gate_log_sha256": _sha(args.gate_log),
        "clean_root": str(args.clean_root),
        "url": args.url,
        "node": str(args.node),
        "match_sha256": _sha(Path(__file__).with_name("match.mjs")),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    config = args.out / "screen-config.json"
    if config.exists() and _json(config) != record:
        raise ValueError("existing screen output belongs to a different frozen candidate")
    config.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fault_root = args.out / "fault"
    fault_root.mkdir(exist_ok=True)
    for seed in seeds:
        for seat in ("blue", "red"):
            if not gate._completed(args.clean_root / f"s{seed}-{seat}"):
                raise ValueError(f"missing frozen clean pair: {seed}, {seat}")
            run = fault_root / f"s{seed}-{seat}"
            if gate._completed(run):
                continue
            command = [
                os.fspath(args.node),
                os.fspath(Path(__file__).with_name("match.mjs")),
                f"--seed={seed}",
                f"--seat={seat}",
                f"--decision-server={args.url}",
                f"--out={run}",
                "--seconds=30",
                "--interval-ms=200",
                "--opponent-interval-ms=300",
            ]
            started = time.monotonic()
            try:
                # GPU contention can make a fixed 30-second simulation take much longer in real time.
                result = subprocess.run(
                    command, cwd=args.repo, capture_output=True, text=True, timeout=1800, check=False
                )
                attempt = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
            except (OSError, subprocess.TimeoutExpired) as error:
                attempt = {"error": repr(error)}
            attempt.update(
                seed=seed,
                seat=seat,
                run=str(run),
                command=command,
                elapsed_seconds=round(time.monotonic() - started, 3),
            )
            with (args.out / "attempts.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(attempt) + "\n")
            if attempt.get("returncode") != 0:
                raise RuntimeError(f"weight-fault match failed: {run}")
    result = validate_fault(args.clean_root, fault_root, seeds, "weights")
    result["candidate"] = candidate
    result["candidate_sha256"] = _sha(args.candidate)
    (args.out / "screen.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-plan", type=Path, required=True)
    parser.add_argument("--gate-log", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--node", type=Path, default=Path("node"))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = screen(args)
    print(json.dumps({"out": str(args.out / "screen.json"), "retained": result["retained"]}))


if __name__ == "__main__":
    main()
