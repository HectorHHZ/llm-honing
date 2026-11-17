"""Run frozen read or probe diagnosis; probes may be generated after read."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import runpy
import subprocess
from pathlib import Path
from typing import Any

SOURCE = runpy.run_path(str(Path(__file__).with_name("run-critical-episode-only.py")))
one = SOURCE["one"]
BIN_DIR = Path("/home/xinkaiz/.codex/packages/standalone/current/bin")


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-blind", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tools", type=Path, required=True)
    parser.add_argument("--mode", choices=("read", "probe"), required=True)
    args = parser.parse_args()
    source, out, mode = args.source_blind, args.out, args.mode
    freeze = read(out / "freeze.json")
    if freeze["status"] != "frozen_before_diagnosis_answers" or mode not in freeze["modes"]:
        raise ValueError("diagnosis mode was not frozen")
    for name in ("protocol", "prompt", "schema"):
        suffix = "txt" if name == "prompt" else "json"
        if digest(out / f"{name}.{suffix}") != freeze[f"{name}_sha256"]:
            raise ValueError(f"frozen {name} changed")
    if (
        digest(source / "sample-freeze.json") != freeze["source_sample_freeze_sha256"]
        or digest(source / "replays" / "summary.json") != freeze["source_replay_summary_sha256"]
        or digest(source / "probe-tools.sha256") != freeze["probe_tools_manifest_sha256"]
        or digest(args.tools / "build-critical-probes.py") != freeze["probe_builder_sha256"]
        or digest(args.tools / "audit-sustained-probes.py") != freeze["probe_audit_sha256"]
        or digest(Path(__file__)) != freeze["run_script_sha256"]
    ):
        raise ValueError("frozen sample or execution tool changed")
    subprocess.run(["sha256sum", "--check", str(source / "probe-tools.sha256")], capture_output=True, check=True)
    protocol = read(out / "protocol.json")
    if (freeze["model"], freeze["effort"]) != (protocol["model"], protocol["effort"]):
        raise ValueError("model or effort differs from protocol")
    cli_version = subprocess.run([str(BIN_DIR / "codex"), "--version"], capture_output=True, text=True, check=True)
    if cli_version.stdout.strip() != protocol["cli_version"]:
        raise ValueError("Codex CLI version changed")
    cases = freeze["cases"]
    if mode == "probe":
        summary = read(source / "probes" / "summary.json")
        audit = read(source / "probe-audit.json")
        if summary["status"] != "complete" or summary["cases"] != len(cases) or audit["valid"] != len(cases):
            raise ValueError("probe or identity gate incomplete")
        probe_hashes = {row["episode_id"]: row["probe_sha256"] for row in summary["records"]}
        if set(probe_hashes) != {case["episode_id"] for case in cases}:
            raise ValueError("probe set differs from frozen cases")
        cases = [{**case, "probes_sha256": probe_hashes[case["episode_id"]]} for case in cases]
    destination = out / mode
    destination.mkdir(exist_ok=False)
    prompt = (out / "prompt.txt").read_text()
    attempts = []
    for offset in range(0, len(cases), 2):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            batch = list(pool.map(lambda case: one(case, mode, source, out, prompt), cases[offset : offset + 2]))
        attempts.extend(batch)
        if any(
            item.get("exit_code") != 0
            or item.get("output_validation_error")
            or item.get("tool_call")
            or item.get("usage") is None
            for item in batch
        ):
            break
        if (
            sum(item["usage"]["input_tokens"] for item in attempts) > freeze["maximum_input_tokens_per_mode"]
            or sum(item["usage"]["output_tokens"] for item in attempts) > freeze["maximum_output_tokens_per_mode"]
        ):
            break
    input_used = sum((item.get("usage") or {}).get("input_tokens", 0) for item in attempts)
    output_used = sum((item.get("usage") or {}).get("output_tokens", 0) for item in attempts)
    if input_used > freeze["maximum_input_tokens_per_mode"] or output_used > freeze["maximum_output_tokens_per_mode"]:
        status = "stopped_budget"
    elif len(attempts) == len(cases):
        status = "complete"
    else:
        status = "stopped_invalid"
    summary = {
        "status": status,
        "mode": mode,
        "completed": len(attempts),
        "expected": len(cases),
        "input_tokens": input_used,
        "output_tokens": output_used,
        "elapsed_seconds_sum": sum(item["elapsed_seconds"] for item in attempts),
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
