"""Seal read/probe rules before either sustained-fault diagnosis runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-blind", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tools", type=Path, required=True)
    args = parser.parse_args()
    source, out = args.source_blind, args.out
    if (out / "freeze.json").exists() or (out / "read").exists() or (out / "probe").exists():
        raise FileExistsError("diagnosis already frozen or started")
    sample = read(source / "sample-freeze.json")
    protocol = read(out / "protocol.json")
    if sample["status"] != "frozen_before_model_answers" or len(sample["cases"]) != 40:
        raise ValueError("expected 40 sealed source games")
    if read(source / "replays" / "summary.json")["valid"] != 40:
        raise ValueError("full replay gate incomplete")
    if protocol["modes"] != ["read", "probe"] or protocol["maximum_calls"] != 80:
        raise ValueError("diagnosis protocol differs")
    cases = []
    for case in sample["cases"]:
        input_path = source / "inputs" / f"{case['episode_id']}.json"
        if digest(input_path) != case["input_sha256"]:
            raise ValueError(f"blind input changed: {case['episode_id']}")
        cases.append({"episode_id": case["episode_id"], "input_sha256": case["input_sha256"]})
    freeze = {
        "status": "frozen_before_diagnosis_answers",
        "model": protocol["model"],
        "effort": protocol["effort"],
        "modes": protocol["modes"],
        "maximum_calls": protocol["maximum_calls"],
        "maximum_input_tokens_per_mode": protocol["maximum_input_tokens_per_mode"],
        "maximum_output_tokens_per_mode": protocol["maximum_output_tokens_per_mode"],
        "source_sample_freeze_sha256": digest(source / "sample-freeze.json"),
        "source_replay_summary_sha256": digest(source / "replays" / "summary.json"),
        "protocol_sha256": digest(out / "protocol.json"),
        "prompt_sha256": digest(out / "prompt.txt"),
        "schema_sha256": digest(out / "schema.json"),
        "probe_tools_manifest_sha256": digest(source / "probe-tools.sha256"),
        "probe_builder_sha256": digest(args.tools / "build-critical-probes.py"),
        "probe_audit_sha256": digest(args.tools / "audit-sustained-probes.py"),
        "run_script_sha256": digest(args.tools / "run-sustained-diagnosis.py"),
        "cases": cases,
    }
    (out / "freeze.json").write_text(json.dumps(freeze, indent=2) + "\n")
    print(json.dumps({"cases": len(cases), "freeze_sha256": digest(out / "freeze.json")}))


if __name__ == "__main__":
    main()
