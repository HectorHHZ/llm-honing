"""Freeze episode-only prompts and already-recorded Tank War inputs for a sensitivity run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-blind", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", choices=("gpt-6-sol", "gpt-6-astra"), default="gpt-6-sol")
    parser.add_argument("--effort", choices=("medium", "high"), default="high")
    args = parser.parse_args()
    source, out = args.source_blind, args.out
    if (out / "freeze.json").exists() or any((out / mode).exists() for mode in ("read", "probe")):
        raise FileExistsError("episode-only sensitivity run already frozen or started")
    for filename in ("protocol.json", "prompt.txt", "schema.json"):
        if not (out / filename).exists():
            raise FileNotFoundError(out / filename)
    protocol = read(out / "protocol.json")
    if (protocol["model"], protocol["effort"]) != (args.model, args.effort):
        raise ValueError("requested model or effort differs from protocol")
    source_freeze = read(source / "sample-freeze.json")
    if source_freeze["status"] != "frozen_before_model_answers":
        raise ValueError("source blind sample is not sealed")
    if read(source / "replays" / "summary.json")["valid"] != len(source_freeze["cases"]):
        raise ValueError("source replay gate incomplete")
    if read(source / "probes" / "summary.json")["status"] != "complete":
        raise ValueError("source probe gate incomplete")
    cases = []
    for case in source_freeze["cases"]:
        episode_id = case["episode_id"]
        input_path = source / "inputs" / f"{episode_id}.json"
        probe_path = source / "probes" / episode_id / "probes.json"
        if sha(input_path) != case["input_sha256"]:
            raise ValueError(f"source input changed: {episode_id}")
        cases.append(
            {
                "episode_id": episode_id,
                "input_sha256": sha(input_path),
                "probes_sha256": sha(probe_path),
            }
        )
    if len(cases) != 40 or len({case["episode_id"] for case in cases}) != 40:
        raise ValueError("expected 40 unique frozen games")
    freeze = {
        "status": "frozen_before_episode_only_answers",
        "interpretation": "post-test prompt sensitivity on the same 40 games; not an independent blind sample",
        "model": args.model,
        "effort": args.effort,
        "modes": ["read", "probe"],
        "calls_per_case_per_mode": 1,
        "maximum_calls": 80,
        "maximum_input_tokens_per_mode": protocol.get("maximum_input_tokens_per_mode"),
        "maximum_output_tokens_per_mode": protocol.get("maximum_output_tokens_per_mode"),
        "source_sample_freeze_sha256": sha(source / "sample-freeze.json"),
        "protocol_sha256": sha(out / "protocol.json"),
        "prompt_sha256": sha(out / "prompt.txt"),
        "schema_sha256": sha(out / "schema.json"),
        "cases": cases,
    }
    (out / "freeze.json").write_text(json.dumps(freeze, indent=2) + "\n")
    print(json.dumps({"cases": len(cases), "freeze_sha256": sha(out / "freeze.json")}, indent=2))


if __name__ == "__main__":
    main()
