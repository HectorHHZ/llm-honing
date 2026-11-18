"""Freeze four game-level diagnosis modes for two models before any answers."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

RUNS = (
    ("sol-bare", "gpt-6-sol", "high", ("read", "probe"), "bare"),
    ("sol-fewshot", "gpt-6-sol", "high", ("read+fewshot", "probe+fewshot"), "fewshot"),
    ("astra-bare", "gpt-6-astra", "medium", ("read", "probe"), "bare"),
    ("astra-fewshot", "gpt-6-astra", "medium", ("read+fewshot", "probe+fewshot"), "fewshot"),
)


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(
    source: Path, registration_path: Path, bare: Path, fewshot: Path, schema: Path, out: Path
) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(out)
    registration = read(registration_path)
    sample = read(source / "sample-freeze.json")
    if (
        sample["status"] != "frozen_before_model_answers"
        or sample["pre_registration_sha256"] != sha(registration_path)
        or registration["prompt_sha256"] != {"bare": sha(bare), "fewshot": sha(fewshot)}
        or registration["schema_sha256"] != sha(schema)
    ):
        raise ValueError("preregistered sample, prompt, or schema differs")
    replay = read(source / "replays/summary.json")
    probes = read(source / "probes/summary.json")
    audit = read(source / "probe-audit.json")
    n = len(sample["cases"])
    if replay["valid"] != n or probes["status"] != "complete" or probes["cases"] != n or audit["valid"] != n:
        raise ValueError("replay or probe evidence incomplete")
    probe_hashes = {row["episode_id"]: row["probe_sha256"] for row in probes["records"]}
    if set(probe_hashes) != {case["episode_id"] for case in sample["cases"]}:
        raise ValueError("probe set does not match frozen cases")
    cases = []
    for case in sample["cases"]:
        episode = case["episode_id"]
        if sha(source / "inputs" / f"{episode}.json") != case["input_sha256"]:
            raise ValueError(f"anonymous input changed: {episode}")
        if sha(source / "probes" / episode / "probes.json") != probe_hashes[episode]:
            raise ValueError(f"controlled checks changed: {episode}")
        cases.append(
            {
                "episode_id": episode,
                "input_sha256": case["input_sha256"],
                "probes_sha256": probe_hashes[episode],
            }
        )
    out.mkdir(parents=True)
    runs = {}
    for name, model, effort, modes, prompt_kind in RUNS:
        if name.startswith("sol") != (model == registration["primary_model"]["id"]):
            raise ValueError("model assignment differs from preregistration")
        expected = registration["primary_model"] if name.startswith("sol") else registration["secondary_model"]
        if (model, effort) != (expected["id"], expected["effort"]):
            raise ValueError("model effort differs from preregistration")
        target = out / name
        target.mkdir()
        shutil.copyfile(bare if prompt_kind == "bare" else fewshot, target / "prompt.txt")
        shutil.copyfile(schema, target / "schema.json")
        protocol = {
            "version": "tank183-v4-multifamily-diagnosis-v1",
            "model": model,
            "effort": effort,
            "modes": list(modes),
            "cases": n,
            "pre_registration_sha256": sha(registration_path),
            "source_sample_freeze_sha256": sha(source / "sample-freeze.json"),
            "maximum_input_tokens_per_mode": registration["limits"]["maximum_input_tokens_per_model_mode"],
            "maximum_output_tokens_per_mode": registration["limits"]["maximum_output_tokens_per_model_mode"],
        }
        (target / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
        frozen = {
            "status": "frozen_before_episode_only_answers",
            "model": model,
            "effort": effort,
            "modes": list(modes),
            "maximum_calls": n * len(modes),
            "maximum_input_tokens_per_mode": protocol["maximum_input_tokens_per_mode"],
            "maximum_output_tokens_per_mode": protocol["maximum_output_tokens_per_mode"],
            "source_sample_freeze_sha256": sha(source / "sample-freeze.json"),
            "source_replay_summary_sha256": sha(source / "replays/summary.json"),
            "source_probe_summary_sha256": sha(source / "probes/summary.json"),
            "protocol_sha256": sha(target / "protocol.json"),
            "prompt_sha256": sha(target / "prompt.txt"),
            "schema_sha256": sha(target / "schema.json"),
            "cases": cases,
        }
        (target / "freeze.json").write_text(json.dumps(frozen, indent=2) + "\n")
        runs[name] = {"freeze_sha256": sha(target / "freeze.json"), "model": model, "modes": list(modes)}
    manifest = {
        "version": "tank183-v4-multifamily-diagnosis-v1",
        "status": "frozen_before_any_new_answers",
        "pre_registration_sha256": sha(registration_path),
        "sample_freeze_sha256": sha(source / "sample-freeze.json"),
        "cases": n,
        "maximum_calls": n * len(registration["modes"]) * 2,
        "runs": runs,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-blind", type=Path, required=True)
    parser.add_argument("--pre-registration", type=Path, required=True)
    parser.add_argument("--bare-prompt", type=Path, required=True)
    parser.add_argument("--fewshot-prompt", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            freeze(
                args.source_blind, args.pre_registration, args.bare_prompt, args.fewshot_prompt, args.schema, args.out
            )
        )
    )


if __name__ == "__main__":
    main()
