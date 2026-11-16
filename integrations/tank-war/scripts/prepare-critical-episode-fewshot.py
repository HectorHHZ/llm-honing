"""Freeze the two secondary few-shot modes on the existing 40 Tank War games."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FORBIDDEN = ("inject", "fault", "clean", "gold", "manifest", "source_run", "seed")
MODES = ("read+fewshot", "probe+fewshot")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict | list:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-blind", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source, prior, out = args.source_blind, args.source_run, args.out
    if out.exists():
        raise FileExistsError(out)
    prior_freeze = read(prior / "freeze.json")
    source_freeze = read(source / "sample-freeze.json")
    if not isinstance(prior_freeze, dict) or not isinstance(source_freeze, dict):
        raise ValueError("invalid source freeze")
    if prior_freeze["status"] != "frozen_before_episode_only_answers":
        raise ValueError("episode-only source is not frozen")
    if source_freeze["status"] != "frozen_before_model_answers":
        raise ValueError("test split is not frozen")
    if sha(source / "sample-freeze.json") != prior_freeze["source_sample_freeze_sha256"]:
        raise ValueError("test split changed")
    if len(prior_freeze["cases"]) != 40 or len({case["episode_id"] for case in prior_freeze["cases"]}) != 40:
        raise ValueError("expected 40 unique games")
    if any(
        sha(prior / name) != prior_freeze[f"{name.split('.')[0]}_sha256"]
        for name in ("prompt.txt", "schema.json")
    ):
        raise ValueError("episode-only prompt or schema changed")
    examples = read(args.examples)
    if not isinstance(examples, list) or len(examples) != 3:
        raise ValueError("expected three previously frozen development examples")
    if [example["answer"]["cause"] for example in examples] != ["harness", "weights", "controller"]:
        raise ValueError("unexpected development example labels")
    shown_examples = []
    for example in examples:
        shown = {key: value for key, value in example.items() if key != "answer"}
        shown["answer"] = {
            "cause": example["answer"]["cause"],
            "confidence": example["answer"]["confidence"],
            "explanation": example["answer"]["evidence"],
        }
        shown_examples.append(shown)
    prompt = (prior / "prompt.txt").read_text().replace(
        "EPISODE JSON:\n",
        "These are three separate, previously reviewed development games. Each answer is for its game as a whole. "
        "Their selected excerpts illustrate evidence; a test game may have a different cause or no clear cause. "
        "Do not assume that any component was deliberately changed.\n\n"
        "DEVELOPMENT EXAMPLES JSON:\n"
        + json.dumps(shown_examples, separators=(",", ":"))
        + "\n\nEPISODE JSON:\n",
    )
    if prompt.count("DEVELOPMENT EXAMPLES JSON:") != 1 or any(term in prompt.lower() for term in FORBIDDEN):
        raise ValueError("invalid or revealing few-shot prompt")
    protocol = {
        "version": "tank183-episode-only-fewshot-v1",
        "purpose": "secondary post-test few-shot sensitivity on the same 40 frozen games",
        "independent_blind_sample": False,
        "model": "gpt-6-sol",
        "effort": "high",
        "modes": list(MODES),
        "calls_per_case_per_mode": 1,
        "maximum_calls": 80,
        "maximum_parallel_calls": 2,
        "maximum_input_tokens_per_mode": 4_000_000,
        "maximum_output_tokens_per_mode": 100_000,
        "examples_sha256": sha(args.examples),
        "example_source": "three frozen, previously human-reviewed development games; no test game",
        "input": "unchanged anonymous replay and controlled checks from blind-v7; no hidden test labels",
        "output": "one game-level cause or uncertain abstention; no decision index",
        "paired_comparison": "probe+fewshot minus read+fewshot on the same 40 games",
        "interpretation_limit": (
            "Secondary post-test analysis on previously diagnosed games, not independent confirmation."
        ),
    }
    out.mkdir(parents=True)
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (out / "prompt.txt").write_text(prompt)
    (out / "schema.json").write_bytes((prior / "schema.json").read_bytes())
    freeze = {
        "status": "frozen_before_episode_only_answers",
        "interpretation": protocol["interpretation_limit"],
        "model": protocol["model"],
        "effort": protocol["effort"],
        "modes": list(MODES),
        "calls_per_case_per_mode": 1,
        "maximum_calls": 80,
        "maximum_input_tokens_per_mode": protocol["maximum_input_tokens_per_mode"],
        "maximum_output_tokens_per_mode": protocol["maximum_output_tokens_per_mode"],
        "source_sample_freeze_sha256": sha(source / "sample-freeze.json"),
        "source_run_freeze_sha256": sha(prior / "freeze.json"),
        "fewshot_examples_sha256": sha(args.examples),
        "protocol_sha256": sha(out / "protocol.json"),
        "prompt_sha256": sha(out / "prompt.txt"),
        "schema_sha256": sha(out / "schema.json"),
        "cases": prior_freeze["cases"],
    }
    (out / "freeze.json").write_text(json.dumps(freeze, indent=2) + "\n")
    print(json.dumps({"cases": len(freeze["cases"]), "freeze_sha256": sha(out / "freeze.json")}, indent=2))


if __name__ == "__main__":
    main()
