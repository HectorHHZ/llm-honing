"""Score the sealed, game-level read/probe diagnosis of sustained Tank War faults."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

LAYERS = ("harness", "weights", "controller")
MODES = ("read", "probe")
ANSWERS = (*LAYERS, "uncertain")


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-blind", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    sample = read(args.source_blind / "sample-freeze.json")
    run = read(args.diagnosis / "freeze.json")
    if sample["status"] != "frozen_before_model_answers" or run["source_sample_freeze_sha256"] != digest(
        args.source_blind / "sample-freeze.json"
    ):
        raise ValueError("sample or diagnosis freeze changed")
    if len(sample["cases"]) != 40 or len({case["seed"] for case in sample["cases"]}) != 40:
        raise ValueError("expected 40 distinct-seed cases")
    if any(read(args.diagnosis / mode / "summary.json")["status"] != "complete" for mode in MODES):
        raise ValueError("both diagnosis modes must complete")
    cases: list[dict[str, Any]] = []
    for case in sample["cases"]:
        episode = case["episode_id"]
        answers: dict[str, str] = {}
        for mode in MODES:
            path = args.diagnosis / mode / episode
            attempt, answer = read(path / "attempt.json"), read(path / "answer.json")
            if (
                attempt.get("exit_code") != 0
                or attempt.get("tool_call")
                or attempt.get("output_validation_error")
                or attempt.get("usage") is None
                or attempt.get("input_sha256") != case["input_sha256"]
                or answer.get("cause") not in ANSWERS
            ):
                raise ValueError(f"invalid diagnosis: {mode}/{episode}")
            answers[mode] = answer["cause"]
        cases.append(
            {
                "episode_id": episode,
                "technical_label": case["technical_label"],
                "technical_reference_index": case["technical_reference_index"],
                "read": answers["read"],
                "probe": answers["probe"],
            }
        )
    faults = [case for case in cases if case["technical_label"] in LAYERS]
    controls = [case for case in cases if case["technical_label"] == "no_injection"]
    if len(faults) != 30 or len(controls) != 10:
        raise ValueError("unexpected class balance")

    def correct(row: dict[str, Any], mode: str) -> bool:
        return row[mode] == row["technical_label"]

    groups = [[row for row in faults if row["technical_label"] == layer] for layer in LAYERS]
    if any(len(group) != 10 for group in groups):
        raise ValueError("expected ten verified cases per fault class")
    rng = random.Random(183)
    differences = sorted(
        sum(
            int(correct(row, "probe")) - int(correct(row, "read"))
            for group in groups
            for row in rng.choices(group, k=10)
        )
        / 30
        for _ in range(10_000)
    )
    accurate = {mode: sum(correct(row, mode) for row in faults) for mode in MODES}
    lower, upper = differences[249], differences[9749]
    result = {
        "version": "tank183-sustained-diagnosis-v1",
        "sample_freeze_sha256": digest(args.source_blind / "sample-freeze.json"),
        "diagnosis_freeze_sha256": digest(args.diagnosis / "freeze.json"),
        "cases": {"verified_faults": 30, "no_injection_losses": 10},
        "accuracy": {mode: {"correct": accurate[mode], "total": 30} for mode in MODES},
        "majority_baseline": {"correct": 10, "total": 30},
        "by_layer": {
            layer: {mode: {"correct": sum(correct(row, mode) for row in group), "total": 10} for mode in MODES}
            for layer, group in zip(LAYERS, groups, strict=True)
        },
        "confusion": {
            mode: {
                label: dict(Counter(row[mode] for row in cases if row["technical_label"] == label))
                for label in (*LAYERS, "no_injection")
            }
            for mode in MODES
        },
        "paired_probe_minus_read": {
            "point": (accurate["probe"] - accurate["read"]) / 30,
            "class_stratified_bootstrap_95pct": [lower, upper],
            "resamples": 10_000,
            "rng_seed": 183,
        },
        "controls": {
            mode: {
                "unsupported_harness_or_controller": sum(row[mode] in ("harness", "controller") for row in controls),
                "weights_answer_on_unmodified_loss": sum(row[mode] == "weights" for row in controls),
                "uncertain": sum(row[mode] == "uncertain" for row in controls),
                "total": 10,
            }
            for mode in MODES
        },
        "scoped_go": lower > 0 and accurate["probe"] > 10,
        "full_issue_183_confirmed": False,
        "limitations": [
            "Only one sustained fault family per layer was sampled.",
            "The harness strategy also participates in post-choice plan revalidation.",
            "A same-state local effect is not proof that it caused the loss.",
        ],
        "per_case": cases,
        "usage": {mode: read(args.diagnosis / mode / "summary.json") for mode in MODES},
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("accuracy", "paired_probe_minus_read", "controls", "scoped_go")}))


if __name__ == "__main__":
    main()
