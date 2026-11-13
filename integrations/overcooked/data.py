"""Audited train-only teacher decisions for Honing's typed_decisions trainers."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from honing.core import DataContract, DecisionQuestion, DecisionSample, TrainingBatch
from integrations.overcooked.engine import ROOT, Scenario, write_json
from integrations.overcooked.match import run_match
from integrations.overcooked.recording import replay
from integrations.overcooked.scenarios import Suite, load_suite


def teacher_rows(run: Path, suite: Suite, group: str) -> list[dict[str, Any]]:
    audit = replay(run)
    manifest = audit["manifest"]
    allowed = [suite.scenario(task.task_id).record() for task in suite.tasks("train")]
    if manifest["controller"] != "teacher" or manifest["kind"] != "teacher-demonstration":
        raise ValueError("only explicitly recorded rule-teacher demonstrations provide SFT labels")
    if manifest["scenario"] not in allowed:
        raise ValueError("demonstrations must match a train task; val/heldout scenes are refused")
    return [
        {"request": row["request"], "chosen": row["chosen"], "group": group, "source_harness": manifest["spec_id"]}
        for row in (json.loads(line) for line in (run / "decisions.jsonl").read_text().splitlines())
    ]


def balanced(rows: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    if type(cap) is not int or cap < 1:
        raise ValueError("per-action cap must be a positive integer")
    counts: Counter[str] = Counter()
    result = []
    for row in rows:
        chosen = row["chosen"]
        if counts[chosen] < cap:
            result.append(row)
            counts[chosen] += 1
    return result


def training_batch(rows: list[dict[str, Any]], batch_id: str) -> TrainingBatch:
    decisions = []
    for row in rows:
        request = row["request"]
        question = request["questions"]["action"]
        decisions.append(
            DecisionSample(
                parts=tuple((name, tuple(lines)) for name, lines in request["parts"].items()),
                questions=(
                    (
                        "action",
                        DecisionQuestion("choice", question["instructions"], tuple(question["criteria"].items())),
                    ),
                ),
                chosen=(("action", row["chosen"]),),
                group=row["group"],
                source_harness=row["source_harness"],
            )
        )
    return TrainingBatch(batch_id, DataContract("typed_decisions"), decisions=tuple(decisions))


def build_dataset(
    fit_runs: list[Path], internal_val_runs: list[Path], out: Path, fit_cap: int = 32, val_cap: int = 8
) -> dict[str, Any]:
    if {p.resolve() for p in fit_runs} & {p.resolve() for p in internal_val_runs}:
        raise ValueError("fit and internal validation must use separate teacher episodes")
    suite = load_suite(ROOT / "tasks/benchmark-v1.jsonl")
    fit, val = [], []
    for index, run in enumerate(fit_runs):
        fit.extend(teacher_rows(run, suite, f"fit:{index}"))
    for index, run in enumerate(internal_val_runs):
        val.extend(teacher_rows(run, suite, f"internal-val:{index}"))
    selected_fit, selected_val = balanced(fit, fit_cap), balanced(val, val_cap)
    result = {
        "kind": "overcooked-train-only-sft-v1",
        "teacher": "public-state rule baseline",
        "fit_runs": [str(p.resolve()) for p in fit_runs],
        "internal_val_runs": [str(p.resolve()) for p in internal_val_runs],
        "selection": {"fit_per_action_cap": fit_cap, "internal_val_per_action_cap": val_cap},
        "raw_counts": {"fit": len(fit), "internal_val": len(val)},
        "selected_counts": {"fit": len(selected_fit), "internal_val": len(selected_val)},
        "action_counts": {
            "fit": dict(Counter(r["chosen"] for r in selected_fit)),
            "internal_val": dict(Counter(r["chosen"] for r in selected_val)),
        },
        "rows": selected_fit + selected_val,
    }
    training_batch(result["rows"], "overcooked-sft-validated")
    write_json(out, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new directory for train-only episodes and dataset")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    scenarios = {
        "fit-role": Scenario(),
        "fit-stay": Scenario(partner="stay"),
        "internal-val": Scenario(controlled_player=1),
    }
    for name, scenario in scenarios.items():
        result = run_match(scenario, args.out / name, policy="teacher", record_images=False)
        print(json.dumps({"teacher_episode": name, "result": result}), flush=True)
    result = build_dataset(
        [args.out / "fit-role", args.out / "fit-stay"], [args.out / "internal-val"], args.out / "dataset.json"
    )
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}), flush=True)


if __name__ == "__main__":
    main()
