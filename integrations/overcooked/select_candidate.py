"""Apply Honing's win/loss selector to the frozen pilot validation pair."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from honing.core import ArchitectureSpec, CheckpointRef, EvaluationResult, LayerShape, ModelSnapshot, UpdateCandidate
from honing.release.selectors import WinLossSelector
from integrations.overcooked.engine import ROOT, digest, write_json


def compare(incumbent: dict[str, Any], candidate: dict[str, Any], training: dict[str, Any]) -> dict[str, Any]:
    protocol = json.loads((ROOT / "experiments/pilot-v1.json").read_text())
    suite_text = (ROOT / protocol["suite"]).read_text()
    identity = digest({"protocol": protocol, "suite": suite_text})
    expected = protocol["pilot_tasks"]["val"]
    indexed = []
    for summary in (incumbent, candidate):
        if not summary["complete"] or summary["split"] != "val" or summary["controller"] != "model":
            raise ValueError("selection requires completed model validation episodes")
        if summary["protocol_sha256"] != identity:
            raise ValueError("selection requires the frozen protocol and task suite")
        episodes = {row["task_id"]: row for row in summary["episodes"]}
        if len(episodes) != len(summary["episodes"]) or set(episodes) != set(expected):
            raise ValueError("validation task identities differ from the frozen pilot")
        if not all(row["verified"] and row["complete"] for row in episodes.values()):
            raise ValueError("validation includes an unverified or unfinished episode")
        indexed.append(episodes)
    if incumbent["spec_id"] != candidate["spec_id"]:
        raise ValueError("weight comparison must use the identical harness")
    saved = training["snapshot"]
    if incumbent["model"]["weightsSha256"] != training["base"]["checkpoint"]["checksum"]:
        raise ValueError("incumbent is not the candidate's training parent")
    if candidate["model"]["weightsSha256"] != saved["checkpoint"]["checksum"]:
        raise ValueError("candidate is not the trained checkpoint")
    snapshot = ModelSnapshot(
        CheckpointRef(**saved["checkpoint"]),
        ArchitectureSpec(
            saved["architecture"]["hidden_size"],
            tuple(LayerShape(**layer) for layer in saved["architecture"]["layers"]),
        ),
        saved["parameters"],
    )
    update = UpdateCandidate("train", "overcooked-pilot-base", snapshot)
    evaluation = EvaluationResult(
        tuple(expected),
        tuple(indexed[1][key]["grade"] for key in expected),
        tuple(indexed[0][key]["grade"] for key in expected),
        "master",
    )
    decision = WinLossSelector().decide(update, evaluation)
    return {
        "protocol_sha256": identity,
        "decision": asdict(decision),
        "evaluation": asdict(evaluation),
        "selected_checksum": (
            candidate["model"]["weightsSha256"] if decision.selected else incumbent["model"]["weightsSha256"]
        ),
        "native_score_pairs": [
            {"task_id": key, "base": indexed[0][key]["score"], "candidate": indexed[1][key]["score"]}
            for key in expected
        ],
        "publication": "local pilot recommendation only; no shared release registry changed",
        "limits": "Two deterministic validation conditions; the interval does not establish generalization.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("selection output must be new")
    report = compare(*(json.loads(path.read_text()) for path in (args.incumbent, args.candidate, args.training)))
    write_json(args.out, report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
