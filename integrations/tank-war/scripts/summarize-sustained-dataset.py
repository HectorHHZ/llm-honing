"""Recount complete sustained-fault games and write a private data card."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

LAYERS = ("harness", "weights", "controller")


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset
    summary = read(root / "summary.json")
    labels = rows(root / "labels.jsonl")
    controls = rows(root / "controls.jsonl")
    if len(labels) != 480 or len(controls) != 160 or summary["fault_loss_utilization"]["retained"] != 450:
        raise ValueError("expected 480 complete fault games and 160 controls")
    if len({(row["layer"], row["seed"], row["seat"]) for row in labels}) != 480:
        raise ValueError("duplicate fault game")
    if sum(row["fault_loss"] for row in labels) != 450:
        raise ValueError("fault loss count differs from assembly")
    recheck: dict[str, Any] = {
        "status": "post_collection_descriptive_check_not_original_retention_gate",
        "bootstrap": {"resamples": 10_000, "seed": 183, "unit": "two-seat seed block"},
        "layers": {},
    }
    for layer in LAYERS:
        group = [row for row in labels if row["layer"] == layer]
        by_seed: dict[int, dict[str, float]] = defaultdict(dict)
        for row in group:
            by_seed[row["seed"]][row["seat"]] = row["fault_score_minus_clean"]
        if len(group) != 160 or len(by_seed) != 80 or any(set(seats) != {"blue", "red"} for seats in by_seed.values()):
            raise ValueError(f"{layer} lacks 80 complete two-seat seed groups")
        means = [(seats["blue"] + seats["red"]) / 2 for _, seats in sorted(by_seed.items())]
        rng = random.Random(183)
        draws = sorted(sum(rng.choices(means, k=80)) / 80 for _ in range(10_000))
        recheck["layers"][layer] = {
            "complete_seed_blocks": 80,
            "mean_fault_minus_clean": sum(means) / 80,
            "paired_bootstrap_95pct": [draws[249], draws[9749]],
        }
    (root / "independent-recheck.json").write_text(json.dumps(recheck, indent=2) + "\n")
    card = {
        "dataset_version": summary["version"],
        "unit": "complete Tank War game; one technical reference at the first same-state local action difference",
        "game_count": len(labels) + len(controls),
        "fault_games": len(labels),
        "fault_losses_retained": 450,
        "verified_fault_loss_labels": sum(row["cause_label"] in LAYERS for row in labels),
        "no_injection_controls": {"games": len(controls), "losses": sum(row["outcome"] == -1 for row in controls)},
        "source": "Private labels index original raw games, requests, replies, states, attempts, and event logs",
        "label_provenance": (
            "screened intervention manifest plus same-state raw-state, request, "
            "selected action, and command comparison"
        ),
        "annotation_status": "machine evidence only; outside human reviews have not been linked to these sample IDs",
        "blind_test": (
            "40 earlier collected games were frozen separately before any v2 System 0 answer; "
            "same-family follow-up only"
        ),
        "limitations": summary["limitations"],
        "hashes": {
            "protocol": digest(root / "protocol.json"),
            "summary": digest(root / "summary.json"),
            "labels": digest(root / "labels.jsonl"),
            "controls": digest(root / "controls.jsonl"),
            "independent_recheck": digest(root / "independent-recheck.json"),
        },
    }
    (root / "data-card.json").write_text(json.dumps(card, indent=2) + "\n")
    print(json.dumps({"layers": recheck["layers"], "data_card": str(root / "data-card.json")}))


if __name__ == "__main__":
    main()
