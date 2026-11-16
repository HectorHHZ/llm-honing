"""Rescore sealed Tank War answers by one changed-layer label per episode."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

LAYERS = ("harness", "weights", "controller")
ANSWERS = (*LAYERS, "uncertain")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed-score", type=Path, required=True)
    parser.add_argument("--technical-labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    score = json.loads(args.sealed_score.read_text())
    episode_only_run = "source_run_freeze_sha256" in score
    label_rows = [json.loads(line) for line in args.technical_labels.read_text().splitlines() if line]
    labels = {row["episode_id"]: row for row in label_rows}
    rows = score["records"]
    if len(rows) != 40 or len(labels) != 40 or {row["episode_id"] for row in rows} != set(labels):
        raise ValueError("sealed answer and audited label sets differ")
    if any(row["label_unit"] != "episode" or row["technical_label"] == "uncertain" for row in label_rows):
        raise ValueError("episode labels are not fully verified")
    for row in rows:
        if any(not row[mode]["valid"] or row[mode]["answer"]["cause"] not in ANSWERS for mode in ("read", "probe")):
            raise ValueError(f"invalid sealed diagnosis: {row['episode_id']}")
    faults = [row for row in rows if labels[row["episode_id"]]["technical_label"] in LAYERS]
    controls = [row for row in rows if labels[row["episode_id"]]["technical_label"] == "no_injection"]
    if len(faults) != 30 or len(controls) != 10:
        raise ValueError("unexpected class balance")

    def correct(row: dict, mode: str) -> bool:
        return row[mode]["answer"]["cause"] == labels[row["episode_id"]]["technical_label"]

    by_layer: dict[str, dict] = {}
    confusion: dict[str, dict] = {}
    for mode in ("read", "probe"):
        confusion[mode] = {}
        for layer in (*LAYERS, "no_injection"):
            group = [row for row in rows if labels[row["episode_id"]]["technical_label"] == layer]
            confusion[mode][layer] = dict(Counter(row[mode]["answer"]["cause"] for row in group))
            if layer in LAYERS:
                by_layer.setdefault(layer, {})[mode] = {
                    "correct": sum(correct(row, mode) for row in group),
                    "total": 10,
                }
    rng = random.Random(183)
    groups = [[row for row in faults if labels[row["episode_id"]]["technical_label"] == layer] for layer in LAYERS]
    differences = sorted(
        sum(
            int(correct(row, "probe")) - int(correct(row, "read"))
            for group in groups
            for row in rng.choices(group, k=10)
        )
        / 30
        for _ in range(10_000)
    )
    provenance = {
        "sealed_score_sha256": sha256(args.sealed_score),
        "private_technical_labels_sha256": sha256(args.technical_labels),
        "no_new_model_calls": not episode_only_run,
        "post_test_endpoint_change": True,
    }
    if episode_only_run:
        provenance.update(
            source_run_freeze_sha256=score["source_run_freeze_sha256"],
            prompt_also_requested_a_window=False,
        )
    else:
        provenance["original_prompt_also_requested_a_window"] = True
    summary = {
        "result_version": "episode-only-prompt-v2" if episode_only_run else "episode-attribution-rescore-v1",
        "unit": "one cause label per complete game",
        "provenance": provenance,
        "cases": {"faults": 30, "no_injection_controls": 10},
        "correct_classification": {
            mode: {"correct": sum(correct(row, mode) for row in faults), "total": 30} for mode in ("read", "probe")
        },
        "by_layer": by_layer,
        "confusion": confusion,
        "paired_probe_minus_read": {
            "point": (sum(correct(row, "probe") for row in faults) - sum(correct(row, "read") for row in faults)) / 30,
            "stratified_bootstrap_95pct": [differences[249], differences[9749]],
            "resamples": 10_000,
            "seed": 183,
        },
        "controls": {
            mode: {
                "unsupported_harness_or_controller": sum(
                    row[mode]["answer"]["cause"] in ("harness", "controller") for row in controls
                ),
                "total": 10,
                "weights_answer_on_unmodified_loss": sum(
                    row[mode]["answer"]["cause"] == "weights" for row in controls
                ),
            }
            for mode in ("read", "probe")
        },
        "interpretation": (
            "Post-test episode-only prompt sensitivity on the same games; not an independent blind confirmation."
            if episode_only_run
            else "Exploratory episode-level rescore of sealed answers; not a new preregistered experiment."
        ),
    }
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: summary[key]
                for key in ("correct_classification", "by_layer", "paired_probe_minus_read", "controls")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
