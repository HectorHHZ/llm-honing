"""Score the secondary few-shot arms against the already sealed bare arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

LAYERS = ("harness", "weights", "controller")
MODES = ("read", "probe", "read+fewshot", "probe+fewshot")


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_discordance_pvalue(improved: int, worsened: int) -> float:
    trials = improved + worsened
    if trials == 0:
        return 1.0
    tail = sum(math.comb(trials, index) for index in range(min(improved, worsened) + 1)) / 2**trials
    return min(1.0, 2 * tail)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bare-score-input", type=Path, required=True)
    parser.add_argument("--fewshot-score-input", type=Path, required=True)
    parser.add_argument("--technical-labels", type=Path, required=True)
    parser.add_argument("--bare-run", type=Path, required=True)
    parser.add_argument("--fewshot-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    bare, fewshot = read(args.bare_score_input), read(args.fewshot_score_input)
    if bare.get("modes", ["read", "probe"]) != ["read", "probe"]:
        raise ValueError("bare score input has unexpected modes")
    if fewshot.get("modes") != ["read+fewshot", "probe+fewshot"]:
        raise ValueError("few-shot score input has unexpected modes")
    if bare["source_run_freeze_sha256"] != sha(args.bare_run / "freeze.json") or fewshot[
        "source_run_freeze_sha256"
    ] != sha(args.fewshot_run / "freeze.json"):
        raise ValueError("a frozen run changed")
    bare_rows = {row["episode_id"]: row for row in bare["records"]}
    fewshot_rows = {row["episode_id"]: row for row in fewshot["records"]}
    label_rows = [json.loads(line) for line in args.technical_labels.read_text().splitlines() if line]
    labels = {row["episode_id"]: row["technical_label"] for row in label_rows}
    if (
        len(bare_rows) != len(fewshot_rows)
        or len(labels) != 40
        or set(bare_rows) != set(fewshot_rows) != set(labels)
    ):
        raise ValueError("different or repeated test games")
    if any(
        row["label_unit"] != "episode" or row["technical_label"] not in (*LAYERS, "no_injection")
        for row in label_rows
    ):
        raise ValueError("technical labels incomplete")
    rows = {key: {**bare_rows[key], **fewshot_rows[key]} for key in bare_rows}
    if any(
        not row[mode]["valid"] or row[mode]["answer"]["cause"] not in (*LAYERS, "uncertain")
        for row in rows.values()
        for mode in MODES
    ):
        raise ValueError("invalid answer")
    groups = {layer: [key for key in rows if labels[key] == layer] for layer in (*LAYERS, "no_injection")}
    if any(len(group) != 10 for group in groups.values()):
        raise ValueError("expected ten games per condition")

    def correct(key: str, mode: str) -> int:
        return int(rows[key][mode]["answer"]["cause"] == labels[key])

    rng = random.Random(18306)
    comparisons = {}
    for name, left, right in (
        ("probe_minus_read", "read", "probe"),
        ("fewshot_probe_minus_fewshot_read", "read+fewshot", "probe+fewshot"),
        ("fewshot_read_minus_read", "read", "read+fewshot"),
        ("fewshot_probe_minus_probe", "probe", "probe+fewshot"),
    ):
        observed = sum(correct(key, right) - correct(key, left) for layer in LAYERS for key in groups[layer]) / 30
        samples = sorted(
            sum(
                correct(key, right) - correct(key, left)
                for layer in LAYERS
                for key in rng.choices(groups[layer], k=10)
            )
            / 30
            for _ in range(10_000)
        )
        comparisons[name] = {
            "point": observed,
            "stratified_episode_bootstrap_95pct": [samples[249], samples[9749]],
            "right_better": sum(correct(key, right) > correct(key, left) for layer in LAYERS for key in groups[layer]),
            "right_worse": sum(correct(key, right) < correct(key, left) for layer in LAYERS for key in groups[layer]),
            "same": sum(correct(key, right) == correct(key, left) for layer in LAYERS for key in groups[layer]),
        }
    secondary_names = (
        "fewshot_probe_minus_fewshot_read",
        "fewshot_read_minus_read",
        "fewshot_probe_minus_probe",
    )
    secondary_order = sorted(
        secondary_names,
        key=lambda name: exact_discordance_pvalue(
            comparisons[name]["right_better"], comparisons[name]["right_worse"]
        ),
    )
    prior_adjusted = 0.0
    for rank, name in enumerate(secondary_order):
        item = comparisons[name]
        raw = exact_discordance_pvalue(item["right_better"], item["right_worse"])
        adjusted = max(prior_adjusted, min(1.0, (len(secondary_order) - rank) * raw))
        item["exact_discordance_two_sided_p"] = raw
        item["holm_three_secondary_comparisons_p"] = adjusted
        prior_adjusted = adjusted
    by_mode = {}
    for mode in MODES:
        run = args.bare_run if mode in ("read", "probe") else args.fewshot_run
        usage = read(run / mode / "summary.json")
        if usage["status"] != "complete" or usage["completed"] != 40:
            raise ValueError(f"mode incomplete: {mode}")
        confusion = {
            layer: dict(Counter(rows[key][mode]["answer"]["cause"] for key in groups[layer]))
            for layer in groups
        }
        by_mode[mode] = {
            "verified_fault_correct": sum(correct(key, mode) for layer in LAYERS for key in groups[layer]),
            "verified_fault_total": 30,
            "per_cause_correct": {layer: sum(correct(key, mode) for key in groups[layer]) for layer in LAYERS},
            "control_unsupported_harness_or_controller": sum(
                rows[key][mode]["answer"]["cause"] in ("harness", "controller") for key in groups["no_injection"]
            ),
            "control_total": 10,
            "confusion": confusion,
            "usage": {
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "elapsed_seconds_sum": usage["elapsed_seconds_sum"],
                "uncached_api_equivalent_usd": round(
                    2 * usage["input_tokens"] / 1_000_000 + 10 * usage["output_tokens"] / 1_000_000, 6
                ),
            },
        }
    result = {
        "result_version": "episode-only-fewshot-secondary-v1",
        "interpretation": (
            "Post-test secondary analysis on the same 40 previously diagnosed games; not independent confirmation."
        ),
        "cases": {"verified_faults": 30, "no_injection_controls": 10},
        "provenance": {
            "bare_score_input_sha256": sha(args.bare_score_input),
            "fewshot_score_input_sha256": sha(args.fewshot_score_input),
            "technical_labels_sha256": sha(args.technical_labels),
            "fewshot_freeze_sha256": sha(args.fewshot_run / "freeze.json"),
        },
        "modes": by_mode,
        "paired_comparisons": comparisons,
        "uncertainty_limit": (
            "Intervals resample episodes within three causes; there is one tested fault family per cause, "
            "so they do not estimate transfer to new fault families."
        ),
        "cost_reference": (
            "GPT-6 Sol standard-tier uncached API equivalent at frozen $2/M input and $10/M output; "
            "not the ChatGPT-login bill."
        ),
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "accuracy": {mode: by_mode[mode]["verified_fault_correct"] for mode in MODES},
                "paired_comparisons": comparisons,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
