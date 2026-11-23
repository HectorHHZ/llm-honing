"""Score saved REA-Issac read/probe diagnoses without network or model credentials."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from honing.backends.rea_issac_agent.collect_exp0 import verify

COMPONENTS = ("harness", "weights", "controller")
LABELS = (*COMPONENTS, "abstain")
MODES = ("read", "probe")


def _metrics(pairs: list[tuple[str | None, str]]) -> dict[str, Any]:
    confusion = {actual: dict.fromkeys(LABELS, 0) for actual in (*COMPONENTS, "no_injection")}
    for actual, predicted in pairs:
        confusion[actual or "no_injection"][predicted] += 1
    verified = [(actual, predicted) for actual, predicted in pairs if actual is not None]
    controls = confusion["no_injection"]
    per_class = {}
    for name in COMPONENTS:
        true = confusion[name][name]
        support = sum(confusion[name].values())
        predicted = sum(confusion[actual][name] for actual in COMPONENTS)
        per_class[name] = {"correct": true, "support": support, "f1": 2 * true / (support + predicted)}
    return {
        "component_accuracy": mean(actual == predicted for actual, predicted in verified),
        "macro_f1": mean(row["f1"] for row in per_class.values()),
        "per_class": per_class,
        "confusion": confusion,
        "control_false_positive_rate": sum(controls[name] for name in COMPONENTS) / sum(controls.values()),
        "coverage": mean(predicted != "abstain" for _, predicted in pairs),
        "abstentions": sum(predicted == "abstain" for _, predicted in pairs),
    }


def _paired_interval(differences: list[float]) -> list[float]:
    rng = random.Random(184)
    samples = sorted(mean(rng.choices(differences, k=len(differences))) for _ in range(10000))
    return [samples[250], samples[9749]]


def score(index: list[dict[str, Any]], replies: list[dict[str, Any]]) -> dict[str, Any]:
    expected = {(episode["episode_id"], mode) for episode in index for mode in MODES}
    found = {}
    for reply in replies:
        key = (reply["episode_id"], reply["mode"])
        if key in found or key not in expected or reply["label"] not in LABELS:
            raise ValueError("duplicate, unexpected, or invalid diagnosis")
        cost = reply.get("cost_usd")
        if cost is not None and (type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0):
            raise ValueError("diagnosis cost must be a nonnegative finite number or null")
        found[key] = reply
    if set(found) != expected:
        raise ValueError("one saved diagnosis per episode and mode is required")
    results = {}
    for mode in MODES:
        selected = [found[row["episode_id"], mode] for row in index]
        pairs = [(row["verified_component"], reply["label"]) for row, reply in zip(index, selected, strict=True)]
        costs = [reply["cost_usd"] for reply in selected if reply.get("cost_usd") is not None]
        results[mode] = {
            **_metrics(pairs),
            "known_cost_usd": sum(costs),
            "diagnoses_with_known_cost": len(costs),
            "total_cost_usd": sum(costs) if len(costs) == len(selected) else None,
        }
    groups: dict[str, list[int]] = defaultdict(list)
    for row in index:
        actual = row["verified_component"]
        if actual is not None:
            episode = row["episode_id"]
            probe = found[episode, "probe"]["label"] == actual
            read = found[episode, "read"]["label"] == actual
            groups[row["reset_group"]].append(int(probe) - int(read))
    differences = [mean(group) for group in groups.values()]
    actual_labels = [row["verified_component"] for row in index]
    constant = {label: _metrics([(actual, label) for actual in actual_labels]) for label in LABELS}
    majority = max(COMPONENTS, key=actual_labels.count)
    return {
        **results,
        "baselines": {
            "expected_uniform_random_component_accuracy": 1 / len(COMPONENTS),
            "majority_label": majority,
            "majority": constant[majority],
            "constant": constant,
        },
        "paired_probe_minus_read": {
            "component_accuracy": mean(differences),
            "reset_group_bootstrap_95": _paired_interval(differences),
            "reset_groups": len(groups),
            "limitation": "Conditional on the fixed fault instance per component; no between-fault variance estimate.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--replies", type=Path, required=True)
    args = parser.parse_args()
    verify(args.collection)
    index = json.loads((args.collection / "episode_index.json").read_text(encoding="utf-8"))
    replies = [json.loads(line) for line in args.replies.read_text(encoding="utf-8").splitlines()]
    print(json.dumps(score(index, replies), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
