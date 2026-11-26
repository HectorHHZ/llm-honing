"""Score saved per-episode Exp 0 v0 diagnoses without a model or an API key (Griddly v0 ``score.py``, extended).

  python3 score.py EPISODES.json READ_DIAGNOSES.jsonl PROBE_DIAGNOSES.jsonl

Standard library only; the same inputs always give the same JSON (fixed bootstrap seed 200, as Griddly).
Per mode, on the 30 verified interventions: component accuracy (abstentions and invalid replies count as
incorrect), macro-F1, per-class precision/recall/F1, confusion matrix, abstention rate, coverage,
calibration of the stated confidence (Brier score, five reliability bins, expected calibration error) where
a component was asserted; on the 10 no-injection controls: the false-positive rate (any asserted
component) and the causes given; token usage totals and dollar cost (``null`` unless every reply has one).
Baselines: uniform random (expected accuracy and macro-F1), majority, and each constant component. Paired
``probe`` minus ``read`` accuracy on the verified episodes, with percentile bootstrap intervals over reset
groups and over fault instances (one instance per component in v0, so the latter is coarse).
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

COMPONENTS = ("harness", "weights", "controller")
MODES = ("read", "probe")
OUTPUTS = (*COMPONENTS, "uncertain", "invalid")
EPISODES, PER_CLASS = 40, 10
BINS = ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0000001))


def diagnosis_label(reply: dict[str, Any]) -> str:
    if reply.get("status") in {"invalid", "error"}:
        return "invalid"
    cause = reply.get("cause")
    if cause not in OUTPUTS:
        raise ValueError("diagnosis has an unsupported cause")
    return cause


def rate(count: int, total: int) -> float | None:
    return count / total if total else None


def calibration(pairs: list[tuple[float, bool]]) -> dict[str, Any]:
    bins = []
    weighted_gap = 0.0
    for low, high in BINS:
        members = [(c, hit) for c, hit in pairs if low <= c < high]
        mean = math.fsum(c for c, _ in members) / len(members) if members else None
        accuracy = sum(hit for _, hit in members) / len(members) if members else None
        if members:
            weighted_gap += len(members) * abs(accuracy - mean)
        bins.append(
            {"range": [low, min(high, 1.0)], "count": len(members), "mean_confidence": mean, "accuracy": accuracy}
        )
    return {
        "confidence_brier": math.fsum((c - int(hit)) ** 2 for c, hit in pairs) / len(pairs) if pairs else None,
        "confidence_count": len(pairs),
        "expected_calibration_error": weighted_gap / len(pairs) if pairs else None,
        "reliability_bins": bins,
    }


def mode_scores(index: list[dict[str, Any]], replies: dict[str, dict[str, Any]]) -> dict[str, Any]:
    labeled = [row for row in index if row.get("verified_component") in COMPONENTS]
    controls = [row for row in index if row["annotation_status"] == "control"]
    matrix = {component: dict.fromkeys(OUTPUTS, 0) for component in COMPONENTS}
    pairs: list[tuple[float, bool]] = []
    for row in labeled:
        reply = replies[row["episode_id"]]
        label, cause = row["verified_component"], diagnosis_label(reply)
        matrix[label][cause] += 1
        confidence = reply.get("confidence")
        if cause in COMPONENTS and type(confidence) in (float, int) and 0 <= confidence <= 1:
            pairs.append((float(confidence), cause == label))
    per_class = {}
    for component in COMPONENTS:
        hit = matrix[component][component]
        support = sum(matrix[component].values())
        predictions = sum(counts[component] for counts in matrix.values())
        per_class[component] = {
            "support": support,
            "predicted": predictions,
            "precision": rate(hit, predictions),
            "recall": rate(hit, support),
            "f1": rate(2 * hit, predictions + support),
        }
    control_causes = Counter(diagnosis_label(replies[row["episode_id"]]) for row in controls)
    abstentions = sum(row["uncertain"] for row in matrix.values())
    invalid = sum(row["invalid"] for row in matrix.values())
    coverage = sum(sum(row[cause] for cause in COMPONENTS) for row in matrix.values())
    costs = [reply["cost_usd"] for reply in replies.values() if reply.get("cost_usd") is not None]
    if any(type(cost) not in (float, int) or not math.isfinite(cost) or cost < 0 for cost in costs):
        raise ValueError("saved reply has an invalid cost")
    usage_totals: Counter[str] = Counter()
    recorded_usage = 0
    for reply in replies.values():
        usage = reply.get("usage")
        if usage is None:
            continue
        if not isinstance(usage, dict) or any(type(value) is not int or value < 0 for value in usage.values()):
            raise ValueError("saved reply has invalid token usage")
        usage_totals.update(usage)
        recorded_usage += 1
    return {
        "component_accuracy": rate(sum(matrix[label][label] for label in COMPONENTS), len(labeled)),
        "correct": sum(matrix[label][label] for label in COMPONENTS),
        "verified_episodes": len(labeled),
        "macro_f1": math.fsum(row["f1"] or 0 for row in per_class.values()) / len(COMPONENTS),
        "per_class": per_class,
        "confusion_matrix": matrix,
        "abstention_rate": rate(abstentions, len(labeled)),
        "invalid_rate": rate(invalid, len(labeled)),
        "coverage": rate(coverage, len(labeled)),
        **calibration(pairs),
        "control_false_positive_rate": rate(sum(control_causes[c] for c in COMPONENTS), len(controls)),
        "control_causes": {cause: control_causes[cause] for cause in OUTPUTS},
        "attempts_total": sum(int(reply.get("attempts") or 0) for reply in replies.values()),
        "cost_usd": math.fsum(costs) if len(costs) == len(replies) else None,
        "recorded_cost_count": len(costs),
        "reported_usage_totals": dict(sorted(usage_totals.items())),
        "recorded_usage_count": recorded_usage,
    }


def grouped_interval(groups: dict[str, list[int]]) -> dict[str, Any]:
    units = [groups[key] for key in sorted(groups)]
    if len(units) < 2:
        return {"groups": len(units), "interval_95": None}
    rng = random.Random(200)
    samples = []
    for _ in range(5000):
        panel = rng.choices(units, k=len(units))
        samples.append(sum(map(sum, panel)) / sum(map(len, panel)))
    samples.sort()
    return {"groups": len(units), "interval_95": [samples[125], samples[4874]]}


def baselines(groups: Counter) -> dict[str, Any]:
    total = sum(groups[c] for c in COMPONENTS)
    share = {c: groups[c] / total for c in COMPONENTS}
    uniform_f1 = math.fsum(2 * share[c] * (1 / 3) / (share[c] + 1 / 3) for c in COMPONENTS) / len(COMPONENTS)
    constant_f1 = {c: (2 * share[c] / (share[c] + 1)) / len(COMPONENTS) for c in COMPONENTS}
    return {
        "uniform_expected_accuracy": 1 / len(COMPONENTS),
        "uniform_expected_macro_f1": uniform_f1,
        "majority_accuracy": max(share.values()),
        "constant_component_accuracy": share,
        "constant_component_macro_f1": constant_f1,
        "constant_component_control_false_positive_rate": 1.0,
    }


def score(index: list[dict[str, Any]], rows: list[dict[str, Any]], per_class: int | None = None) -> dict[str, Any]:
    """Scores of one index. A v0 release index has 40 episodes, 10 per component and 10 controls (the default);
    the bundle and the verifier hold a release to that. ``per_class`` other than 10 is only for a release-chain
    dry run on pilot / smoke data (contract 0.2.0; the plan's ``per_condition``); the CLI derives it from the index."""
    per_class = PER_CLASS if per_class is None else per_class
    episodes = 4 * per_class
    ids = {row["episode_id"] for row in index}
    if len(index) != episodes or len(ids) != episodes:
        raise ValueError(f"the v0 index must have {episodes} unique episodes")
    groups = Counter(row.get("verified_component", "control") for row in index)
    if groups != Counter(dict.fromkeys((*COMPONENTS, "control"), per_class)):
        raise ValueError(f"the v0 index must have {per_class} episodes for each component and control condition")
    replies: dict[str, dict[str, dict[str, Any]]] = {mode: {} for mode in MODES}
    for row in rows:
        mode, episode_id = row["mode"], row["episode_id"]
        if mode not in MODES or episode_id not in ids or episode_id in replies[mode]:
            raise ValueError("duplicate or unrecognized episode/mode pair")
        replies[mode][episode_id] = row
    if any(set(replies[mode]) != ids for mode in MODES):
        raise ValueError("both modes must cover the exact same episode IDs")
    if len({row.get("model") for row in rows}) != 1:
        raise ValueError("both modes must use the same diagnosis model")
    scenario_groups: dict[str, list[int]] = defaultdict(list)
    fault_instances: dict[str, list[int]] = defaultdict(list)
    differences = []
    for row in index:
        target = row.get("verified_component")
        if target not in COMPONENTS:
            continue
        read, probe = (diagnosis_label(replies[mode][row["episode_id"]]) for mode in MODES)
        delta = int(probe == target) - int(read == target)
        scenario_groups[row["scenario_group"]].append(delta)
        fault_instances[row["fault_instance"]].append(delta)
        differences.append(delta)
    return {
        **{mode: mode_scores(index, replies[mode]) for mode in MODES},
        "baselines": baselines(groups),
        "paired_probe_minus_read": {
            "estimate": sum(differences) / len(differences),
            "scenario_reset_bootstrap": grouped_interval(scenario_groups),
            "fault_instance_bootstrap": grouped_interval(fault_instances),
        },
        "model": rows[0].get("model"),
        "episodes": len(index),
        **({} if per_class == PER_CLASS else {"per_class": per_class, "dry_run_size": True}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("index", type=Path)
    parser.add_argument("diagnoses", type=Path, nargs="+")
    args = parser.parse_args()
    index = json.loads(args.index.read_text(encoding="utf-8"))
    rows = [json.loads(line) for path in args.diagnoses for line in path.read_text(encoding="utf-8").splitlines()]
    controls = sum(row.get("annotation_status") == "control" for row in index)
    per_class = None if controls == PER_CLASS else controls  # a dry-run index has fewer (the bundle says so)
    print(json.dumps(score(index, rows, per_class), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
