"""Fill the card templates (``templates/*.md``) for one game: computed numbers plus the game's own text.

  python3 fill_cards.py ROOT AUDIT OUT_DIR --values VALUES.json [--scores SCORES.json] [--templates DIR]

Computed from the collection (no model call): outcome counts per condition, attempts and exclusions, reset
groups, scored model decisions / native windows / decision bytes / model-server input tokens, the weights
evidence range; with ``--scores`` also every RESULTS number (correct, macro-F1, abstentions, control false
positives, tokens, paired difference and intervals, baselines). ``VALUES.json`` maps every other
placeholder name (e.g. ``GAME_NAME``, ``EPISODE_DEFINITION``, ``HARNESS_FAULT_RULE``) to its text; a value
given there overrides a computed one. Placeholders left unfilled are listed, and ``bundle.py release``
refuses cards that still contain any.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bundle import _model_input_tokens
from contract import read_json, read_rows

HERE = Path(__file__).resolve().parent
PLACEHOLDER = re.compile(r"\{\{([A-Z0-9_]+)\}\}")
PREFIX = {"no_injection": "CONTROL", "harness": "HARNESS", "weights": "WEIGHTS", "controller": "CONTROLLER"}


def _contract_version() -> str:
    from contract import CONTRACT_VERSION

    return CONTRACT_VERSION


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _interval(value: dict[str, Any]) -> str:
    interval = value.get("interval_95")
    if not interval:
        return "n/a (fewer than two groups)"
    return f"{interval[0] * 100:+.1f} to {interval[1] * 100:+.1f} percentage points ({value['groups']} groups)"


def computed(root: Path, audit: Path, scores: dict[str, Any] | None) -> dict[str, str]:
    index = read_json(audit / "episodes.json")
    attempts = read_rows(audit / "attempts.jsonl")
    values: dict[str, str] = {}
    for condition, prefix in PREFIX.items():
        counts = Counter(row["native_outcome"]["status"] for row in index if row["condition"] == condition)
        for status, name in (("win", "WIN"), ("loss", "LOSS"), ("timeout", "TIMEOUT"), ("other_terminal", "OTHER")):
            values[f"{prefix}_{name}"] = str(counts.get(status, 0))
    values["ATTEMPTS"] = str(len(attempts))
    values["EXCLUDED"] = str(sum(row["annotation_status"] == "excluded" for row in attempts))
    values["RESET_GROUPS"] = str(len({row["scenario_group"] for row in index}))
    decisions = windows = size = tokens = 0
    for row in index:
        run = root / row["run"]
        rows = read_rows(run / "decisions.jsonl")
        decisions += sum(item["kind"] == "model" for item in rows)
        windows += sum(len((item["executed"] or {}).get("windows", [])) for item in rows)
        size += (run / "decisions.jsonl").stat().st_size
        tokens += _model_input_tokens(run, rows)
    questions = 0
    for row in index:
        questions += sum(
            len((item["request"] or {}).get("questions") or {})
            for item in read_rows(root / row["run"] / "decisions.jsonl")
            if item["kind"] == "model"
        )
    values.update(
        MODEL_DECISIONS=f"{decisions:,}",
        NATIVE_WINDOWS=f"{windows:,}",
        DECISION_BYTES=f"{size:,}",
        MODEL_INPUT_TOKENS=f"{tokens:,}",
        MODEL_QUESTIONS=f"{questions:,}",
        SCORED_EPISODES=str(len(index)),
        PER_CONDITION=str(len(index) // 4),
        CONTRACT_VERSION=_contract_version(),
    )
    weights = [row["evidence"] for row in index if row["condition"] == "weights"]
    if weights:
        differences = [item["max_probability_difference"] for item in weights]
        values.update(
            WEIGHTS_DP_MIN=f"{min(differences):.4f}",
            WEIGHTS_DP_MAX=f"{max(differences):.4f}",
            WEIGHTS_CHOICE_CHANGED=str(sum(item["choice_changed"] for item in weights)),
        )
    if scores:
        for mode in ("read", "probe"):
            part, prefix = scores[mode], mode.upper()
            matrix = part["confusion_matrix"]
            values[f"{prefix}_CORRECT"] = str(part["correct"])
            values[f"{prefix}_MACRO_F1"] = _fmt(part["macro_f1"])
            values[f"{prefix}_ABSTAIN"] = str(sum(row["uncertain"] for row in matrix.values()))
            values[f"{prefix}_CONTROL_FP"] = str(
                sum(part["control_causes"][c] for c in ("harness", "weights", "controller"))
            )
            usage = part["reported_usage_totals"]
            values[f"{prefix}_INPUT_TOKENS"] = f"{usage.get('input_tokens', 0):,}" if usage else "unknown"
            values[f"{prefix}_OUTPUT_TOKENS"] = f"{usage.get('output_tokens', 0):,}" if usage else "unknown"
        paired = scores["paired_probe_minus_read"]
        values["PAIRED_ESTIMATE"] = f"{paired['estimate'] * 100:+.1f} percentage points"
        values["GROUP_INTERVAL"] = _interval(paired["scenario_reset_bootstrap"])
        values["INSTANCE_INTERVAL"] = _interval(paired["fault_instance_bootstrap"])
        values["UNIFORM_ACC"] = _fmt(scores["baselines"]["uniform_expected_accuracy"])
        values["MAJORITY_ACC"] = _fmt(scores["baselines"]["majority_accuracy"])
    return values


def fill(templates: Path, out: Path, values: dict[str, str]) -> dict[str, list[str]]:
    out.mkdir(parents=True, exist_ok=True)
    missing: dict[str, list[str]] = {}
    for template in sorted(templates.glob("*.md")):
        text = PLACEHOLDER.sub(
            lambda match: values.get(match.group(1), match.group(0)), template.read_text(encoding="utf-8")
        )
        left = sorted(set(PLACEHOLDER.findall(text)))
        if left:
            missing[template.name] = left
        (out / template.name).write_text(text, encoding="utf-8")
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("audit", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--values", type=Path, required=True)
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--templates", type=Path, default=HERE / "templates")
    args = parser.parse_args()
    values = computed(args.root, args.audit, read_json(args.scores) if args.scores else None)
    values.update({key: str(value) for key, value in read_json(args.values).items()})
    missing = fill(args.templates, args.out, values)
    for name, keys in missing.items():
        print(f"{name}: unfilled {', '.join(keys)}")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
