"""Audit private, whole-episode Tank War fault labels against same-state probes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

LAYERS = ("harness", "weights", "controller")


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def changed(row: dict) -> bool:
    return any(
        row[key] for key in ("choice_changed", "selected_action_changed", "command_changed", "execution_changed")
    )


def derive_label(row: dict, manifest: dict, tolerance: float) -> str:
    """Use measured local signatures; consult checkpoint provenance only for weights."""
    same_input = row["same_model_input"] and row["same_question_text"] and not row["option_set_changed"]
    probability_delta = row["max_abs_probability_change"]
    selected_changed = row["selected_action_changed"]
    if not same_input and selected_changed and row["same_model_input"] is False:
        return "harness"
    checkpoint = manifest.get("critical_checkpoint")
    if (
        same_input
        and probability_delta > tolerance
        and selected_changed
        and checkpoint
        and checkpoint.get("weights_sha256") != manifest["model"]["weights_sha256"]
    ):
        return "weights"
    if (
        same_input
        and probability_delta <= tolerance
        and not row["choice_changed"]
        and not selected_changed
        and (row["command_changed"] or row["execution_changed"])
    ):
        return "controller"
    if same_input and probability_delta <= tolerance and not changed(row):
        return "no_injection"
    return "uncertain"


def audit_case(root: Path, case: dict, probe_audit: dict) -> dict:
    episode_id = case["episode_id"]
    source = Path(case["source_run"])
    manifest_path = source / "manifest.json"
    if sha256(manifest_path) != case["source_hashes"]["manifest.json"]:
        raise ValueError(f"source manifest hash changed: {episode_id}")
    manifest = read(manifest_path)
    probe_path = root / "probes" / episode_id / "probes.json"
    probes = read(probe_path)
    if probes["probe_version"] != "tank183-critical-all-decisions-v1":
        raise ValueError(f"unexpected probe version: {episode_id}")
    if [row["decision_index"] for row in probes["indices"]] != list(range(case["decisions"])):
        raise ValueError(f"incomplete per-decision checks: {episode_id}")
    audit = probe_audit[episode_id]
    if not audit["expected_signature"] or audit["off_target_differences"]:
        raise ValueError(f"probe isolation failed: {episode_id}")
    reference = case["technical_reference_index"]
    if reference is None and case["layer"] == "control":
        if not all(
            derive_label(item, manifest, probes["identity_tolerance"]) == "no_injection" for item in probes["indices"]
        ):
            raise ValueError(f"control has a changed decision: {episode_id}")
        row = probes["indices"][0]
        label = "no_injection"
    else:
        row = probes["indices"][reference]
        label = derive_label(row, manifest, probes["identity_tolerance"])
    expected = "no_injection" if case["layer"] == "control" else case["layer"]
    if label != expected:
        raise ValueError(f"measured signature does not match frozen category: {episode_id}: {label} != {expected}")
    declared = manifest.get("critical_layer")
    if (declared or "no_injection") != label:
        raise ValueError(f"manifest disagrees with measured signature: {episode_id}")
    if label == "harness" and (manifest.get("critical_checkpoint") or not manifest.get("critical_harness_files")):
        raise ValueError(f"harness layer isolation failed: {episode_id}")
    if label == "weights" and (manifest.get("critical_harness_files") or not manifest.get("critical_checkpoint")):
        raise ValueError(f"weights layer isolation failed: {episode_id}")
    if label in ("controller", "no_injection") and (
        manifest.get("critical_harness_files") or manifest.get("critical_checkpoint")
    ):
        raise ValueError(f"post-selection layer isolation failed: {episode_id}")
    if (label == "no_injection") != (manifest.get("critical_fault_sha256") is None):
        raise ValueError(f"fault manifest presence inconsistent: {episode_id}")
    return {
        "episode_id": episode_id,
        "label_unit": "episode",
        "technical_label": label,
        "reference_decision_index_for_audit_only": reference,
        "evidence": {
            key: row[key]
            for key in (
                "same_model_input",
                "same_question_text",
                "max_abs_probability_change",
                "choice_changed",
                "selected_action_changed",
                "command_changed",
                "execution_changed",
            )
        },
        "source_manifest_sha256": sha256(manifest_path),
        "same_state_probes_sha256": sha256(probe_path),
        "checkpoint_sha256": manifest["model"]["weights_sha256"],
        "critical_checkpoint_sha256": (
            manifest["critical_checkpoint"]["weights_sha256"] if manifest.get("critical_checkpoint") else None
        ),
        "paired_clean_outcome_better": case["outcome_improved_without_fault"],
        "failure_causality_claim": False,
        "label_source": "same_state_technical_audit_v1",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.blind_root
    freeze = read(root / "sample-freeze.json")
    audit = read(root / "probe-audit.json")
    if freeze["status"] != "frozen_before_model_answers" or audit["off_target_cases"]:
        raise ValueError("frozen sample or probe isolation invalid")
    probe_audit = {row["episode_id"]: row for row in audit["records"]}
    if len(probe_audit) != len(freeze["cases"]) or len({c["seed"] for c in freeze["cases"]}) != len(freeze["cases"]):
        raise ValueError("case count or seed independence invalid")
    labels = [audit_case(root, case, probe_audit) for case in freeze["cases"]]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    label_path = args.output_dir / "labels.jsonl"
    label_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in labels))
    summary = {
        "label_unit": "episode",
        "labels_sha256": sha256(label_path),
        "sample_freeze_sha256": sha256(root / "sample-freeze.json"),
        "probe_audit_sha256": sha256(root / "probe-audit.json"),
        "counts": dict(Counter(row["technical_label"] for row in labels)),
        "verified_all_cases": len(labels) == 40 and all(row["technical_label"] != "uncertain" for row in labels),
        "failure_causality": "separate; a verified layer does not imply that the intervention caused a loss",
        "original_human_review_requirement": "not met by this technical audit",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
