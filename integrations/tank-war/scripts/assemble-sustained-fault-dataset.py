"""Audit previously screened sustained Tank War faults and label independent games.

The output is private evaluation data. A changed match outcome is recorded
separately from a same-state local effect; it is not required for a cause label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

LAYERS = ("harness", "weights", "controller")
SEATS = ("blue", "red")
SCORE = {-1: 0.0, 0: 0.5, 1: 1.0}


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def policy_reply(reply: dict[str, Any]) -> dict[str, Any]:
    return {
        "answers": {
            head: {"choice": answer["choice"], "probabilities": answer["probabilities"]}
            for head, answer in reply.get("answers", {}).items()
        },
        "branches": {key: policy_reply(value) for key, value in reply.get("branches", {}).items()},
    }


def screen_check(path: Path, expected_sha: str) -> tuple[dict[str, Any], set[int]]:
    if digest(path) != expected_sha:
        raise ValueError(f"screen changed: {path}")
    screen = read(path)
    observations = screen["observations"]
    pairs = {(row["seed"], row["seat"]) for row in observations}
    seeds = {seed for seed, _ in pairs}
    if len(seeds) != 128 or len(pairs) != 256 or pairs != {(seed, seat) for seed in seeds for seat in SEATS}:
        raise ValueError(f"screen lacks 128 complete two-seat blocks: {path}")
    mean = sum(row["fault"] - row["clean"] for row in observations) / len(observations)
    if (
        not screen["retained"]
        or mean > -0.05
        or screen["ci95"][1] >= 0
        or abs(mean - screen["mean_fault_minus_clean"]) > 1e-12
    ):
        raise ValueError(f"screen did not demonstrate a stable score drop: {path}")
    return screen, seeds


def same_build(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return all(
        a[key] == b[key]
        for key in ("code_commit", "sources", "seconds", "interval_ms", "opponent_interval_ms", "opponent")
    )


def same_variant(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (
        same_build(a, b)
        and all(a[key] == b[key] for key in ("harness_sha256", "controller_sha256", "agent_seat"))
        and a["model"]["weights_sha256"] == b["model"]["weights_sha256"]
    )


def first_local_difference(clean: Path, fault: Path, layer: str) -> dict[str, Any]:
    files = [(clean / name).open() for name in ("states.jsonl", "execution.jsonl", "model_calls.jsonl")] + [
        (fault / name).open() for name in ("states.jsonl", "execution.jsonl", "model_calls.jsonl")
    ]
    try:
        for index, lines in enumerate(zip(*files, strict=False)):
            c_state, c_action, c_call, f_state, f_action, f_call = map(json.loads, lines)
            if c_state["decision_index"] != index or f_state["decision_index"] != index:
                raise ValueError(f"decision sequence mismatch at {index}: {fault}")
            if c_state["state_hash"] != f_state["state_hash"]:
                return {"verified": False, "reason": "states diverged before a local action difference"}
            selected_changed = c_action["selected_action"] != f_action["selected_action"]
            command_changed = c_action["acknowledged_command"] != f_action["acknowledged_command"]
            if not (selected_changed or command_changed):
                continue
            request_changed = c_call["request"] != f_call["request"]
            reply_changed = policy_reply(c_call["reply"]) != policy_reply(f_call["reply"])
            verified = (
                (layer == "harness" and request_changed and selected_changed)
                or (layer == "weights" and not request_changed and selected_changed)
                or (
                    layer == "controller"
                    and not request_changed
                    and not reply_changed
                    and not selected_changed
                    and command_changed
                )
            )
            return {
                "verified": verified,
                "reason": None if verified else "first local difference does not isolate the declared layer",
                "decision_index": index,
                "state_hash": c_state["state_hash"],
                "request_changed": request_changed,
                "reply_changed": reply_changed,
                "selected_changed": selected_changed,
                "command_changed": command_changed,
            }
        return {"verified": False, "reason": "no same-state local action difference"}
    finally:
        for file in files:
            file.close()


def label_pair(clean: Path, fault: Path, layer: str, screen_manifest: dict[str, Any]) -> dict[str, Any]:
    clean_result, fault_result = read(clean / "result.json"), read(fault / "result.json")
    clean_manifest, fault_manifest = read(clean / "manifest.json"), read(fault / "manifest.json")
    if not same_build(clean_manifest, fault_manifest) or not same_variant(screen_manifest, fault_manifest):
        raise ValueError(f"build or screened variant mismatch: {fault}")
    if (clean_result["seed"], clean_result["seat"]) != (fault_result["seed"], fault_result["seat"]):
        raise ValueError(f"clean/fault identity mismatch: {fault}")
    if (clean_manifest["seed"], clean_manifest["seat"]) != (clean_result["seed"], clean_result["seat"]):
        raise ValueError(f"clean manifest identity mismatch: {clean}")
    if (fault_manifest["seed"], fault_manifest["seat"]) != (fault_result["seed"], fault_result["seat"]):
        raise ValueError(f"fault manifest identity mismatch: {fault}")
    clean_weights = clean_manifest["model"]["weights_sha256"]
    fault_weights = fault_manifest["model"]["weights_sha256"]
    if not (
        (
            layer == "harness"
            and clean_weights == fault_weights
            and clean_manifest["harness_sha256"] != fault_manifest["harness_sha256"]
        )
        or (
            layer == "weights"
            and clean_weights != fault_weights
            and clean_manifest["harness_sha256"] == fault_manifest["harness_sha256"]
            and clean_manifest["controller_sha256"] == fault_manifest["controller_sha256"]
        )
        or (
            layer == "controller"
            and clean_weights == fault_weights
            and clean_manifest["harness_sha256"] == fault_manifest["harness_sha256"]
            and clean_manifest["controller_sha256"] != fault_manifest["controller_sha256"]
        )
    ):
        raise ValueError(f"layer configuration mismatch: {fault}")
    evidence = first_local_difference(clean, fault, layer)
    seat = fault_result["seat"]
    clean_score, fault_score = SCORE[clean_result["outcome"][seat]], SCORE[fault_result["outcome"][seat]]
    return {
        "sample_id": hashlib.sha256(f"{layer}:{fault_result['seed']}:{seat}".encode()).hexdigest()[:24],
        "seed": fault_result["seed"],
        "seat": seat,
        "layer": layer,
        "fault_run": str(fault),
        "clean_run": str(clean),
        "source_hashes": {
            "fault_result": digest(fault / "result.json"),
            "clean_result": digest(clean / "result.json"),
            "fault_manifest": digest(fault / "manifest.json"),
            "clean_manifest": digest(clean / "manifest.json"),
        },
        "fault_outcome": fault_result["outcome"][seat],
        "clean_outcome": clean_result["outcome"][seat],
        "fault_score_minus_clean": fault_score - clean_score,
        "fault_loss": fault_result["outcome"][seat] == -1,
        "verified_local_effect": evidence["verified"],
        "technical_reference": evidence,
        "cause_label": layer if fault_result["outcome"][seat] == -1 and evidence["verified"] else None,
        "annotation_status": "machine_verified" if evidence["verified"] else "uncertain",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    protocol = read(args.protocol)
    if set(protocol["layers"]) != set(LAYERS):
        raise ValueError("expected harness, weights, and controller")
    clean_root = Path(protocol["clean_games"])
    clean_runs = {path.parent.name: path.parent for path in clean_root.glob("s*-*/result.json")}
    if len(clean_runs) != 160:
        raise ValueError("expected 160 distinct clean games")
    records: list[dict[str, Any]] = []
    screens: dict[str, Any] = {}
    all_validation_seeds: set[int] = set()
    for layer in LAYERS:
        setting = protocol["layers"][layer]
        screen, validation_seeds = screen_check(Path(setting["screen"]), setting["screen_sha256"])
        all_validation_seeds |= validation_seeds
        screens[layer] = {
            "instance": screen["candidate"]["instance"],
            "mean_fault_minus_clean": screen["mean_fault_minus_clean"],
            "ci95": screen["ci95"],
            "screen_sha256": setting["screen_sha256"],
        }
        screen_manifest = read(next(Path(setting["screen_fault_games"]).glob("s*-*/manifest.json")))
        fault_runs = sorted(Path(setting["fault_games"]).glob("s*-*/result.json"))
        if not fault_runs:
            raise ValueError(f"no collected games: {layer}")
        for result_path in fault_runs:
            fault = result_path.parent
            if fault.name not in clean_runs:
                raise ValueError(f"missing clean counterpart: {fault}")
            record = label_pair(clean_runs[fault.name], fault, layer, screen_manifest)
            if record["seed"] in validation_seeds:
                raise ValueError(f"collection reused a validation seed: {fault}")
            records.append(record)
    if len({record["sample_id"] for record in records}) != len(records):
        raise ValueError("duplicate fault sample")
    if {int(name.split("-")[0][1:]) for name in clean_runs} & all_validation_seeds:
        raise ValueError("clean control reused a validation seed")
    counts: dict[str, Any] = {}
    for layer in LAYERS:
        group = [record for record in records if record["layer"] == layer]
        counts[layer] = {
            "collected_games": len(group),
            "retained_games": len(group),
            "losses": sum(record["fault_loss"] for record in group),
            "verified_local_effects": sum(record["verified_local_effect"] for record in group),
            "verified_fault_losses": sum(record["cause_label"] == layer for record in group),
            "clean_better_pairs": sum(record["fault_score_minus_clean"] < 0 for record in group),
            "fault_better_pairs": sum(record["fault_score_minus_clean"] > 0 for record in group),
            "equal_pairs": sum(record["fault_score_minus_clean"] == 0 for record in group),
            "unverified_reasons": dict(
                Counter(
                    record["technical_reference"]["reason"] for record in group if not record["verified_local_effect"]
                )
            ),
        }
    controls = []
    for _name, path in sorted(clean_runs.items()):
        result = read(path / "result.json")
        controls.append(
            {
                "sample_id": hashlib.sha256(f"control:{result['seed']}:{result['seat']}".encode()).hexdigest()[:24],
                "seed": result["seed"],
                "seat": result["seat"],
                "source_run": str(path),
                "result_sha256": digest(path / "result.json"),
                "manifest_sha256": digest(path / "manifest.json"),
                "outcome": result["outcome"][result["seat"]],
                "control_condition": "no_injection",
                "cause_label": None,
            }
        )
    args.out.mkdir(parents=True)
    (args.out / "protocol.json").write_bytes(args.protocol.read_bytes())
    (args.out / "labels.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    (args.out / "controls.jsonl").write_text("".join(json.dumps(row) + "\n" for row in controls))
    summary = {
        "version": protocol["version"],
        "protocol_sha256": digest(args.out / "protocol.json"),
        "status": "retrospective_assembly_of_previously_screened_and_collected_games",
        "screens": screens,
        "layers": counts,
        "controls": {"games": len(controls), "losses": sum(row["outcome"] == -1 for row in controls)},
        "fault_loss_utilization": {
            "retained": sum(row["fault_loss"] for row in records),
            "complete": sum(row["fault_loss"] for row in records),
        },
        "verified_fault_losses": sum(row["cause_label"] is not None for row in records),
        "limitations": [
            "One sustained fault family per layer; this is not a family-transfer test.",
            "The harness strategy is used in observation and plan revalidation; its functional boundary is mixed.",
            "A local change does not prove that it caused the whole-game loss.",
            "Previously collected games may have appeared in development; do not claim a fresh blind test.",
        ],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
