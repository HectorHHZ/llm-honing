"""Stage and verify the frozen 40-game Tank War Exp 0 v0 evidence package.

This is game-local release tooling. It never writes into the source recordings.
The staged package is private until its contents and publication are reviewed.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

LAYERS = ("harness", "weights", "controller", "no_injection")
SOURCE_FILES = (
    "manifest.json",
    "result.json",
    "states.jsonl",
    "decisions.jsonl",
    "execution.jsonl",
    "events.jsonl",
    "model_calls.jsonl",
)
FORBIDDEN_INPUT_TERMS = ("inject", "fault", "clean", "gold", "manifest", "source_run", "seed")
DIAGNOSIS_MODES = {
    "sol-bare": ("read", "probe"),
    "sol-fewshot": ("read+fewshot", "probe+fewshot"),
    "astra-bare": ("read", "probe"),
    "astra-fewshot": ("read+fewshot", "probe+fewshot"),
}
DEFAULT_PINS = Path(__file__).resolve().parents[1] / "exp0/v0/freeze.json"


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    return digest(path.read_bytes())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def copy_recordings(source: Path, target: Path, expected: dict[str, str]) -> dict[str, dict[str, Any]]:
    target.mkdir(parents=True)
    copied: dict[str, dict[str, Any]] = {}
    for name in SOURCE_FILES:
        original = source / name
        require(original.is_file() and not original.is_symlink(), f"missing or linked source: {original}")
        data = original.read_bytes()
        require(digest(data) == expected[name], f"source hash differs: {original}")
        if name.endswith(".jsonl"):
            relative = name + ".gz"
            with (
                (target / relative).open("wb") as output,
                gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as compressed,
            ):
                compressed.write(data)
        else:
            relative = name
            (target / relative).write_bytes(data)
        copied[name] = {
            "path": str((target / relative).relative_to(target.parents[2])),
            "source_sha256": expected[name],
            "package_sha256": file_digest(target / relative),
            "bytes": len(data),
        }
    return copied


def validate_pins(layer: str, manifest: dict[str, Any], pins: dict[str, Any], instance: str | None) -> str | None:
    expected_settings = {
        "kind": "evaluation",
        "edition": "tank-war",
        "opponent": pins["game"]["opponent"],
        "seconds": pins["game"]["sim_seconds"],
        "interval_ms": pins["game"]["model_interval_ms"],
        "opponent_interval_ms": pins["game"]["opponent_interval_ms"],
        "sample_seed": None,
        "node": pins["game"]["node_version"],
    }
    for field, expected in expected_settings.items():
        require(field in manifest and manifest[field] == expected, f"match setting differs: {field}")
    seat_settings = {
        "controllerMode": pins["clean_harness"]["controller_mode"],
        "contextVersion": pins["clean_harness"]["context_version"],
        "assistEnabled": pins["clean_harness"]["assist_enabled"],
        "assistProfile": pins["clean_harness"]["assist_profile"],
        "fireGuardEnabled": pins["clean_harness"]["fire_guard_enabled"],
    }
    for field, expected in seat_settings.items():
        require(manifest["agent_seat"][field] == expected, f"agent seat setting differs: {field}")
        require(manifest["opponent_seat"][field] == expected, f"opponent seat setting differs: {field}")
    require(manifest["opponent_seat"]["decisionVersion"] == "baseline", "opponent decision version differs")
    require(manifest["code_commit"] == pins["game"]["recorded_code_commit"], "game revision differs")
    require(manifest["sources"] == pins["game"]["sources_sha256"], "game source hashes differ")
    model = manifest["model"]
    require(model["source_commit"] == pins["model"]["source_revision"], "model source differs")
    require(model["context_limit"] == pins["model"]["max_input_tokens"], "model context limit differs")
    clean = pins["clean_harness"]
    if layer == "weights":
        expected = {item["instance"]: item["weights_sha256"] for item in pins["selected_faults"]["weights"]}
        actual = [name for name, sha256 in expected.items() if model["weights_sha256"] == sha256]
        require(len(actual) == 1, "edited weights do not match a frozen checkpoint")
        verified_checkpoint_variant = actual[0]
    else:
        require(model["weights_sha256"] == pins["model"]["clean_weights_sha256"], "clean weights differ")
        verified_checkpoint_variant = None
    if layer == "harness":
        fault = pins["selected_faults"]["harness"]
        require(instance == fault["instance"], "harness instance differs")
        harness_sha = fault["recorded_harness_sha256"]
        controller_sha = fault["recorded_controller_sha256"]
    elif layer == "controller":
        fault = pins["selected_faults"]["controller"]
        require(instance == fault["instance"], "controller instance differs")
        harness_sha = clean["recorded_harness_sha256"]
        controller_sha = fault["recorded_controller_sha256"]
    else:
        harness_sha = clean["recorded_harness_sha256"]
        controller_sha = clean["recorded_controller_sha256"]
    require(manifest["harness_sha256"] == harness_sha, "harness pin differs")
    require(manifest["controller_sha256"] == controller_sha, "controller pin differs")
    return verified_checkpoint_variant


def valid_probe(layer: str, row: dict[str, Any]) -> bool:
    shift = row["max_abs_probability_change"] or 0
    if layer == "harness":
        return not row["same_model_input"] and row["selected_action_changed"]
    if layer == "weights":
        return row["same_model_input"] and row["selected_action_changed"] and (row["choice_changed"] or shift > 0.01)
    if layer == "controller":
        return (
            row["same_model_input"]
            and not row["selected_action_changed"]
            and row["command_changed"]
            and not row["option_set_changed"]
            and shift <= 0.01
        )
    return (
        row["same_model_input"]
        and not row["option_set_changed"]
        and shift <= 0.01
        and not row["selected_action_changed"]
        and not row["command_changed"]
        and not row["execution_changed"]
    )


def model_input_tokens(calls: list[dict[str, Any]]) -> int:
    """Sum the native input-token audits for each recorded question, not billing usage."""
    total = 0
    for call in calls:
        reply = call["reply"]
        for branch in (reply, *reply.get("branches", {}).values()):
            for audit in branch["tokenAudit"].values():
                require(not audit["stateTruncated"], "native input was truncated")
                require(audit["inputTokens"] <= audit["maxInputTokens"], "native input exceeds limit")
                total += audit["inputTokens"]
    return total


def sanitize_attempt(row: dict[str, Any], unmatched: dict[str, dict[str, Any]]) -> dict[str, Any]:
    status = row["status"]
    if status == "error":
        error_path = Path(row["fault_run"]) / "error.json"
        reason = read(error_path)["error"] if error_path.is_file() else "error_record_missing"
    elif status == "incomplete":
        reason = "result_record_missing"
    elif status == "unmatched":
        require(row["fault_run"] in unmatched, "unmatched game missing exclusion record")
        reasons = unmatched[row["fault_run"]]["errors"]
        require(reasons and all("build or screened variant mismatch" in item for item in reasons), "unknown mismatch")
        reason = "build_or_screened_variant_mismatch"
    else:
        require(status == "labeled", f"unknown attempt status: {status}")
        reason = None
    return {
        "attempt_id": digest(row["fault_run"].encode())[:20],
        "instance": row["instance"],
        "layer": row["layer"],
        "status": status,
        "exclusion_reason": reason,
        "fault_outcome": row.get("fault_outcome"),
        "fault_source_hashes": row.get("fault_source_hashes"),
        "sample_id": row.get("sample_id"),
    }


def audit_diagnoser(
    source: Path, freeze: dict[str, Any], out: Path, packaged_probe_sha256: dict[str, str]
) -> dict[str, Any]:
    diagnosis = source / "prospective-diagnosis-v2"
    manifest = read(diagnosis / "manifest.json")
    require(
        manifest["sample_freeze_sha256"] == file_digest(source / "prospective-blind-v2/sample-freeze.json"),
        "diagnosis sample differs",
    )
    records = []
    require(set(manifest["runs"]) == set(DIAGNOSIS_MODES), "diagnosis arms differ")
    frozen_cases = {case["episode_id"]: case for case in freeze["cases"]}
    for arm, arm_spec in manifest["runs"].items():
        arm_root = diagnosis / arm
        arm_freeze = read(arm_root / "freeze.json")
        require(file_digest(arm_root / "freeze.json") == arm_spec["freeze_sha256"], "diagnosis freeze differs")
        require(
            {case["episode_id"] for case in arm_freeze["cases"]} == {case["episode_id"] for case in freeze["cases"]},
            "diagnosis cases differ",
        )
        arm_cases = {case["episode_id"]: case for case in arm_freeze["cases"]}
        require(len(arm_cases) == len(freeze["cases"]), "diagnosis cases duplicate")
        require(set(arm_spec["modes"]) == set(DIAGNOSIS_MODES[arm]), "diagnosis modes differ")
        for episode_id, case in arm_cases.items():
            require(case["input_sha256"] == frozen_cases[episode_id]["input_sha256"], "diagnosis input freeze differs")
            require(
                case["probes_sha256"] == packaged_probe_sha256[episode_id],
                f"packaged probe differs from diagnosis: {episode_id}",
            )
        for mode in arm_spec["modes"]:
            mode_root = arm_root / mode
            require(
                {path.name for path in mode_root.iterdir() if path.is_dir()}
                == {case["episode_id"] for case in freeze["cases"]},
                "diagnosis call count differs",
            )
            for case in freeze["cases"]:
                episode_id = case["episode_id"]
                call_root = mode_root / episode_id
                attempt = read(call_root / "attempt.json")
                require(attempt["exit_code"] == 0 and not attempt["tool_call"], "diagnosis failed or used a tool")
                require(attempt["episode_id"] == episode_id and attempt["mode"] == mode, "diagnosis identity differs")
                require(attempt["model"] == arm_spec["model"], "diagnosis model differs")
                require(attempt["input_sha256"] == case["input_sha256"], "diagnosis input differs")
                require(attempt["prompt_sha256"] == file_digest(arm_root / "prompt.txt"), "diagnosis prompt differs")
                require(attempt["schema_sha256"] == file_digest(arm_root / "schema.json"), "diagnosis schema differs")
                expected_checks = packaged_probe_sha256[episode_id] if mode.startswith("probe") else digest(b"null")
                require(attempt["checks_sha256"] == expected_checks, "diagnosis checks differ")
                require(attempt["usage"]["input_tokens"] > 0, "diagnosis usage missing")
                require((call_root / "answer.json").is_file(), "diagnosis answer missing")
                command = attempt["command"]
                require(command[0] == "bwrap" and command.count("--bind") == 1, "diagnosis sandbox differs")
                bind_position = command.index("--bind")
                require(
                    command[bind_position + 1] == str(call_root) and command[bind_position + 2] == "/work",
                    "diagnosis work mount differs",
                )
                approved_read_targets = {
                    "/run/systemd/resolve",
                    "/usr",
                    "/lib",
                    "/lib64",
                    "/etc",
                    "/app",
                    "/home/sandbox/.codex/auth.json",
                    "/schema.json",
                }
                read_binds = [
                    (command[position + 1], command[position + 2])
                    for position, token in enumerate(command)
                    if token == "--ro-bind"
                ]
                require({target for _, target in read_binds} == approved_read_targets, "diagnosis read mounts differ")
                require(
                    all(
                        source_path == target or target in {"/app", "/home/sandbox/.codex/auth.json", "/schema.json"}
                        for source_path, target in read_binds
                    ),
                    "unexpected diagnosis read mount",
                )
                require(
                    next(source_path for source_path, target in read_binds if target == "/schema.json")
                    == str(arm_root / "schema.json"),
                    "diagnosis schema mount differs",
                )
                require(
                    not any(
                        token in str(command)
                        for token in ("labels.jsonl", "sample-freeze.json", "full-audit-v5", "/gold/")
                    ),
                    "diagnosis command exposes gold",
                )
                records.append(
                    {
                        "episode_id": episode_id,
                        "arm": arm,
                        "mode": mode,
                        "valid": True,
                        "model": attempt["model"],
                        "effort": attempt["effort"],
                        "input_sha256": attempt["input_sha256"],
                        "checks_sha256": attempt["checks_sha256"],
                        "prompt_sha256": attempt["prompt_sha256"],
                        "schema_sha256": attempt["schema_sha256"],
                        "sandbox_command_sha256": digest(json.dumps(command).encode()),
                        "tool_call": False,
                        "input_tokens": attempt["usage"]["input_tokens"],
                        "output_tokens": attempt["usage"]["output_tokens"],
                    }
                )
    require(len(records) == manifest["maximum_calls"] == 320, "diagnosis total differs")
    write_jsonl(out / "checks/diagnoser-isolation.jsonl", records)
    result = {
        "calls": len(records),
        "valid": len(records),
        "tool_calls": 0,
        "arms": dict(Counter(row["arm"] for row in records)),
        "input_tokens": sum(row["input_tokens"] for row in records),
        "output_tokens": sum(row["output_tokens"] for row in records),
    }
    write_json(out / "checks/diagnoser-isolation-summary.json", result)
    return result


def audit_identity(gate: Path, pins: dict[str, Any], out: Path) -> dict[str, Any]:
    plan = read(gate / "plan.json")
    summary = read(gate / "summary.json")
    require(plan["total_calls"] == summary["calls"] == summary["expected_calls"] == 891, "identity count differs")
    require(summary["failed_calls"] == 0 and summary["tolerance"] == 0.01, "identity gate failed")
    require(summary["maximum_probability_difference"] <= 0.01, "identity probability tolerance failed")
    rows = []
    for group, job in zip(plan["jobs"], summary["jobs"], strict=True):
        call_path = gate / f"gpu{job['gpu']}-calls.jsonl"
        require(file_digest(call_path) == job["output_sha256"], "identity call log differs")
        calls = jsonl(call_path)
        require(len(calls) == sum(item["calls"] for item in group), "identity group count differs")
        for item in group:
            manifest_path = Path(item["path"]) / "manifest.json"
            require(file_digest(manifest_path) == item["manifest_sha256"], "identity source manifest differs")
            validate_pins("no_injection", read(manifest_path), pins, None)
        require(
            all(call["passed"] and not call["too_long"] and not call["truncated"] for call in calls),
            "identity call failed",
        )
        rows.extend(calls)
    require(len(rows) == 891, "identity call total differs")
    write_jsonl(out / "checks/identity-calls.jsonl", rows)
    result = {
        "calls": len(rows),
        "failed_calls": 0,
        "tolerance": 0.01,
        "maximum_probability_difference": summary["maximum_probability_difference"],
        "source_summary_sha256": file_digest(gate / "summary.json"),
    }
    write_json(out / "checks/identity-summary.json", result)
    return result


def stage(source: Path, pins_path: Path, identity_gate: Path, out: Path) -> dict[str, Any]:
    require(not out.exists(), f"output already exists: {out}")
    blind = source / "prospective-blind-v2"
    audit = source / "full-audit-v5"
    pins = read(pins_path)
    freeze_path = blind / "sample-freeze.json"
    freeze = read(freeze_path)
    require(file_digest(freeze_path) == pins["existing_sample"]["sample_sha256"], "sample freeze differs")
    require(
        Counter(case["technical_label"] for case in freeze["cases"]) == pins["existing_sample"]["counts"],
        "composition differs",
    )
    require(
        len({case["seed"] for case in freeze["cases"]}) == pins["existing_sample"]["independent_seed_blocks"],
        "seed groups differ",
    )
    labels = {(row["fault_instance"], row["seed"], row["seat"]): row for row in jsonl(audit / "labels.jsonl")}
    attempts = jsonl(audit / "attempts.jsonl")
    unmatched = {row["fault_run"]: row for row in jsonl(audit / "unverified-games.jsonl")}
    replay_summary = read(blind / "replays/summary.json")
    probe_summary = read(blind / "probes/summary.json")
    probe_audit = read(blind / "probe-audit.json")
    require(replay_summary["valid"] == 40 and not replay_summary["invalid"], "replay summary incomplete")
    require(probe_summary["cases"] == 40 and probe_summary["decisions"] == 2151, "probe summary incomplete")
    require(probe_audit["valid"] == 40 and not probe_audit["invalid"], "probe audit incomplete")
    require(len(attempts) == 1330, "attempt ledger differs")

    out.mkdir(parents=True)
    index = []
    provenance = []
    probe_checks = []
    replay_checks = []
    packaged_probe_sha256 = {}
    selected_recording_bytes = 0
    paired_recording_bytes = 0
    selected_input_tokens = 0
    paired_input_tokens = 0
    for case in freeze["cases"]:
        episode_id = case["episode_id"]
        layer = case["technical_label"]
        require(layer in LAYERS, f"unknown class: {layer}")
        source_run = Path(case["source_run"])
        source_hashes = case["source_hashes"]
        run_files = copy_recordings(source_run, out / "episodes" / episode_id / "run", source_hashes)
        selected_recording_bytes += sum(record["bytes"] for record in run_files.values())
        manifest = read(source_run / "manifest.json")
        result = read(source_run / "result.json")
        verified_checkpoint_variant = validate_pins(layer, manifest, pins, case["fault_instance"])
        require(result["seed"] == case["seed"] and result["seat"] == case["seat"], "result identity differs")
        require(result["ended"] and not result["truncated"], f"incomplete episode: {episode_id}")
        require(result["outcome"][case["seat"]] == -1, f"non-loss in selected sample: {episode_id}")
        states = jsonl(source_run / "states.jsonl")
        decisions = jsonl(source_run / "decisions.jsonl")
        execution = jsonl(source_run / "execution.jsonl")
        calls = jsonl(source_run / "model_calls.jsonl")
        require(len(states) == len(execution) == case["decisions"], "decision alignment differs")
        require(len(decisions) == 2 * len(states), "question count differs")
        require(len(calls) == len(states), "model-call count differs")
        native_input_tokens = model_input_tokens(calls)
        selected_input_tokens += native_input_tokens
        require(
            all(
                states[i]["decision_index"] == execution[i]["decision_index"] == i
                and all(
                    decisions[2 * i + question]["decision_index"] == i
                    and decisions[2 * i + question]["state_hash"] == states[i]["state_hash"]
                    for question in (0, 1)
                )
                for i in range(len(states))
            ),
            "state or question alignment differs",
        )
        input_path = blind / "inputs" / f"{episode_id}.json"
        require(input_path.is_file() and not input_path.is_symlink(), f"blind input missing or linked: {episode_id}")
        input_bytes = input_path.read_bytes()
        require(digest(input_bytes) == case["input_sha256"], f"blind input differs: {episode_id}")
        serialized_input = input_bytes.decode()
        require(
            not any(term in serialized_input.lower() for term in FORBIDDEN_INPUT_TERMS)
            and str(case["seed"]) not in serialized_input
            and str(source_run) not in serialized_input,
            f"blind input leaks construction metadata: {episode_id}",
        )
        (out / "inputs").mkdir(exist_ok=True)
        (out / "inputs" / f"{episode_id}.json").write_bytes(input_bytes)
        probe_path = blind / "probes" / episode_id / "probes.json"
        require(probe_path.is_file() and not probe_path.is_symlink(), f"probe missing or linked: {episode_id}")
        probe_bytes = probe_path.read_bytes()
        probe = json.loads(probe_bytes)
        indices = probe["indices"]
        require([row["decision_index"] for row in indices] == list(range(len(states))), "probe coverage differs")
        (out / "probes").mkdir(exist_ok=True)
        (out / "probes" / f"{episode_id}.json").write_bytes(probe_bytes)
        packaged_probe_sha256[episode_id] = digest(probe_bytes)
        if layer == "no_injection":
            require(all(valid_probe(layer, row) for row in indices), f"control probe differs: {episode_id}")
            label = None
            clean_files = None
            reference = None
            clean_outcome = None
            outcome_status = "untested"
        else:
            label = labels[(case["fault_instance"], case["seed"], case["seat"])]
            require(label["fault_run"] == str(source_run), "label source differs")
            require(label["cause_label"] == layer and label["verified_local_effect"], "component not verified")
            require(label["annotation_status"] == "machine_verified", "label provenance differs")
            require(label["fault_loss"] and label["fault_outcome"] == -1, "fault outcome differs")
            require(label["source_hashes"]["fault"] == source_hashes, "label source hashes differ")
            reference = label["technical_reference"]
            position = case["technical_reference_index"]
            require(reference["verified"] and reference["decision_index"] == position, "reference differs")
            require(states[position]["state_hash"] == reference["state_hash"], "reference state differs")
            require(valid_probe(layer, indices[position]), f"layer evidence differs: {episode_id}")
            clean_run = Path(label["clean_run"])
            clean_files = copy_recordings(
                clean_run, out / "episodes" / episode_id / "paired_clean", label["source_hashes"]["clean"]
            )
            paired_recording_bytes += sum(record["bytes"] for record in clean_files.values())
            paired_input_tokens += model_input_tokens(jsonl(clean_run / "model_calls.jsonl"))
            clean_manifest = read(clean_run / "manifest.json")
            validate_pins("no_injection", clean_manifest, pins, None)
            require(
                all(
                    clean_manifest[field] == manifest[field]
                    for field in (
                        "seed",
                        "seat",
                        "opponent",
                        "seconds",
                        "interval_ms",
                        "opponent_interval_ms",
                        "sample_seed",
                    )
                )
                and clean_manifest["opponent_seat"] == manifest["opponent_seat"],
                "paired match settings differ",
            )
            clean_result = read(clean_run / "result.json")
            require(
                clean_result["seed"] == case["seed"] and clean_result["seat"] == case["seat"], "clean identity differs"
            )
            clean_outcome = clean_result["outcome"][case["seat"]]
            require(clean_outcome == label["clean_outcome"], "paired outcome differs")
            outcome_status = "paired_outcome_differs" if clean_outcome != -1 else "paired_outcome_same"
        replay = read(blind / "replays" / episode_id / "attempt.json")
        require(
            replay["valid"]
            and all(replay[f] for f in ("result_equal", "states_equal", "execution_equal", "events_equal")),
            "replay differs",
        )
        require(replay["episode_id"] == episode_id and replay["returncode"] == 0, "replay failed")
        replay_checks.append(
            {
                "episode_id": episode_id,
                "valid": True,
                "source_hashes": source_hashes,
                "elapsed_seconds": replay["elapsed_seconds"],
            }
        )
        probe_checks.append(
            {
                "episode_id": episode_id,
                "valid": True,
                "decisions": len(indices),
                "identity_tolerance": probe["identity_tolerance"],
                "probe_sha256": packaged_probe_sha256[episode_id],
            }
        )
        index.append(
            {
                "episode_id": episode_id,
                "split": "v0_development_scored",
                "scenario_reset_group": str(case["seed"]),
                "seed": case["seed"],
                "seat": case["seat"],
                "game_commit": manifest["code_commit"],
                "checkpoint_sha256": manifest["model"]["weights_sha256"],
                "verified_checkpoint_variant": verified_checkpoint_variant,
                "harness_sha256": manifest["harness_sha256"],
                "controller_sha256": manifest["controller_sha256"],
                "native_outcome": -1,
                "paired_clean_outcome": clean_outcome,
                "paired_outcome_status": outcome_status,
                "decision_count": len(states),
                "native_request_input_tokens": native_input_tokens,
                "recording_raw_bytes": sum(record["bytes"] for record in run_files.values()),
                "files": {"run": run_files, "paired_clean": clean_files},
                "blind_input_sha256": case["input_sha256"],
            }
        )
        provenance.append(
            {
                "episode_id": episode_id,
                "injection_layer": layer if label else None,
                "fault_family": label["fault_family"] if label else None,
                "fault_instance": case["fault_instance"],
                "changed_artifact_sha256": (
                    manifest["harness_sha256"]
                    if layer == "harness"
                    else (
                        manifest["controller_sha256"]
                        if layer == "controller"
                        else manifest["model"]["weights_sha256"] if layer == "weights" else None
                    )
                ),
                "verified_checkpoint_variant": verified_checkpoint_variant,
                "source_instance_name_mismatch": (
                    layer == "weights" and verified_checkpoint_variant != case["fault_instance"]
                ),
                "verified_component": layer if label else None,
                "annotation_status": label["annotation_status"] if label else "no_injection_control",
                "annotator": "full-audit-v5 automated saved-state comparison" if label else "no-injection setup",
                "verifier": "package-exp0-v0.py pinned-source and probe-signature audit",
                "verification_procedure": (
                    "saved-state repeat plus recorded replay" if label else "unchanged-checkpoint repeat"
                ),
                "reference": reference,
                "evidence_references": {
                    "run": f"episodes/{episode_id}/run/",
                    "paired_clean": f"episodes/{episode_id}/paired_clean/" if label else None,
                    "probe": f"probes/{episode_id}.json",
                    "replay_check": "checks/replay.jsonl",
                },
                "probe_signature_sha256": packaged_probe_sha256[episode_id],
                "paired_outcome_status": outcome_status,
                "sample_id": label["sample_id"] if label else None,
                "source_label_sha256": digest(json.dumps(label, sort_keys=True).encode()) if label else None,
                "disagreements": "not_collected; v0 does not require human window review",
                "label_revision": (
                    "checkpoint_strength_name_resolved_by_sha256"
                    if layer == "weights" and verified_checkpoint_variant != case["fault_instance"]
                    else "source_label_preserved"
                ),
            }
        )
    require(len(index) == 40 and len({row["episode_id"] for row in index}) == 40, "sample is not 40 distinct games")
    require(sum(row["decisions"] for row in probe_checks) == 2151, "decision total differs")
    write_jsonl(out / "index.jsonl", index)
    write_jsonl(out / "labels.jsonl", provenance)
    write_jsonl(out / "attempts.jsonl", [sanitize_attempt(row, unmatched) for row in attempts])
    write_json(out / "checks/replay-summary.json", replay_summary)
    write_json(out / "checks/probe-summary.json", probe_summary)
    write_json(out / "checks/probe-audit.json", probe_audit)
    write_jsonl(out / "checks/replay.jsonl", replay_checks)
    write_jsonl(out / "checks/probe-identity.jsonl", probe_checks)
    write_json(
        out / "sample-selection.json",
        {
            "source_sample_sha256": file_digest(freeze_path),
            "counts": freeze["counts"],
            "independent_seed_blocks": freeze["independent_seed_blocks"],
            "cases": [{key: value for key, value in case.items() if key != "source_run"} for case in freeze["cases"]],
        },
    )
    shutil.copyfile(blind / "protocol.json", out / "protocol.json")
    shutil.copyfile(blind / "prompt.txt", out / "prompt.txt")
    shutil.copyfile(blind / "schema.json", out / "diagnosis-schema.json")
    isolation = audit_diagnoser(source, freeze, out, packaged_probe_sha256)
    identity = audit_identity(identity_gate, pins, out)
    write_json(
        out / "schema.json",
        {
            "version": "tank-war-exp0-v0-stage-1",
            "unit": "complete episode",
            "index": "index.jsonl",
            "labels": "labels.jsonl; development labels only; never mount for diagnosis",
            "label_fields": [
                "injection_layer",
                "fault_family",
                "fault_instance",
                "changed_artifact_sha256",
                "verified_component",
                "annotation_status",
                "annotator",
                "verifier",
                "verification_procedure",
                "evidence_references",
                "reference",
                "paired_outcome_status",
                "disagreements",
                "label_revision",
            ],
            "attempts": "attempts.jsonl; all 1330 original attempts with paths replaced by opaque IDs",
            "trace_encoding": "UTF-8 JSONL compressed as deterministic gzip; source and package SHA-256 in index",
            "native_request_input_tokens": (
                "sum of recorded per-question inputTokens audits; not GPU work or billing usage"
            ),
            "diagnoser_input_allowlist": ["inputs/<episode_id>.json", "probes/<episode_id>.json"],
            "outcome_statuses": ["paired_outcome_differs", "paired_outcome_same", "untested"],
        },
    )
    all_files = sorted(path for path in out.rglob("*") if path.is_file())
    file_hashes = {str(path.relative_to(out)): file_digest(path) for path in all_files}
    write_json(out / "file-hashes.json", file_hashes)
    summary = {
        "status": "private_v0_stage_for_review",
        "sample_sha256": file_digest(freeze_path),
        "counts": dict(Counter(row["verified_component"] or "no_injection" for row in provenance)),
        "episodes": len(index),
        "independent_seed_groups": len({row["scenario_reset_group"] for row in index}),
        "decisions": sum(row["decision_count"] for row in index),
        "questions": 2 * sum(row["decision_count"] for row in index),
        "selected_native_request_input_tokens": selected_input_tokens,
        "paired_clean_native_request_input_tokens": paired_input_tokens,
        "historical_diagnoser_input_tokens": isolation["input_tokens"],
        "historical_diagnoser_output_tokens": isolation["output_tokens"],
        "selected_recording_raw_bytes": selected_recording_bytes,
        "paired_clean_recording_raw_bytes": paired_recording_bytes,
        "package_hashed_file_bytes": sum(path.stat().st_size for path in all_files),
        "all_attempts": len(attempts),
        "attempt_status_counts": dict(Counter(row["status"] for row in attempts)),
        "replays_valid": len(replay_checks),
        "probe_cases_valid": len(probe_checks),
        "probe_decisions": sum(row["decisions"] for row in probe_checks),
        "diagnoser_isolation_calls_valid": isolation["valid"],
        "identity_calls_valid": identity["calls"],
        "paired_outcome_differs": sum(row["paired_outcome_status"] == "paired_outcome_differs" for row in index),
        "checkpoint_name_mismatches": sum(row["source_instance_name_mismatch"] for row in provenance),
        "all_selected_outcomes": "loss",
        "source_pin_file_sha256": file_digest(pins_path),
        "file_hashes_sha256": file_digest(out / "file-hashes.json"),
    }
    write_json(out / "stage-summary.json", summary)
    return summary


def rows_by_episode(path: Path, episode_ids: set[str]) -> dict[str, dict[str, Any]]:
    rows = jsonl(path)
    by_episode = {row["episode_id"]: row for row in rows}
    require(len(rows) == len(by_episode) and set(by_episode) == episode_ids, f"episode rows differ: {path}")
    return by_episode


def verified_recording(
    out: Path, episode_id: str, kind: str, files: dict[str, Any], expected_hashes: dict[str, str]
) -> dict[str, bytes]:
    require(set(files) == set(SOURCE_FILES), f"recording files incomplete: {episode_id}/{kind}")
    contents = {}
    for name in SOURCE_FILES:
        record = files[name]
        relative = f"episodes/{episode_id}/{kind}/{name}{'.gz' if name.endswith('.jsonl') else ''}"
        require(record["path"] == relative, f"recording path differs: {episode_id}/{kind}/{name}")
        path = out / relative
        require(record["package_sha256"] == expected_hashes[relative], f"recording package hash differs: {relative}")
        payload = path.read_bytes()
        contents[name] = gzip.decompress(payload) if name.endswith(".jsonl") else payload
        require(digest(contents[name]) == record["source_sha256"], f"recording source hash differs: {relative}")
        require(len(contents[name]) == record["bytes"], f"recording byte count differs: {relative}")
    return contents


def decoded_jsonl(data: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in data.splitlines() if line]


def verify_diagnoser_checks(
    out: Path, index_by_episode: dict[str, dict[str, Any]], probe_sha256: dict[str, str], summary: dict[str, Any]
) -> None:
    diagnosis = jsonl(out / "checks/diagnoser-isolation.jsonl")
    expected_calls = {
        (arm, mode, episode_id)
        for arm, modes in DIAGNOSIS_MODES.items()
        for mode in modes
        for episode_id in index_by_episode
    }
    require(
        len(diagnosis) == summary["diagnoser_isolation_calls_valid"] == 320
        and {(row["arm"], row["mode"], row["episode_id"]) for row in diagnosis} == expected_calls,
        "diagnoser isolation calls differ",
    )
    require(
        all(
            row["valid"] is True
            and row["tool_call"] is False
            and row["input_sha256"] == index_by_episode[row["episode_id"]]["blind_input_sha256"]
            and row["checks_sha256"]
            == (probe_sha256[row["episode_id"]] if row["mode"].startswith("probe") else digest(b"null"))
            for row in diagnosis
        ),
        "diagnoser evidence differs",
    )
    diagnosis_summary = read(out / "checks/diagnoser-isolation-summary.json")
    require(
        diagnosis_summary["calls"] == diagnosis_summary["valid"] == 320
        and diagnosis_summary["tool_calls"] == 0
        and diagnosis_summary["arms"] == Counter(row["arm"] for row in diagnosis)
        and diagnosis_summary["input_tokens"]
        == summary["historical_diagnoser_input_tokens"]
        == sum(row["input_tokens"] for row in diagnosis)
        and diagnosis_summary["output_tokens"]
        == summary["historical_diagnoser_output_tokens"]
        == sum(row["output_tokens"] for row in diagnosis),
        "diagnoser summary differs",
    )


def verify_identity_checks(out: Path, pins: dict[str, Any], summary: dict[str, Any]) -> None:
    identity = jsonl(out / "checks/identity-calls.jsonl")
    identity_summary = read(out / "checks/identity-summary.json")
    tolerance = pins["diagnosis"]["probability_identity_tolerance"]
    require(
        len(identity) == summary["identity_calls_valid"] == identity_summary["calls"] == 891
        and identity_summary["failed_calls"] == 0
        and identity_summary["tolerance"] == tolerance
        and identity_summary["maximum_probability_difference"] <= tolerance
        and all(
            row["passed"] is True
            and row["too_long"] is False
            and row["truncated"] is False
            and row["maximum_probability_difference"] <= tolerance
            for row in identity
        ),
        "identity calls failed",
    )


def verify(out: Path, pins_path: Path = DEFAULT_PINS) -> dict[str, Any]:
    summary = read(out / "stage-summary.json")
    pins = read(pins_path)
    require(file_digest(pins_path) == summary["source_pin_file_sha256"], "frozen pin file differs")
    require(summary["sample_sha256"] == pins["existing_sample"]["sample_sha256"], "sample pin differs")
    hashes = read(out / "file-hashes.json")
    actual_files = {str(path.relative_to(out)) for path in out.rglob("*") if path.is_file() or path.is_symlink()}
    require(actual_files == set(hashes) | {"file-hashes.json", "stage-summary.json"}, "package file list incomplete")
    for name, expected in hashes.items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, f"unsafe package path: {name}")
        path = out / name
        require(
            path.is_file() and not path.is_symlink() and file_digest(path) == expected, f"package file differs: {name}"
        )
    require(file_digest(out / "file-hashes.json") == summary["file_hashes_sha256"], "file hash list differs")
    index = jsonl(out / "index.jsonl")
    labels = jsonl(out / "labels.jsonl")
    require(len(index) == len(labels) == summary["episodes"] == 40, "index/label count differs")
    index_by_episode = {row["episode_id"]: row for row in index}
    episode_ids = set(index_by_episode)
    require(len(episode_ids) == 40, "duplicate episode in index")
    require(
        all(Path(episode_id).name == episode_id and episode_id not in {".", ".."} for episode_id in episode_ids),
        "unsafe episode id",
    )
    label_by_episode = {row["episode_id"]: row for row in labels}
    require(len(label_by_episode) == 40 and set(label_by_episode) == episode_ids, "label episodes differ")
    selection = read(out / "sample-selection.json")
    selected = {case["episode_id"]: case for case in selection["cases"]}
    require(
        len(selection["cases"]) == len(selected) == 40 and set(selected) == episode_ids, "selection episodes differ"
    )
    require(selection["source_sample_sha256"] == summary["sample_sha256"], "selection hash differs")
    require(selection["counts"] == pins["existing_sample"]["counts"], "selection composition differs")
    require(
        selection["independent_seed_blocks"] == pins["existing_sample"]["independent_seed_blocks"],
        "selection seed groups differ",
    )
    require(
        Counter(case["technical_label"] for case in selected.values()) == pins["existing_sample"]["counts"],
        "selected technical labels differ",
    )
    require(
        Counter(row["verified_component"] or "no_injection" for row in labels)
        == summary["counts"]
        == pins["existing_sample"]["counts"],
        "label composition differs",
    )
    replay = rows_by_episode(out / "checks/replay.jsonl", episode_ids)
    probe_checks = rows_by_episode(out / "checks/probe-identity.jsonl", episode_ids)
    replay_summary = read(out / "checks/replay-summary.json")
    probe_summary = read(out / "checks/probe-summary.json")
    probe_audit = read(out / "checks/probe-audit.json")
    require(all(row["valid"] is True for row in replay.values()), "replay check failed")
    require(all(row["valid"] is True for row in probe_checks.values()), "saved-state probe check failed")
    require(replay_summary["valid"] == 40 and not replay_summary["invalid"], "replay summary failed")
    require(probe_summary["cases"] == 40 and probe_summary["decisions"] == 2151, "probe summary failed")
    require(probe_audit["valid"] == 40 and not probe_audit["invalid"], "probe audit failed")
    require(summary["replays_valid"] == summary["probe_cases_valid"] == 40, "episode check count differs")
    probe_sha256 = {}
    selected_bytes = paired_bytes = selected_tokens = paired_tokens = 0
    for row in index:
        episode_id = row["episode_id"]
        case = selected[episode_id]
        label = label_by_episode[episode_id]
        layer = case["technical_label"]
        require(layer in LAYERS, f"unknown technical label: {episode_id}")
        require(
            row["seed"] == case["seed"]
            and row["seat"] == case["seat"]
            and row["scenario_reset_group"] == str(case["seed"])
            and row["decision_count"] == case["decisions"]
            and row["blind_input_sha256"] == case["input_sha256"],
            f"index selection differs: {episode_id}",
        )
        require(
            label["verified_component"] == (None if layer == "no_injection" else layer), f"label differs: {episode_id}"
        )
        require(label["injection_layer"] == label["verified_component"], f"injection label differs: {episode_id}")
        require(label["paired_outcome_status"] == row["paired_outcome_status"], f"outcome label differs: {episode_id}")
        require(replay[episode_id]["source_hashes"] == case["source_hashes"], f"replay source differs: {episode_id}")
        require(
            {name: record["source_sha256"] for name, record in row["files"]["run"].items()} == case["source_hashes"],
            f"selected source hashes differ: {episode_id}",
        )
        run = verified_recording(out, episode_id, "run", row["files"]["run"], hashes)
        selected_bytes += sum(record["bytes"] for record in row["files"]["run"].values())
        manifest = json.loads(run["manifest.json"])
        result = json.loads(run["result.json"])
        verified_variant = validate_pins(layer, manifest, pins, case["fault_instance"])
        require(
            manifest["seed"] == result["seed"] == row["seed"] and manifest["seat"] == result["seat"] == row["seat"],
            f"run identity differs: {episode_id}",
        )
        require(
            result["ended"] and not result["truncated"] and result["outcome"][row["seat"]] == -1,
            f"selected outcome differs: {episode_id}",
        )
        require(
            row["native_outcome"] == -1
            and row["game_commit"] == manifest["code_commit"]
            and row["checkpoint_sha256"] == manifest["model"]["weights_sha256"]
            and row["harness_sha256"] == manifest["harness_sha256"]
            and row["controller_sha256"] == manifest["controller_sha256"]
            and row["verified_checkpoint_variant"] == label["verified_checkpoint_variant"] == verified_variant,
            f"run metadata differs: {episode_id}",
        )
        states = decoded_jsonl(run["states.jsonl"])
        decisions = decoded_jsonl(run["decisions.jsonl"])
        execution = decoded_jsonl(run["execution.jsonl"])
        calls = decoded_jsonl(run["model_calls.jsonl"])
        require(
            len(states) == len(execution) == len(calls) == row["decision_count"],
            f"decision count differs: {episode_id}",
        )
        require(len(decisions) == 2 * len(states), f"question count differs: {episode_id}")
        require(
            all(
                states[i]["decision_index"] == execution[i]["decision_index"] == i
                and all(
                    decisions[2 * i + question]["decision_index"] == i
                    and decisions[2 * i + question]["state_hash"] == states[i]["state_hash"]
                    for question in (0, 1)
                )
                for i in range(len(states))
            ),
            f"decision alignment differs: {episode_id}",
        )
        native_tokens = model_input_tokens(calls)
        selected_tokens += native_tokens
        require(row["native_request_input_tokens"] == native_tokens, f"native token count differs: {episode_id}")
        require(
            row["recording_raw_bytes"] == sum(record["bytes"] for record in row["files"]["run"].values()),
            f"recording byte count differs: {episode_id}",
        )
        probe_path = out / "probes" / f"{episode_id}.json"
        probe_sha256[episode_id] = file_digest(probe_path)
        probe = read(probe_path)
        indices = probe["indices"]
        require(
            [item["decision_index"] for item in indices] == list(range(len(states))),
            f"probe coverage differs: {episode_id}",
        )
        require(
            probe["identity_tolerance"]
            == probe_checks[episode_id]["identity_tolerance"]
            == pins["diagnosis"]["probability_identity_tolerance"],
            f"probe tolerance differs: {episode_id}",
        )
        require(
            probe_checks[episode_id]["decisions"] == row["decision_count"]
            and probe_checks[episode_id]["probe_sha256"]
            == label["probe_signature_sha256"]
            == probe_sha256[episode_id],
            f"probe hash or count differs: {episode_id}",
        )
        input_path = out / "inputs" / f"{episode_id}.json"
        require(file_digest(input_path) == row["blind_input_sha256"], f"blind input differs: {episode_id}")
        require(
            not any(term in input_path.read_text().lower() for term in FORBIDDEN_INPUT_TERMS),
            f"blind input leaks construction metadata: {episode_id}",
        )
        if layer == "no_injection":
            require(row["files"]["paired_clean"] is None, f"control has clean pair: {episode_id}")
            require(
                row["paired_clean_outcome"] is None and row["paired_outcome_status"] == "untested",
                f"control outcome differs: {episode_id}",
            )
            require(all(valid_probe(layer, item) for item in indices), f"control probe differs: {episode_id}")
            require(label["reference"] is None, f"control reference differs: {episode_id}")
            continue
        require(label["annotation_status"] == "machine_verified", f"label provenance differs: {episode_id}")
        reference = label["reference"]
        position = case["technical_reference_index"]
        require(
            reference["verified"] is True
            and reference["decision_index"] == position
            and states[position]["state_hash"] == reference["state_hash"]
            and valid_probe(layer, indices[position]),
            f"local component evidence differs: {episode_id}",
        )
        clean_files = row["files"]["paired_clean"]
        require(isinstance(clean_files, dict), f"clean recording missing: {episode_id}")
        clean = verified_recording(out, episode_id, "paired_clean", clean_files, hashes)
        paired_bytes += sum(record["bytes"] for record in clean_files.values())
        paired_tokens += model_input_tokens(decoded_jsonl(clean["model_calls.jsonl"]))
        clean_manifest = json.loads(clean["manifest.json"])
        clean_result = json.loads(clean["result.json"])
        validate_pins("no_injection", clean_manifest, pins, None)
        require(
            all(
                clean_manifest[field] == manifest[field]
                for field in (
                    "seed",
                    "seat",
                    "opponent",
                    "seconds",
                    "interval_ms",
                    "opponent_interval_ms",
                    "sample_seed",
                )
            )
            and clean_manifest["opponent_seat"] == manifest["opponent_seat"],
            f"paired match settings differ: {episode_id}",
        )
        require(
            clean_result["seed"] == row["seed"] and clean_result["seat"] == row["seat"],
            f"clean identity differs: {episode_id}",
        )
        require(clean_result["ended"] and not clean_result["truncated"], f"clean match incomplete: {episode_id}")
        clean_outcome = clean_result["outcome"][row["seat"]]
        expected_status = "paired_outcome_differs" if clean_outcome != -1 else "paired_outcome_same"
        require(
            row["paired_clean_outcome"] == clean_outcome and row["paired_outcome_status"] == expected_status,
            f"paired outcome differs: {episode_id}",
        )
    require(summary["all_selected_outcomes"] == "loss", "selected outcome summary differs")
    require(
        summary["independent_seed_groups"] == len({row["seed"] for row in index}) == 19, "seed group total differs"
    )
    require(summary["decisions"] == sum(row["decision_count"] for row in index) == 2151, "decision total differs")
    require(summary["questions"] == 2 * summary["decisions"], "question total differs")
    require(summary["selected_recording_raw_bytes"] == selected_bytes, "selected recording bytes differ")
    require(summary["paired_clean_recording_raw_bytes"] == paired_bytes, "paired recording bytes differ")
    require(summary["selected_native_request_input_tokens"] == selected_tokens, "selected native tokens differ")
    require(summary["paired_clean_native_request_input_tokens"] == paired_tokens, "paired native tokens differ")
    require(
        summary["paired_outcome_differs"]
        == sum(row["paired_outcome_status"] == "paired_outcome_differs" for row in index),
        "paired outcome total differs",
    )
    require(
        summary["checkpoint_name_mismatches"] == sum(row["source_instance_name_mismatch"] for row in labels),
        "checkpoint mismatch total differs",
    )
    attempts = jsonl(out / "attempts.jsonl")
    require(len(attempts) == summary["all_attempts"] == 1330, "attempt count differs")
    require(Counter(row["status"] for row in attempts) == summary["attempt_status_counts"], "attempt statuses differ")
    require(
        sum(row["decisions"] for row in probe_checks.values()) == summary["probe_decisions"] == 2151,
        "probe count differs",
    )
    verify_diagnoser_checks(out, index_by_episode, probe_sha256, summary)
    verify_identity_checks(out, pins, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Existing v4 private source root; stage only")
    parser.add_argument(
        "--pins", type=Path, help="Merged v0 freeze.json; defaults to the checked-in file for verification"
    )
    parser.add_argument("--identity-gate", type=Path, help="Archived clean-checkpoint identity gate; stage only")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.source or args.identity_gate:
        require(
            args.source is not None and args.pins is not None and args.identity_gate is not None,
            "stage needs --source, --pins, and --identity-gate",
        )
        stage(args.source, args.pins, args.identity_gate, args.out)
    print(json.dumps(verify(args.out, args.pins or DEFAULT_PINS), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
