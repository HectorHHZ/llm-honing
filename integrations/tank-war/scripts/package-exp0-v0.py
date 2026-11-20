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


def audit_diagnoser(source: Path, freeze: dict[str, Any], out: Path) -> dict[str, Any]:
    diagnosis = source / "prospective-diagnosis-v2"
    manifest = read(diagnosis / "manifest.json")
    require(
        manifest["sample_freeze_sha256"] == file_digest(source / "prospective-blind-v2/sample-freeze.json"),
        "diagnosis sample differs",
    )
    records = []
    for arm, arm_spec in manifest["runs"].items():
        arm_root = diagnosis / arm
        arm_freeze = read(arm_root / "freeze.json")
        require(file_digest(arm_root / "freeze.json") == arm_spec["freeze_sha256"], "diagnosis freeze differs")
        require(
            {case["episode_id"] for case in arm_freeze["cases"]} == {case["episode_id"] for case in freeze["cases"]},
            "diagnosis cases differ",
        )
        arm_cases = {case["episode_id"]: case for case in arm_freeze["cases"]}
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
                expected_checks = (
                    arm_cases[episode_id]["probes_sha256"] if mode.startswith("probe") else digest(b"null")
                )
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
                    }
                )
    require(len(records) == manifest["maximum_calls"] == 320, "diagnosis total differs")
    write_jsonl(out / "checks/diagnoser-isolation.jsonl", records)
    result = {
        "calls": len(records),
        "valid": len(records),
        "tool_calls": 0,
        "arms": dict(Counter(row["arm"] for row in records)),
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
    for case in freeze["cases"]:
        episode_id = case["episode_id"]
        layer = case["technical_label"]
        require(layer in LAYERS, f"unknown class: {layer}")
        source_run = Path(case["source_run"])
        source_hashes = case["source_hashes"]
        run_files = copy_recordings(source_run, out / "episodes" / episode_id / "run", source_hashes)
        manifest = read(source_run / "manifest.json")
        result = read(source_run / "result.json")
        verified_checkpoint_variant = validate_pins(layer, manifest, pins, case["fault_instance"])
        require(result["seed"] == case["seed"] and result["seat"] == case["seat"], "result identity differs")
        require(result["ended"] and not result["truncated"], f"incomplete episode: {episode_id}")
        require(result["outcome"][case["seat"]] == -1, f"non-loss in selected sample: {episode_id}")
        states = jsonl(source_run / "states.jsonl")
        decisions = jsonl(source_run / "decisions.jsonl")
        execution = jsonl(source_run / "execution.jsonl")
        require(len(states) == len(execution) == case["decisions"], "decision alignment differs")
        require(len(decisions) == 2 * len(states), "question count differs")
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
        require(file_digest(input_path) == case["input_sha256"], f"blind input differs: {episode_id}")
        serialized_input = input_path.read_text()
        require(
            not any(term in serialized_input.lower() for term in FORBIDDEN_INPUT_TERMS)
            and str(case["seed"]) not in serialized_input
            and str(source_run) not in serialized_input,
            f"blind input leaks construction metadata: {episode_id}",
        )
        (out / "inputs").mkdir(exist_ok=True)
        shutil.copyfile(input_path, out / "inputs" / f"{episode_id}.json")
        probe_path = blind / "probes" / episode_id / "probes.json"
        probe = read(probe_path)
        indices = probe["indices"]
        require([row["decision_index"] for row in indices] == list(range(len(states))), "probe coverage differs")
        (out / "probes").mkdir(exist_ok=True)
        shutil.copyfile(probe_path, out / "probes" / f"{episode_id}.json")
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
            validate_pins("no_injection", read(clean_run / "manifest.json"), pins, None)
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
                "probe_sha256": file_digest(probe_path),
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
                "verified_checkpoint_variant": verified_checkpoint_variant,
                "source_instance_name_mismatch": (
                    layer == "weights" and verified_checkpoint_variant != case["fault_instance"]
                ),
                "verified_component": layer if label else None,
                "annotation_status": label["annotation_status"] if label else "no_injection_control",
                "verification_procedure": (
                    "saved-state repeat plus recorded replay" if label else "unchanged-checkpoint repeat"
                ),
                "reference": reference,
                "probe_signature_sha256": file_digest(probe_path),
                "paired_outcome_status": outcome_status,
                "sample_id": label["sample_id"] if label else None,
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
    isolation = audit_diagnoser(source, freeze, out)
    identity = audit_identity(identity_gate, pins, out)
    write_json(
        out / "schema.json",
        {
            "version": "tank-war-exp0-v0-stage-1",
            "unit": "complete episode",
            "index": "index.jsonl",
            "labels": "labels.jsonl; development labels only; never mount for diagnosis",
            "attempts": "attempts.jsonl; all 1330 original attempts with paths replaced by opaque IDs",
            "trace_encoding": "UTF-8 JSONL compressed as deterministic gzip; source and package SHA-256 in index",
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


def verify(out: Path) -> dict[str, Any]:
    summary = read(out / "stage-summary.json")
    hashes = read(out / "file-hashes.json")
    for name, expected in hashes.items():
        path = out / name
        require(path.is_file() and file_digest(path) == expected, f"package file differs: {name}")
    require(file_digest(out / "file-hashes.json") == summary["file_hashes_sha256"], "file hash list differs")
    index = jsonl(out / "index.jsonl")
    labels = jsonl(out / "labels.jsonl")
    require(len(index) == len(labels) == summary["episodes"] == 40, "index/label count differs")
    require(
        Counter(row["verified_component"] or "no_injection" for row in labels) == summary["counts"],
        "label composition differs",
    )
    for row in index:
        for files in row["files"].values():
            if files is None:
                continue
            for name, record in files.items():
                path = out / record["path"]
                data = gzip.decompress(path.read_bytes()) if name.endswith(".jsonl") else path.read_bytes()
                require(digest(data) == record["source_sha256"], f"unpacked source differs: {path}")
    require(len(jsonl(out / "attempts.jsonl")) == summary["all_attempts"] == 1330, "attempt count differs")
    require(len(jsonl(out / "checks/replay.jsonl")) == summary["replays_valid"] == 40, "replay count differs")
    require(
        sum(row["decisions"] for row in jsonl(out / "checks/probe-identity.jsonl"))
        == summary["probe_decisions"]
        == 2151,
        "probe count differs",
    )
    require(
        len(jsonl(out / "checks/diagnoser-isolation.jsonl")) == summary["diagnoser_isolation_calls_valid"] == 320,
        "diagnoser isolation count differs",
    )
    require(
        len(jsonl(out / "checks/identity-calls.jsonl")) == summary["identity_calls_valid"] == 891,
        "identity count differs",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Existing v4 private source root; stage only")
    parser.add_argument("--pins", type=Path, help="Merged v0 freeze.json; stage only")
    parser.add_argument("--identity-gate", type=Path, help="Archived clean-checkpoint identity gate; stage only")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.source or args.pins or args.identity_gate:
        require(
            args.source is not None and args.pins is not None and args.identity_gate is not None,
            "stage needs --source, --pins, and --identity-gate",
        )
        stage(args.source, args.pins, args.identity_gate, args.out)
    print(json.dumps(verify(args.out), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
