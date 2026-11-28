"""The Tank War v0 package verifies source bytes and keeps checkpoint identity explicit."""

from __future__ import annotations

import gzip
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "integrations/tank-war/scripts/package-exp0-v0.py"
FREEZE = Path(__file__).resolve().parents[1] / "integrations/tank-war/exp0/v0/freeze.json"
SPEC = importlib.util.spec_from_file_location("tank_exp0_v0_package", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def pinned_manifest(pins: dict) -> dict:
    seat = {
        "controllerMode": pins["clean_harness"]["controller_mode"],
        "contextVersion": pins["clean_harness"]["context_version"],
        "assistEnabled": pins["clean_harness"]["assist_enabled"],
        "assistProfile": pins["clean_harness"]["assist_profile"],
        "fireGuardEnabled": pins["clean_harness"]["fire_guard_enabled"],
    }
    return {
        "kind": "evaluation",
        "edition": "tank-war",
        "opponent": pins["game"]["opponent"],
        "seconds": pins["game"]["sim_seconds"],
        "interval_ms": pins["game"]["model_interval_ms"],
        "opponent_interval_ms": pins["game"]["opponent_interval_ms"],
        "sample_seed": None,
        "node": pins["game"]["node_version"],
        "agent_seat": seat,
        "opponent_seat": {**seat, "decisionVersion": "baseline"},
        "code_commit": pins["game"]["recorded_code_commit"],
        "sources": pins["game"]["sources_sha256"],
        "model": {
            "source_commit": pins["model"]["source_revision"],
            "context_limit": pins["model"]["max_input_tokens"],
            "weights_sha256": pins["model"]["clean_weights_sha256"],
        },
        "harness_sha256": pins["clean_harness"]["recorded_harness_sha256"],
        "controller_sha256": pins["clean_harness"]["recorded_controller_sha256"],
    }


@pytest.mark.unit
def test_checkpoint_variant_follows_recorded_digest() -> None:
    pins = json.loads(FREEZE.read_text())
    manifest = pinned_manifest(pins)
    manifest["model"]["weights_sha256"] = pins["selected_faults"]["weights"][1]["weights_sha256"]
    assert module.validate_pins("weights", manifest, pins, "reverse_strength_3p0-v1") == "reverse_strength_4p0-v1"
    manifest["model"]["weights_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="frozen checkpoint"):
        module.validate_pins("weights", manifest, pins, "reverse_strength_3p0-v1")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "altered"),
    [("seconds", 60), ("interval_ms", 400), ("opponent_interval_ms", 600), ("sample_seed", 7)],
)
def test_clean_pair_rejects_changed_match_settings(field: str, altered: int) -> None:
    pins = json.loads(FREEZE.read_text())
    clean = pinned_manifest(pins)
    clean[field] = altered
    with pytest.raises(ValueError, match=f"match setting differs: {field}"):
        module.validate_pins("no_injection", clean, pins, None)


@pytest.mark.unit
def test_diagnoser_requires_the_packaged_probe_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    blind = source / "prospective-blind-v2"
    diagnosis = source / "prospective-diagnosis-v2"
    module.write_json(blind / "sample-freeze.json", {"cases": 40})
    cases = [{"episode_id": f"e{i}", "input_sha256": "a" * 64} for i in range(40)]
    packaged_hashes = {case["episode_id"]: "b" * 64 for case in cases}
    runs = {}
    for arm, modes in module.DIAGNOSIS_MODES.items():
        root = diagnosis / arm
        module.write_json(
            root / "freeze.json",
            {
                "cases": [
                    {"episode_id": case["episode_id"], "input_sha256": case["input_sha256"], "probes_sha256": "a" * 64}
                    for case in cases
                ]
            },
        )
        runs[arm] = {"freeze_sha256": module.file_digest(root / "freeze.json"), "modes": modes}
    module.write_json(
        diagnosis / "manifest.json",
        {
            "sample_freeze_sha256": module.file_digest(blind / "sample-freeze.json"),
            "runs": runs,
        },
    )
    with pytest.raises(ValueError, match="packaged probe differs from diagnosis"):
        module.audit_diagnoser(source, {"cases": cases}, tmp_path / "output", packaged_hashes)


def synthetic_invalid_package(tmp_path: Path, *, replay_valid: bool) -> Path:
    out = tmp_path / "package"
    pins = json.loads(FREEZE.read_text())
    layers = [layer for layer in module.LAYERS for _ in range(10)]
    cases = [
        {
            "episode_id": f"e{i}",
            "technical_label": layer,
            "seed": i % 19,
            "seat": "blue",
            "decisions": 53 if i < 39 else 84,
            "input_sha256": "a" * 64,
            "source_hashes": {},
        }
        for i, layer in enumerate(layers)
    ]
    index = [
        {
            "episode_id": case["episode_id"],
            "seed": case["seed"],
            "seat": case["seat"],
            "scenario_reset_group": str(case["seed"]),
            "decision_count": case["decisions"],
            "blind_input_sha256": case["input_sha256"],
            "paired_outcome_status": (
                "untested" if case["technical_label"] == "no_injection" else "paired_outcome_same"
            ),
            "files": {"run": {}, "paired_clean": None},
        }
        for case in cases
    ]
    labels = [
        {
            "episode_id": case["episode_id"],
            "verified_component": None if case["technical_label"] == "no_injection" else case["technical_label"],
            "injection_layer": None if case["technical_label"] == "no_injection" else case["technical_label"],
            "paired_outcome_status": (
                "untested" if case["technical_label"] == "no_injection" else "paired_outcome_same"
            ),
        }
        for case in cases
    ]
    module.write_jsonl(out / "index.jsonl", index)
    module.write_jsonl(out / "labels.jsonl", labels)
    module.write_json(
        out / "sample-selection.json",
        {
            "cases": cases,
            "counts": pins["existing_sample"]["counts"],
            "source_sample_sha256": pins["existing_sample"]["sample_sha256"],
            "independent_seed_blocks": 19,
        },
    )
    module.write_jsonl(
        out / "checks/replay.jsonl",
        [{"episode_id": case["episode_id"], "valid": replay_valid, "source_hashes": {}} for case in cases],
    )
    module.write_jsonl(
        out / "checks/probe-identity.jsonl",
        [{"episode_id": case["episode_id"], "valid": True} for case in cases],
    )
    module.write_json(out / "checks/replay-summary.json", {"valid": 40, "invalid": []})
    module.write_json(out / "checks/probe-summary.json", {"cases": 40, "decisions": 2151})
    module.write_json(out / "checks/probe-audit.json", {"valid": 40, "invalid": []})
    module.write_json(
        out / "file-hashes.json",
        {str(path.relative_to(out)): module.file_digest(path) for path in out.rglob("*") if path.is_file()},
    )
    module.write_json(
        out / "stage-summary.json",
        {
            "source_pin_file_sha256": module.file_digest(FREEZE),
            "sample_sha256": pins["existing_sample"]["sample_sha256"],
            "file_hashes_sha256": module.file_digest(out / "file-hashes.json"),
            "episodes": 40,
            "counts": dict(Counter(label["verified_component"] or "no_injection" for label in labels)),
            "replays_valid": 40,
            "probe_cases_valid": 40,
        },
    )
    return out


@pytest.mark.unit
def test_offline_verifier_rejects_failed_replays_even_when_hashes_match(tmp_path: Path) -> None:
    out = synthetic_invalid_package(tmp_path, replay_valid=False)
    with pytest.raises(ValueError, match="replay check failed"):
        module.verify(out)


@pytest.mark.unit
def test_offline_verifier_requires_recording_files_for_every_episode(tmp_path: Path) -> None:
    out = synthetic_invalid_package(tmp_path, replay_valid=True)
    with pytest.raises(ValueError, match="recording files incomplete"):
        module.verify(out)


@pytest.mark.unit
def test_offline_verifier_rejects_failed_identity_calls(tmp_path: Path) -> None:
    pins = json.loads(FREEZE.read_text())
    out = tmp_path / "package"
    rows = [
        {"passed": True, "too_long": False, "truncated": False, "maximum_probability_difference": 0.0}
        for _ in range(891)
    ]
    rows[0]["passed"] = False
    module.write_jsonl(out / "checks/identity-calls.jsonl", rows)
    module.write_json(
        out / "checks/identity-summary.json",
        {"calls": 891, "failed_calls": 0, "tolerance": 0.01, "maximum_probability_difference": 0.0},
    )
    with pytest.raises(ValueError, match="identity calls failed"):
        module.verify_identity_checks(out, pins, {"identity_calls_valid": 891})


@pytest.mark.unit
@pytest.mark.parametrize("alteration", ["failed_call", "changed_probe"])
def test_offline_verifier_rejects_invalid_diagnoser_evidence(tmp_path: Path, alteration: str) -> None:
    out = tmp_path / "package"
    index = {f"e{i}": {"blind_input_sha256": "a" * 64} for i in range(40)}
    probes = dict.fromkeys(index, "b" * 64)
    rows = [
        {
            "arm": arm,
            "mode": mode,
            "episode_id": episode_id,
            "valid": True,
            "tool_call": False,
            "input_sha256": "a" * 64,
            "checks_sha256": probes[episode_id] if mode.startswith("probe") else module.digest(b"null"),
            "input_tokens": 1,
            "output_tokens": 1,
        }
        for arm, modes in module.DIAGNOSIS_MODES.items()
        for mode in modes
        for episode_id in index
    ]
    if alteration == "failed_call":
        rows[0]["valid"] = False
    else:
        probes["e0"] = "c" * 64
    module.write_jsonl(out / "checks/diagnoser-isolation.jsonl", rows)
    with pytest.raises(ValueError, match="diagnoser evidence differs"):
        module.verify_diagnoser_checks(out, index, probes, {"diagnoser_isolation_calls_valid": 320})


@pytest.mark.unit
def test_recordings_preserve_original_hashes_and_reject_symlinks(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    expected = {}
    for name in module.SOURCE_FILES:
        content = ("{}\n" if name.endswith(".jsonl") else "{}").encode()
        (source / name).write_bytes(content)
        expected[name] = module.digest(content)
    output = tmp_path / "package" / "episodes" / "case" / "run"
    copied = module.copy_recordings(source, output, expected)
    for name, record in copied.items():
        packaged = tmp_path / "package" / record["path"]
        data = gzip.decompress(packaged.read_bytes()) if name.endswith(".jsonl") else packaged.read_bytes()
        assert module.digest(data) == expected[name]
    (source / "states.jsonl").unlink()
    (source / "states.jsonl").symlink_to(source / "decisions.jsonl")
    with pytest.raises(ValueError, match="linked source"):
        module.copy_recordings(source, tmp_path / "other" / "episodes" / "case" / "run", expected)


@pytest.mark.unit
def test_component_probe_signatures_stay_separate() -> None:
    row = {
        "same_model_input": True,
        "selected_action_changed": False,
        "choice_changed": False,
        "command_changed": True,
        "execution_changed": True,
        "option_set_changed": False,
        "max_abs_probability_change": 0,
    }
    assert module.valid_probe("controller", row)
    assert not module.valid_probe("harness", row)
    assert not module.valid_probe("weights", row)


@pytest.mark.unit
def test_native_token_accounting_includes_branches_and_rejects_truncation() -> None:
    def audit(tokens: int) -> dict:
        return {"inputTokens": tokens, "maxInputTokens": 8192, "stateTruncated": False}

    call = {
        "reply": {"tokenAudit": {"motion": audit(100)}, "branches": {"forward": {"tokenAudit": {"weapon": audit(25)}}}}
    }
    assert module.model_input_tokens([call]) == 125
    call["reply"]["branches"]["forward"]["tokenAudit"]["weapon"]["stateTruncated"] = True
    with pytest.raises(ValueError, match="truncated"):
        module.model_input_tokens([call])
