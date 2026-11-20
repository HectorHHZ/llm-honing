"""The Tank War v0 package verifies source bytes and keeps checkpoint identity explicit."""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "integrations/tank-war/scripts/package-exp0-v0.py"
FREEZE = Path(__file__).resolve().parents[1] / "integrations/tank-war/exp0/v0/freeze.json"
SPEC = importlib.util.spec_from_file_location("tank_exp0_v0_package", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


@pytest.mark.unit
def test_checkpoint_variant_follows_recorded_digest() -> None:
    pins = json.loads(FREEZE.read_text())
    manifest = {
        "code_commit": pins["game"]["recorded_code_commit"],
        "sources": pins["game"]["sources_sha256"],
        "model": {
            "source_commit": pins["model"]["source_revision"],
            "context_limit": pins["model"]["max_input_tokens"],
            "weights_sha256": pins["selected_faults"]["weights"][1]["weights_sha256"],
        },
        "harness_sha256": pins["clean_harness"]["recorded_harness_sha256"],
        "controller_sha256": pins["clean_harness"]["recorded_controller_sha256"],
    }
    assert module.validate_pins("weights", manifest, pins, "reverse_strength_3p0-v1") == "reverse_strength_4p0-v1"
    manifest["model"]["weights_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="frozen checkpoint"):
        module.validate_pins("weights", manifest, pins, "reverse_strength_3p0-v1")


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
