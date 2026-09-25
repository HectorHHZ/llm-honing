"""Lossless evidence archiving and recovery with the optional native zstd CLI."""

import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/ra2/league"))
from archive_logs import archive
from storage import digest

pytestmark = pytest.mark.skipif(not shutil.which("zstd"), reason="Optional zstd CLI unavailable")


def verified_log(directory):
    original = b'{"tick":3,"text":"normal observation only"}\n' * 4000
    path = directory / "decisions.jsonl.gz"
    path.write_bytes(gzip.compress(original))
    evidence = {
        "auditPassed": True,
        "replayVerified": True,
        "artifacts": {"decisions.jsonl": hashlib.sha256(original).hexdigest()},
    }
    (directory / "evidence.json").write_text(json.dumps(evidence))
    (directory / "result.json").write_text('{"ended":true}')
    (directory / "log-storage.json").write_text('{"format":"gzip-jsonl"}')
    return original, path, (directory / "evidence.json").read_bytes()


def test_roundtrip_keeps_evidence_and_supports_current_and_frozen_readers(tmp_path):
    original, path, evidence = verified_log(tmp_path)
    report = archive(tmp_path)
    assert report["losslessVerified"] and not path.exists()
    assert digest(tmp_path / "decisions.jsonl") == hashlib.sha256(original).hexdigest()
    assert (tmp_path / "evidence.json").read_bytes() == evidence
    assert list(tmp_path.glob("log-storage.before-*.json"))
    restored = archive(tmp_path, restore=True)
    assert restored["format"] == "gzip-jsonl"
    assert gzip.decompress(path.read_bytes()) == original
    assert (tmp_path / "decisions.jsonl.zst").exists()
    assert (tmp_path / "evidence.json").read_bytes() == evidence
    # Re-archive a restored copy without overwriting the existing verified archive.
    compressed = (tmp_path / "decisions.jsonl.zst").read_bytes()
    archive(tmp_path)
    assert not path.exists() and (tmp_path / "decisions.jsonl.zst").read_bytes() == compressed


def test_corrupt_archive_never_removes_source(tmp_path):
    _, path, evidence = verified_log(tmp_path)
    before = path.read_bytes()
    (tmp_path / "decisions.jsonl.zst").write_bytes(b"not zstd")
    with pytest.raises(ValueError, match="Invalid zstd"):
        archive(tmp_path)
    assert path.read_bytes() == before and (tmp_path / "evidence.json").read_bytes() == evidence


def test_mismatched_or_unverified_evidence_never_removes_source(tmp_path):
    _, path, _ = verified_log(tmp_path)
    evidence_path = tmp_path / "evidence.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["artifacts"]["decisions.jsonl"] = "0" * 64
    evidence_path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="differs from verified"):
        archive(tmp_path)
    evidence["auditPassed"] = False
    evidence_path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="Only completed"):
        archive(tmp_path)
    assert path.exists() and not (tmp_path / "decisions.jsonl.zst").exists()
