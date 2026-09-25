"""CPU checks for evidence gates and resumable job supervision."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

LEAGUE = Path(__file__).resolve().parents[1] / "examples/ra2/league"


def load(name):
    spec = importlib.util.spec_from_file_location("ra2_league_" + name, LEAGUE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_promotion_rejects_teacher_truncation_and_incomplete_matrix():
    gate = load("gates")
    protocol = json.loads((LEAGUE / "protocol.json").read_text())
    spec = protocol["sealedTest"]
    rows = [
        {
            "map": m,
            "seed": seed,
            "seat": seat,
            "controller": "model",
            "normalView": True,
            "fallbackCount": 0,
            "auditPassed": True,
            "replayVerified": True,
            "error": None,
            "split": "sealedTest",
            "checkpointSha256": "fixture-checkpoint",
            "ended": True,
            "truncated": False,
            "outcome": {seat: 1},
            **{key: protocol[key] for key in ("engineCommit", "baselineRevision", "cadenceTicks", "engineActionCap")},
            "decisionInterval": protocol["cadenceTicks"],
        }
        for m in spec["maps"]
        for seed in spec["seeds"]
        for seat in spec["seats"]
    ]
    assert gate.evaluate_gate(rows, protocol)["passed"]
    assert not gate.evaluate_gate(rows[:-1], protocol)["passed"]
    assert not gate.evaluate_gate([*rows, rows[0]], protocol)["passed"]
    bad = [dict(row, controller="teacher") for row in rows]
    assert not gate.evaluate_gate(bad, protocol)["passed"]
    bad = [dict(row, truncated=True, ended=False) for row in rows]
    result = gate.evaluate_gate(bad, protocol)
    assert result["wins"] == 0 and not result["passed"]
    bad = [dict(row, checkpointSha256=str(i)) for i, row in enumerate(rows)]
    assert "mixed_checkpoints" in gate.evaluate_gate(bad, protocol)["failures"]
    assert 0.89 < gate.wilson_lower(94, 96) < 0.95
    assert (
        "protocol_mismatch" in gate.evaluate_gate([dict(r, engineCommit="other") for r in rows], protocol)["failures"]
    )


@pytest.mark.unit
def test_campaign_resume_does_not_execute_completed_job_twice(tmp_path):
    campaign = load("campaign")
    result = tmp_path / "result.json"
    command = [
        sys.executable,
        "-c",
        "from pathlib import Path; p=Path('result.json'); "
        "p.write_text(str(int(p.read_text())+1) if p.exists() else '1')",
    ]
    (tmp_path / "plan.json").write_text(
        json.dumps({"jobs": [{"id": "once", "command": command, "cwd": str(tmp_path), "result": str(result)}]})
    )
    campaign.run(tmp_path, 1)
    campaign.run(tmp_path, 1)
    assert result.read_text() == "1"
    state = json.loads((tmp_path / "progress.json").read_text())
    assert state["done"] == ["once"] and not state["failed"]


@pytest.mark.unit
def test_campaign_vanished_process_requires_inspection(tmp_path):
    campaign = load("campaign")
    spec = {"id": "vanished", "command": ["false"], "cwd": str(tmp_path), "result": str(tmp_path / "never")}
    (tmp_path / "plan.json").write_text(json.dumps({"jobs": [spec]}))
    (tmp_path / "status").mkdir()
    (tmp_path / "status/vanished.live.json").write_text(json.dumps({"pid": 999999999, "processStart": "1"}))
    with pytest.raises(RuntimeError, match="inspection"):
        campaign.run(tmp_path, 1)


@pytest.mark.unit
def test_training_input_deduplication_removes_cross_split_overlap():
    dataset = load("dataset")
    a = {"ids": [1, 2], "markers": [1], "label": 0}
    b = {"ids": [1, 3], "markers": [1], "label": 0}
    training, validation, audit = dataset.separate_unique([a, a], [a, b])
    assert training == [a] and validation == [b]
    assert audit["removedOverlap"] == 1 and audit["novelValidation"] == 1
    with pytest.raises(ValueError, match="Conflicting labels"):
        dataset.separate_unique([a], [dict(a, label=1)])


@pytest.mark.unit
def test_campaign_enforces_dependencies_before_dispatch(tmp_path):
    campaign = load("campaign")
    first, second = tmp_path / "first", tmp_path / "second"
    jobs = [
        {
            "id": "second",
            "dependsOn": ["first"],
            "cwd": str(tmp_path),
            "result": str(second),
            "command": [
                sys.executable,
                "-c",
                "from pathlib import Path; assert Path('first').exists(); Path('second').touch()",
            ],
        },
        {
            "id": "first",
            "cwd": str(tmp_path),
            "result": str(first),
            "command": [sys.executable, "-c", "from pathlib import Path; Path('first').touch()"],
        },
    ]
    (tmp_path / "plan.json").write_text(json.dumps({"jobs": jobs}))
    campaign.run(tmp_path, 2)
    assert first.exists() and second.exists()


@pytest.mark.unit
def test_campaign_rejects_cyclic_dependencies(tmp_path):
    campaign = load("campaign")
    jobs = [
        {"id": name, "dependsOn": [other], "cwd": str(tmp_path), "result": str(tmp_path / name), "command": ["false"]}
        for name, other in [("a", "b"), ("b", "a")]
    ]
    (tmp_path / "plan.json").write_text(json.dumps({"jobs": jobs}))
    with pytest.raises(ValueError, match="Cyclic"):
        campaign.run(tmp_path, 2)


@pytest.mark.unit
def test_completed_log_compression_preserves_content_hash(tmp_path):
    storage = load("storage")
    data = b'{"tick":3,"context":"repeat"}\n' * 100
    (tmp_path / "result.json").write_text('{"tick": 3, "truncated": true}')
    log = tmp_path / "decisions.jsonl"
    log.write_bytes(data)
    before = storage.digest(log)
    record = storage.compress(tmp_path)
    assert not log.exists() and record["losslessVerified"]
    assert storage.digest(log) == before == record["contentSha256"]
    with storage.open_content(log) as stream:
        assert stream.read() == data
    assert storage.compress(tmp_path)["alreadyCompressed"]


@pytest.mark.unit
def test_compression_never_discards_original_for_mismatching_archive(tmp_path):
    import gzip

    storage = load("storage")
    (tmp_path / "result.json").write_text("{}")
    original = tmp_path / "decisions.jsonl"
    original.write_bytes(b"original")
    with gzip.open(tmp_path / "decisions.jsonl.gz", "wb") as stream:
        stream.write(b"different")
    with pytest.raises(ValueError, match="original retained"):
        storage.compress(tmp_path)
    assert original.read_bytes() == b"original"


@pytest.mark.unit
def test_gpu_binding_uses_available_allowed_device_and_preserves_plan(tmp_path, monkeypatch):
    campaign = load("campaign")

    def gpu_query(argv, **kwargs):
        if "--query-gpu=index,uuid" in argv:
            return "1, uuid-one\n5, uuid-five\n"
        return "uuid-five\n"

    monkeypatch.setattr(campaign.subprocess, "check_output", gpu_query)
    command = [
        sys.executable,
        "-c",
        "import sys; from pathlib import Path; Path('result').write_text(sys.argv[-1])",
        "--gpu=5",
    ]
    spec = {"id": "gpu-job", "gpu": "5", "command": command, "cwd": str(tmp_path), "result": str(tmp_path / "result")}
    (tmp_path / "plan.json").write_text(json.dumps({"jobs": [spec]}))
    campaign.run(tmp_path, 2, ["1", "5"])
    assert (tmp_path / "result").read_text() == "--gpu=1"
    binding = json.loads((tmp_path / "status/gpu-job.execution.json").read_text())
    assert binding["requestedGpu"] == "5" and binding["assignedGpu"] == "1"
    assert json.loads((tmp_path / "jobs/gpu-job.json").read_text()) == spec
