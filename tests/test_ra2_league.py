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
            "maxTicks": spec["maxTicks"],
            "tick": spec["maxTicks"] - 3,
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
    assert (
        "tick_budget_mismatch"
        in gate.evaluate_gate([dict(r, maxTicks=spec["maxTicks"] * 2) for r in rows], protocol)["failures"]
    )
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
    monkeypatch.setenv("LAYA_LOCK_DIR", str(tmp_path / "locks"))

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


@pytest.mark.unit
def test_sealed_cases_are_consumed_before_launch_and_exact_resume_is_allowed(tmp_path):
    sealed = load("sealed")
    protocol = json.loads((LEAGUE / "protocol.json").read_text())
    ledger = tmp_path / "heldout.json"
    record = sealed.reserve(ledger, protocol, "a" * 64, "b" * 64, tmp_path / "final")
    assert record["consumedBeforeLaunch"]
    resumed = sealed.reserve(ledger, protocol, "a" * 64, "b" * 64, tmp_path / "final")
    assert resumed["id"] == record["id"]
    assert len(json.loads(ledger.read_text())["reservations"]) == 1
    for checkpoint, source, directory in [
        ("c" * 64, "b" * 64, "final"),
        ("a" * 64, "c" * 64, "final"),
        ("a" * 64, "b" * 64, "second-attempt"),
    ]:
        with pytest.raises(ValueError, match="already consumed"):
            sealed.reserve(ledger, protocol, checkpoint, source, tmp_path / directory)
    # Renaming the protocol or weakening its threshold must not recycle cases.
    protocol["version"] = "renamed"
    protocol["sealedTest"]["minimumWinRate"] = 0.1
    with pytest.raises(ValueError, match="already consumed"):
        sealed.reserve(ledger, protocol, "a" * 64, "b" * 64, tmp_path / "final")


@pytest.mark.unit
def test_sealed_matrix_rejects_training_overlap_and_duplicate_cases():
    sealed = load("sealed")
    protocol = json.loads((LEAGUE / "protocol.json").read_text())
    protocol["sealedTest"]["seeds"][0] = protocol["train"]["seeds"][0]
    with pytest.raises(ValueError, match="overlap"):
        sealed.cases(protocol)
    protocol = json.loads((LEAGUE / "protocol.json").read_text())
    protocol["sealedTest"]["maps"].append(protocol["sealedTest"]["maps"][0])
    with pytest.raises(ValueError, match="unique"):
        sealed.cases(protocol)


@pytest.mark.unit
def test_final_driver_requires_full_development_success_before_reservation(monkeypatch):
    monkeypatch.syspath_prepend(str(LEAGUE))
    final = load("final_gate")
    protocol = json.loads((LEAGUE / "protocol.json").read_text())
    rows = [
        {
            "map": m,
            "seed": seed,
            "seat": seat,
            "split": "dev",
            "checkpointSha256": "checkpoint",
            "maxTicks": protocol["dev"]["maxTicks"],
            "tick": protocol["dev"]["maxTicks"] - 3,
            "controller": "model",
            "normalView": True,
            "auditPassed": True,
            "replayVerified": True,
            "fallbackCount": 0,
            "error": None,
            "ended": True,
            "truncated": False,
            "outcome": {seat: 1},
        }
        for m in protocol["dev"]["maps"]
        for seed in protocol["dev"]["seeds"]
        for seat in ("red", "blue")
    ]
    assert final.development_gate(rows, protocol, "checkpoint")["passed"]
    with pytest.raises(ValueError, match="tick budget"):
        final.development_gate([dict(r, maxTicks=36000) for r in rows], protocol, "checkpoint")
    with pytest.raises(ValueError, match="complete unique"):
        final.development_gate(rows[:4], protocol, "checkpoint")
    with pytest.raises(ValueError, match="provenance"):
        final.development_gate(rows, protocol, "other-checkpoint")
    with pytest.raises(ValueError, match="win rate"):
        final.development_gate([dict(r, ended=False, truncated=True) for r in rows], protocol, "checkpoint")


@pytest.mark.unit
def test_training_report_exposes_root_action_failures_despite_high_wait_accuracy(monkeypatch):
    monkeypatch.syspath_prepend(str(LEAGUE))
    trainer = load("train_laya")
    items = [{"bucket": "root_wait", "label": 0} for _ in range(99)]
    items += [{"bucket": "root_action", "label": 1}]
    report = trainer.classification_breakdown(items, [0] * 100)
    assert report["root_wait"] == {"correct": 99, "total": 99, "accuracy": 1.0}
    assert report["root_action"] == {"correct": 0, "total": 1, "accuracy": 0.0}
    assert trainer.classification_breakdown([], []) == {}


@pytest.mark.unit
def test_tick_budget_requires_recorded_limit_and_a_valid_terminal_tick():
    gate = load("gates")
    spec = {"maxTicks": 18000}
    assert gate.tick_budget_matches({"maxTicks": 18000, "tick": 16017}, spec)
    assert gate.tick_budget_matches({"maxTicks": 18000, "tick": 18000}, spec)
    for record in [
        {"tick": 16017},
        {"maxTicks": 36000, "tick": 16017},
        {"maxTicks": 18000, "tick": 18003},
        {"maxTicks": 18000, "tick": -1},
        {"maxTicks": 18000, "tick": True},
    ]:
        assert not gate.tick_budget_matches(record, spec)


@pytest.mark.unit
def test_diversity_reports_duplicate_actions_across_seeds_not_different_final_hashes():
    gate = load("gates")
    rows = [
        {"map": "m", "seat": "red", "seed": seed, "hash": seed, "artifacts": {"actions.jsonl": "same"}}
        for seed in (1000, 1001)
    ]
    rows.append({"map": "m", "seat": "blue", "seed": 1000, "artifacts": {"actions.jsonl": "same"}})
    rows.append({"map": "m", "seat": "blue", "seed": 1001})
    result = gate.trajectory_diversity(rows)
    assert result["recordedGames"] == 3
    assert result["missingActionTraces"] == 1
    assert result["uniqueMapSeatActionTraces"] == 2
    assert result["repeatedTraceGroups"] == [
        {"map": "m", "seat": "red", "actionTraceSha256": "same", "seeds": [1000, 1001]}
    ]


@pytest.mark.unit
def test_laya_qualification_and_final_cases_are_disjoint_and_reserved_separately(tmp_path):
    sealed = load("sealed")
    protocol = json.loads((LEAGUE / "protocol.json").read_text())
    qualification = sealed.cases(protocol, "layaQualification")
    final = sealed.cases(protocol)
    assert len(qualification) == 36 and len(final) == 96
    assert not {c[0] for c in qualification} & {c[0] for c in final}
    ledger = tmp_path / "heldout.json"
    first = sealed.reserve(ledger, protocol, "a" * 64, "b" * 64, tmp_path / "qualification", "layaQualification")
    second = sealed.reserve(ledger, protocol, "c" * 64, "d" * 64, tmp_path / "final")
    assert first["spec"]["stage"] == "layaQualification"
    assert second["spec"]["stage"] == "sealedTest"
    assert len(json.loads(ledger.read_text())["reservations"]) == 2
    with pytest.raises(ValueError, match="already consumed"):
        sealed.reserve(ledger, protocol, "c" * 64, "d" * 64, tmp_path / "retry", "layaQualification")
    protocol["layaQualification"]["maps"][0] = protocol["sealedTest"]["maps"][0]
    with pytest.raises(ValueError, match="overlap"):
        sealed.cases(protocol, "layaQualification")


@pytest.mark.unit
def test_qualification_gate_uses_its_own_matrix_and_budget():
    gate = load("gates")
    protocol = json.loads((LEAGUE / "protocol.json").read_text())
    spec = protocol["layaQualification"]
    rows = [
        {
            "map": m,
            "seed": seed,
            "seat": seat,
            "split": "layaQualification",
            "controller": "model",
            "normalView": True,
            "fallbackCount": 0,
            "auditPassed": True,
            "replayVerified": True,
            "error": None,
            "checkpointSha256": "a" * 64,
            "ended": True,
            "truncated": False,
            "outcome": {seat: 1},
            "maxTicks": 18000,
            "tick": 16000,
            "decisionInterval": 3,
            **{k: protocol[k] for k in ("engineCommit", "baselineRevision", "cadenceTicks", "engineActionCap")},
        }
        for m in spec["maps"]
        for seed in spec["seeds"]
        for seat in spec["seats"]
    ]
    assert gate.evaluate_gate(rows, protocol, "layaQualification")["passed"]
    assert not gate.evaluate_gate(rows, protocol)["passed"]
    assert (
        "tick_budget_mismatch"
        in gate.evaluate_gate([dict(r, maxTicks=36000) for r in rows], protocol, "layaQualification")["failures"]
    )


@pytest.mark.unit
def test_gpu_lease_detects_cpu_preparation_and_survives_fd_handoff(tmp_path, monkeypatch):
    import fcntl
    import subprocess

    campaign = load("campaign")
    monkeypatch.setenv("LAYA_LOCK_DIR", str(tmp_path))
    with (tmp_path / "uuid-one.lock").open("a") as worker:
        fcntl.flock(worker, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert campaign.try_device_lease("uuid-one") is None
    lease = campaign.try_device_lease("uuid-one")
    assert lease is not None
    # Use a real child and pipe to prove the parent can hand off its lock
    # across exec without a gap while another controller probes admission.
    child = subprocess.Popen(
        [sys.executable, "-c", "import sys; print('ready', flush=True); sys.stdin.readline()"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        pass_fds=(lease.fileno(),),
    )
    lease.close()
    try:
        assert child.stdout.readline().strip() == "ready"
        assert campaign.try_device_lease("uuid-one") is None
    finally:
        child.communicate("finish\n", timeout=5)
    with campaign.try_device_lease("uuid-one") as available:
        assert available is not None


@pytest.mark.unit
def test_gpu_scheduler_skips_worker_lock_even_when_nvidia_reports_idle(tmp_path, monkeypatch):
    import fcntl

    campaign = load("campaign")
    directory = tmp_path / "locks"
    directory.mkdir()
    monkeypatch.setenv("LAYA_LOCK_DIR", str(directory))
    monkeypatch.setattr(
        campaign.subprocess,
        "check_output",
        lambda args, **kw: "1, uuid-one\n5, uuid-five\n" if "--query-gpu=index,uuid" in args else "",
    )
    command = [sys.executable, "-c", "from pathlib import Path; Path('result').write_text('ok')", "--gpu=1"]
    spec = {"id": "gpu-job", "gpu": "1", "command": command, "cwd": str(tmp_path), "result": str(tmp_path / "result")}
    (tmp_path / "plan.json").write_text(json.dumps({"jobs": [spec]}))
    with (directory / "uuid-one.lock").open("a") as worker:
        fcntl.flock(worker, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaign.run(tmp_path, 1, ["1", "5"])
    binding = json.loads((tmp_path / "status/gpu-job.execution.json").read_text())
    assert binding["assignedGpu"] == "5"
    assert (tmp_path / "result").read_text() == "ok"


@pytest.mark.unit
def test_worker_from_older_controller_acquires_lease_for_bound_device(tmp_path, monkeypatch):
    campaign = load("campaign")
    monkeypatch.setenv("LAYA_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.setattr(
        campaign.subprocess,
        "check_output",
        lambda args, **kw: "1, uuid-one\n5, uuid-five\n" if "--query-gpu=index,uuid" in args else "",
    )
    for directory in ("jobs", "logs", "status"):
        (tmp_path / directory).mkdir()
    command = [sys.executable, "-c", "from pathlib import Path; Path('result').write_text('ok')"]
    spec = {"id": "gpu-job", "gpu": "1", "command": command, "cwd": str(tmp_path), "result": str(tmp_path / "result")}
    (tmp_path / "jobs/gpu-job.json").write_text(json.dumps(spec))
    (tmp_path / "status/gpu-job.execution.json").write_text(json.dumps({"assignedGpu": "5", "command": command}))
    campaign.execute(tmp_path, "gpu-job")
    assert json.loads((tmp_path / "status/gpu-job.exit.json").read_text())["exitCode"] == 0
    assert (tmp_path / "locks/uuid-five.dispatch.lock").exists()
    assert not (tmp_path / "locks/uuid-one.dispatch.lock").exists()
    with campaign.try_device_lease("uuid-five") as available:
        assert available is not None


@pytest.mark.unit
def test_development_resume_preserves_original_protocol_across_heldout_changes(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(LEAGUE))
    optimize = load("optimize")
    protocol = json.loads((LEAGUE / "protocol.json").read_text())
    current = tmp_path / "installed.json"
    current.write_text(json.dumps(protocol))
    root = tmp_path / "run"
    root.mkdir()
    assert optimize.archived_protocol(root, current) == protocol
    original_hash = optimize.digest(root / "protocol.json")
    protocol["qualificationNote"] = "New descriptive metadata outside this development loop"
    current.write_text(json.dumps(protocol))
    assert optimize.archived_protocol(root, current)["qualificationNote"] != protocol["qualificationNote"]
    assert optimize.digest(root / "protocol.json") == original_hash
    protocol["dev"]["maxTicks"] *= 2
    current.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="differs"):
        optimize.archived_protocol(root, current)


@pytest.mark.unit
def test_legacy_resume_requires_exact_protocol_recovery_before_snapshot(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(LEAGUE))
    optimize = load("optimize")
    current = tmp_path / "installed.json"
    current.write_bytes((LEAGUE / "protocol.json").read_bytes())
    root = tmp_path / "run"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"protocolSha256": "wrong"}))
    with pytest.raises(ValueError, match="recover the exact"):
        optimize.archived_protocol(root, current)
    assert not (root / "protocol.json").exists()
    (root / "config.json").write_text(json.dumps({"protocolSha256": optimize.digest(current)}))
    optimize.archived_protocol(root, current)
    assert (root / "protocol.json").read_bytes() == current.read_bytes()
