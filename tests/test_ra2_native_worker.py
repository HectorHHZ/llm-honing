"""Native worker provenance, resource and transport checks without GPU/model execution."""

import io
import json
import os
from pathlib import Path

import pytest
from examples.ra2.integration.models import worker


def prepared(tmp_path, family="nanojev"):
    directory = tmp_path / "model"
    directory.mkdir()
    if family == "nanojev":
        names = [
            "best.safetensors",
            "config.json",
            "backbone_config/config.json",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
        ]
    else:
        names = [
            "backbone/config.json",
            "backbone/model.safetensors.index.json",
            "decision_config.json",
            "head.pt",
            "runtime_buffers.pt",
            "load_model.py",
            "tokenizer.json",
            "tokenizer_config.json",
        ]
    hashes = {}
    for name in names:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"weight_map": {"x": "unregistered.safetensors"}})
            if name.endswith("index.json")
            else "fixture only"
        )
        hashes[name] = worker.digest(path)
    pin = {"source": "fixture-source", "model": "fixture-model", "modelRevision": "fixture-revision"}
    manifest = {"kind": "prepared-upstream-model", **pin, "path": str(directory), "filesSha256": hashes}
    path = tmp_path / "prepared.json"
    path.write_text(json.dumps(manifest))
    return path, manifest, {family: pin}


def test_prepared_bytes_and_pins_are_verified_before_native_import(tmp_path):
    path, manifest, pins = prepared(tmp_path)
    _, metadata = worker.verify_prepared(path, "nanojev", pins)
    assert metadata["verifiedFilesSha256"] == manifest["filesSha256"]
    assert metadata["preparedManifestSha256"] == worker.digest(path)
    assert "weightsSha256" not in metadata  # A set digest is not a single weight-file hash.
    pins["nanojev"]["modelRevision"] = "changed"
    with pytest.raises(ValueError, match="pin mismatch"):
        worker.verify_prepared(path, "nanojev", pins)
    pins["nanojev"]["modelRevision"] = manifest["modelRevision"]
    (Path(manifest["path"]) / "best.safetensors").write_text("changed weights")
    with pytest.raises(ValueError, match="hash mismatch"):
        worker.verify_prepared(path, "nanojev", pins)


def test_manifest_cannot_omit_native_inputs_or_escape_directory(tmp_path):
    path, manifest, pins = prepared(tmp_path)
    original = dict(manifest["filesSha256"])
    del manifest["filesSha256"]["best.safetensors"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="required native input"):
        worker.verify_prepared(path, "nanojev", pins)
    manifest["filesSha256"] = {"../outside": "a" * 64, **original}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="relative paths"):
        worker.verify_prepared(path, "nanojev", pins)


def test_omni_requires_every_referenced_weight_shard(tmp_path):
    path, manifest, pins = prepared(tmp_path, "jev-omni")
    with pytest.raises(ValueError, match="unverified weight shard"):
        worker.verify_prepared(path, "jev-omni", pins)
    shard = Path(manifest["path"]) / "backbone/unregistered.safetensors"
    shard.write_text("fixture shard")
    manifest["filesSha256"]["backbone/unregistered.safetensors"] = worker.digest(shard)
    path.write_text(json.dumps(manifest))
    _, metadata = worker.verify_prepared(path, "jev-omni", pins)
    assert len(metadata["weightSetSha256"]) == 64


def test_nano_requires_clean_pinned_tracked_source(tmp_path, monkeypatch):
    pin = {"nanojev": {"sourceRevision": "pinned"}}
    replies = iter(["pinned\n", " M scripts/predict_toy_decisions.py\n"])
    monkeypatch.setattr(worker.subprocess, "check_output", lambda *a, **k: next(replies))
    with pytest.raises(ValueError, match="unmodified tracked source"):
        worker.verify_nano_source(tmp_path, pin)
    replies = iter(["pinned\n", ""])
    assert worker.verify_nano_source(tmp_path, pin) == (tmp_path, "pinned")


class FixtureAdapter:
    def __init__(self):
        self.calls = 0

    def predict(self, parts, questions):
        self.calls += 1
        return {"answers": {"fixture": True}, "parts": parts, "questions": questions}


def test_jsonl_preserves_request_binding_and_rejects_replayed_ids():
    adapter, outputs = FixtureAdapter(), []
    request = {"id": 1, "parts": {"essential": ["normal"]}, "questions": {"x": "fixture"}}
    worker.requests(adapter, io.StringIO(json.dumps(request) + "\n"), outputs.append)
    assert outputs[0]["id"] == 1 and outputs[0]["parts"] == request["parts"]
    adapter, outputs = FixtureAdapter(), []
    with pytest.raises(ValueError, match="strictly increasing"):
        worker.requests(adapter, io.StringIO((json.dumps(request) + "\n") * 2), outputs.append)
    assert adapter.calls == len(outputs) == 1


@pytest.mark.parametrize(
    "text",
    [
        '{"id":1,"id":2,"parts":{},"questions":{}}',
        '{"id":NaN,"parts":{},"questions":{}}',
        '{"id":true,"parts":{},"questions":{}}',
        '{"id":1,"parts":{},"questions":{},"extra":0}',
    ],
)
def test_malformed_transport_never_calls_model(text):
    adapter, outputs = FixtureAdapter(), []
    with pytest.raises(ValueError):
        worker.requests(adapter, io.StringIO(text), outputs.append)
    assert adapter.calls == 0 and outputs == []


def test_gpu_worker_lock_and_occupancy_fail_without_changing_device(tmp_path, monkeypatch):
    monkeypatch.setenv("LAYA_LOCK_DIR", str(tmp_path))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "previous")

    def query(command, **_):
        return "5, GPU-fixture\n" if "--query-gpu=index,uuid" in command else ""

    monkeypatch.setattr(worker.subprocess, "check_output", query)
    with worker.gpu_lock("5") as uuid:
        assert uuid == "GPU-fixture" and os.environ["CUDA_VISIBLE_DEVICES"] == uuid
        with pytest.raises(BlockingIOError), worker.gpu_lock("5"):
            pytest.fail("Second worker acquired the same GPU")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "previous"
    monkeypatch.setattr(
        worker.subprocess,
        "check_output",
        lambda command, **_: "5, GPU-fixture\n" if "--query-gpu=index,uuid" in command else "GPU-fixture\n",
    )
    with pytest.raises(RuntimeError, match="compute process"), worker.gpu_lock("5"):
        pytest.fail("Occupied GPU was accepted")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "previous"
