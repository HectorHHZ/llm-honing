"""Control-boundary regressions using labelled transport fixtures, not game/model claims."""

from __future__ import annotations

import copy
import importlib
import json
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

INTEGRATION = Path(__file__).resolve().parents[1] / "integrations/soul-knight"


@pytest.fixture
def bridge(monkeypatch):
    monkeypatch.syspath_prepend(str(INTEGRATION))
    return SimpleNamespace(
        **{name: importlib.import_module(f"soul_knight_bridge.{name}") for name in ("native", "controller", "runtime")}
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "action",
    [
        {"unknown": True},
        {"move": [True, 0]},
        {"move": [2, 0]},
        {"move": [0]},
        {"aim": [float("nan"), 0]},
        {"aim": [float("inf"), 0]},
        {"aim": [1e300, 0]},
        {"aim": [10**1000, 0]},
        {"aim": [1_000_001, 0]},
        {"attack": 1},
    ],
)
def test_invalid_actions_do_not_write_ipc(bridge, tmp_path, action):
    client = bridge.native.NativeClient(tmp_path)
    with pytest.raises(ValueError):
        client.step(action)
    assert not (tmp_path / "request.json").exists()
    assert client.last_request is None


@pytest.mark.unit
@pytest.mark.parametrize("frames", [True, 0, 251, 1.5])
def test_invalid_frame_batches_are_atomic(bridge, tmp_path, frames):
    with pytest.raises(ValueError):
        bridge.native.NativeClient(tmp_path).step({}, frames=frames)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.unit
def test_native_transport_ignores_stale_reply_and_retains_consumed_envelopes(bridge, tmp_path):
    (tmp_path / "response.json").write_text(json.dumps({"id": "stale", "ok": True, "state": {"bad": True}}))
    seen = []

    def consume_transport_fixture():
        deadline = time.monotonic() + 2
        path = tmp_path / "request.json"
        while not path.exists() and time.monotonic() < deadline:
            time.sleep(0.005)
        request = json.loads(path.read_text())
        path.unlink()
        seen.append(request)
        response = {"id": request["id"], "ok": True, "state": {"bridge_frame": 5, "label": "transport fixture"}}
        temporary = tmp_path / "response.tmp"
        temporary.write_text(json.dumps(response))
        temporary.replace(tmp_path / "response.json")

    worker = threading.Thread(target=consume_transport_fixture)
    worker.start()
    client = bridge.native.NativeClient(tmp_path, timeout_seconds=2)
    state = client.step({"move": [1, 0], "attack": True}, frames=5)
    worker.join(timeout=2)
    assert state == {"bridge_frame": 5, "label": "transport fixture"}
    assert not worker.is_alive()
    assert client.last_request == seen[0]
    assert client.last_response["id"] == seen[0]["id"]
    assert client.last_request["action"]["aim"] == [1, 0]
    assert not (tmp_path / "request.json").exists()


@pytest.mark.unit
def test_timeout_preserves_pending_operation_without_automatic_retry(bridge, tmp_path):
    client = bridge.native.NativeClient(tmp_path, timeout_seconds=0.03)
    with pytest.raises(TimeoutError, match="no retry"):
        client.step({}, frames=1)
    original = (tmp_path / "request.json").read_bytes()
    with pytest.raises(RuntimeError, match="still pending"):
        client.state()
    assert (tmp_path / "request.json").read_bytes() == original
    assert client.last_response is None


@pytest.mark.unit
def test_prompt_filters_inactive_targets_and_preserves_exact_native_snapshot(bridge):
    state = {
        "bridge_frame": 0,
        "player": {"x": 1.234567, "y": 2, "hp": 20},
        "enemies": [
            {"id": 1, "x": 3.141592, "y": 4, "active": True, "dead": False},
            {"id": 2, "x": 9, "y": 0, "active": False, "dead": False},
        ],
        "landmarks": [{"name": "transport fixture door", "x": 5, "y": 6}],
        "progress": {},
    }
    original = copy.deepcopy(state)
    request, aims = bridge.controller.make_request(state, 1, [])
    assert state == original
    assert aims["enemy_0"] == [3.141592, 4.0]
    assert "enemy_1" not in aims
    assert "landmarks" in "\n".join(request["parts"]["details"])
    assert "1.235" in "\n".join(request["parts"]["essential"])


@pytest.mark.unit
def test_model_argmax_maps_to_native_controls_without_policy_override(bridge):
    state = {"player": {"x": 0, "y": 0}, "enemies": []}
    request, aims = bridge.controller.make_request(state, 1, [])
    selected = {"move": "west", "button": "attack", "aim": "east"}
    response = {
        "answers": {
            key: {
                "choice": selected[key],
                "probabilities": {option: float(option == selected[key]) for option in question["criteria"]},
            }
            for key, question in request["questions"].items()
        }
    }
    choices, action = bridge.controller.choose(response, request, aims)
    assert choices == selected
    assert action == {
        "move": [-1, 0],
        "aim": [5.0, 0.0],
        "attack": True,
        "interact": False,
        "skill": False,
        "switch_weapon": False,
    }
    response["answers"]["move"]["choice"] = "east"
    with pytest.raises(ValueError, match="argmax"):
        bridge.controller.choose(response, request, aims)


@pytest.mark.unit
def test_receipts_tolerate_only_aim_float32_round_trip(bridge):
    expected = bridge.native.normalize_action({"aim": [-26.739999771118164, 3.200000047683716]})
    actual = {**expected, "aim": [-26.74, 3.2]}
    assert bridge.controller.actions_match(expected, actual)
    assert not bridge.controller.actions_match(expected, {**actual, "attack": True})
    assert not bridge.controller.actions_match(expected, {**actual, "move": [1, 0]})
    assert not bridge.controller.actions_match(expected, {**actual, "aim": [-26.7, 3.2]})


@pytest.mark.unit
def test_prepare_rejects_output_nested_in_original_before_copy(bridge, tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    with pytest.raises(ValueError, match="outside the original"):
        bridge.runtime.prepare_game(original, original / "recursive", "mono", "mcs", tmp_path / "cecil.dll")
    assert list(original.iterdir()) == []


@pytest.mark.unit
def test_launch_rechecks_pristine_runtime_identity_before_starting_processes(bridge, tmp_path, monkeypatch):
    root, game = tmp_path / "integration", tmp_path / "game"
    root.mkdir()
    game.mkdir()
    original = game / "Soul_Knight.exe"
    original.write_bytes(b"labelled release identity fixture")
    expected = {"Soul_Knight.exe": {"bytes": original.stat().st_size, "sha256": bridge.runtime.sha256(original)}}
    (root / "pins.json").write_text(json.dumps({"game_files": expected}))
    (game / "honing-prepared.json").write_text(json.dumps({"generated_files": {}}))
    original.write_bytes(b"tampered identity")
    monkeypatch.setattr(bridge.runtime, "ROOT", root)
    monkeypatch.setattr(bridge.runtime.subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("must not launch"))
    with pytest.raises(ValueError, match="Pinned author-release"):
        bridge.runtime.launch_game(game, tmp_path / "new-run", "wine")


@pytest.mark.unit
def test_cleanup_refuses_reused_pid_and_scopes_wineserver_to_owned_prefix(bridge, tmp_path, monkeypatch):
    prefix = tmp_path / "wine-prefix"
    prefix.mkdir()
    server = tmp_path / "wineserver-fixture"
    server.write_text("labelled cleanup fixture")
    expected = {"pid": 999999, "start_ticks": "old", "command": ["owned-game"]}
    receipt = {
        "kind": "honing.soul-knight.launch.v1",
        "run_dir": str(tmp_path),
        "processes": [expected],
        "wineserver": str(server),
        "wineserver_sha256": bridge.runtime.sha256(server),
        "wine_prefix_identity": {"device": prefix.stat().st_dev, "inode": prefix.stat().st_ino},
    }
    path = tmp_path / "launch.json"
    path.write_text(json.dumps(receipt))
    monkeypatch.setattr(bridge.runtime, "process_identity", lambda _pid: {**expected, "start_ticks": "new"})
    monkeypatch.setattr(bridge.runtime.os, "kill", lambda *_args: pytest.fail("must not signal a reused PID"))
    calls = []

    def cleanup_fixture(command, **kwargs):
        calls.append((command, kwargs["env"]["WINEPREFIX"]))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(bridge.runtime.subprocess, "run", cleanup_fixture)
    result = bridge.runtime.stop_launch(path)
    assert result["processes"][0]["reason"] == "Process identity changed"
    assert calls == [([str(server), "-k"], str(prefix)), ([str(server), "-w"], str(prefix))]


@pytest.mark.unit
def test_generated_assets_cannot_be_written_inside_checkout(bridge):
    with pytest.raises(ValueError, match="outside the Honing"):
        bridge.runtime.external_output(bridge.runtime.CHECKOUT / "accidental-game-package")
