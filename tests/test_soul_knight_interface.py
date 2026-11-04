"""Control-boundary regressions using labelled transport fixtures, not game/model claims."""

from __future__ import annotations

import copy
import importlib
import io
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


def model_response_fixture(request):
    """Labelled serving-shaped response, with coherent synthetic token counts."""
    selected = {"move": "west", "button": "attack", "aim": "east"}
    state_text = "\n".join(
        text for section in ("essential", "details", "memory", "logs") for text in request["parts"][section]
    )
    state_tokens = len(state_text.split())
    return {
        "id": request["id"],
        "stateText": state_text,
        "trimmed": [],
        "tokenAudit": {
            key: {
                "stateTokens": state_tokens,
                "stateRoom": 4096 - 16 - len(question["criteria"]),
                "inputTokens": state_tokens + 16 + len(question["criteria"]),
                "options": len(question["criteria"]),
                "stateTruncated": False,
            }
            for key, question in request["questions"].items()
        },
        "answers": {
            key: {
                "type": "choice",
                "choice": selected[key],
                "probabilities": {option: float(option == selected[key]) for option in question["criteria"]},
            }
            for key, question in request["questions"].items()
        },
    }


@pytest.fixture
def model_driver(bridge, tmp_path, monkeypatch):
    """Exercise the real HTTP reader and driver against labelled in-memory boundaries."""
    state = {
        "bridge_frame": 0,
        "player": {"x": 0, "y": 0, "hp": 20},
        "enemies": [{"id": 1, "x": 3, "y": 4, "active": True, "dead": False}],
    }
    steps, exchanges = [], []

    class PausedNativeFixture:
        def __init__(self, *_args, **_kwargs):
            self.last_request = self.last_response = None

        def state(self):
            return copy.deepcopy(state)

        def step(self, action, *, frames):
            steps.append(copy.deepcopy(action))
            state["bridge_frame"] += frames
            self.last_request = {"id": f"fixture:{len(steps)}", "action": action}
            self.last_response = {"id": self.last_request["id"], "ok": True}
            return {"executed_action": action, "advanced_frames": frames, "bridge_frame": state["bridge_frame"]}

    monkeypatch.setattr(bridge.controller, "NativeClient", PausedNativeFixture)
    output = tmp_path / "trace"

    def run(mutate=None, *, raw_response=None, decisions=1):
        def http_fixture(request, **_kwargs):
            if request.full_url.endswith("/v1/health"):
                return io.BytesIO(b'{"ready":true,"metadata":{"repo":"labelled model fixture"}}')
            submitted = json.loads(request.data)
            response = model_response_fixture(submitted)
            if mutate:
                mutate(response, submitted)
            raw = raw_response if raw_response is not None else json.dumps(response).encode()
            exchanges.append((submitted, raw))
            return io.BytesIO(raw)

        monkeypatch.setattr(bridge.controller, "urlopen", http_fixture)
        return bridge.controller.run_episode(
            tmp_path / "ipc", "http://fixture.invalid", output, decisions=decisions, frames=5, run_id="fixture"
        )

    return SimpleNamespace(run=run, output=output, steps=steps, state=state, exchanges=exchanges)


def assert_rejected_without_native_action(driver, reason):
    assert driver.steps == []
    assert driver.state["bridge_frame"] == 0
    request, raw = driver.exchanges[-1]
    assert (driver.output / "current_decision.response.raw").read_bytes() == raw
    rejected = json.loads((driver.output / "current_decision.rejected.json").read_text())
    assert rejected["request"] == request
    assert rejected["request"]["id"] == "fixture:0"
    assert rejected["decision_id"] == 0
    assert reason in rejected["rejection_reason"]
    assert rejected["response_rejected"] is True
    assert rejected["validation_pending"] is False
    assert rejected["execution_pending"] is False
    assert "selected_action" not in rejected
    assert json.loads((driver.output / "current_decision.pending.json").read_text()) == rejected
    status = json.loads((driver.output / "status.json").read_text())
    assert status["stage"] == "error"
    assert status["decisions_completed"] == 0
    assert (driver.output / "decisions.jsonl").read_text() == ""
    assert not (driver.output / "current_decision.json").exists()


@pytest.mark.unit
@pytest.mark.parametrize("value", [0, 2, True, -0.1, float("nan"), float("inf"), "1", None, 10**400])
def test_invalid_model_probabilities_never_advance_native_state(model_driver, value):
    def invalid(response, _request):
        for answer in response["answers"].values():
            answer["probabilities"] = dict.fromkeys(answer["probabilities"], value)

    with pytest.raises(ValueError, match="probability"):
        model_driver.run(invalid)
    assert_rejected_without_native_action(model_driver, "probability")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        pytest.param(lambda r, _q: r.pop("id"), "request id", id="missing-id"),
        pytest.param(lambda r, _q: r.update(id=True), "request id", id="boolean-id"),
        pytest.param(lambda r, _q: r.update(id="other:0"), "request id", id="wrong-id"),
        pytest.param(lambda r, _q: r.pop("answers"), "submitted questions", id="missing-answers"),
        pytest.param(lambda r, _q: r["answers"].pop("move"), "submitted questions", id="missing-question"),
        pytest.param(lambda r, _q: r["answers"].update(extra={}), "submitted questions", id="extra-question"),
        pytest.param(lambda r, _q: r["answers"].update(move=[]), "Invalid model answer", id="malformed-answer"),
        pytest.param(lambda r, _q: r["answers"]["move"].pop("probabilities"), "support", id="missing-probabilities"),
        pytest.param(
            lambda r, _q: r["answers"]["move"].update(probabilities=[]), "support", id="malformed-probabilities"
        ),
        pytest.param(lambda r, _q: r["answers"]["move"]["probabilities"].pop("stay"), "support", id="missing-option"),
        pytest.param(
            lambda r, _q: r["answers"]["move"]["probabilities"].update(extra=0), "support", id="extra-option"
        ),
        pytest.param(
            lambda r, _q: r["answers"]["move"]["probabilities"].update(west=0.9994),
            "total",
            id="outside-rounding-bound",
        ),
        pytest.param(lambda r, _q: r["answers"]["move"].update(choice="east"), "argmax", id="argmax-mismatch"),
        pytest.param(lambda r, _q: r["answers"]["move"].update(choice="unoffered"), "argmax", id="unoffered-choice"),
    ],
)
def test_invalid_model_answers_are_preserved_before_native_execution(model_driver, mutate, reason):
    with pytest.raises(ValueError, match=reason):
        model_driver.run(mutate)
    assert_rejected_without_native_action(model_driver, reason)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        pytest.param(lambda r, _q: r.pop("stateText"), "stateText", id="missing-retained-text"),
        pytest.param(lambda r, _q: r.update(stateText="player only"), "stateText", id="omitted-required-text"),
        pytest.param(lambda r, _q: r.pop("trimmed"), "trimmed", id="missing-removal-report"),
        pytest.param(lambda r, _q: r.update(trimmed={}), "trimmed", id="malformed-removal-report"),
        pytest.param(lambda r, _q: r.update(trimmed=[{}]), "removal entry", id="malformed-removal-entry"),
        pytest.param(
            lambda r, q: r.update(trimmed=[{"section": "essential", "text": q["parts"]["essential"][-1]}]),
            "required essential",
            id="removed-current-player-state",
        ),
        pytest.param(
            lambda r, q: r.update(trimmed=[{"section": "details", "text": q["parts"]["details"][-1]}]),
            "required details",
            id="removed-enemy-state",
        ),
        pytest.param(
            lambda r, _q: r.update(trimmed=[{"section": "logs", "text": "not submitted"}]),
            "removal order",
            id="fabricated-history-removal",
        ),
        pytest.param(lambda r, _q: r.pop("tokenAudit"), "tokenAudit", id="missing-audit"),
        pytest.param(lambda r, _q: r["tokenAudit"].pop("move"), "tokenAudit", id="missing-question-audit"),
        pytest.param(lambda r, _q: r["tokenAudit"].update(move=[]), "counts", id="malformed-question-audit"),
        pytest.param(lambda r, _q: r["tokenAudit"]["move"].pop("options"), "counts", id="missing-option-count"),
        pytest.param(lambda r, _q: r["tokenAudit"]["move"].update(options=True), "counts", id="boolean-option-count"),
        pytest.param(
            lambda r, _q: r["tokenAudit"]["move"].update(options=1), "option count", id="lost-option-markers"
        ),
        pytest.param(
            lambda r, _q: r["tokenAudit"]["move"].update(stateTokens=-1), "counts", id="negative-token-count"
        ),
        pytest.param(
            lambda r, _q: r["tokenAudit"]["move"].update(stateTruncated=True), "truncation", id="clipped-state"
        ),
        pytest.param(
            lambda r, _q: r["tokenAudit"]["move"].pop("stateTruncated"), "truncation", id="missing-clipping-audit"
        ),
        pytest.param(
            lambda r, _q: r["tokenAudit"]["move"].update(stateTruncated=0),
            "truncation",
            id="nonboolean-clipping-audit",
        ),
        pytest.param(
            lambda r, _q: r["tokenAudit"]["move"].update(stateRoom=0), "truncation", id="contradictory-clipping-audit"
        ),
        pytest.param(
            lambda r, _q: r["tokenAudit"]["move"].update(inputTokens=1), "Inconsistent", id="impossible-input-count"
        ),
        pytest.param(
            lambda r, _q: r["tokenAudit"]["move"].update(stateTokens=1), "disagrees", id="different-state-size"
        ),
        pytest.param(
            lambda r, _q: r["tokenAudit"]["move"].update(stateRoom=4096), "disagrees", id="different-token-budget"
        ),
    ],
)
def test_missing_or_truncated_model_context_never_reaches_native_execution(model_driver, mutate, reason):
    with pytest.raises(ValueError, match=reason):
        model_driver.run(mutate)
    assert_rejected_without_native_action(model_driver, reason)


@pytest.mark.unit
def test_audit_without_room_for_option_markers_never_reaches_native_execution(model_driver):
    def omit_question_tokens(response, _request):
        for audit in response["tokenAudit"].values():
            audit.update(inputTokens=audit["stateTokens"], stateRoom=2048)

    with pytest.raises(ValueError, match="Inconsistent"):
        model_driver.run(omit_question_tokens)
    assert_rejected_without_native_action(model_driver, "Inconsistent")


@pytest.mark.unit
@pytest.mark.parametrize("raw", [b'{"id":', b"\xff", b"[]"])
def test_unparseable_or_nonobject_http_response_retains_exact_rejected_bytes(model_driver, raw):
    with pytest.raises(ValueError):
        model_driver.run(raw_response=raw)
    assert_rejected_without_native_action(model_driver, "")


@pytest.mark.unit
@pytest.mark.parametrize("rounded", [False, True])
def test_valid_model_response_crosses_native_boundary_without_probability_rewriting(model_driver, rounded):
    def distribution(response, _request):
        if rounded:
            probabilities = response["answers"]["move"]["probabilities"]
            response["answers"]["move"]["probabilities"] = {
                key: 0.3333 if key in ("stay", "east", "west") else 0 for key in probabilities
            }

    result = model_driver.run(distribution)
    assert result["decisions_completed"] == 1
    assert model_driver.steps == [
        {"move": [-1, 0], "aim": [5.0, 0.0], "attack": True, "interact": False, "skill": False, "switch_weapon": False}
    ]
    assert model_driver.state["bridge_frame"] == 5
    record = json.loads((model_driver.output / "current_decision.json").read_text())
    assert all(record["execution_checks"].values())
    assert record["response"] == json.loads(model_driver.exchanges[-1][1])
    assert (model_driver.output / "current_decision.response.raw").read_bytes() == model_driver.exchanges[-1][1]
    assert not (model_driver.output / "current_decision.pending.json").exists()
    assert not (model_driver.output / "current_decision.rejected.json").exists()


@pytest.mark.unit
def test_optional_history_removal_preserves_current_state_before_native_execution(model_driver):
    def remove_history(response, request):
        response["trimmed"] = [{"section": "logs", "text": text} for text in request["parts"]["logs"]]
        retained = copy.deepcopy(request)
        retained["parts"]["logs"] = []
        expected = model_response_fixture(retained)
        response.update(stateText=expected["stateText"], tokenAudit=expected["tokenAudit"])

    result = model_driver.run(remove_history, decisions=2)
    assert result["decisions_completed"] == 2
    assert len(model_driver.steps) == 2
    record = json.loads((model_driver.output / "current_decision.json").read_text())
    assert record["response"]["trimmed"] == [{"section": "logs", "text": record["request"]["parts"]["logs"][0]}]
    assert all(text in record["response"]["stateText"] for text in record["request"]["parts"]["details"])


@pytest.mark.unit
def test_context_removals_match_submitted_history_order_and_content(bridge):
    request, aims = bridge.controller.make_request({"player": {"x": 0, "y": 0}}, 0, [])
    request["parts"].update(logs=["old log", "new log"], memory=["old memory", "new memory"])
    response = model_response_fixture(request)
    response["trimmed"] = [
        {"section": "logs", "text": "old log"},
        {"section": "logs", "text": "new log"},
        {"section": "memory", "text": "new memory"},
    ]
    retained = copy.deepcopy(request)
    retained["parts"].update(logs=[], memory=["old memory"])
    expected = model_response_fixture(retained)
    response.update(stateText=expected["stateText"], tokenAudit=expected["tokenAudit"])
    assert bridge.controller.validate_response(response, request, aims)[0]["move"] == "west"
    for invalid in (
        [{"section": "memory", "text": "new memory"}],
        [{"section": "logs", "text": "new log"}],
    ):
        response["trimmed"] = invalid
        with pytest.raises(ValueError, match=r"removal (order|text)"):
            bridge.controller.validate_response(response, request, aims)


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
