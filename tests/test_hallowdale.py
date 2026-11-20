"""The hallowdale client refuses bad decisions and owns its processes; fixtures are not gameplay evidence."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import patch

import pytest

from tests.conftest import load_script

HERE = Path(__file__).parents[1] / "integrations/hallowdale"
run = load_script("integrations/hallowdale/run.py")
encode = load_script("integrations/hallowdale/encode.py")
pytestmark = pytest.mark.unit


def observation(identity: int = 1, tick: int = 0) -> dict[str, Any]:
    return {
        "schema": "honing.visible.v1",
        "profile": "visible",
        "capabilities": {"pixels_to_model": False},
        "frame": {"id": identity, "tick": tick, "simulation_time": 21.8},
        "view": {"screen_rect": [0, 0, 960, 523], "world_from_screen": [1, 0, 0, -1, 40, -90]},
        "terrain": {"cell_size": [40, 40], "rows": ["ĀĀ", "ĂĂ"], "legend": {"solid": "#"}},
        "objects": [
            {"id": "forgotten_deans-pass/f1", "kind": "wall", "bounds": [0, 480, 960, 43], "look": "terrain"},
        ],
        "self": {"object_id": "forgotten_deans-pass/player", "screen_box": [460, 300, 24, 41], "facing": "right"},
        "hud": {"health": {"value": 5, "evidence": "hud"}, "text": []},
        "rules": ["static rule text"],
        "disclosure": {"debug": "static disclosure text"},
    }


def response_for(request: dict[str, Any], choice: str = "right") -> dict[str, Any]:
    return {
        "id": request["id"],
        "stateText": "\n".join(request["parts"]["essential"]),
        "trimmed": [],
        "tokenAudit": {
            "action": {
                "stateTokens": 700,
                "stateRoom": 1700,
                "inputTokens": 1000,
                "options": len(run.ACTIONS),
                "stateTruncated": False,
            }
        },
        "answers": {
            "action": {
                "type": "choice",
                "choice": choice,
                "probabilities": {name: float(name == choice) for name in run.ACTIONS},
            }
        },
    }


class PausedGame:
    """Records native commands; a rejected response must leave this list empty."""

    def __init__(self, session: Path) -> None:
        self.session = session
        (session / "responses").mkdir(parents=True, exist_ok=True)
        self.commands: list[dict[str, Any]] = []

    def step(self, before: dict[str, Any], command: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        self.commands.append(command)
        return observation(before["frame"]["id"] + 1, before["frame"]["tick"] + command["frames"]), {
            "status": "applied"
        }


class FixtureClient:
    def __init__(self, response: Any) -> None:
        self.response = response

    def decide(self, request: dict[str, Any], *, raw_path: Path) -> Any:
        raw_path.write_text(json.dumps(self.response))
        return self.response


def damage(response: dict[str, Any], request: dict[str, Any], kind: str) -> None:
    answer = response["answers"]["action"]
    audit = response["tokenAudit"]["action"]
    if kind == "wrong_id":
        response["id"] += 1
    elif kind == "boolean_id":
        response["id"] = True
    elif kind == "changed_state":
        response["stateText"] += " altered"
    elif kind == "missing_state":
        del response["stateText"]
    elif kind == "trimmed_required":
        response["trimmed"] = [{"section": "essential", "text": request["parts"]["essential"][1]}]
    elif kind == "truncated":
        audit["stateTruncated"] = True
    elif kind == "missing_counts":
        del audit["stateTokens"]
    elif kind == "wrong_options":
        audit["options"] = 11
    elif kind == "all_zero":
        answer["probabilities"] = dict.fromkeys(run.ACTIONS, 0)
    elif kind == "all_two":
        answer["probabilities"] = dict.fromkeys(run.ACTIONS, 2)
    elif kind == "all_true":
        answer["probabilities"] = dict.fromkeys(run.ACTIONS, True)
    elif kind == "not_argmax":
        answer["choice"] = "left"
    elif kind == "unoffered":
        answer["choice"] = "teleport"
    elif kind == "missing_option":
        del answer["probabilities"]["wait"]
    elif kind == "extra_option":
        answer["probabilities"]["teleport"] = 0
    else:
        raise AssertionError(kind)


@pytest.mark.parametrize(
    "kind",
    [
        "wrong_id",
        "boolean_id",
        "changed_state",
        "missing_state",
        "trimmed_required",
        "truncated",
        "missing_counts",
        "wrong_options",
        "all_zero",
        "all_two",
        "all_true",
        "not_argmax",
        "unoffered",
        "missing_option",
        "extra_option",
    ],
)
def test_rejected_response_keeps_its_bytes_and_plays_nothing(tmp_path: Path, kind: str) -> None:
    game = PausedGame(tmp_path)
    before = observation()
    response = response_for(run.request_for(before, 12))
    damage(response, run.request_for(before, 12), kind)
    with pytest.raises(ValueError):
        run.decide_and_step(game, FixtureClient(response), before, 12)
    assert game.commands == []
    assert json.loads((tmp_path / "responses/000001.raw").read_text()) == response
    record = json.loads((tmp_path / "decisions.jsonl").read_text())
    assert record["responseRawFile"] == "responses/000001.raw" and "ValueError" in record["error"]


def test_model_argmax_is_the_native_command(tmp_path: Path) -> None:
    game = PausedGame(tmp_path)
    before = observation()
    after = run.decide_and_step(
        game, FixtureClient(response_for(run.request_for(before, 12), "attack_up")), before, 12
    )
    assert game.commands == [{"action": "attack_up", "frames": 12}]
    assert after["frame"] == {"id": 2, "tick": 12, "simulation_time": 21.8}


@pytest.mark.parametrize(
    ("choice", "action"),
    [("short_jump", "jump"), ("short_jump_left", "jump_left"), ("short_jump_right", "jump_right")],
)
def test_low_hop_choices_set_the_jump_hold(choice: str, action: str) -> None:
    assert run.native_command(choice, 12) == {"action": action, "frames": 12, "jump_frames": run.SHORT_JUMP_FRAMES}
    assert run.native_command(choice, 2)["jump_frames"] == 2


def test_full_jump_choices_carry_no_jump_hold() -> None:
    assert run.native_command("jump_left", 12) == {"action": "jump_left", "frames": 12}


def test_unoffered_choice_has_no_native_command() -> None:
    with pytest.raises(ValueError, match="Not an offered action"):
        run.native_command("pause", 12)


def test_request_carries_only_the_visible_state() -> None:
    request = run.request_for(observation(), 12)
    prompt, state_text = request["parts"]["essential"]
    state = json.loads(state_text)
    assert set(state) == {"screen", "rows", "objects", "self", "hud"}
    assert state["objects"] == [["f1", "wall", "terrain", 0, 480, 960, 43]]
    assert state["self"] == {"screen_box": [460, 300, 24, 41], "facing": "right"}
    assert "40 pixel cells" in prompt and "12 frames" in prompt
    assert set(request["questions"]["action"]["criteria"]) == set(run.ACTIONS)
    assert "world_from_screen" not in state_text and "forgotten_deans-pass" not in state_text


@pytest.mark.parametrize(
    "change",
    [
        {"profile": "global_debug"},
        {"schema": "other"},
        {"capabilities": {"pixels_to_model": True}},
        {"capabilities": {}},
        {"frame": {"id": True, "tick": 0}},
        {"frame": {"id": 1}},
    ],
)
def test_only_the_visible_profile_can_reach_a_model(change: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=r"visible|frame"):
        run.require_visible({**observation(), **change})


@pytest.mark.parametrize(("timeout", "seconds"), [(180, 210), (0.1, 31), (299.5, 330), (3570, 3600)])
def test_native_idle_deadline_is_the_model_timeout_plus_a_margin(timeout: float, seconds: int) -> None:
    assert run.native_idle_seconds(timeout) == seconds


@pytest.mark.parametrize("timeout", [True, 0, -1, float("inf"), float("nan"), 3570.1, "300"])
def test_invalid_model_timeout_is_refused(timeout: Any) -> None:
    with pytest.raises(ValueError, match="timeout"):
        run.native_idle_seconds(timeout)


def test_inherited_bridge_settings_cannot_change_what_the_model_sees(tmp_path: Path) -> None:
    game = run.Game(tmp_path / "game", tmp_path / "session", idle_seconds=210, cell_pixels=40)
    inherited = {
        "HONING_OBSERVATION_PROFILE": "global_debug",
        "HONING_VIEW_ANCHOR": "feet:1,2",
        "HONING_MODEL_TIMEOUT_SECONDS": "1",
    }
    with patch.dict(os.environ, inherited):
        environment = game.environment(":77")
    assert environment["HONING_OBSERVATION_PROFILE"] == "visible"
    assert environment["HONING_MODEL_TIMEOUT_SECONDS"] == "210"
    assert "HONING_VIEW_ANCHOR" not in environment


ACK = {
    "id": 1,
    "status": "applied",
    "action": "jump",
    "frames": 12,
    "jump_frames": 3,
    "frame_before": 0,
    "frame_after": 12,
    "observation_id": 2,
}


@pytest.mark.parametrize(
    "change",
    [
        {"action": "left"},
        {"frames": 11},
        {"jump_frames": None},
        {"status": "refused"},
        {"id": 2},
        {"frame_after": 13},
        {"observation_id": 3},
    ],
)
def test_acknowledgement_of_another_window_stops_the_run(change: dict[str, Any]) -> None:
    command = {"action": "jump", "frames": 12, "jump_frames": 3}
    run.check_transition(observation(1, 0), observation(2, 12), command, ACK)
    with pytest.raises(ValueError, match="different window"):
        run.check_transition(observation(1, 0), observation(2, 12), command, {**ACK, **change})


@pytest.mark.parametrize("after", [observation(3, 12), observation(2, 13)])
def test_observation_that_did_not_advance_exactly_stops_the_run(after: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="advance"):
        run.check_transition(observation(1, 0), after, {"action": "wait", "frames": 12}, ACK)


# ---- owned processes, with stub executables in place of Xvfb and the game ----------------------------------
XVFB = """#!{python}
import os, sys, time
Path = __import__("pathlib").Path
descriptor = int(sys.argv[sys.argv.index("-displayfd") + 1])
Path(os.environ["STUB_DIR"], "xvfb.pid").write_text(str(os.getpid()))
mode = os.environ.get("STUB_XVFB", "ready")
if mode == "ready":
    os.write(descriptor, b"77\\n")
elif mode == "parts":
    os.write(descriptor, b"7")
    time.sleep(0.2)
    os.write(descriptor, b"7\\n")
elif mode == "garbage":
    os.write(descriptor, b"not-a-display\\n")
elif mode == "closed":
    os.close(descriptor)
time.sleep(120)
"""

GAME = """#!{python}
import json, os, time
from pathlib import Path
session = Path(os.environ["HONING_SESSION_DIR"])
Path(os.environ["STUB_DIR"], "game.pid").write_text(str(os.getpid()))
mode = os.environ.get("STUB_GAME", "play")
def publish(name, value):
    temporary = session / (name + ".tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(session / name)
def observe(identity, tick):
    publish("observation.json", {{
        "schema": "honing.visible.v1", "profile": "visible", "capabilities": {{"pixels_to_model": False}},
        "frame": {{"id": identity, "tick": tick}}, "view": {{"screen_rect": [0, 0, 960, 523]}},
        "terrain": {{"cell_size": [40, 40], "rows": ["Ā"]}}, "objects": [], "self": {{}}, "hud": {{"text": []}},
    }})
if mode == "exit":
    raise SystemExit(3)
identity, tick = 1, 0
observe(identity, tick)
while True:
    try:
        command = json.loads((session / "action.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        time.sleep(0.005)
        continue
    if command.get("id") != identity:
        time.sleep(0.005)
        continue
    if command["action"] == "stop":
        raise SystemExit(0)
    if mode == "die":
        raise SystemExit(4)
    if mode == "hang":
        time.sleep(120)
    if mode == "refuse":
        publish("ack.json", {{"ack": 1, "id": identity, "status": "refused", "reason": "stub refusal"}})
        time.sleep(120)
    action = "wait" if mode == "swap" else command["action"]
    publish("ack.json", {{
        "ack": identity, "id": identity, "status": "applied", "action": action, "frames": command["frames"],
        "jump_frames": command.get("jump_frames"), "frame_before": tick, "frame_after": tick + command["frames"],
        "observation_id": identity + 1, "started": [], "children": [], "keys": {{}},
    }})
    identity, tick = identity + 1, tick + command["frames"]
    observe(identity, tick)
"""


def alive(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


@pytest.fixture
def stubs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "stubs"
    directory.mkdir()
    for name, text in (("Xvfb", XVFB), ("game", GAME)):
        path = directory / name
        path.write_text(
            text.format(python=sys.executable) if name == "game" else text.replace("{python}", sys.executable)
        )
        path.chmod(0o755)
    monkeypatch.setenv("PATH", f"{directory}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("STUB_DIR", str(directory))
    return directory


def owned_pids(stubs: Path) -> list[int]:
    return [int((stubs / name).read_text()) for name in ("game.pid", "xvfb.pid") if (stubs / name).exists()]


def wait_gone(pids: list[int]) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and any(alive(pid) for pid in pids):
        time.sleep(0.01)
    assert not any(alive(pid) for pid in pids)


def new_game(stubs: Path, tmp_path: Path, **kwargs: Any) -> Any:
    return run.Game(stubs / "game", tmp_path / "session", observation_timeout=5, **kwargs)


@pytest.mark.parametrize("mode", ["ready", "parts"])
@pytest.mark.integration
def test_display_is_used_after_its_complete_ready_line(
    stubs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setenv("STUB_XVFB", mode)
    with (tmp_path / "display.log").open("wb") as log:
        process, display = run.start_xvfb(log, timeout=5)
    try:
        assert display == ":77" and process.poll() is None
    finally:
        run.stop_owned(process)
    assert process.poll() is not None


@pytest.mark.parametrize(
    ("mode", "error"), [("garbage", RuntimeError), ("closed", RuntimeError), ("silent", TimeoutError)]
)
@pytest.mark.integration
def test_display_that_never_becomes_ready_is_stopped(
    stubs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, error: type[Exception]
) -> None:
    monkeypatch.setenv("STUB_XVFB", mode)
    with (tmp_path / "display.log").open("wb") as log, pytest.raises(error):
        run.start_xvfb(log, timeout=1)
    wait_gone(owned_pids(stubs))


@pytest.mark.integration
def test_game_that_exits_during_boot_leaves_no_display(
    stubs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STUB_GAME", "exit")
    game = new_game(stubs, tmp_path)
    with pytest.raises(RuntimeError, match="exited with code 3"):
        game.start()
    wait_gone(owned_pids(stubs))
    receipt = json.loads((tmp_path / "session/cleanup.json").read_text())
    assert receipt["game"]["returncode"] == 3 and receipt["display"]["reaped"] is True


@pytest.mark.integration
def test_launch_receipt_failure_stops_both_children_and_no_other_process(stubs: Path, tmp_path: Path) -> None:
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    original = run.write_json

    def failing(path: Path, value: Any) -> None:
        if path.name == "launch.json":
            raise OSError("injected receipt failure")
        original(path, value)

    game = new_game(stubs, tmp_path)
    try:
        with patch.object(run, "write_json", failing), pytest.raises(OSError, match="injected"):
            game.start()
        wait_gone(owned_pids(stubs))
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


@pytest.mark.integration
def test_existing_output_directory_is_left_untouched(stubs: Path, tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    (session / "caller.txt").write_bytes(b"caller data\x00")
    game = new_game(stubs, tmp_path)
    with pytest.raises(ValueError, match="new directory"):
        game.start()
    game.close()
    assert [path.name for path in session.iterdir()] == ["caller.txt"]
    assert (session / "caller.txt").read_bytes() == b"caller data\x00"
    assert owned_pids(stubs) == []


@pytest.mark.integration
def test_game_session_steps_and_stops_cleanly(stubs: Path, tmp_path: Path) -> None:
    game = new_game(stubs, tmp_path)
    try:
        first = game.start()
        after, ack = game.step(first, run.native_command("short_jump_left", 12))
        assert after["frame"] == {"id": 2, "tick": 12} and ack["jump_frames"] == run.SHORT_JUMP_FRAMES
        game.stop(after)
    finally:
        game.close()
    wait_gone(owned_pids(stubs))
    receipt = json.loads((tmp_path / "session/cleanup.json").read_text())
    assert receipt["game"] == {"pid": receipt["game"]["pid"], "returncode": 0, "reaped": True}


@pytest.mark.parametrize(
    ("mode", "error", "message"),
    [
        ("refuse", ValueError, "stub refusal"),
        ("swap", ValueError, "different window"),
        ("die", RuntimeError, "exited with code 4"),
        ("hang", TimeoutError, "No new observation"),
    ],
)
@pytest.mark.integration
def test_native_failure_stops_the_run_and_reaps_the_children(
    stubs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, error: type[Exception], message: str
) -> None:
    monkeypatch.setenv("STUB_GAME", mode)
    game = run.Game(stubs / "game", tmp_path / "session", observation_timeout=1)
    try:
        first = game.start()
        with pytest.raises(error, match=message):
            game.step(first, {"action": "right", "frames": 12})
    finally:
        game.close()
    wait_gone(owned_pids(stubs))


@contextmanager
def decision_server(respond: Any) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        bodies: ClassVar[list[bytes]] = []

        def log_message(self, *_args: Any) -> None:
            pass

        def do_GET(self) -> None:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"ready":true}')

        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            body = respond(request)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def cli(stubs: Path, output: Path, url: str, decisions: int = 2) -> list[str]:
    return [
        "run.py",
        "--game",
        str(stubs / "game"),
        "--output",
        str(output),
        "--decision-url",
        url,
        "--decisions",
        str(decisions),
    ]


@pytest.mark.integration
def test_complete_run_keeps_the_existing_trace_layout(stubs: Path, tmp_path: Path) -> None:
    def respond(request: dict[str, Any]) -> bytes:
        return json.dumps(response_for(request, "right")).encode()

    output = tmp_path / "session"
    with decision_server(respond) as url, patch.object(sys, "argv", cli(stubs, output, url)):
        run.main()
    wait_gone(owned_pids(stubs))
    records = [json.loads(line) for line in (output / "decisions.jsonl").read_text().splitlines()]
    assert [record["choice"] for record in records] == ["right", "right"]
    assert [record["responseRawFile"] for record in records] == ["responses/000001.raw", "responses/000002.raw"]
    assert all({"request", "response", "ack"} <= set(record) for record in records)
    assert json.loads((output / "result.json").read_text()) == {
        "decisions": 2,
        "capture": False,
        "heuristicFallback": False,
    }
    assert (output / "model.json").is_file() and (output / "cleanup.json").is_file()


@pytest.mark.integration
def test_rejected_http_response_leaves_no_result_and_no_command(stubs: Path, tmp_path: Path) -> None:
    raw = b'{"id": 1, "answers": {"action": {"type": "choice", "choice": "right"}}, "extra": 1e309}'
    output = tmp_path / "session"
    with (
        decision_server(lambda _request: raw) as url,
        patch.object(sys, "argv", cli(stubs, output, url)),
        pytest.raises(ValueError),
    ):
        run.main()
    wait_gone(owned_pids(stubs))
    assert (output / "responses/000001.raw").read_bytes() == raw
    assert not (output / "result.json").exists() and not (output / "action.json").exists()


@pytest.mark.parametrize(
    "arguments", [["--timeout", "0"], ["--timeout", "3571"], ["--frames", "121"], ["--decisions", "0"]]
)
def test_invalid_cli_values_start_nothing(tmp_path: Path, arguments: list[str]) -> None:
    argv = ["run.py", "--game", "game", "--output", str(tmp_path / "output"), *arguments]
    with patch.object(sys, "argv", argv), patch.object(run.subprocess, "Popen") as popen, pytest.raises(SystemExit):
        run.main()
    assert not popen.called and not (tmp_path / "output").exists()


def capture_session(tmp_path: Path, frames: int, names: list[str]) -> Path:
    (tmp_path / "frames").mkdir()
    for name in names:
        (tmp_path / "frames" / name).write_bytes(b"png")
    (tmp_path / "result.json").write_text(json.dumps({"capture": True, "heuristicFallback": False}))
    (tmp_path / "observations.jsonl").write_text(json.dumps({"frame": frames}) + "\n")
    return tmp_path


def test_capture_with_every_frame_can_be_encoded(tmp_path: Path) -> None:
    assert encode.validate_capture(capture_session(tmp_path, 2, ["000000.png", "000001.png"])) == 2


@pytest.mark.parametrize("names", [["000000.png"], ["000000.png", "000002.png"], []])
def test_capture_with_missing_frames_is_refused(tmp_path: Path, names: list[str]) -> None:
    with pytest.raises(ValueError, match="contiguous"):
        encode.validate_capture(capture_session(tmp_path, 2, names))


@pytest.mark.integration
def test_review_sample_passes_its_offline_check() -> None:
    check = HERE / "validation/review-sample/check.py"
    result = subprocess.run([sys.executable, "-S", str(check)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["checks"] == "passed"


def test_request_builder_does_not_modify_the_observation() -> None:
    before = observation()
    snapshot = copy.deepcopy(before)
    run.request_for(before, 12)
    assert before == snapshot
