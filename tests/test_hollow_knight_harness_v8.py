"""The Hollow Knight harness v8 experiment: its response guard, launcher, visible profile and committed evidence."""

from __future__ import annotations

import gzip
import importlib
import importlib.util
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tests.conftest import REPO_ROOT, load_script

INTEGRATION = REPO_ROOT / "integrations/hollow-knight"
EXPERIMENT = INTEGRATION / "experiments/harness-v8"
TRACE = EXPERIMENT / "data/eval_laya_v8_L1L2"
SAMPLE = INTEGRATION / "validation/review-sample"
UNKNOWN_BIT = 64


@contextmanager
def experiment_imports() -> Iterator[None]:
    """Import from the experiment without leaving its generic `core` and `harness` names importable."""

    def is_generic(name: str) -> bool:
        return name in ("core", "harness") or name.startswith(("core.", "harness."))

    saved_modules = {name: module for name, module in sys.modules.items() if is_generic(name)}
    saved_path = list(sys.path)
    for name in saved_modules:
        del sys.modules[name]
    sys.path.insert(0, str(EXPERIMENT))
    try:
        yield
    finally:
        sys.path[:] = saved_path
        for name in [name for name in sys.modules if is_generic(name)]:
            del sys.modules[name]
        sys.modules.update(saved_modules)


@pytest.fixture(scope="module")
def laya() -> ModuleType:
    with experiment_imports():
        return importlib.import_module("core.laya")


@pytest.fixture(scope="module")
def env_module() -> ModuleType:
    with experiment_imports():
        return importlib.import_module("core.env")


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    with experiment_imports():
        spec = importlib.util.spec_from_file_location("harness_v8_run_agent", EXPERIMENT / "run_agent.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def strict_json_loads():
    return load_script("integrations/_shared/decision.py").strict_json_loads


def trace_rows(name: str) -> list[dict]:
    text = gzip.decompress((TRACE / f"{name}.jsonl.gz").read_bytes()).decode()
    return [json.loads(line) for line in text.splitlines()]


@pytest.fixture(scope="module")
def decisions() -> list[dict]:
    return trace_rows("decisions")


@pytest.fixture(scope="module")
def transition() -> dict:
    return json.loads((SAMPLE / "transition.json").read_text())


# ---- response guard ---------------------------------------------------------------------------


@pytest.mark.unit
def test_every_recorded_response_replays_identically_through_the_shared_receiver(
    laya, decisions, strict_json_loads
) -> None:
    assert len(decisions) == 47
    for row in decisions:
        response = strict_json_loads(row["responseRaw"])
        assert laya.checked_choice(response, row["request"]) == (
            row["choice"],
            row["probability"],
            row["probabilities"],
        )


def damage(response: dict, problem: str) -> None:
    answer = response["answers"]["action"]
    probabilities = answer["probabilities"]
    audit = response["tokenAudit"]["action"]
    if problem == "stale_id":
        response["id"] += 1
    elif problem == "changed_state_echo":
        response["stateText"] += " altered"
    elif problem == "removed_context":
        response["trimmed"] = [{"section": "essential", "text": "removed"}]
    elif problem == "no_trimmed_report":
        del response["trimmed"]
    elif problem == "no_audit":
        del response["tokenAudit"]
    elif problem == "audit_without_token_counts":
        del audit["stateTokens"]
    elif problem == "truncated":
        audit["stateTruncated"] = True
    elif problem == "wrong_option_count":
        audit["options"] += 1
    elif problem == "not_a_choice_answer":
        del answer["type"]
    elif problem == "extra_answer":
        response["answers"]["other"] = dict(answer)
    elif problem == "missing_option":
        del probabilities[next(iter(probabilities))]
    elif problem == "unoffered_option":
        probabilities["teleport"] = 0.0
    elif problem == "all_zero":
        probabilities.update(dict.fromkeys(probabilities, 0))
    elif problem == "all_two":
        probabilities.update(dict.fromkeys(probabilities, 2))
    elif problem == "all_true":
        probabilities.update(dict.fromkeys(probabilities, True))
    elif problem == "not_finite":
        probabilities[answer["choice"]] = float("nan")
    elif problem == "negative":
        probabilities[answer["choice"]] = -0.1
    elif problem == "not_the_argmax":
        answer["choice"] = min(probabilities, key=probabilities.get)
    else:
        raise AssertionError(problem)


@pytest.mark.unit
@pytest.mark.parametrize(
    "problem",
    [
        "stale_id",
        "changed_state_echo",
        "removed_context",
        "no_trimmed_report",
        "no_audit",
        "audit_without_token_counts",
        "truncated",
        "wrong_option_count",
        "not_a_choice_answer",
        "extra_answer",
        "missing_option",
        "unoffered_option",
        "all_zero",
        "all_two",
        "all_true",
        "not_finite",
        "negative",
        "not_the_argmax",
    ],
)
def test_guard_rejects_a_damaged_recorded_response(laya, decisions, problem: str) -> None:
    response = json.loads(decisions[0]["responseRaw"])
    damage(response, problem)
    with pytest.raises(laya.GuardRejection):
        laya.checked_choice(response, decisions[0]["request"])


@contextmanager
def decision_server(status: int, body: bytes) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args) -> None:
            pass

        def do_POST(self) -> None:
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(status)
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


@pytest.mark.integration
@pytest.mark.parametrize(
    ("status", "body"),
    [
        (200, b'{"id": 1, broken'),
        (200, b'{"id": 1, "id": 2}'),
        (200, b'{"id": NaN}'),
        (200, b"\xff\xfe"),
        (200, b"[]"),
        (503, b"upstream model failed"),
    ],
)
def test_client_keeps_the_exact_bytes_of_a_rejected_response(laya, tmp_path: Path, status: int, body: bytes) -> None:
    raw_path = tmp_path / "0001.raw"
    with decision_server(status, body) as url, pytest.raises(laya.GuardRejection):
        laya.LayaClient(url, timeout=10).decide({"id": 1}, raw_path=raw_path)
    assert raw_path.read_bytes() == body


@pytest.mark.integration
def test_unreachable_decision_server_is_a_rejection(laya, tmp_path: Path) -> None:
    with decision_server(200, b"{}") as url:
        pass
    with pytest.raises(laya.GuardRejection):
        laya.LayaClient(url, timeout=2).decide({"id": 1}, raw_path=tmp_path / "0001.raw")


# ---- run loop: the model's choice is executed exactly, a rejection executes nothing ---------------


def scripted_run(runner: ModuleType, monkeypatch, tmp_path: Path, row: dict, response: dict) -> SimpleNamespace:
    """Run the real decision loop for one decision around a fake game, agent and model service."""
    seen = SimpleNamespace(executed=[], closed=0, out=tmp_path / "run")
    menu = {
        "essential": row["request"]["parts"]["essential"],
        "instructions": row["request"]["questions"]["action"]["instructions"],
        "options": row["request"]["questions"]["action"]["criteria"],
    }

    class Env:
        frames = 0

        def __init__(self, out: Path, **_kwargs) -> None:
            self.session = out / "session"
            self.session.mkdir()
            self.obs = {"frame": {"id": row["request"]["id"], "tick": 0}}

        def start(self) -> None:
            pass

        def close(self) -> dict:
            seen.closed += 1
            return {"windows": 0, "frames": 0}

    class Agent:
        def __init__(self, _env) -> None:
            pass

        def boot(self) -> None:
            pass

        def menu(self) -> dict:
            return menu

        def execute(self, choice: str, index: int, label: str, probability: float) -> dict:
            seen.executed.append((choice, index, label, probability))
            return {"reason": "done", "windows": 0}

    class Client:
        def health(self) -> dict:
            return {"ready": True}

        def decide(self, _request: dict, raw_path: Path) -> tuple[dict, float]:
            raw_path.write_text(json.dumps(response))
            return response, 0.01

    monkeypatch.setattr(runner, "HKEnv", Env)
    monkeypatch.setattr(runner, "LayaClient", Client)
    monkeypatch.setattr(runner, "importlib", SimpleNamespace(import_module=lambda _name: SimpleNamespace(Agent=Agent)))
    arguments = ["run_agent.py", "--harness", "v8", "--max-decisions", "1", "--no-video", "--out", str(seen.out)]
    monkeypatch.setattr(sys, "argv", arguments)
    runner.main()
    return seen


@pytest.mark.integration
def test_the_validated_argmax_is_the_option_executed(runner, monkeypatch, tmp_path: Path, decisions) -> None:
    row = next(row for row in decisions if row["choice"] != row["offered"][0])
    seen = scripted_run(runner, monkeypatch, tmp_path, row, json.loads(row["responseRaw"]))
    assert seen.executed == [(row["choice"], 1, f"Laya chose {row['choice']}", row["probability"])]
    record = json.loads((seen.out / "session/decisions.jsonl").read_text())
    assert (record["choice"], record["probabilities"]) == (row["choice"], row["probabilities"])
    assert seen.closed == 1


@pytest.mark.integration
def test_a_rejected_response_executes_nothing_and_is_recorded(runner, monkeypatch, tmp_path: Path, decisions) -> None:
    row = decisions[0]
    response = json.loads(row["responseRaw"])
    response["trimmed"] = [{"section": "essential", "text": "removed"}]
    seen = scripted_run(runner, monkeypatch, tmp_path, row, response)
    assert seen.executed == []
    record = json.loads((seen.out / "session/decisions.jsonl").read_text())
    assert record["request"] == row["request"] and record["rejected"]
    assert json.loads((seen.out / "session/responses/0001.raw").read_text()) == response
    assert json.loads((seen.out / "summary.json").read_text())["status"].startswith("guard rejection at decision 1")
    assert seen.closed == 1


@pytest.mark.integration
def test_an_existing_output_directory_is_refused_and_left_alone(
    runner, monkeypatch, tmp_path: Path, decisions
) -> None:
    (tmp_path / "run").mkdir()
    (tmp_path / "run/keep.txt").write_text("caller data")
    with pytest.raises(FileExistsError):
        scripted_run(runner, monkeypatch, tmp_path, decisions[0], json.loads(decisions[0]["responseRaw"]))
    assert [path.name for path in (tmp_path / "run").iterdir()] == ["keep.txt"]
    assert (tmp_path / "run/keep.txt").read_text() == "caller data"


# ---- launcher: owned processes, display readiness, coordinated timeouts ---------------------------


PLAYER = r"""
import json, os, sys, time
from pathlib import Path
session = Path(os.environ["HONING_SESSION_DIR"])
mode = os.environ.get("TEST_PLAYER_MODE", "ok")
names = ["DISPLAY", "HONING_OBSERVATION_PROFILE", "HONING_MODEL_TIMEOUT_SECONDS", "HONING_CAPTURE", "HONING_SCENE"]
(session / "environment.json").write_text(json.dumps({name: os.environ.get(name) for name in names}))
if mode == "die":
    sys.exit(3)
identity, tick = 1, 0
def publish():
    visible = {"schema": "honing.visible.v1", "frame": {"id": identity, "tick": tick}}
    body = {"id": identity, "frame": tick} if mode == "scene_state" else visible
    temporary = session / "observation.tmp"
    temporary.write_text(json.dumps(body))
    temporary.replace(session / "observation.json")
publish()
(session / "debug_observation.json").write_text('{"schema": "debug state the harness must not read"}')
deadline = time.monotonic() + 120
while time.monotonic() < deadline:
    path = session / "action.json"
    if path.exists():
        command = json.loads(path.read_text())
        if command["id"] == identity:
            if command["action"] == "stop":
                sys.exit(0)
            identity, tick = identity + 1, tick + command["frames"] + (mode == "skips_a_frame")
            publish()
    time.sleep(0.005)
"""

DISPLAY = r"""
import os, sys, time
fd = int(sys.argv[sys.argv.index("-displayfd") + 1])
mode = os.environ.get("TEST_DISPLAY_MODE", "ok")
if mode == "exit":
    sys.exit(7)
if mode == "bad":
    os.write(fd, b"invalid\n")
if mode == "ok":
    os.write(fd, b"12")
    time.sleep(0.05)
    os.write(fd, b"34\n")
time.sleep(120)
"""


def script(path: Path, body: str) -> Path:
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(0o755)
    return path


@pytest.fixture
def launched(env_module, tmp_path: Path, monkeypatch) -> list[subprocess.Popen]:
    """Point the launcher at stub programs and keep a handle on every process it starts."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script(bin_dir / "Xvfb", DISPLAY)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("HONING_HK_GAME", str(script(tmp_path / "player", PLAYER)))
    processes: list[subprocess.Popen] = []
    popen = subprocess.Popen

    def recording(*args, **kwargs) -> subprocess.Popen:
        processes.append(popen(*args, **kwargs))
        return processes[-1]

    monkeypatch.setattr(env_module.subprocess, "Popen", recording)
    return processes


def assert_all_stopped(processes: list[subprocess.Popen], count: int) -> None:
    """A return code is set only after the launcher has waited for that child."""
    assert len(processes) == count and all(process.returncode is not None for process in processes)


@pytest.mark.integration
def test_episode_launch_sends_windows_and_stops_its_own_processes(env_module, laya, launched, tmp_path: Path) -> None:
    env = env_module.HKEnv(tmp_path / "run", scene="Level1", capture=False, observation_timeout=30)
    try:
        first = env.start()
        assert first["frame"] == {"id": 1, "tick": 0}
        assert env.step("right", 3, "label")["frame"] == {"id": 2, "tick": 3}
        seen = json.loads((env.session / "environment.json").read_text())
        assert seen["DISPLAY"] == ":1234" and seen["HONING_OBSERVATION_PROFILE"] == "visible"
        # The bridge watchdog must outlast the longest model call it waits for.
        assert float(seen["HONING_MODEL_TIMEOUT_SECONDS"]) >= laya.LayaClient(timeout=240).timeout + 30
        window = json.loads((env.session / "windows.jsonl").read_text())
        assert (window["action"], window["frames"], window["label"]) == ("right", 3, "label")
    finally:
        closed = env.close()
    assert closed["game_exit"] == 0 and closed["windows"] == 1
    assert env.close() == closed
    assert_all_stopped(launched, 2)


@pytest.mark.integration
def test_launch_needs_a_named_player_and_starts_nothing_without_one(
    env_module, launched, tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("HONING_HK_GAME")
    with pytest.raises(RuntimeError, match="HONING_HK_GAME"):
        env_module.HKEnv(tmp_path / "run").start()
    assert launched == []


@pytest.mark.integration
@pytest.mark.parametrize(("mode", "error"), [("exit", RuntimeError), ("bad", RuntimeError), ("hang", TimeoutError)])
def test_display_failure_stops_the_owned_display(env_module, launched, tmp_path, monkeypatch, mode, error) -> None:
    monkeypatch.setenv("TEST_DISPLAY_MODE", mode)
    announced = env_module.read_display_number
    budget = 0.3 if mode == "hang" else 30
    monkeypatch.setattr(env_module, "read_display_number", lambda fd: announced(fd, timeout_seconds=budget))
    with pytest.raises(error):
        env_module.HKEnv(tmp_path / "run").start()
    assert_all_stopped(launched, 1)


@pytest.mark.integration
@pytest.mark.parametrize(("mode", "error"), [("die", RuntimeError), ("scene_state", ValueError)])
def test_failed_game_start_stops_everything_it_started(
    env_module, launched, tmp_path, monkeypatch, mode, error
) -> None:
    monkeypatch.setenv("TEST_PLAYER_MODE", mode)
    with pytest.raises(error):
        env_module.HKEnv(tmp_path / "run", observation_timeout=30).start()
    assert_all_stopped(launched, 2)


@pytest.mark.integration
def test_a_bridge_that_skips_a_frame_is_refused(env_module, launched, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TEST_PLAYER_MODE", "skips_a_frame")
    env = env_module.HKEnv(tmp_path / "run", observation_timeout=30)
    env.start()
    try:
        with pytest.raises(ValueError, match="exactly one window"):
            env.step("right", 3)
    finally:
        env.close()
    assert_all_stopped(launched, 2)


@pytest.mark.unit
def test_display_number_waits_for_the_complete_line(env_module) -> None:
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, b"7")
        timer = threading.Timer(0.05, os.write, (write_fd, b"7\n"))
        timer.start()
        assert env_module.read_display_number(read_fd, timeout_seconds=5) == "77"
        timer.join()
    finally:
        os.close(read_fd)
        os.close(write_fd)


# ---- the visible profile as the harness receives it -----------------------------------------------


def cell_mask(row: str, column: int) -> int:
    return ord(row[column]) - 0x100


def assert_documented_envelope(envelope: dict) -> None:
    assert (envelope["schema"], envelope["adapter"], envelope["profile"]) == (
        "honing.visible.v1",
        "hollow-knight-visible-v1",
        "visible",
    )
    expected = {"schema", "adapter", "profile", "game", "frame", "view", "terrain", "objects", "self", "hud"}
    assert set(envelope) == expected | {"rules", "capabilities", "disclosure"}
    assert set(envelope["frame"]) == {"id", "tick", "simulation_time"}
    assert envelope["view"]["screen_rect"] == [0, 0, 960, 540] and len(envelope["view"]["world_from_screen"]) == 6
    terrain = envelope["terrain"]
    assert terrain["cell_size"] == [20, 20] and len(terrain["rows"]) == 27
    for row in terrain["rows"]:
        assert len(row) == 48 and all(1 <= ord(cell) - 0x100 <= 255 for cell in row)
    kinds = {"player", "visible_equipment", "enemy", "visible_hazard", "platform", "wall", "interactable"}
    for item in envelope["objects"]:
        assert set(item) == {"id", "kind", "bounds"} and re.fullmatch(r"v\d+", item["id"])
        assert item["kind"] in kinds | {"visible_scenery"}
        x, y, width, height = item["bounds"]
        assert width > 0 and height > 0 and x >= 0 and x + width <= 960 and y >= 0 and y + height <= 540
    identity = envelope["self"]["object_id"]
    assert identity == "" or [item["kind"] for item in envelope["objects"] if item["id"] == identity] == ["player"]
    assert envelope["capabilities"]["pixels_to_model"] is False


@pytest.mark.unit
def test_every_recorded_visible_observation_keeps_the_documented_boundary() -> None:
    rows = trace_rows("visible_observations")
    assert [row["frame"]["id"] for row in rows] == list(range(1, 1168))
    scene_state = {"health", "enemies", "exits", "vx", "vy", "grounded", "jumps", "x", "y", "scene"}
    for envelope in rows:
        assert_documented_envelope(envelope)
        assert not scene_state & set(envelope)
        # The recording's caption box covers x 8..667 and y 8..55: those cells can never claim to be clear.
        for row in envelope["terrain"]["rows"][:3]:
            assert all(cell_mask(row, column) & UNKNOWN_BIT for column in range(34))


@pytest.mark.unit
def test_harness_perception_reads_the_recorded_envelope(transition) -> None:
    pytest.importorskip("numpy")
    with experiment_imports():
        perception = importlib.import_module("harness.v8.perception")
    scene = perception.Scene(transition["visibleBefore"])
    assert scene.hp == 5 and scene.self_ok and scene.masks.shape == (perception.ROWS, perception.COLS)
    assert scene.support() is not None


@pytest.mark.unit
def test_bridge_and_harness_agree_on_the_grid_encoding() -> None:
    pytest.importorskip("numpy")
    source = (INTEGRATION / "VisibleObservation.cs").read_text()
    assert "const int Width = 960, Height = 540;" in source and "int cell=20;" in source
    alphabet = re.search(r"char\[\] alphabet=\{(.*?)\};", source).group(1).replace("'", "").split(",")
    assert alphabet == [".", "#", "=", "!", "@", "+", "?", ":"]
    with experiment_imports():
        perception = importlib.import_module("harness.v8.perception")
    bits = (perception.BG, perception.WALL, perception.PLAT, perception.HAZ)
    bits += (perception.ACTOR, perception.INTER, perception.UNK, perception.SCEN)
    assert bits == tuple(1 << index for index in range(8))
    assert (perception.CELL, perception.COLS, perception.ROWS) == (20, 48, 27)


# ---- committed evidence ----------------------------------------------------------------------------


@pytest.mark.integration
def test_committed_decision_table_regenerates_from_the_committed_trace(tmp_path: Path) -> None:
    output = tmp_path / "table.md"
    command = [sys.executable, "tools/decision_table.py", "data/eval_laya_v8_L1L2", "--md", str(output)]
    subprocess.run(command, cwd=EXPERIMENT, check=True, capture_output=True)
    assert output.read_text() == (TRACE / "decision_table.md").read_text()


@pytest.mark.unit
def test_results_table_matches_the_committed_trace(decisions) -> None:
    runs = {row["run"]: row for row in json.loads((EXPERIMENT / "data/runs.json").read_text())["runs"]}
    main = runs["eval_laya_v8_L1L2"]
    debug = trace_rows("observations")
    windows = trace_rows("windows")
    deaths = sum(a["health"] > 0 and b["health"] == 0 for a, b in itertools.pairwise(debug))
    assert (main["decisions"], main["windows"], main["frames"]) == (len(decisions), len(windows), debug[-1]["frame"])
    assert (main["deaths"], main["minHp"]) == (deaths, min(row["health"] for row in debug)) == (0, 4)
    assert sum(window["frames"] for window in windows) == main["frames"]
    assert debug[-1]["scene"] == "Level2" and debug[-1]["x"] >= 124
    assert [row["scene"] for row in debug[:1]] == ["Level1"] and main["outcome"] == "pass"


@pytest.mark.integration
def test_review_sample_passes_its_offline_check() -> None:
    command = [sys.executable, "-S", str(SAMPLE / "check.py")]
    completed = subprocess.run(command, check=True, capture_output=True, text=True, cwd=REPO_ROOT)
    report = json.loads(completed.stdout)
    assert (report["checks"], report["choice"], report["freshModelCall"]) == ("passed", "advance_right", False)


@pytest.mark.integration
def test_review_sample_check_refuses_changed_response_bytes(tmp_path: Path) -> None:
    copy_root = tmp_path / "integrations"
    shutil.copytree(REPO_ROOT / "integrations/_shared", copy_root / "_shared")
    target = copy_root / "hollow-knight/validation/review-sample"
    shutil.copytree(SAMPLE, target)
    raw = (target / "response.raw").read_bytes()
    (target / "response.raw").write_bytes(raw.replace(b'"choice": "advance_right"', b'"choice": "retreat_left"', 1))
    completed = subprocess.run([sys.executable, "-S", str(target / "check.py")], capture_output=True, text=True)
    assert completed.returncode != 0 and "Sample bytes changed: response.raw" in completed.stderr


@pytest.mark.unit
def test_recorded_request_is_rebuilt_exactly_by_the_committed_request_builder(laya, decisions) -> None:
    for row in decisions:
        request = row["request"]
        question = request["questions"]["action"]
        rebuilt = laya.build_request(
            request["id"], request["parts"]["essential"], question["instructions"], question["criteria"]
        )
        assert rebuilt == request and json.dumps(rebuilt) == json.dumps(request)


@pytest.mark.unit
def test_every_recorded_response_kept_the_complete_context_and_menu(decisions) -> None:
    for row in decisions:
        request = row["request"]
        response = json.loads(row["responseRaw"])
        assert response["stateText"] == "\n".join(request["parts"]["essential"]) and response["trimmed"] == []
        assert 3 <= len(request["questions"]["action"]["criteria"]) <= 16
        assert list(request["questions"]["action"]["criteria"]) == row["offered"]


@pytest.mark.unit
def test_demo_stays_within_the_repository_file_limit() -> None:
    assert (TRACE / "demo_cut.mp4").stat().st_size <= 1024 * 1024
