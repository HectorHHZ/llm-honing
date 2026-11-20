"""Game-owned transport and lifecycle regressions; no model or native game assets."""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "integrations/taisei"
spec = importlib.util.spec_from_file_location("taisei_native_contract", HERE / "run.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


@pytest.fixture(scope="module")
def command_probe(tmp_path_factory):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("C compiler required for the actual native command header")
    output = tmp_path_factory.mktemp("taisei-command")
    source = output / "probe.c"
    source.write_text(
        '#include "honing_command.h"\n'
        "int main(int n, char **v) { if(n != 3) return 2;\n"
        'if(!strcmp(v[1], "timeout")) { uint64_t ms;\n'
        "if(!honing_parse_idle_timeout(v[2], &ms)) return 1;\n"
        'printf("%llu", (unsigned long long)ms); return 0; }\n'
        "HoningCommand c; if(!honing_parse_command(v[2], 7, &c)) return 1;\n"
        'printf("%u %u %d", c.flags, c.frames, c.stop); return 0; }\n'
    )
    binary = output / "probe"
    subprocess.run([compiler, "-std=c11", "-I", str(HERE), str(source), "-o", str(binary)], check=True)
    return binary


@pytest.mark.parametrize(
    "action,direction",
    [
        ("stay", 0),
        ("left", 4),
        ("right", 8),
        ("up", 1),
        ("down", 2),
        ("up_left", 5),
        ("up_right", 9),
        ("down_left", 6),
        ("down_right", 10),
        ("stop", 0),
    ],
)
def test_actual_native_masks(command_probe, action, direction):
    command = json.dumps({"id": 7, "action": action, "frames": 120}, separators=(",", ":"))
    result = subprocess.check_output([command_probe, "command", command], text=True)
    assert result == f"{direction | 16 | 32} 120 {int(action == 'stop')}"


@pytest.mark.parametrize(
    "command",
    [
        '{"id":6,"action":"right","frames":30}',
        '{"id":7,"action":"invalid","frames":30}',
        '{"id":7,"action":"right","frames":0}',
        '{"id":7,"action":"right","frames":121}',
        '{"id":7,"action":"right","frames":-1}',
        '{"id":7,"action":"right","frames":30,"extra":0}',
        '{"id":7,"action":"right","frames":30}garbage',
    ],
)
def test_actual_native_command_rejection(command_probe, command):
    assert subprocess.run([command_probe, "command", command]).returncode == 1


@pytest.mark.parametrize("value", ["", "-1", "+1", "nan", "inf", "1.5", "0", " 1", "18446744073709551616"])
def test_native_watchdog_rejects_bad_environment(command_probe, value):
    assert subprocess.run([command_probe, "timeout", value]).returncode == 1


@pytest.mark.parametrize("value", ["1", "330000", "18446744073709551615"])
def test_native_watchdog_accepts_uint64_milliseconds(command_probe, value):
    assert subprocess.check_output([command_probe, "timeout", value], text=True) == value


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), 1e308, 2**64])
def test_http_watchdog_validation(timeout):
    with pytest.raises(ValueError):
        demo.idle_timeout_milliseconds(timeout)


def test_http_watchdog_margin():
    assert demo.idle_timeout_milliseconds(300) == 330000
    assert demo.idle_timeout_milliseconds(0.0001) == 30001


@pytest.fixture
def fake_display(tmp_path, monkeypatch):
    script = tmp_path / "Xvfb"
    script.write_text(
        f"#!{sys.executable}\n"
        "import os, sys, time\n"
        "from pathlib import Path\n"
        "Path(os.environ['DISPLAY_TEST_PID']).write_text(str(os.getpid()))\n"
        "fd = int(sys.argv[sys.argv.index('-displayfd')+1])\n"
        "mode = os.environ.get('DISPLAY_TEST_MODE', 'slow')\n"
        "if mode == 'exit': sys.exit(7)\n"
        "if mode == 'bad': os.write(fd, b'invalid\\n')\n"
        "if mode == 'slow':\n"
        " time.sleep(.65)\n"
        " os.write(fd, b'123')\n"
        " time.sleep(.1)\n"
        " os.write(fd, b'4\\n')\n"
        "time.sleep(60)\n"
    )
    script.chmod(0o755)
    pid_path = tmp_path / "display.pid"
    monkeypatch.setenv("DISPLAY_TEST_PID", str(pid_path))
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    return pid_path


def assert_reaped(pid_path):
    pid = int(pid_path.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_display_waits_for_slow_complete_readiness(fake_display):
    with open(os.devnull, "wb") as log:
        started = time.monotonic()
        process, display = demo.start_xvfb(log, timeout=3)
        try:
            assert display == ":1234"
            assert time.monotonic() - started >= 0.7
            assert process.poll() is None
        finally:
            demo.stop_owned(process)
    assert_reaped(fake_display)


@pytest.mark.parametrize("mode,error", [("exit", RuntimeError), ("bad", RuntimeError), ("hang", TimeoutError)])
def test_display_failure_reaps_owned_child(fake_display, monkeypatch, mode, error):
    monkeypatch.setenv("DISPLAY_TEST_MODE", mode)
    with open(os.devnull, "wb") as log, pytest.raises(error):
        demo.start_xvfb(log, timeout=0.3)
    assert_reaped(fake_display)


def test_start_failure_cleans_display_and_propagates_timeout(fake_display, monkeypatch, tmp_path):
    monkeypatch.setattr(demo.DecisionClient, "health", lambda _: {"ready": True})
    seen = {}
    real_popen = subprocess.Popen

    def popen(command, **kwargs):
        if command[0] != "Xvfb":
            seen.update(kwargs["env"])
            raise FileNotFoundError("deliberate game launch failure")
        return real_popen(command, **kwargs)

    monkeypatch.setattr(demo.subprocess, "Popen", popen)
    args = argparse.Namespace(
        output=tmp_path / "run",
        timeout=300,
        decision_url="http://localhost:8809",
        resources=tmp_path,
        game=tmp_path / "missing-game",
        capture=False,
        frames=30,
        decisions=1,
    )
    with pytest.raises(FileNotFoundError, match="deliberate"):
        demo.run(args)
    assert seen["HONING_IDLE_TIMEOUT_MS"] == "330000"
    assert seen["DISPLAY"] == ":1234"
    assert_reaped(fake_display)


def test_request_text_describes_actual_movement_window():
    state = {"id": 7, "frame": 60, "player": {"x": 10, "y": 20}}
    request = demo.request_for(state, 30)
    assert request["questions"]["action"]["criteria"] == demo.ACTIONS
    assert len(demo.ACTIONS) == 9
    text, serialized = request["parts"]["essential"]
    assert "30 ticks at 60Hz" in text
    assert "Each action holds movement, shot and focus" in text
    assert "other bullets, lasers and enemies are not in this partial sensor" in text
    assert json.loads(serialized) == state
