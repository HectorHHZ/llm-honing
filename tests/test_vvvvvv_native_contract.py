"""Compile actual bridge routines with fixture engine data, and probe launcher ownership."""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "integrations/vvvvvv"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("vvvvvv_native_contract", HERE / "run.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


@pytest.fixture(scope="module")
def bridge_probe(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler required for actual bridge routines")
    output = tmp_path_factory.mktemp("vvvvvv-bridge")
    bridge = (HERE / "HoningBridge.inc").read_text()
    tile = bridge.split("static int honing_tile", 1)[1].split("bool honing_active", 1)[0]
    observe = bridge.split("static void honing_observe", 1)[1].split("\nvoid honing_run", 1)[0]
    inputs = bridge.split("void honing_input", 1)[1].split("static void honing_one_tick", 1)[0]
    parser = bridge.split("std::istringstream command(line);", 1)[1].split("        if (!valid)", 1)[0]
    source = output / "probe.cpp"
    source.write_text(
        "#include <vector>\n#include <string>\n#include <fstream>\n#include <sstream>\n"
        "#include <iomanip>\n#include <iostream>\n#include <cstdlib>\n#include <cstdio>\n"
        "struct entclass { int xp,yp,w,h,onground,onroof,invis,type; double vx,vy; bool harmful; };\n"
        "struct Objects { std::vector<entclass> entities; int getplayer() { return 0; } } obj;\n"
        "struct Game { int framecounter,roomx,roomy,gravitycontrol,deathseq,deathcounts,totalflips,\n"
        "saverx,savery,savex,savey,savegc; bool hascontrol,press_left,press_right,press_action,press_interact;\n"
        "int get_timestep() { return 34; } } game;\n"
        "struct Map { bool collide(int x,int y,bool spikes) {\n"
        "return spikes ? (x==2 && y==1) : (x==0 && y==0); } } map;\n"
        'static std::string honing_dir,honing_action="wait",honing_last_action="reset";\n'
        "static int honing_id=1,honing_tick=0,honing_epoch=0,honing_milliseconds=0,\n"
        "honing_last_ticks=0,honing_action_tick=0;\n"
        "enum { EntityType_INVALID=-1 }; void VVV_exit(int code) { exit(code); }\n"
        "bool honing_active() { return true; }\n"
        + "static int honing_tile"
        + tile
        + "void honing_input"
        + inputs
        + "static void honing_observe"
        + observe
        + "int main(int argc,char**argv) { if(argc<3) return 2; std::string mode=argv[1];\n"
        'if(mode=="parse") { std::string line=argv[2]; std::istringstream command(line);\n'
        "std::string extra,action;\n" + parser + "\nstd::cout << valid; return 0; }\n"
        'if(mode=="input") { honing_action=argv[2]; honing_action_tick=atoi(argv[3]); honing_input();\n'
        "std::cout << game.press_left << game.press_right << game.press_action << game.press_interact; return 0; }\n"
        'if(mode=="observe") { honing_dir=argv[2]; entclass e={}; e.xp=atoi(argv[3]); e.yp=atoi(argv[4]);\n'
        "obj.entities.push_back(e); honing_observe(); return 0; } return 2; }\n"
    )
    binary = output / "probe"
    subprocess.run([compiler, "-std=c++98", str(source), "-o", str(binary)], check=True)
    return binary


@pytest.mark.parametrize(
    "x,y",
    [
        (-17, 0),
        (-16, 0),
        (-15, 0),
        (-14, 0),
        (-9, 0),
        (-8, 0),
        (-5, 0),
        (-1, 0),
        (0, -9),
        (0, -8),
        (0, -5),
        (0, -2),
        (0, -1),
        (0, 0),
        (7, 7),
        (8, 8),
        (307, 231),
        (308, 232),
        (319, 238),
        (320, 240),
    ],
)
@pytest.mark.functional
def test_actual_observation_grid_uses_floor(bridge_probe, tmp_path, x, y):
    subprocess.run([bridge_probe, "observe", str(tmp_path), str(x), str(y)], check=True)
    state = json.loads((tmp_path / "observation.json").read_text())
    origin = [x // 8 - 4, y // 8 - 4]
    assert state["localOriginTiles"] == origin
    rows = []
    for ty in range(origin[1], origin[1] + 9):
        row = ""
        for tx in range(origin[0], origin[0] + 9):
            row += (
                "?"
                if tx < 0 or tx >= 40 or ty < 0 or ty >= 29
                else "#" if (tx, ty) == (0, 0) else "!" if (tx, ty) == (2, 1) else "."
            )
        rows.append(row)
    assert state["localSolidRows"] == rows


@pytest.mark.parametrize(
    "action,left,right",
    [
        ("wait", False, False),
        ("left", True, False),
        ("right", False, True),
        ("flip", False, False),
        ("flip_left", True, False),
        ("flip_right", False, True),
    ],
)
@pytest.mark.functional
def test_actual_input_masks_and_release_edge(bridge_probe, action, left, right):
    for tick in (0, 1, 2):
        bits = subprocess.check_output([bridge_probe, "input", action, str(tick)], text=True)
        assert bits == f"{int(left)}{int(right)}{int(tick == 1 and action.startswith('flip'))}0"


@pytest.mark.parametrize(
    "command,accepted",
    [
        ("1 wait 2", True),
        ("1 flip_right 120", True),
        ("1 stop 0", True),
        ("1 reset 0", True),
        ("0 right 15", False),
        ("2 right 15", False),
        ("1 invented 15", False),
        ("1 right 1", False),
        ("1 right 121", False),
        ("1 stop 1", False),
        ("1 right 15 trailing", False),
        ("1 right 15" + " " * 129, False),
    ],
)
@pytest.mark.functional
def test_actual_native_command_parser(bridge_probe, command, accepted):
    assert subprocess.check_output([bridge_probe, "parse", command], text=True) == str(int(accepted))


@pytest.mark.functional
@pytest.mark.parametrize("launch_write_failure", [False, True])
def test_missing_procfs_and_launch_failure_reap_owned_game(tmp_path, monkeypatch, launch_write_failure):
    game = tmp_path / "game"
    game.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(60)\n")
    game.chmod(0o755)
    output = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--game",
            str(game),
            "--assets",
            str(tmp_path / "data.zip"),
            "--decision-url",
            "http://localhost:8809",
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(demo.DecisionClient, "health", lambda _: {"ready": True})
    original_read = Path.read_text

    def no_procfs(path, *args, **kwargs):
        if str(path).startswith("/proc/"):
            raise FileNotFoundError("simulated host without procfs")
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", no_procfs)
    children = []
    original_popen = subprocess.Popen
    unrelated = original_popen([sys.executable, "-c", "import time; time.sleep(60)"])

    def popen(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(demo.subprocess, "Popen", popen)
    monkeypatch.setattr(demo.NativeClient, "observe", lambda *_: (_ for _ in ()).throw(RuntimeError("probe complete")))
    if launch_write_failure:
        original_write = demo.write

        def failing_write(path, value):
            if path.name == "launch.json":
                raise OSError("deliberate receipt write failure")
            original_write(path, value)

        monkeypatch.setattr(demo, "write", failing_write)
    try:
        with pytest.raises((RuntimeError, OSError)):
            demo.main()
        assert len(children) == 1 and children[0].poll() is not None
        assert unrelated.poll() is None
        assert json.loads((output / "cleanup.json").read_text())["ownedPid"] == children[0].pid
        if not launch_write_failure:
            receipt = json.loads((output / "launch.json").read_text())
            assert receipt["procStartTicks"] is None and receipt["pid"] == children[0].pid
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


@pytest.mark.unit
def test_procfs_identity_handles_parenthesized_process_names(monkeypatch):
    class Process:
        def __init__(self):
            self.pid = 42
            self.args = ["fixture-game"]

    fields = ["S"] + ["0"] * 18 + ["123456"]
    monkeypatch.setattr(Path, "read_text", lambda *_: "42 (game with ) and spaces) " + " ".join(fields))
    assert demo.process_receipt(Process()) == {
        "pid": 42,
        "command": ["fixture-game"],
        "procStartTicks": "123456",
    }


@pytest.mark.unit
def test_request_text_exposes_actual_tick_edge_and_grid():
    state = {"id": 9, "localOriginTiles": [-5, -4], "localSolidRows": ["?" * 9] * 9}
    request = demo.request_for(state, 15)
    assert len(request["questions"]["action"]["criteria"]) == 6
    text, serialized = request["parts"]["essential"]
    assert "15 native ticks; flip is a single edge at tick 2" in text
    assert "? outside room" in text and "localOriginTiles" in text
    assert json.loads(serialized) == state
