"""Compile production bridge parsing/masks; substitute engine and renderer callbacks.

HONING_HURRICAN_SOURCE also executes the real patched native priority branch.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

HERE = Path(__file__).parents[1] / "integrations/hurrican"


def load_run():
    spec = importlib.util.spec_from_file_location("hurrican_controls_run", HERE / "run.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.functional
def test_compiled_production_parser_and_jump_masks(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler required")
    for name in ("HoningBridge.cpp", "HoningBridge.hpp"):
        shutil.copyfile(HERE / name, tmp_path / name)
    # Only the game's types/callbacks are fixtures: production bridge is included unchanged.
    stubs = r"""
#pragma once
#include <bitset>
#include <memory>
#include <vector>
enum { AKTION_LINKS, AKTION_RECHTS, AKTION_JUMP, AKTION_SHOOT, AKTION_DUCKEN, MAX_AKTIONEN };
struct SDL_Window {};
struct P {
    int xpos=0,ypos=0,xspeed=0,yspeed=0,Energy=100,Lives=3,ShotDelay=0,AutoFireCount=0;
    int Score=0,Handlung=0,Blickrichtung=1;
    bool JumpPossible=true;
    std::bitset<MAX_AKTIONEN> Aktion;
};
static P Player[1];
namespace Direction { static int asInt(int d) { return d; } }
struct Enemy {
    bool Active=true,Destroyable=true; int xPos=0,yPos=0,Energy=0,GegnerArt=0;
    bool IsOnScreen() const { return true; }
};
using GegnerClass=Enemy;
static struct { std::vector<std::shared_ptr<Enemy>> enemies; } Gegner;
struct Shot { P *pParent; int ShotArt; };
static struct { std::vector<Shot*> projectiles; } Projectiles;
enum { SPREADSHOT=0, LASERSHOTBIG2=15, SKILL_MEDIUM=1 };
static int Stage,NewStage,NUMPLAYERS,Skill,SpielZustand;
static bool RunningTutorial;
static struct { int Timelimit=100,XOffset=0,YOffset=0; } TileEngine;
static struct { void update() {} void setSpeedFactor(float) {} float GetMoveSpeed() { return 10; } } Timer;
static bool GameInit2() { return true; }
static void InitNewGame() {} static void InitNewGameLevel() {} static void Heartbeat() {}
static void SDL_GL_GetDrawableSize(SDL_Window*, int *w,int *h) { *w=*h=0; }
enum { GL_PACK_ALIGNMENT,GL_RGB,GL_UNSIGNED_BYTE };
static void glPixelStorei(int,int) {} static void glReadPixels(int,int,int,int,int,int,void*) {}
"""
    (tmp_path / "Engine.hpp").write_text(stubs)
    for name in (
        "Player.hpp",
        "Gameplay.hpp",
        "Main.hpp",
        "Timer.hpp",
        "Tileengine.hpp",
        "GegnerClass.hpp",
        "Gegner_Helper.hpp",
        "Projectiles.hpp",
        "opengl.hpp",
    ):
        (tmp_path / name).write_text('#include "Engine.hpp"\n')
    (tmp_path / "controls.cpp").write_text(
        '#include "HoningBridge.cpp"\n'
        'int main() { setenv("HONING_HURRICAN_SESSION", ".", 1);\n'
        "initialized=true; while(HoningBeforeTick()) {\n"
        'Player[0].Aktion=HoningActions(); std::cout << "MASK " << Player[0].Aktion.to_ulong() << "\\n";\n'
        "HoningAfterTick(); } }\n"
    )
    binary = tmp_path / "controls"
    compile_result = subprocess.run(
        [compiler, "-std=c++17", str(tmp_path / "controls.cpp"), "-o", str(binary)],
        capture_output=True,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stdout + compile_result.stderr
    result = subprocess.run(
        [str(binary)],
        input="step 0 3 7\nstep 1 3 8\nstep 2 1 6\nstep 3 1 6\nstep 3 2 6\nstep 4 0 6\nstop 4\n",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    masks = [int(line[5:]) for line in result.stdout.splitlines() if line.startswith("MASK ")]
    assert masks == [13, 13, 13, 10, 14, 14, 0, 4]
    observations = [json.loads(line[7:]) for line in result.stdout.splitlines() if line.startswith("HONING ")]
    assert [(obs["id"], obs["tick"]) for obs in observations] == [
        (1, 3),
        (2, 6),
        (3, 7),
        (4, 8),
        (4, 8),
        (4, 8),
    ]
    assert observations[-2]["error"] == "stale or malformed command"
    assert observations[-1]["error"] == "invalid step"


@pytest.mark.unit
def test_original_forced_action_priority(tmp_path: Path) -> None:
    source = os.environ.get("HONING_HURRICAN_SOURCE")
    compiler = shutil.which("c++")
    if not source or compiler is None:
        pytest.skip("set HONING_HURRICAN_SOURCE to the pinned source with current patch; C++ compiler required")
    revision = subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD"], text=True).strip()
    assert revision == "0d91d1f2d40f38d21e7f41afc01369ee55649fb6"
    native = (Path(source) / "Hurrican/src/Player.cpp").read_text()
    # Execute the actual native branch, including the hook's placement, without copying its logic.
    start = native.index("if (DEMOPlaying)", native.index("bool PlayerClass::GetPlayerInput()"))
    end = native.index("// oder Spieler spielt ?", start)
    branch = native[start:end] + "else Aktion[0]=true;"
    harness = (
        r"""
#include <bitset>
#include <cassert>
static bool DEMOPlaying,DoFesteAktion;
static int FesteAktion,modelCalls;
static std::bitset<4> Aktion;
static bool HoningEnabled() { return true; }
static std::bitset<4> HoningActions() { modelCalls++; return std::bitset<4>(8); }
static void PlayDemo() { Aktion[1]=true; }
static void input() { Aktion.reset();
"""
        + branch
        + r"""
}
int main() {
    DoFesteAktion=true; FesteAktion=2; input();
    assert(Aktion.to_ulong()==4 && modelCalls==0); /* end-level walkout */
    FesteAktion=-1; input(); assert(Aktion.none() && modelCalls==0); /* dragon ride */
    DoFesteAktion=false; input(); assert(Aktion.to_ulong()==8 && modelCalls==1);
    DEMOPlaying=true; input(); assert(Aktion.to_ulong()==2 && modelCalls==1);
}
"""
    )
    path = tmp_path / "priority.cpp"
    path.write_text(harness)
    binary = tmp_path / "priority"
    result = subprocess.run(
        [compiler, "-std=c++17", str(path), "-o", str(binary)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    subprocess.run([str(binary)], check=True)


@pytest.mark.unit
@pytest.mark.parametrize("timeout", [300, 0.1, 3570])
def test_native_timeout_margin_overrides_inherited_setting(tmp_path: Path, timeout: float) -> None:
    run = load_run()
    process = MagicMock()
    process.pid, process.args = 123, ["fixture-game"]
    with (
        patch.object(run.subprocess, "Popen", return_value=process) as popen,
        patch.object(run.threading, "Thread"),
        patch.dict(run.os.environ, {"HONING_HURRICAN_IDLE_SECONDS": "1"}),
    ):
        native = run.NativeClient(Path("game"), Path("data"), tmp_path, timeout=timeout)
    assert native.timeout == timeout
    assert popen.call_args.kwargs["env"]["HONING_HURRICAN_IDLE_SECONDS"] == str(run.math.ceil(timeout) + 30)
    native.log.close()


@pytest.mark.unit
@pytest.mark.parametrize("timeout", [True, 0, -1, float("inf"), float("nan"), 3570.1, "300"])
def test_invalid_native_timeout_has_no_process_or_output(tmp_path: Path, timeout) -> None:
    run = load_run()
    with patch.object(run.subprocess, "Popen") as popen, pytest.raises(ValueError):
        run.NativeClient(Path("game"), Path("data"), tmp_path, timeout=timeout)
    assert not popen.called and list(tmp_path.iterdir()) == []


@pytest.mark.unit
def test_rendered_request_describes_counted_release() -> None:
    request = load_run().request_for({"id": 0, "tick": 0}, 1)
    text = request["parts"]["essential"][0]
    assert "1 fixed60Hz native updates" in text and "releases jump on its first update" in text


@pytest.mark.parametrize("failure", ["launch_receipt", "thread_start", "popen"])
@pytest.mark.functional
def test_constructor_failure_reaps_only_owned_child_and_closes_log(tmp_path: Path, failure: str) -> None:
    run = load_run()
    game = tmp_path / "fixture-game"
    game.write_text(f"#!{sys.executable}\nimport sys\nfor line in sys.stdin: pass\n")
    game.chmod(0o755)
    session = tmp_path / "caller-session"
    session.mkdir()
    sentinel = session / "caller-owned.txt"
    sentinel.write_bytes(b"unmodified caller data\x00")
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    native = run.NativeClient.__new__(run.NativeClient)
    if failure == "launch_receipt":
        injection = patch.object(run, "write_json", side_effect=OSError("injected receipt storage failure"))
    elif failure == "thread_start":
        injection = patch.object(run.threading.Thread, "start", side_effect=RuntimeError("injected thread failure"))
    else:
        injection = patch.object(run.subprocess, "Popen", side_effect=OSError("injected spawn failure"))
    try:
        with injection, pytest.raises((OSError, RuntimeError), match="injected"):
            native.__init__(game, tmp_path, session)
        assert native.log.closed and not (session / "cleanup.json").exists()
        assert sentinel.read_bytes() == b"unmodified caller data\x00"
        assert unrelated.poll() is None
        if failure != "popen":
            assert native.process.poll() is not None
            assert native.process.stdin.closed and native.process.stdout.closed
        else:
            assert not hasattr(native, "process")
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)
