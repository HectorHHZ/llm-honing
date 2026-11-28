"""Draw the frames of one logged VVVVVV attempt by replaying its executed inputs natively (presentation only).

  python3 -B capture_replay.py RUN_DIR FRAMES_DIR

A fresh game process (own Xvfb in :430-:439) with the bridge's frame capture on (`HONING_CAPTURE=1`) receives only
the logged executed inputs of every row (model and auto rows), as adapter.replay does; every resulting bridge
observation is compared with the logged one, so the frames show exactly the logged episode. No model is called.
The frames (BMP, one per native tick) go to FRAMES_DIR; they are for the local videos only (delete them after).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from adapter import ASSETS, GAME, REPLAY_DISPLAYS, SDL_LIB, VOLATILE
from procs import OwnedXvfb, proc_start_ticks, stop_owned
from run import NativeClient


def capture(run_dir: Path, frames: Path) -> dict:
    rows = [json.loads(line) for line in (run_dir / "decisions.jsonl").read_text().splitlines() if line.strip()]
    result = json.loads((run_dir / "result.json").read_text())
    work = frames.parent / f"{frames.name}-session"
    session = work / "session"
    (session / "user").mkdir(parents=True, exist_ok=False)
    (session / "frames").mkdir()
    xvfb = OwnedXvfb(work / "xvfb.log", size="800x600x24", displays=REPLAY_DISPLAYS)
    process = None
    strip = lambda o: {k: v for k, v in o.items() if k not in VOLATILE}  # noqa: E731
    try:
        display = xvfb.start()
        env = {
            **os.environ,
            "HONING_SESSION_DIR": str(session),
            "DISPLAY": f":{display}",
            "LD_LIBRARY_PATH": SDL_LIB,
            "SDL_AUDIODRIVER": "dummy",
            "SDL_RENDER_DRIVER": "software",
            "HONING_CAPTURE": "1",
        }
        process = subprocess.Popen(
            [str(GAME), "-assets", str(ASSETS), "-basedir", str(session / "user")],
            env=env,
            stdout=(work / "player.log").open("wb"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        native = NativeClient(session, process)
        state = native.observe()
        steps = 0
        for row in rows:
            for window in (row.get("executed") or {}).get("windows", []):
                state = native.step(state, window["native_action"], window["ticks"])
                steps += 1
            logged = (row.get("extra") or {}).get("after_observation")
            if logged is None or strip(state) != strip(logged):
                raise ValueError(f"native state differs from the log after row {row['index']}")
        native.command(state, "stop", 0)
        process.wait(timeout=15)
        if state["tick"] != result["terminal_tick"]:
            raise ValueError("the replayed terminal tick differs from result.json")
    finally:
        if process is not None:
            stop_owned(process, proc_start_ticks(process.pid))
        xvfb.stop()
    shutil.move(str(session / "frames"), str(frames))
    shutil.rmtree(work)
    return {
        "run": run_dir.name,
        "native_windows": steps,
        "frames": len(list(frames.iterdir())),
        "observations_match": True,
    }


if __name__ == "__main__":
    print(json.dumps(capture(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())))
