"""Probe/audit adapter for VVVVVV (integrations/_shared/exp0/adapter.py interface).

rerender(state, history): the clean integration request builder on the saved bridge observation. Deterministic;
no network, no game process.

replay(run_dir): native replay, as Griddly v0 did. Starts a fresh game (own Xvfb in :430-439 since 2026-11-26; it
was :350-399 in phase 1), sends the logged executed inputs of every row (model and auto rows) in order, and compares
every resulting bridge observation with the logged one (`extra.after_observation`), then the outcome. No model is
called. Work directories go to $EXP0_REPLAY_WORK_ROOT when set (else work/vvvvvv/replays/ of the repository).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE))
from procs import OwnedXvfb, proc_start_ticks, stop_owned
from run import NativeClient, request_for
from runtime import ASSETS, GAME, SDL_LIB

VOLATILE = ("id",)  # the bridge's command sequence restarts per process; everything else must match
REPLAY_DISPLAYS = range(430, 440)  # inside the VVVVVV v0 collection's :420..:439; collectors use :420..:429


class VVVVVVAdapter:
    game_id = "vvvvvv"

    def rerender(self, state: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
        return request_for(state["observation"], state["ticks"])

    def replay(self, run_dir: Path, work: Path | None = None) -> dict[str, Any]:
        run_dir = Path(run_dir)
        rows = [
            json.loads(l) for l in (run_dir / "decisions.jsonl").read_text().splitlines() if l.strip()  # noqa: E741
        ]
        result = json.loads((run_dir / "result.json").read_text())
        logged_start = json.loads((run_dir / "extra" / "start_observation.json").read_text())
        if work is None:  # outside the collection root, so that no tool mistakes it for an attempt
            import time

            base = (
                Path(os.environ["EXP0_REPLAY_WORK_ROOT"])
                if os.environ.get("EXP0_REPLAY_WORK_ROOT")
                else HERE.parents[3] / "work" / "vvvvvv" / "replays"
            )
            work = base / f"{run_dir.parent.name}-{run_dir.name}-{time.strftime('%Y%m%d%H%M%S')}"
        work = Path(work)
        session = work / "session"
        (session / "user").mkdir(parents=True, exist_ok=False)
        xvfb = OwnedXvfb(work / "xvfb.log", size="800x600x24", displays=REPLAY_DISPLAYS)
        process = None
        try:
            display = xvfb.start()
            env = {
                **os.environ,
                "HONING_SESSION_DIR": str(session),
                "DISPLAY": f":{display}",
                "LD_LIBRARY_PATH": SDL_LIB,
                "SDL_AUDIODRIVER": "dummy",
                "SDL_RENDER_DRIVER": "software",
            }
            env.pop("HONING_CAPTURE", None)
            process = subprocess.Popen(
                [str(GAME), "-assets", str(ASSETS), "-basedir", str(session / "user")],
                env=env,
                stdout=(work / "player.log").open("wb"),
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            native = NativeClient(session, process)
            state = native.observe()
            strip = lambda o: {k: v for k, v in o.items() if k not in VOLATILE}  # noqa: E731
            if strip(state) != strip(logged_start):
                raise ValueError("the replayed start observation differs from the logged one")
            steps = 0
            for row in rows:
                executed = row.get("executed")
                if executed is None:
                    break
                for window in executed["windows"]:
                    state = native.step(state, window["native_action"], window["ticks"])
                    steps += 1
                logged = (row.get("extra") or {}).get("after_observation")
                if logged is None or strip(state) != strip(logged):
                    raise ValueError(f"native state differs from the log after row {row['index']}")
            native.command(state, "stop", 0)
            process.wait(timeout=15)
            if state["tick"] != result["terminal_tick"]:
                raise ValueError("the replayed terminal tick differs from result.json")
            # "method" is the index schema's enum value (contract 0.2.0 lets an adapter name it); the procedure in words
            return {
                "verified": True,
                "status": result["status"],
                "reason": result["reason"],
                "native_steps": steps,
                "method": "native_replay",
                "procedure": "fresh-process native replay of the logged executed inputs; every observation compared",
            }
        finally:
            if process is not None:
                stop_owned(process, proc_start_ticks(process.pid))
            xvfb.stop()


ADAPTER = VVVVVVAdapter()
