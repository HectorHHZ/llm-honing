"""The game replaced by its recording: a stand-in for core.env_keys.KeysEnv that serves the logged observations.

Used to verify everything above the game without a game process (the HK stand-in for Griddly's native replay,
as code/exp0/hk_full/replaylib.py did for the old harness): the harness snapshot runs again with the recorded
choices, and every bridge window it asks for must equal the logged request, window by window.

Sources (written during the run, nothing else is read):
  session/visible_observations.jsonl  every visible observation the bridge wrote, in order (one per bridge window,
                                      the first at the handover)
  session/windows.jsonl               every harness window with the env's frame id / tick, the keys the harness
                                      asked for (`requested`, or the keys themselves in logs without a controller
                                      layer) and the number of adapter windows played with it
  session/bridge_windows.jsonl        every command the game received (for the adapter windows' frames)
The observation the harness got after a window is the bridge's latest one, with the env's own frame id and tick
(`ForestDangoEnv._present` for units=game, kinds=raw, which KeysEnv uses).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path


def rows(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


class ReplayMismatch(ValueError):  # noqa: N818
    pass


class ReplayEnv:
    def __init__(self, session: Path, strict: bool = True):
        session = Path(session)
        self.stream = rows(session / "visible_observations.jsonl")
        self.records = rows(session / "windows.jsonl")
        self.bridge = rows(session / "bridge_windows.jsonl")
        if not self.stream:
            raise ValueError(f"{session} has no visible_observations.jsonl")
        self.strict = strict
        lead = len(self.bridge) - sum(1 + r.get("adapter_windows", 0) for r in self.records)
        if lead < 0:
            raise ValueError("window accounting failed: more harness windows than bridge windows")
        self.p = lead  # bridge windows played so far = index of the latest observation
        self.b = lead  # bridge window index of the next command
        self.k = 0  # next harness window record
        self.auto_frames = sum(int(w["frames"]) for w in self.bridge[:lead])
        self.windows = 0
        self.obs = self._present(self.stream[self.p], 1, 0)
        self.asked: list[dict] = []
        self._tick = 0
        self.key_filter = None  # as core.env_keys.KeysEnv (offline fault tests only: the recording is clean)
        self.window_tag = None  # as core.env_keys.KeysEnv (prefix / absent windows; not used in a replay)
        self.sent: list[dict] = []

    @staticmethod
    def _present(raw: dict, ident: int, tick: int) -> dict:
        obs = copy.deepcopy(raw)
        obs["frame"]["id"], obs["frame"]["tick"] = ident, tick
        return obs

    def step_keys(self, frames: int, hold=(), press=(), first=(), label: str = "", meta: dict | None = None) -> dict:
        if self.k >= len(self.records):
            raise ReplayMismatch(f"the harness asked for window {self.k + 1}, the recording has {len(self.records)}")
        r = self.records[self.k]
        logged = r.get("requested") or {
            "frames": r["frames"],
            "hold": r["hold"],
            "press": r["press"],
            "first": r["first"],
        }
        asked = {"frames": int(frames), "hold": list(hold), "press": list(press), "first": list(first)}
        self.asked.append(asked)
        self._tick = r["tick"]
        if self.key_filter is not None:
            sent, _change = self.key_filter(json.loads(json.dumps(asked)), dict(meta or {}))
            self.sent.append(sent)
        if self.strict and asked != {k: logged[k] for k in ("frames", "hold", "press", "first")}:
            raise ReplayMismatch(
                f"window {self.k + 1} (decision {r.get('decision')}): the harness asked for {asked}, "
                f"the recording has {logged}"
            )
        adapter = int(r.get("adapter_windows", 0))
        for w in self.bridge[self.b : self.b + 1 + adapter]:
            if str(w.get("why", "")).startswith("adapter:"):
                self.auto_frames += int(w["frames"])
        self.b += 1 + adapter
        self.p += 1 + adapter
        if self.p >= len(self.stream):
            raise ReplayMismatch("the recording ends before this window's observation")
        self.obs = self._present(self.stream[self.p], r["obs_id"] + 1, r["tick"] + r["frames"])
        self.k += 1
        self.windows += 1
        return self.obs
