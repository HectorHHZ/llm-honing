"""The full Hollow Knight build driven by raw keys (v9_keys, 2026-11-25; a new file, nothing existing is changed).

`KeysEnv` is core.env_forestdango.ForestDangoEnv plus one method, `step_keys`: a bridge window given as the keys
themselves (the bridge's "keys" action with `hold` / `press` / `first`, see HoningBridge.cs) instead of one of the
named native actions. Everything else is the parent's: the adapter's automatic play of segments in which no
player can act (flow=auto, text=auto), its frame ids, its logs. `step_keys` does the same bookkeeping as the
parent's `step` with protocol=native, and writes the same `windows.jsonl` rows with the keys in them.

    hold   keys that are down in every frame of the window
    press  a fresh press in the first frame, held one frame (JUMP: held to the end of the window; with the bridge
           option jump=window it is let go when the window ends unless the next window holds it)
    first  direction keys that are down in the first frame instead of the directions of `hold`

Exp 0 v0 copy (code/exp0/hk_v9/core, 2026-11-25, Exp 0 prep subagent of Honing_CC_Deli), copied from
code/hk-hitl/core/env_keys.py (unchanged there; reference copy in code/exp0/hk_v9/source_ref/core/). Changes:
  * the display: the parent's explicit `display` option (Xvfb :N, N >= 300) replaces the tools/keys_xvfb shim;
  * the controller layer of Exp 0 (`key_filter`): when the collector sets a filter (only for a controller-fault
    attempt, after the model's selection), every window the harness asks for is passed through it before it is
    sent; the filter returns the window that is really sent and, when it differs, a change record. The harness
    never sees the filter. windows.jsonl keeps both: `requested` (what the harness asked for) and the keys and
    frames actually sent (`hold`, `press`, `first`, `frames`), plus `controller_fault` with the reason. The native
    clock (tick) advances by the frames really sent;
  * `t_wall` (Unix time when the window was done) in every windows.jsonl record;
  * (2026-11-26, Exp 0 v0 HK collection) `window_tag`: when the collector sets it (a dict), it is merged into every
    windows.jsonl record after the harness's own meta, so that the windows of a scripted start prefix
    ({"prefix": k, "decision": None}) and the harness's no-key windows while the knight is not on screen
    ({"absent_before": k}) can be told apart from the windows of a model decision. Nothing is sent differently.
Without a filter the behaviour and the logs are those of the original (plus `t_wall` and `requested`).
"""

from __future__ import annotations

import json
import time

from .env_forestdango import ForestDangoEnv

KEYS = ("LEFT", "RIGHT", "UP", "DOWN", "JUMP", "NAIL", "DASH", "CAST", "QUICK_CAST", "SUPER_DASH", "INVENTORY")


class KeysEnv(ForestDangoEnv):
    def __init__(
        self,
        out_dir,
        scene: str = "new",
        capture: bool = False,
        options: dict | None = None,
        display: int | None = None,
        bridge_env: dict | None = None,
        **kw,
    ):
        options = {
            "protocol": "native",
            "view": "window",
            "units": "game",
            "kinds": "raw",
            "jump": "window",
            **(options or {}),
        }
        if display is not None:
            options["display"] = str(int(display))
        super().__init__(out_dir, scene=scene, capture=capture, options=options, bridge_env=bridge_env, **kw)
        self.display_number = display
        self.key_filter = None  # Exp 0 controller layer: callable(window, meta) -> (sent_window, change | None)
        self.filter_changes = 0
        self.window_tag = None  # Exp 0: extra fields for the windows.jsonl records (prefix / absent windows)

    def step_keys(self, frames: int, hold=(), press=(), first=(), label: str = "", meta: dict | None = None) -> dict:
        """One bridge window of exactly `frames` frames with these keys; returns the next visible observation."""
        frames = int(frames)
        if not 1 <= frames <= 120:
            raise ValueError("frames must be within 1..120")
        for k in (*hold, *press, *first):
            if k not in KEYS:
                raise ValueError(f"not a key: {k}")
        requested = {"frames": frames, "hold": list(hold), "press": list(press), "first": list(first)}
        sent, change = dict(requested), None
        if self.key_filter is not None:
            sent, change = self.key_filter(json.loads(json.dumps(requested)), dict(meta or {}))
            if not 1 <= int(sent["frames"]) <= 120 or any(
                k not in KEYS for k in (*sent["hold"], *sent["press"], *sent["first"])
            ):
                raise ValueError(f"the controller filter produced an invalid window: {sent}")
            if change is not None:
                self.filter_changes += 1
        frames, hold, press, first = int(sent["frames"]), sent["hold"], sent["press"], sent["first"]
        pid, ptick = self._id, self._tick
        record = {
            "obs_id": pid,
            "tick": ptick,
            "action": "keys",
            "hold": list(hold),
            "press": list(press),
            "first": list(first),
            "frames": frames,
            "label": label,
            "requested": requested,
        }
        if change is not None:
            record["controller_fault"] = change
        before = self._advance_flow(label)
        command = {"action": "keys", "frames": frames, "hold": list(hold), "press": list(press)}
        if first:
            command["first"] = list(first)
        self._send(command, label)
        self.real_frames += frames
        record["real_frames"] = frames
        after = self._advance_flow(label)
        if before or after:
            record["adapter_windows"] = before + after
        self._id, self._tick = pid + 1, ptick + frames
        self.obs = self._present(self.raw)
        self.windows += 1
        self.frames += frames
        if meta:
            record.update(meta)
        if self.window_tag:
            record.update(self.window_tag)
        record["t_wall"] = round(time.time(), 4)  # Exp 0: timing
        with (self.session / "windows.jsonl").open("a") as log:
            log.write(json.dumps(record) + "\n")
        return self.obs
