"""Own one Xvfb display + one game process of the full Hollow Knight build (ForestDango/Hollow-Knight-Demo with
the Honing bridge, code/forestdango-hk-demo-work) and speak its file bridge. Same interface as core.env.HKEnv,
so that run_agent.py and the harness versions can drive it (`run_agent.py --game forestdango`).

Exp 0 v0 copy for the new (no-autopilot) harness (code/exp0/hk_v9/core, 2026-11-25, Exp 0 prep subagent of
Honing_CC_Deli). Copied from code/hk-hitl/core/env_forestdango.py (unchanged there; reference copy in
code/exp0/hk_v9/source_ref/core/). Additions only; with the defaults below it behaves as the original:
  * option `display` (a number >= 300): Xvfb is started on exactly that display (`Xvfb :N -displayfd ...`; the
    start fails when the display is taken) instead of the first free one, so this collector only uses its own
    displays (:350-:379); the original's tools/keys_xvfb shim is not needed;
  * constructor argument `bridge_env` ({"HONING_...": value}): extra environment variables for the game process,
    for the knobs of a new build (e.g. a faster, deterministic build); they are applied last and recorded in
    launch.json; names must start with HONING_;
  * every windows.jsonl record has `t_wall` (Unix time when the window was done), for timing.
The game executable stays the `player` option (default: the linked build variant with the bridge), so another
build is dropped in by passing its path.

Added 2026-11-18 (phase 2 of experiments/hk-forestdango-demo); nothing in core/env.py or core/env_hallowdale.py
is changed. The structure follows core/env_hallowdale.py on purpose: the game's bridge mirrors the Hollow Knight
bridge (observation.json in schema honing.visible.v1, action.json, debug_observation.json / observations.jsonl,
frames/), and what differs between the games is absorbed here by explicit options, so that every run says which
difference it compensates:

  protocol  hk      v1-v8 call step(action, sim_frames + 1): in the HK bridge an N-frame window plays N-1
                    frames, and a 1-frame window plays none but takes the button press ("zero-time press",
                    Honing P37). Emulated: N-1 real frames are played; a 1-frame window plays nothing and its
                    button press is put on the first frame of the next window.
            native  step(action, frames) plays exactly `frames` bridge frames.
  view      window  the observation's screen is the game window (taken as 960 x 540; 1 unit = 33.3 px)
            hk      a player-centred cut of the window, magnified so that the knight's body is as wide as in
                    our HK game (35.6 px) with the feet at (480, 413.6) of a 960 x 540 screen: what a harness
                    built for that game's rigid camera expects. Only removes information (13.5 x 7.6 of the
                    window's 28.8 x 16.2 units are left).
            hkwide  the same magnification on a 1440 x 800 screen with the feet at (720, 440) (v8h's screen)
            hkfull  the same magnification on a 2080 x 1160 screen with the feet at (1040, 750): the whole
                    game window fits when the camera sits on the knight (for a harness whose screen-size
                    constants can be set)
  units     game    world units as the bridge gives them: 1 unit = 1 Unity unit of the game (body 0.5 x 1.28)
            hk      1 unit = 54 px of the observation's screen (the scale of our HK game: body 0.66 units wide)
  kinds     raw     object kinds as the bridge gives them: all solid terrain is "wall"
            standable  terrain is "platform": every solid block of this game can be stood on, and v1-v8 plan
                    landings only on "platform" objects
  jump      sticky  the jump key stays down until the knight stops rising, so every jump is a full jump (the
                    HK bridge never sent a jump release); window = the key is let go when the window ends
  flow      auto    while the game is in a segment that no player can act in (room transition, cutscene,
                    death and respawn, respawn after spikes, a scripted sequence) this adapter plays "wait"
                    windows until control is back, the same way for every policy, and logs them as adapter
                    windows. A death is first shown to the harness (one observation with 0 health).
                    off = every such frame is left to the harness.
  text      auto    a dialogue box that blocks the game is advanced with the jump key by this adapter (logged);
                    off = left to the harness
  size      WxH     size of the game window that is drawn (default 960x540; 16:9). The observation does not
                    depend on it: it is computed from collision geometry, so a small window (480x270) gives the
                    same observations faster. Videos are recorded at this size.
  step      0.5     raster step of the bridge's terrain geometry in world units
  seed      42      random seed set when the bridge hands over
  hz        30      bridge frames per second of game time
  player    path    the game executable (default: the linked build variant with the bridge)
  camfix    on      (added 2026-11-19) the bridge's camera lock fix (HONING_CAMERA_LOCK_FIX, see HoningBridge.cs): lock
                    areas of an additively loaded scene (the False Knight arena) get the default bounds their own
                    code never sets, so the camera does not fly to the room's corner; off = the build as it is
  frames    png     (added 2026-11-19) picture format of the captured frames (png | jpg; jpg is ~8x smaller)

Information boundary: this module reads observation.json only. The debug files are written by the game for
EpisodeWatch / evaluation tools and are not opened here.
Processes: one Xvfb (started with -displayfd; its number is read from the pipe) and one game process, both
started here in their own sessions and stopped by their Popen handles only. A failure anywhere in start()
stops both. The Laya service is not touched.
"""

from __future__ import annotations

import copy
import json
import math
import os
import select
import subprocess
import time
from pathlib import Path

from .env import NATIVE_ACTIONS, WORKSPACE, atomic_json, default_ffmpeg

# no absolute local path in the code: the default build and a relative `player` option are resolved against the
# repository root
PLAYER = WORKSPACE / "work" / "hollow-knight-forestdango" / "linux64_bridge" / "HollowKnightDemo.x86_64"
HK_BODY_W = 35.6  # how wide the knight's body is on the screen v1-v8 were built for
HK_PPU = 54.0  # ... and that screen's pixels per world unit
VIEWS = {
    "hk": ((480.0, 413.6), (960, 540)),
    "hkwide": ((720.0, 440.0), (1440, 800)),
    "hkfull": ((1040.0, 750.0), (2080, 1160)),
}  # feet anchor, screen size
BODY_W = 0.5  # this game: width of the knight's body collider in world units
WINDOW_PPU = 540.0 / 16.1966  # this game: pixels per world unit when the window is taken as 960 x 540
# (the camera shows 16.197 units of height; checked against the first observation)
EXTRA_ACTIONS = (
    "jump",
    "attack",
    "up",
    "down",
    "dash",
    "focus",
    "cast",
    "cast_up",
    "cast_down",
    "quick_cast",
    "super_dash",
    "interact",
    "inventory",
)
DEFAULTS = {
    "protocol": "hk",
    "view": "window",
    "units": "game",
    "kinds": "raw",
    "jump": "sticky",
    "flow": "auto",
    "text": "auto",
    "size": "960x540",
    "step": "0.5",
    "seed": "42",
    "hz": "30",
    "player": "",
    "threads": "8",
    "camfix": "on",
    "frames": "png",
    "display": "",
}
# direction and button of a native action (for the zero-time press of the HK protocol)
PRESS = {"jump": "JUMP", "attack": "NAIL", "dash": "DASH", "cast": "CAST", "interact": "UP"}
DIRS = {"left": "LEFT", "right": "RIGHT", "up": "UP", "down": "DOWN"}
AUTO_SEGMENTS = ("transition", "cutscene", "hazard_respawn", "scripted", "loading", "dead")
AUTO_FRAMES = 6  # length of one adapter "wait" window
AUTO_CAP = 600  # at most this many adapter windows in a row (120 s of game time)


class ForestDangoEnv:
    """One game session. `obs` is always the latest visible observation as the harness sees it."""

    def __init__(
        self,
        out_dir: Path,
        scene: str = "new",
        capture: bool = True,
        options: dict | None = None,
        model_timeout: float = 600.0,
        observation_timeout: float = 300.0,
        bridge_env: dict | None = None,
    ):
        self.bridge_env = {str(k): str(v) for k, v in (bridge_env or {}).items()}  # Exp 0: knobs of a new build
        if any(not k.startswith("HONING_") or k == "HONING_SESSION_DIR" for k in self.bridge_env):
            raise ValueError("extra bridge environment variables must start with HONING_ (and not move the session)")
        self.out = Path(out_dir).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.session = self.out / "session"
        self.session.mkdir(exist_ok=False)
        self.scene = scene
        self.capture = capture
        self.opt = {**DEFAULTS, **(options or {})}
        unknown = set(self.opt) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown forestdango option(s): {sorted(unknown)}")
        if not (math.isfinite(model_timeout) and 0 < model_timeout <= 3570):
            raise ValueError("model_timeout must be within (0, 3570] seconds")
        self.model_timeout = model_timeout
        self.observation_timeout = observation_timeout
        self.display = None
        self.game = None
        self.obs = None  # what the harness sees (transformed, with the harness-side frame ids)
        self.raw = None  # the bridge's latest observation
        self.windows = 0
        self.frames = 0  # frames as the harness counts them (HK protocol: sim_frames + 1 per window)
        self.real_frames = 0  # bridge frames really played in harness windows
        self.auto_frames = 0  # bridge frames played by this adapter (flow / text)
        self.auto_log: list[dict] = []
        self.zoom = 1.0
        self.unit = 1.0
        self._pending = None  # zero-time press waiting for the next window (HK protocol)
        self._death_shown = False
        self._id = 0
        self._tick = 0
        self._xvfb_log = None
        self._game_log = None

    # ---- lifecycle -------------------------------------------------------------------------
    def start(self) -> dict:
        try:
            return self._start()
        except BaseException:
            self.close()  # owned processes only; never leaves a display or a game behind
            raise

    def _start(self) -> dict:
        opt = self.opt
        width, height = (int(v) for v in opt["size"].split("x"))
        if width * 9 != height * 16:
            raise ValueError("size must be 16:9 (the observation takes the camera's rectangle as 960 x 540)")
        player = Path(opt["player"]) if opt["player"] else PLAYER
        if not player.is_absolute():
            player = WORKSPACE / player
        if not player.exists():
            raise FileNotFoundError(player)
        fixed = []  # Exp 0: an explicit display number (>= 300) instead of the first free one
        if str(opt.get("display", "")).strip():
            if int(opt["display"]) < 300:
                raise ValueError("Exp 0 copy: displays below :300 are not ours")
            fixed = [f":{int(opt['display'])}"]
        read_fd, write_fd = os.pipe()
        self._xvfb_log = (self.out / "xvfb.log").open("w")
        try:
            self.display = subprocess.Popen(
                [
                    "Xvfb",
                    *fixed,
                    "-displayfd",
                    str(write_fd),
                    "-screen",
                    "0",
                    f"{width}x{height}x24",
                    "-nolisten",
                    "tcp",
                ],
                pass_fds=[write_fd],
                stdout=self._xvfb_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            os.close(write_fd)
        try:
            # the display is ready when Xvfb has written its number and a newline (not before)
            text = b""
            deadline = time.monotonic() + 30
            while not text.endswith(b"\n"):
                left = deadline - time.monotonic()
                if left <= 0 or self.display.poll() is not None or not select.select([read_fd], [], [], left)[0]:
                    raise RuntimeError("Xvfb did not report a display")
                chunk = os.read(read_fd, 64)
                if not chunk:
                    raise RuntimeError("Xvfb closed its display pipe")
                text += chunk
            number = int(text.decode().strip())
        finally:
            os.close(read_fd)
        home = self.out / "home"
        home.mkdir(exist_ok=True)
        env = {
            **os.environ,
            "DISPLAY": f":{number}",
            "LIBGL_ALWAYS_SOFTWARE": "1",
            "LP_NUM_THREADS": str(opt["threads"]),
            "HOME": str(home),
            "XDG_CONFIG_HOME": str(home / ".config"),
            "XDG_DATA_HOME": str(home / ".local/share"),
            "XDG_CACHE_HOME": str(home / ".cache"),
            "SDL_AUDIODRIVER": "dummy",
            "HONING_SESSION_DIR": str(self.session),
            "HONING_CAPTURE": "1" if self.capture else "0",
            "HONING_SCENE": self.scene,
            # the game must outwait the model: its idle limit is the model's timeout plus a margin
            "HONING_MODEL_TIMEOUT_SECONDS": str(int(math.ceil(self.model_timeout)) + 30),  # noqa: RUF046
            "HONING_OBSERVATION_PROFILE": "visible",
            "HONING_JUMP_MODE": opt["jump"],
            "HONING_SEED": str(opt["seed"]),
            "HONING_FRAME_HZ": str(opt["hz"]),
            "HONING_GEOM_STEP": str(opt["step"]),
        }
        if opt["camfix"] not in ("on", "off") or opt["frames"] not in ("png", "jpg"):
            raise ValueError("camfix must be on or off, frames png or jpg")
        env["HONING_CAMERA_LOCK_FIX"] = "1" if opt["camfix"] == "on" else "0"
        env["HONING_CAPTURE_FORMAT"] = opt["frames"]
        for name in (
            "HONING_DEBUG_DIR",
            "HONING_CAPTURE_DIR",
            "HONING_FIXED_FPS",
            "HONING_VIEW_ANCHOR",
            "HONING_VIEW_ZOOM",
            "HONING_VIEW_SIZE",
        ):
            env.pop(name, None)
        if opt["view"] in VIEWS:
            feet, size = VIEWS[opt["view"]]
            self.zoom = HK_BODY_W / (BODY_W * WINDOW_PPU)
            env.update(
                {
                    "HONING_VIEW_ANCHOR": f"feet:{feet[0]},{feet[1]}",
                    "HONING_VIEW_ZOOM": repr(self.zoom),
                    "HONING_VIEW_SIZE": f"{size[0]}x{size[1]}",
                }
            )
        elif opt["view"] != "window":
            raise ValueError("view must be window, hk, hkwide or hkfull")
        env.update(self.bridge_env)  # Exp 0: knobs of a new build, applied last (recorded in launch.json)
        # world units of the game per harness unit
        self.unit = (HK_PPU / (WINDOW_PPU * self.zoom)) if opt["units"] == "hk" else 1.0
        if opt["units"] not in ("hk", "game"):
            raise ValueError("units must be game or hk")
        command = [
            str(player),
            "-screen-width",
            str(width),
            "-screen-height",
            str(height),
            "-screen-fullscreen",
            "0",
            "-force-glcore",
            "-logFile",
            str(self.session / "player.log"),
        ]
        self._game_log = (self.session / "stdout.log").open("w")
        self.game = subprocess.Popen(
            command,
            env=env,
            cwd=str(player.parent),
            stdout=self._game_log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        atomic_json(
            self.out / "launch.json",
            {
                "game": "forestdango",
                "display": f":{number}",
                "xvfb_pid": self.display.pid,
                "game_pid": self.game.pid,
                "command": command,
                "scene": self.scene,
                "capture": self.capture,
                "options": self.opt,
                "window": [width, height],
                "zoom": self.zoom,
                "game_units_per_unit": self.unit,
                "bridge_env": {k: v for k, v in env.items() if k.startswith("HONING_")},
            },
        )
        self.raw = self._wait_raw(0)
        ppu = 1.0 / self.raw["view"]["world_from_screen"][0]
        if abs(ppu - WINDOW_PPU * self.zoom) > 0.02 * ppu:
            raise RuntimeError(
                f"the bridge's scale ({ppu:.3f} px per unit) is not the one this adapter assumes "
                f"({WINDOW_PPU * self.zoom:.3f}): the game's camera differs from the measured one"
            )
        self._advance_flow("boot")
        self._id, self._tick = 1, 0
        self.obs = self._present(self.raw)
        return self.obs

    def _wait_raw(self, previous: int) -> dict:
        deadline = time.monotonic() + self.observation_timeout
        path = self.session / "observation.json"
        while time.monotonic() < deadline:
            if self.game.poll() is not None:
                raise RuntimeError(
                    f"game exited with code {self.game.returncode}; see {self.session / 'player.log'} "
                    f"and {self.session / 'bridge.log'}"
                )
            try:
                obs = json.loads(path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                time.sleep(0.003)
                continue
            if obs.get("schema") != "honing.visible.v1":
                raise ValueError("expected the visible observation profile")
            if obs["frame"]["id"] > previous:
                return obs
            time.sleep(0.003)
        raise TimeoutError("no new visible observation; see player.log / bridge.log")

    # ---- the bridge window -----------------------------------------------------------------
    def _send(self, command: dict, why: str) -> dict:
        """One real window on the bridge; returns the bridge's next observation."""
        raw = self.raw
        rid, rtick = raw["frame"]["id"], raw["frame"]["tick"]
        command = {"id": rid, "model": "harness", "probability": 0.0, **command}
        atomic_json(self.session / "action.json", command)
        nxt = self._wait_raw(rid)
        if nxt["frame"]["id"] != rid + 1 or nxt["frame"]["tick"] != rtick + command["frames"]:
            raise ValueError(
                f"bridge did not advance exactly one window: {rid}->{nxt['frame']['id']}, "
                f"tick {rtick}->{nxt['frame']['tick']} (frames={command['frames']})"
            )
        self.raw = nxt
        with (self.session / "bridge_windows.jsonl").open("a") as log:
            log.write(
                json.dumps(
                    {
                        "bridge_id": rid,
                        "bridge_tick": rtick,
                        "why": why,
                        **{k: v for k, v in command.items() if k not in ("id", "model", "probability")},
                    }
                )
                + "\n"
            )
        return nxt

    def _segment(self) -> tuple[str, bool]:
        flow = self.raw.get("flow") or {}
        return flow.get("segment", "play"), bool(flow.get("can_act", True))

    def _health(self):
        health = (self.raw.get("hud") or {}).get("health")
        return health["value"] if isinstance(health, dict) else None

    def _advance_flow(self, why: str) -> int:
        """Options flow=auto / text=auto: play "wait" windows while no player could act (see the module text).
        Decided from the visible observation's `flow` and `hud.text`. Returns the number of adapter windows."""
        count = 0
        entry = None
        while count < AUTO_CAP:
            segment, can_act = self._segment()
            blocking = any(t.get("blocking") for t in (self.raw.get("hud") or {}).get("text", []))
            if blocking and self.opt["text"] == "auto":
                self._send({"action": "jump", "frames": 1}, f"adapter: advance text ({why})")
                self._send({"action": "wait", "frames": 8}, f"adapter: advance text ({why})")
                self.auto_frames += 9
                count += 2
                if entry is None or entry["segment"] != "dialogue":
                    entry = {
                        "segment": "dialogue",
                        "why": why,
                        "bridge_tick": self.raw["frame"]["tick"],
                        "windows": 0,
                        "frames": 0,
                    }
                    self.auto_log.append(entry)
                entry["windows"] += 2
                entry["frames"] += 9
                continue
            if can_act or segment not in AUTO_SEGMENTS or self.opt["flow"] != "auto":
                break
            if segment == "dead" and not self._death_shown:
                break  # the harness has to see this death first (0 health on the HUD)
            self._send({"action": "wait", "frames": AUTO_FRAMES}, f"adapter: auto-advance {segment} ({why})")
            self.auto_frames += AUTO_FRAMES
            count += 1
            if entry is None or entry["segment"] != segment:
                entry = {
                    "segment": segment,
                    "why": why,
                    "bridge_tick": self.raw["frame"]["tick"],
                    "windows": 0,
                    "frames": 0,
                }
                self.auto_log.append(entry)
            entry["windows"] += 1
            entry["frames"] += AUTO_FRAMES
        if self._segment()[0] != "dead" and (self._health() or 0) > 0:
            self._death_shown = False
        return count

    def _present(self, raw: dict) -> dict:
        """The bridge observation as the harness gets it: frame ids of this adapter, units, kinds."""
        obs = copy.deepcopy(raw)
        obs["frame"]["id"], obs["frame"]["tick"] = self._id, self._tick
        if self.unit != 1.0:
            obs["view"]["world_from_screen"] = [v / self.unit for v in obs["view"]["world_from_screen"]]
            obs["view"]["world_units"] = (
                f"1 unit = {self.unit:.4f} world units of the game (= 54 px of this screen, the scale "
                "of the Hollow Knight imitation the harness was built on); x right, y up"
            )
        if self.opt["kinds"] == "standable":
            for o in obs["objects"]:
                if o["kind"] == "wall":
                    o["kind"] = "platform"
        elif self.opt["kinds"] != "raw":
            raise ValueError("kinds must be raw or standable")
        if raw.get("flow", {}).get("segment") == "dead":
            self._death_shown = True
        return obs

    def step(
        self, action: str, frames: int, label: str = "", probability: float = 0.0, meta: dict | None = None
    ) -> dict:
        """Send one native window and return the next visible observation."""
        if action not in NATIVE_ACTIONS and action not in EXTRA_ACTIONS:
            raise ValueError(f"not a native action: {action}")
        if not 1 <= int(frames) <= 120:
            raise ValueError("frames must be within 1..120")
        frames = int(frames)
        pid, ptick = self._id, self._tick
        record = {"obs_id": pid, "tick": ptick, "action": action, "frames": frames, "label": label}
        before = self._advance_flow(label)  # e.g. the death the harness has just been shown: play it out first
        head, _, tail = action.partition("_")
        if self.opt["protocol"] == "hk":
            real = frames - 1
            if real == 0:
                # zero-time press: nothing is played; the press goes onto the first frame of the next window
                if head in PRESS:
                    self._pending = (PRESS[head], DIRS.get(tail))
                record["real_frames"] = 0
            else:
                command = {"action": action, "frames": real}
                if self._pending is not None:
                    button, direction = self._pending
                    self._pending = None
                    command["press"] = [button]
                    if button != "JUMP" and direction is not None and head not in PRESS:
                        # the strike takes its direction / facing in the frame of the press
                        command["first"] = [direction]
                self._send(command, label)
                self.real_frames += real
                record["real_frames"] = real
        elif self.opt["protocol"] == "native":
            self._send({"action": action, "frames": frames}, label)
            self.real_frames += frames
            record["real_frames"] = frames
        else:
            raise ValueError("protocol must be hk or native")
        after = self._advance_flow(label) if record["real_frames"] else 0
        if before or after:
            record["adapter_windows"] = before + after
        self._id, self._tick = pid + 1, ptick + frames
        self.obs = self._present(self.raw)
        self.windows += 1
        self.frames += frames
        if meta:
            record.update(meta)
        record["t_wall"] = round(time.time(), 4)  # Exp 0: timing
        with (self.session / "windows.jsonl").open("a") as log:
            log.write(json.dumps(record) + "\n")
        return self.obs

    def close(self) -> dict:
        info = {}
        game, display = self.game, self.display
        if game is not None:
            if game.poll() is None and self.raw is not None:
                try:
                    atomic_json(
                        self.session / "action.json", {"id": self.raw["frame"]["id"], "action": "stop", "frames": 1}
                    )
                    game.wait(timeout=20)
                except Exception:
                    pass
            if game.poll() is None:
                game.terminate()
                try:
                    game.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    game.kill()
                    game.wait(timeout=10)
            info["game_exit"] = game.returncode
        if display is not None:
            if display.poll() is None:
                display.terminate()
                try:
                    display.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    display.kill()
                    display.wait(timeout=5)
            info["display_exit"] = display.returncode
        for handle in (self._xvfb_log, self._game_log):
            if handle is not None:
                handle.close()
        self._xvfb_log = self._game_log = None
        info.update(
            {
                "windows": self.windows,
                "frames": self.frames,
                "real_frames": self.real_frames,
                "auto_frames": self.auto_frames,
                "auto_segments": self.auto_log,
            }
        )
        try:  # noqa: SIM105
            atomic_json(self.out / "close.json", info)
        except OSError:
            pass
        self.game = self.display = None
        return info

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()


def encode_video(
    session: Path, output: Path, start: int = 0, count: int | None = None, fps: float = 30, ffmpeg: str | None = None
) -> None:
    """Frames -> mp4, like core.env.encode_video. Every bridge frame is in it: the harness's windows and the
    adapter's auto-advanced ones; the time the model thinks is not (the game is stopped then)."""
    ffmpeg = ffmpeg or default_ffmpeg()
    frames = session / "frames"
    pattern = "%06d.png" if any(frames.glob("*.png")) else "%06d.jpg"
    args = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-framerate",
        str(fps),
        "-start_number",
        str(start),
        "-i",
        str(frames / pattern),
    ]
    if count is not None:
        args += ["-frames:v", str(count)]
    args += [
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(args, check=True)
