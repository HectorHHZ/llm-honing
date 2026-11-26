# fmt: off
# isort: skip_file
# ruff: noqa
"""Own one Xvfb display + one Hollow Knight game process and speak the file bridge.

Speaks the same file protocol as integrations/hollow-knight/run.py, with these differences:
  * the harness side only ever reads `observation.json` (schema honing.visible.v1, what a human sees);
    `debug_observation.json` is never opened by this module. The game itself appends the debug state of
    every window to `observations.jsonl`; only tools/progress.py (evaluation) and EpisodeWatch read it.
  * every native window (action, frames, label) is appended to `windows.jsonl`.
Processes: this class starts its own Xvfb and its own game process (each in a new session) and stops only
those two, by the Popen handles it holds. It never touches the Laya service or any other display.
A start that fails at any point stops what it had already started before the error reaches the caller.

The player comes from the environment: HONING_HK_GAME names the Linux build (see the experiment README).
"""
from __future__ import annotations

import json
import os
import select
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
XVFB_READY_SECONDS = 20.0
NATIVE_ACTIONS = ("wait", "left", "right", "jump_left", "jump_right", "dash_left", "dash_right",
                  "attack_left", "attack_right", "attack_up", "attack_down")


def game_binary() -> Path:
    """The built player to launch; the operator names it because game builds are not part of this repository."""
    value = os.environ.get("HONING_HK_GAME")
    if not value:
        raise RuntimeError("Set HONING_HK_GAME to the Linux player built for this experiment (see README.md)")
    return Path(value)


def read_display_number(fd: int, timeout_seconds: float = XVFB_READY_SECONDS) -> str:
    """Wait for the complete line Xvfb writes to -displayfd once it accepts connections."""
    deadline = time.monotonic() + timeout_seconds
    line = b""
    while not line.endswith(b"\n"):
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
            raise TimeoutError("Xvfb launch timeout")
        chunk = os.read(fd, 100)
        if not chunk:
            raise RuntimeError("Xvfb exited before it announced a display")
        line += chunk
    number = line.decode(errors="replace").strip()
    if not number.isdigit():
        raise RuntimeError(f"Xvfb announced an invalid display: {number!r}")
    return number


def atomic_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


class HKEnv:
    """One game session. `obs` is always the latest *visible* observation (dict)."""

    def __init__(self, out_dir: Path, scene: str = "Level1", capture: bool = True,
                 mask_capture: bool = False, model_timeout: float = 600.0, observation_timeout: float = 90.0):
        self.out = Path(out_dir).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.session = self.out / "session"
        self.session.mkdir(exist_ok=False)
        self.scene = scene
        self.capture = capture
        self.mask_capture = mask_capture
        self.model_timeout = model_timeout
        self.observation_timeout = observation_timeout
        self.display = None
        self.game = None
        self.obs = None
        self.windows = 0
        self.frames = 0
        self._xvfb_log = None
        self._game_log = None
        self._close_info = None

    # ---- lifecycle -------------------------------------------------------------------------
    def start(self) -> dict:
        try:
            return self._start()
        except BaseException:
            self.close()
            raise

    def _start(self) -> dict:
        game = game_binary()
        read_fd, write_fd = os.pipe()
        self._xvfb_log = (self.out / "xvfb.log").open("w")
        try:
            self.display = subprocess.Popen(
                ["Xvfb", "-displayfd", str(write_fd), "-screen", "0", "960x540x24", "-nolisten", "tcp"],
                pass_fds=[write_fd], stdout=self._xvfb_log, stderr=subprocess.STDOUT, start_new_session=True)
        finally:
            os.close(write_fd)
        try:
            number = read_display_number(read_fd)
        finally:
            os.close(read_fd)
        env = {**os.environ, "DISPLAY": ":" + number, "LIBGL_ALWAYS_SOFTWARE": "1", "LP_NUM_THREADS": "4",
               "HONING_SESSION_DIR": str(self.session), "HONING_CAPTURE": "1" if self.capture else "0",
               "HONING_SCENE": self.scene, "HONING_MODEL_TIMEOUT_SECONDS": str(self.model_timeout),
               "HONING_OBSERVATION_PROFILE": "visible"}
        if self.mask_capture:
            env["HONING_VISIBLE_MASK_CAPTURE"] = "1"
        else:
            env.pop("HONING_VISIBLE_MASK_CAPTURE", None)
        command = [str(game), "-logFile", str(self.session / "player.log"), "-screen-fullscreen", "0",
                   "-screen-width", "960", "-screen-height", "540", "-force-glcore", "-noaudio",
                   "-job-worker-count", "4"]
        self._game_log = (self.session / "stdout.log").open("w")
        self.game = subprocess.Popen(command, env=env, stdout=self._game_log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        atomic_json(self.out / "launch.json", {"display": ":" + number, "xvfb_pid": self.display.pid,
                                               "game_pid": self.game.pid, "command": command,
                                               "scene": self.scene, "capture": self.capture})
        self.obs = self._wait_visible(0)
        return self.obs

    def _wait_visible(self, previous: int) -> dict:
        deadline = time.monotonic() + self.observation_timeout
        path = self.session / "observation.json"
        while time.monotonic() < deadline:
            if self.game.poll() is not None:
                raise RuntimeError(f"game exited with code {self.game.returncode}; see {self.session/'player.log'}")
            try:
                obs = json.loads(path.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                time.sleep(0.004)
                continue
            if obs.get("schema") != "honing.visible.v1":
                raise ValueError("expected the visible observation profile")
            if obs["frame"]["id"] > previous:
                return obs
            time.sleep(0.004)
        raise TimeoutError("no new visible observation; see player.log")

    def step(self, action: str, frames: int, label: str = "", probability: float = 0.0, meta: dict | None = None) -> dict:
        """Send one native window and return the next visible observation."""
        if action not in NATIVE_ACTIONS:
            raise ValueError(f"not a native action: {action}")
        if not 1 <= int(frames) <= 120:
            raise ValueError("frames must be within 1..120")
        frames = int(frames)
        prev = self.obs
        pid, ptick = prev["frame"]["id"], prev["frame"]["tick"]
        command = {"id": pid, "action": action, "frames": frames, "model": label or "harness",
                   "probability": float(probability)}
        atomic_json(self.session / "action.json", command)
        obs = self._wait_visible(pid)
        if obs["frame"]["id"] != pid + 1 or obs["frame"]["tick"] != ptick + frames:
            raise ValueError(f"bridge did not advance exactly one window: {pid}->{obs['frame']['id']}, "
                             f"tick {ptick}->{obs['frame']['tick']} (frames={frames})")
        self.obs = obs
        self.windows += 1
        self.frames += frames
        record = {"obs_id": pid, "tick": ptick, "action": action, "frames": frames, "label": label}
        if meta:
            record.update(meta)
        with (self.session / "windows.jsonl").open("a") as log:
            log.write(json.dumps(record) + "\n")
        return obs

    def close(self) -> dict:
        if self._close_info is not None:
            return self._close_info
        info = {}
        game, display = self.game, self.display
        if game is not None:
            if game.poll() is None and self.obs is not None:
                try:
                    atomic_json(self.session / "action.json", {"id": self.obs["frame"]["id"], "action": "stop"})
                    game.wait(timeout=15)
                except Exception:
                    pass
            if game.poll() is None:
                game.terminate()
                try:
                    game.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    game.kill()
                    game.wait(timeout=5)
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
        info["windows"], info["frames"] = self.windows, self.frames
        atomic_json(self.out / "close.json", info)
        self.game = self.display = None
        self._close_info = info
        return info

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()


def encode_video(session: Path, output: Path, start: int = 0, count: int | None = None, fps: int = 30,
                 ffmpeg: str | None = None) -> None:
    """Frames -> mp4 (action frames only; model waiting time is not in the video)."""
    ffmpeg = ffmpeg or os.environ.get("HONING_FFMPEG", "ffmpeg")
    frames = session / "frames"
    args = [ffmpeg, "-hide_banner", "-loglevel", "warning", "-y", "-framerate", str(fps), "-start_number", str(start),
            "-i", str(frames / "%06d.png")]
    if count is not None:
        args += ["-frames:v", str(count)]
    args += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    subprocess.run(args, check=True)
