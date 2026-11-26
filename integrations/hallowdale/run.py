"""Run the patched hallowdale game with an existing Honing decision server."""

from __future__ import annotations

import argparse
import json
import math
import os
import select
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

# Resolve the sibling stdlib-only receiver without exposing integrations as importable runtimes.
_shared_import_path = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, _shared_import_path)
try:
    from _shared.decision import DecisionClient as DecisionClient
    from _shared.decision import checked_action as checked_decision_action
    from _shared.decision import finite_json_float as finite_json_float
    from _shared.decision import reject_json_constant as reject_json_constant
    from _shared.decision import strict_json_loads as strict_json_loads
    from _shared.decision import unique_object as unique_object
finally:
    # Remove only our inserted object, preserving preexisting equal path entries.
    for _shared_index, _shared_entry in enumerate(sys.path):
        if _shared_entry is _shared_import_path:
            del sys.path[_shared_index]
            break

ACTIONS = {
    "wait": "Do nothing.",
    "left": "Walk left.",
    "right": "Walk right.",
    "up": "Hold up.",
    "down": "Hold down.",
    "jump": "Full jump straight up.",
    "jump_left": "Full jump to the left.",
    "jump_right": "Full jump to the right.",
    "short_jump": "Low hop straight up.",
    "short_jump_left": "Low hop to the left.",
    "short_jump_right": "Low hop to the right.",
    "attack": "Strike the way you face.",
    "attack_left": "Strike left.",
    "attack_right": "Strike right.",
    "attack_up": "Strike up.",
    "attack_down": "Strike down while in the air.",
    "dash": "Dash the way you face.",
    "dash_left": "Dash left.",
    "dash_right": "Dash right.",
    "interact": "Talk, read or advance text.",
    "focus": "Hold to heal, using soul.",
    "cast": "Cast a spell the way you face.",
    "cast_left": "Cast a spell to the left.",
    "cast_right": "Cast a spell to the right.",
    "cast_up": "Cast the upward spell.",
    "cast_down": "Cast the downward spell.",
    "c_dash": "Hold to charge a long dash; it launches on release.",
    "dream_nail": "Hold to swing the dream nail.",
}

# A low hop holds the jump key for this many bridge frames; a full jump keeps it down.
SHORT_JUMP_FRAMES = 3

PROMPT = (
    "Play hallowdale, a 2D platformer. Explore and leave each room through an opening at one of its edges. "
    "Doors and jars break when struck. Enemies and their projectiles hurt on touch. Spikes hurt and put you "
    "back on the last safe ground. The state is what a player sees on the screen right now. rows is that "
    "screen as a grid of {cell} pixel cells, top row first. Each character is U+0100 plus the sum of what the "
    "cell shows: 1 background, 2 solid, 4 platform, 8 hazard, 16 player or enemy, 32 breakable or lever, "
    "64 unknown (off screen or covered), 128 scenery. objects lists [id, kind, look, x, y, width, height] in "
    "screen pixels, x right and y down. Each action lasts {frames} frames at 30 per second and the game pauses "
    "while you choose. Abilities you have not found yet do nothing."
)


def native_command(choice: str, frames: int) -> dict[str, Any]:
    """Translate one offered choice into the bridge's native action fields."""
    if choice not in ACTIONS:
        raise ValueError(f"Not an offered action: {choice}")
    if choice.startswith("short_jump"):
        return {
            "action": choice.removeprefix("short_"),
            "frames": frames,
            "jump_frames": min(frames, SHORT_JUMP_FRAMES),
        }
    return {"action": choice, "frames": frames}


def model_state(observation: dict[str, Any]) -> dict[str, Any]:
    """Keep what changes between observations; the fixed rules and legend are in the prompt."""
    own_body = observation["self"]
    return {
        "screen": observation["view"]["screen_rect"][2:],
        "rows": observation["terrain"]["rows"],
        "objects": [
            [item["id"].rsplit("/", 1)[-1], item["kind"], item["look"], *item["bounds"]]
            for item in observation["objects"]
        ],
        "self": {key: own_body[key] for key in ("screen_box", "facing") if key in own_body},
        "hud": observation["hud"],
    }


def request_for(observation: dict[str, Any], frames: int) -> dict[str, Any]:
    """Submit the instructions and the visible state as required context, with every action offered."""
    return {
        "id": observation["frame"]["id"],
        "parts": {
            "essential": [
                PROMPT.format(cell=observation["terrain"]["cell_size"][0], frames=frames),
                json.dumps(model_state(observation), separators=(",", ":"), ensure_ascii=False, allow_nan=False),
            ]
        },
        "questions": {
            "action": {
                "type": "choice",
                "instructions": "Choose the next input to make progress and survive.",
                "criteria": dict(ACTIONS),
            }
        },
    }


def checked_action(response: dict[str, Any], request: dict[str, Any]) -> tuple[str, float]:
    """Validate the shared serving contract before the game-specific native input."""
    return checked_decision_action(response, request, ACTIONS)


def write_json(path: Path, value: Any) -> None:
    """Publish a complete file with a same-directory rename."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def native_idle_seconds(timeout: float) -> int:
    """Give the bridge's idle watchdog thirty seconds beyond the model's HTTP timeout."""
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or not 0 < timeout <= 3570
    ):
        raise ValueError("timeout must be finite and greater than 0, at most 3570 seconds")
    return math.ceil(timeout) + 30


def stop_owned(process: subprocess.Popen | None) -> None:
    """Stop the process group of one child started here; other processes are never signalled."""
    if process is None or process.poll() is not None:
        return
    for signal_number in (signal.SIGTERM, signal.SIGKILL):
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal_number)
        try:
            process.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            continue


def start_xvfb(log: Any, timeout: float = 30) -> tuple[subprocess.Popen, str]:
    """Allocate a fresh display and wait for Xvfb's complete readiness line."""
    read_fd, write_fd = os.pipe()
    process = None
    try:
        process = subprocess.Popen(
            ["Xvfb", "-displayfd", str(write_fd), "-screen", "0", "960x540x24", "-nolisten", "tcp"],
            pass_fds=(write_fd,),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        os.close(write_fd)
        write_fd = -1
        deadline = time.monotonic() + timeout
        data = b""
        while b"\n" not in data:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([read_fd], [], [], remaining)[0]:
                raise TimeoutError("Owned Xvfb did not report a ready display")
            chunk = os.read(read_fd, 80)
            if not chunk or len(data) + len(chunk) > 80:
                raise RuntimeError("Owned Xvfb closed or returned an invalid display number")
            data += chunk
        number = data.strip()
        if not number.isdigit() or process.poll() is not None:
            raise RuntimeError("Owned Xvfb failed to allocate a ready display")
        return process, ":" + number.decode("ascii")
    except BaseException:
        stop_owned(process)
        raise
    finally:
        os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)


def require_visible(observation: Any) -> dict[str, Any]:
    """Refuse anything but the human-visible profile before it can reach a model."""
    if (
        not isinstance(observation, dict)
        or observation.get("schema") != "honing.visible.v1"
        or observation.get("profile") != "visible"
        or observation.get("capabilities", {}).get("pixels_to_model") is not False
    ):
        raise ValueError("The bridge did not publish the visible observation profile")
    frame = observation.get("frame")
    if not isinstance(frame, dict) or type(frame.get("id")) is not int or type(frame.get("tick")) is not int:
        raise ValueError("The visible observation has no integer frame id and tick")
    return observation


def check_transition(
    before: dict[str, Any], after: dict[str, Any], command: dict[str, Any], ack: dict[str, Any]
) -> None:
    """Prove that the bridge applied exactly the requested window before the next decision."""
    if after["frame"]["id"] != before["frame"]["id"] + 1:
        raise ValueError("The native window did not advance the observation id by one")
    if after["frame"]["tick"] != before["frame"]["tick"] + command["frames"]:
        raise ValueError("The native window did not advance by the requested frames")
    expected = {
        "id": before["frame"]["id"],
        "status": "applied",
        "action": command["action"],
        "frames": command["frames"],
        "jump_frames": command.get("jump_frames"),
        "frame_before": before["frame"]["tick"],
        "frame_after": after["frame"]["tick"],
        "observation_id": after["frame"]["id"],
    }
    applied = {key: ack.get(key) for key in expected}
    if applied != expected:
        raise ValueError(f"The bridge acknowledged a different window: {applied}")


class Game:
    """Own one Xvfb and one game process; only accepted commands advance the paused game."""

    def __init__(
        self,
        game: Path,
        session: Path,
        *,
        capture: bool = False,
        idle_seconds: float = 210,
        cell_pixels: int = 40,
        seed: int = 42,
        scene: str = "new",
        observation_timeout: float = 300,
    ) -> None:
        self.game = game.resolve()
        self.session = session.resolve()
        self.capture, self.idle_seconds, self.cell_pixels = capture, idle_seconds, cell_pixels
        self.seed, self.scene, self.observation_timeout = seed, scene, observation_timeout
        self.process: subprocess.Popen | None = None
        self.display: subprocess.Popen | None = None
        self.owns_session = False
        self.logs: list[Any] = []

    def environment(self, display: str) -> dict[str, str]:
        """Inherit no bridge setting: an inherited knob must not change what the model is shown."""
        inherited = {key: value for key, value in os.environ.items() if not key.startswith("HONING_")}
        return {
            **inherited,
            "DISPLAY": display,
            "LIBGL_ALWAYS_SOFTWARE": "1",
            "PULSE_SERVER": "unix:/nonexistent-honing-audio",
            "HONING_SESSION_DIR": str(self.session),
            "HONING_CAPTURE": "1" if self.capture else "0",
            "HONING_RENDER": "frame" if self.capture else "window",
            "HONING_SCENE": self.scene,
            "HONING_SEED": str(self.seed),
            "HONING_JUMP_MODE": "sticky",
            "HONING_OBSERVATION_PROFILE": "visible",
            "HONING_VISIBLE_CELL_PIXELS": str(self.cell_pixels),
            "HONING_MODEL_TIMEOUT_SECONDS": str(self.idle_seconds),
        }

    def start(self) -> dict[str, Any]:
        """Start the owned display and game; every failure stops what was started."""
        if self.session.exists():
            raise ValueError("output must be a new directory")
        if not self.game.is_file() or not os.access(self.game, os.X_OK):
            raise ValueError(f"Game executable not found: {self.game}")
        self.session.mkdir(parents=True)
        self.owns_session = True
        (self.session / "responses").mkdir()
        try:
            display_log = (self.session / "xvfb.log").open("wb")
            self.logs.append(display_log)
            self.display, display = start_xvfb(display_log)
            game_log = (self.session / "stdout.log").open("wb")
            self.logs.append(game_log)
            self.process = subprocess.Popen(
                [str(self.game)],
                env=self.environment(display),
                stdout=game_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            write_json(
                self.session / "launch.json",
                {
                    "gamePid": self.process.pid,
                    "displayPid": self.display.pid,
                    "display": display,
                    "command": [str(self.game)],
                    "idleSeconds": self.idle_seconds,
                    "cellPixels": self.cell_pixels,
                    "scene": self.scene,
                    "seed": self.seed,
                },
            )
            return self.wait_observation(0)
        except BaseException:
            self.close()
            raise

    def latest(self) -> dict[str, Any] | None:
        try:
            return require_visible(strict_json_loads((self.session / "observation.json").read_bytes()))
        except FileNotFoundError:
            return None

    def latest_ack(self) -> dict[str, Any] | None:
        try:
            ack = strict_json_loads((self.session / "ack.json").read_bytes())
        except FileNotFoundError:
            return None
        return ack if isinstance(ack, dict) else None

    def wait_observation(self, previous: int, *, command_id: int | None = None, seen_ack: int = 0) -> dict[str, Any]:
        """Wait for a newer observation; report a newly refused command or a dead game at once."""
        deadline = time.monotonic() + self.observation_timeout
        while time.monotonic() < deadline:
            observation = self.latest()
            if observation is not None and observation["frame"]["id"] > previous:
                return observation
            ack = self.latest_ack() if command_id is not None else None
            if ack is not None and ack.get("ack", 0) > seen_ack and ack.get("status") == "refused":
                raise ValueError(f"The bridge refused the command: {ack.get('reason')}")
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(f"The game exited with code {self.process.returncode}; see stdout.log")
            time.sleep(0.005)
        raise TimeoutError(f"No new observation within {self.observation_timeout:g}s; see bridge.log")

    def send(self, command: dict[str, Any]) -> None:
        write_json(self.session / "action.json", command)

    def step(self, before: dict[str, Any], command: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Play one native window and return the next observation with the bridge's acknowledgement."""
        identity = before["frame"]["id"]
        seen_ack = (self.latest_ack() or {}).get("ack", 0)
        self.send({"id": identity, **command})
        after = self.wait_observation(identity, command_id=identity, seen_ack=seen_ack)
        ack = self.latest_ack()
        if ack is None:
            raise ValueError("The bridge published no acknowledgement")
        check_transition(before, after, command, ack)
        return after, ack

    def stop(self, observation: dict[str, Any]) -> None:
        """Ask the bridge to exit; close() still reaps the process when this fails."""
        self.send({"id": observation["frame"]["id"], "action": "stop"})
        if self.process is not None and self.process.wait(timeout=30) != 0:
            raise RuntimeError(f"The game did not exit cleanly: {self.process.returncode}")

    def close(self) -> None:
        """Reap only the processes started here, and leave a receipt in a directory created here."""
        receipts = {}
        for name, process in (("game", self.process), ("display", self.display)):
            if process is not None:
                stop_owned(process)
                receipts[name] = {
                    "pid": process.pid,
                    "returncode": process.returncode,
                    "reaped": process.poll() is not None,
                }
        for log in self.logs:
            log.close()
        self.logs = []
        if self.owns_session and self.session.is_dir():
            write_json(self.session / "cleanup.json", receipts)


def decide_and_step(game: Game, client: DecisionClient, before: dict[str, Any], frames: int) -> dict[str, Any]:
    """Keep the raw response before validating it; a rejected response never reaches the game."""
    request = request_for(before, frames)
    raw_file = f"responses/{before['frame']['id']:06d}.raw"
    record: dict[str, Any] = {"request": request, "responseRawFile": raw_file}
    try:
        response = client.decide(request, raw_path=game.session / raw_file)
        record["response"] = response
        choice, probability = checked_action(response, request)
        record.update(choice=choice, probability=probability)
        after, record["ack"] = game.step(before, native_command(choice, frames))
        return after
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        with (game.session / "decisions.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(record, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", type=Path, required=True, help="Executable printed by build.py")
    parser.add_argument("--output", type=Path, required=True, help="New local directory for the private trace")
    parser.add_argument("--decision-url", default="http://127.0.0.1:8809")
    parser.add_argument("--decisions", type=int, default=12)
    parser.add_argument("--frames", type=int, default=12, help="Bridge frames (1/30 s) per decision")
    parser.add_argument("--timeout", type=float, default=180, help="Model HTTP timeout in seconds")
    parser.add_argument("--cell-pixels", type=int, default=40, help="Size of one grid cell on the 960 px screen")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--capture", action="store_true", help="Save one PNG per bridge frame for encode.py")
    args = parser.parse_args()
    if not 1 <= args.decisions <= 10000 or not 1 <= args.frames <= 120 or not 4 <= args.cell_pixels <= 120:
        parser.error("decisions must be 1..10000, frames 1..120 and cell-pixels 4..120")
    try:
        idle_seconds = native_idle_seconds(args.timeout)
    except ValueError as error:
        parser.error(str(error))
    client = DecisionClient(args.decision_url, args.timeout)
    health = client.health()
    game = Game(
        args.game,
        args.output,
        capture=args.capture,
        idle_seconds=idle_seconds,
        cell_pixels=args.cell_pixels,
        seed=args.seed,
    )
    completed = 0
    try:
        observation = game.start()
        write_json(game.session / "model.json", health)
        for _ in range(args.decisions):
            observation = decide_and_step(game, client, observation, args.frames)
            completed += 1
            print(f"{completed}: tick={observation['frame']['tick']}", flush=True)
        game.stop(observation)
        write_json(
            game.session / "result.json",
            {"decisions": completed, "capture": args.capture, "heuristicFallback": False},
        )
    finally:
        game.close()
    print(game.session)


if __name__ == "__main__":
    main()
