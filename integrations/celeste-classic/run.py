"""Run a paused Celeste Classic C port with Honing decisions and local frame capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import select
import subprocess
import sys
import time
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
    "wait": "Wait without pressing buttons.",
    "left": "Move left.",
    "right": "Move right.",
    "jump": "Hold jump.",
    "jump_left": "Hold jump and left.",
    "jump_right": "Hold jump and right.",
    "dash_left": "Hold dash and left.",
    "dash_right": "Hold dash and right.",
    "dash_up": "Hold dash and up.",
    "dash_down": "Hold dash and down.",
    "dash_up_left": "Hold dash, up and left.",
    "dash_up_right": "Hold dash, up and right.",
    "dash_down_left": "Hold dash, down and left.",
    "dash_down_right": "Hold dash, down and right.",
}
MASKS = dict(zip(ACTIONS, (0, 1, 2, 16, 17, 18, 33, 34, 36, 40, 37, 38, 41, 42), strict=True))


def request_for(observation: dict[str, Any], frames: int) -> dict[str, Any]:
    return {
        "id": observation["id"],
        "parts": {
            "essential": [
                "Play Celeste Classic. Climb to the top of the room and avoid spikes. "
                "Coordinates are room pixels: x right, y down; velocity pixels/tick. "
                "Tiles are 8px: # solid, ^ up / v down / > right / < left pointing spikes, . other. "
                "Entities cover the whole current room. "
                "Dashes refill on ground. Jump/dash trigger on button edges; "
                "Repeated jump/dash first releases that button until the player processes release, "
                "keeping directions held. "
                "Native freeze, cooldown and dash availability still apply. "
                f"One action lasts {frames} game ticks at 30Hz, including any release tick. "
                "Game update and draw pause while you decide.",
                json.dumps(observation, separators=(",", ":"), allow_nan=False),
            ]
        },
        "questions": {
            "action": {
                "type": "choice",
                "instructions": "Choose the next input to climb and survive.",
                "criteria": dict(ACTIONS),
            }
        },
    }


def checked_action(response: dict[str, Any], request: dict[str, Any]) -> tuple[str, float]:
    """Validate the shared serving contract before the game-specific native input."""
    return checked_decision_action(response, request, ACTIONS, optional_sections=("logs",))


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False, indent=2) + "\n")
    temporary.replace(path)


def process_receipt(process: subprocess.Popen) -> dict[str, Any]:
    stat = Path(f"/proc/{process.pid}/stat")
    return {
        "pid": process.pid,
        "argv": process.args,
        "startTicks": stat.read_text().split()[21] if stat.exists() else None,
    }


class Native:
    """Own one player and one Xvfb; only accepted commands advance update/draw."""

    def __init__(
        self,
        game: Path,
        output: Path,
        *,
        capture: bool = False,
        watchdog: int = 240,
        seed: int = 7,
        library_path: str = "",
    ) -> None:
        self.game = game.resolve()
        self.output = output.resolve()
        self.capture, self.watchdog, self.seed, self.library_path = capture, watchdog, seed, library_path
        self.player: subprocess.Popen | None = None
        self.display: subprocess.Popen | None = None
        self.receipts: dict[str, Any] = {}
        self.owns_output = False

    def start(self) -> dict[str, Any]:
        if self.output.exists():
            raise ValueError("output must be a fresh directory")
        if not self.game.is_file():
            raise ValueError("game executable does not exist")
        self.output.mkdir(parents=True)
        self.owns_output = True
        (self.output / "frames").mkdir()
        read_fd, write_fd = os.pipe()
        try:
            with (self.output / "xvfb.log").open("wb") as log:
                self.display = subprocess.Popen(
                    ["Xvfb", "-displayfd", str(write_fd), "-screen", "0", "768x768x24", "-nolisten", "tcp", "-ac"],
                    pass_fds=(write_fd,),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            self.receipts["display"] = process_receipt(self.display)
            write_json(self.output / "processes.json", self.receipts)
            os.close(write_fd)
            write_fd = -1
            if not select.select([read_fd], [], [], 15)[0]:
                raise TimeoutError("owned Xvfb did not allocate a new display")
            display_number = os.read(read_fd, 80).decode().strip()
            if not display_number.isdigit() or self.display.poll() is not None:
                raise RuntimeError("owned Xvfb failed to allocate display")
        finally:
            os.close(read_fd)
            if write_fd >= 0:
                os.close(write_fd)
        env = {
            **os.environ,
            "DISPLAY": ":" + display_number,
            "SDL_AUDIODRIVER": "dummy",
            "HONING_SESSION_DIR": str(self.output),
            "HONING_CAPTURE": "1" if self.capture else "0",
            "HONING_SEED": str(self.seed),
            "HONING_WATCHDOG_SECONDS": str(self.watchdog),
        }
        if self.library_path:
            env["LD_LIBRARY_PATH"] = self.library_path + ":" + env.get("LD_LIBRARY_PATH", "")
        with (self.output / "player.log").open("wb") as log:
            self.player = subprocess.Popen(
                [str(self.game)], cwd=self.game.parent, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        self.receipts["player"] = process_receipt(self.player)
        self.receipts["allocatedDisplay"] = ":" + display_number
        write_json(self.output / "processes.json", self.receipts)
        return self.wait(0)

    def state(self) -> dict[str, Any]:
        return json.loads((self.output / "observation.json").read_text())

    def wait(self, previous: int, timeout: float = 30) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                state = self.state()
                if type(state.get("id")) is int and state["id"] > previous:
                    if state.get("status") != "paused":
                        raise RuntimeError(f"Native bridge is {state.get('status')}")
                    return state
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            if self.player is not None and self.player.poll() is not None:
                raise RuntimeError("Native game exited; see player.log")
            time.sleep(0.01)
        raise TimeoutError("Native observation timed out")

    def send_raw(self, command: bytes) -> None:
        temporary = self.output / "command.tmp"
        temporary.write_bytes(command)
        temporary.replace(self.output / "command.txt")

    def step(self, before: dict[str, Any], action: str, frames: int) -> dict[str, Any]:
        if action not in MASKS or type(frames) is not int or not 1 <= frames <= 30:
            raise ValueError("invalid action or frame duration")
        self.send_raw(f"step {before['id']} {MASKS[action]} {frames}\n".encode())
        after = self.wait(before["id"])
        check_transition(before, after, MASKS[action], frames)
        return after

    def close(self) -> None:
        cleanup: dict[str, Any] = {}
        for name, process in (("player", self.player), ("display", self.display)):
            if process is None:
                continue
            if name == "player" and process.poll() is None:
                try:
                    state = self.state()
                    self.send_raw(f"stop {state['id']} 0 0\n".encode())
                    process.wait(timeout=3)
                except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired):
                    pass
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            cleanup[name] = {
                "pid": process.pid,
                "returncode": process.returncode,
                "reaped": process.poll() is not None,
            }
        if self.owns_output and self.output.is_dir():
            write_json(self.output / "cleanup.json", cleanup)


def check_transition(before: dict[str, Any], after: dict[str, Any], mask: int, frames: int) -> None:
    if after["id"] != before["id"] + 1 or after["lastMask"] != mask or after["lastFrames"] != frames:
        raise ValueError("native action echo mismatch")
    for counter in ("ticks", "episodeTicks", "nativeUpdates", "nativeDraws"):
        if after[counter] - before[counter] != frames:
            raise ValueError(f"native {counter} did not advance by the requested ticks")
    if after["clockFrame"] != (before["clockFrame"] + frames) % 30:
        raise ValueError("original engine clock did not advance")


def decide_and_step(native: Native, client: DecisionClient, before: dict[str, Any], frames: int) -> dict[str, Any]:
    request = request_for(before, frames)
    raw_path = native.output / f"response-{before['id']:06d}.raw"
    record: dict[str, Any] = {"before": before, "request": request, "rawResponse": raw_path.name}
    try:
        response = client.decide(request, raw_path=raw_path)
        record["response"] = response
        action, probability = checked_action(response, request)
        record.update({"action": action, "probability": probability})
        after = native.step(before, action, frames)
        record["after"] = after
        return after
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            line = json.dumps(record, allow_nan=False)
        except ValueError:
            record.pop("response", None)  # The complete rejected bytes remain in rawResponse.
            line = json.dumps(record, allow_nan=False)
        with (native.output / "decisions.jsonl").open("a") as log:
            log.write(line + "\n")


def native_watchdog_seconds(timeout: float) -> int:
    """Keep a full 30-second margin within the native watchdog's supported range."""
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or not 0 < timeout <= 3570
    ):
        raise ValueError("timeout must be finite and greater than 0, at most 3570 seconds")
    return math.ceil(timeout) + 30


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library-path", default="")
    parser.add_argument("--decision-url", default="http://127.0.0.1:8809")
    parser.add_argument("--decisions", type=int, default=20)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.frames <= 30 or not 1 <= args.decisions <= 1000 or not 0 <= args.seed <= 1000000000:
        parser.error("frames 1..30, decisions 1..1000, seed 0..1000000000 required")
    try:
        watchdog = native_watchdog_seconds(args.timeout)
    except ValueError as error:
        parser.error(str(error))
    client = DecisionClient(args.decision_url, args.timeout)
    health = client.health()
    native = Native(
        args.game,
        args.output,
        capture=args.capture,
        watchdog=watchdog,
        seed=args.seed,
        library_path=args.library_path,
    )
    started = time.monotonic()
    try:
        observation = native.start()
        first = observation
        write_json(native.output / "model.json", health)
        source_dir = Path(__file__).resolve().parent
        write_json(
            native.output / "adapter.json",
            {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(source_dir.glob("*.py"))},
        )
        write_json(native.output / "build.json", json.loads((native.game.parent / "build.json").read_text()))
        for _ in range(args.decisions):
            observation = decide_and_step(native, client, observation, args.frames)
            print(f"id={observation['id']} ticks={observation['ticks']} room={observation['room']}", flush=True)
        write_json(
            native.output / "result.json",
            {
                "decisions": args.decisions,
                "frames": observation["ticks"],
                "simulationSeconds": observation["ticks"] / 30,
                "wallSeconds": time.monotonic() - started,
                "initial": first,
                "final": observation,
                "capture": args.capture,
            },
        )
    finally:
        native.close()


if __name__ == "__main__":
    main()
