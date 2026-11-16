"""Run a paused native Taisei segment with an existing Honing decision service."""

from __future__ import annotations

import argparse
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
    "stay": "Stay still, shoot and focus.",
    "left": "Move left, shoot and focus.",
    "right": "Move right, shoot and focus.",
    "up": "Move up, shoot and focus.",
    "down": "Move down, shoot and focus.",
    "up_left": "Move diagonally up-left, shoot and focus.",
    "up_right": "Move diagonally up-right, shoot and focus.",
    "down_left": "Move diagonally down-left, shoot and focus.",
    "down_right": "Move diagonally down-right, shoot and focus.",
}


def request_for(observation: dict[str, Any], frames: int) -> dict[str, Any]:
    """Send a bounded current-state sensor in the existing Honing decision contract."""
    return {
        "id": observation["id"],
        "parts": {
            "essential": [
                "Play Taisei, a vertical bullet-hell shooter. Survive and shoot enemies. "
                "Coordinates are viewport pixels, x right, y down. nearestBullets entries "
                "are [x,y,vx,vy], native movement velocity parameters in pixels/tick, "
                "up to eight nearest on-screen enemy "
                "bullets; other bullets, lasers and enemies are not in this partial sensor. "
                f"Each action holds movement, shot and focus for {frames} ticks at 60Hz. "
                "Diagonal speed is normalized by the game. The game pauses during inference.",
                json.dumps(observation, separators=(",", ":"), allow_nan=False),
            ]
        },
        "questions": {
            "action": {
                "type": "choice",
                "instructions": "Choose the next movement to survive.",
                "criteria": dict(ACTIONS),
            }
        },
    }


def checked_action(response: dict[str, Any], request: dict[str, Any]) -> tuple[str, float]:
    """Validate the shared serving contract before the game-specific native input."""
    return checked_decision_action(response, request, ACTIONS)


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    """Publish a complete command with a same-directory rename."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, allow_nan=False, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)


class NativeClient:
    """Consume one command per paused observation from an isolated stage process."""

    def __init__(self, session: Path, process: subprocess.Popen) -> None:
        self.session = session
        self.process = process
        self.previous = 0

    def observe(self, timeout: float = 90) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"Taisei exited: {self.process.returncode}; see game.log")
            try:
                value = json.loads((self.session / "observation.json").read_text())
            except (OSError, ValueError):
                time.sleep(0.01)
                continue
            if type(value.get("id")) is int and value["id"] > self.previous:
                self.previous = value["id"]
                return value
            time.sleep(0.01)
        raise TimeoutError("Native observation timed out")

    def step(self, observation_id: int, action: str, frames: int) -> None:
        if observation_id != self.previous or action not in {*ACTIONS, "stop"}:
            raise ValueError("Invalid native action or stale observation")
        if type(frames) is not int or not 1 <= frames <= 120:
            raise ValueError("frames must be an integer from 1 through 120")
        atomic_json(self.session / "action.json", {"id": observation_id, "action": action, "frames": frames})


def decision_step(client: DecisionClient, native: NativeClient, observation: dict[str, Any], frames: int) -> str:
    """Preserve rejected bytes before validating; only accepted responses reach native.step."""
    request = request_for(observation, frames)
    raw = native.session / f"response-{observation['id']:06d}.raw"
    atomic_json(native.session / f"request-{observation['id']:06d}.json", request)
    record: dict[str, Any] = {"request": request, "responseRawFile": raw.name}
    try:
        response = client.decide(request, raw_path=raw)
        record["response"] = response
        action, probability = checked_action(response, request)
        record["acceptedAction"] = action
        record["probability"] = probability
    except Exception as error:
        record["rejected"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        with (native.session / "decisions.jsonl").open("a") as trace:
            trace.write(json.dumps(record, allow_nan=False) + "\n")
    native.step(observation["id"], action, frames)
    return action


def idle_timeout_milliseconds(timeout: float) -> int:
    """Leave thirty seconds beyond the HTTP deadline for command delivery."""
    milliseconds = (timeout + 30) * 1000
    if not math.isfinite(timeout) or timeout <= 0 or not math.isfinite(milliseconds):
        raise ValueError("timeout must be positive and finite")
    result = math.ceil(milliseconds)
    if result > 2**64 - 1:
        raise ValueError("timeout exceeds the native uint64 millisecond deadline")
    return result


def stop_owned(process: subprocess.Popen | None) -> None:
    """Reap only the child represented by this Popen handle."""
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def start_xvfb(log: Any, timeout: float = 15) -> tuple[subprocess.Popen, str]:
    """Allocate a fresh display and wait for Xvfb's post-listen readiness line."""
    read_fd, write_fd = os.pipe()
    process = None
    try:
        process = subprocess.Popen(
            ["Xvfb", "-displayfd", str(write_fd), "-screen", "0", "800x600x24", "-nolisten", "tcp"],
            pass_fds=(write_fd,),
            stdout=log,
            stderr=subprocess.STDOUT,
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


def run(args: argparse.Namespace) -> None:
    idle_timeout = idle_timeout_milliseconds(args.timeout)
    args.output.mkdir(parents=True, exist_ok=False)
    session = args.output.resolve()
    client = DecisionClient(args.decision_url, args.timeout)
    atomic_json(session / "model.json", client.health())
    environment = {
        **os.environ,
        "HONING_SESSION_DIR": str(session),
        "HONING_IDLE_TIMEOUT_MS": str(idle_timeout),
        "TAISEI_STORAGE_PATH": str(session / "storage"),
        "TAISEI_CACHE_PATH": str(session / "cache"),
        "TAISEI_RES_PATH": str(args.resources.resolve()),
        "TAISEI_TASKMGR_NUM_THREADS": "4",
        "LP_NUM_THREADS": "4",
        "TAISEI_AUDIO_BACKEND": "null",
        "LIBGL_ALWAYS_SOFTWARE": "1",
        "TAISEI_NOASYNC": "1",
    }
    if args.capture:
        (session / "frames").mkdir()
        environment["TAISEI_FRAMEDUMP"] = str(session / "frames/frame-")
    command = [
        str(args.game.resolve()),
        "--play",
        "--sid",
        "1",
        "--diff",
        "Easy",
        "--shotmode",
        "youmuA",
        "--width",
        "800",
        "--height",
        "600",
    ]
    xvfb = None
    started = time.monotonic()
    completed = 0
    initial = final = None
    with (session / "game.log").open("wb") as game_log, (session / "display.log").open("wb") as display_log:
        process = None
        try:
            xvfb, environment["DISPLAY"] = start_xvfb(display_log)
            process = subprocess.Popen(command, env=environment, stdout=game_log, stderr=subprocess.STDOUT)
            atomic_json(
                session / "launch.json",
                {
                    "gamePid": process.pid,
                    "displayPid": xvfb.pid,
                    "display": environment["DISPLAY"],
                    "command": command,
                    "idleTimeoutMilliseconds": idle_timeout,
                },
            )
            native = NativeClient(session, process)
            initial = native.observe()
            observation = initial
            for _index in range(args.decisions):
                action = decision_step(client, native, observation, args.frames)
                after = native.observe()
                with (session / "actions.jsonl").open("a") as trace:
                    trace.write(
                        json.dumps({"before": observation, "action": action, "frames": args.frames, "after": after})
                        + "\n"
                    )
                completed += 1
                print(f"{completed}: frame {observation['frame']}->{after['frame']} {action}", flush=True)
                observation = after
            final = observation
            native.step(observation["id"], "stop", 1)
            process.wait(timeout=20)
        finally:
            for owned in (process, xvfb):
                stop_owned(owned)
            atomic_json(
                session / "result.json",
                {
                    "completedDecisions": completed,
                    "requestedFramesPerAction": args.frames,
                    "initial": initial,
                    "final": final,
                    "wallSeconds": time.monotonic() - started,
                    "simulationSeconds": (
                        None if initial is None or final is None else (final["frame"] - initial["frame"]) / 60
                    ),
                    "nativeExitCode": None if process is None else process.returncode,
                    "ownedDisplayExitCode": None if xvfb is None else xvfb.returncode,
                    "trainingPerformed": False,
                    "heuristicFallback": False,
                    "controllerAssistance": "shot and focus are held by every offered movement action",
                },
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True, help="Pinned checkout resources directory")
    parser.add_argument("--decision-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decisions", type=int, default=12)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    if args.decisions < 1 or not 1 <= args.frames <= 120:
        parser.error("decisions must be positive, frames 1..120")
    run(args)


if __name__ == "__main__":
    main()
