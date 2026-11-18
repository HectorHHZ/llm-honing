"""Control a caller-supplied Unity game with an existing Honing decision server."""

from __future__ import annotations

import argparse
import json
import math
import os
import plistlib
import subprocess
import sys
import tempfile
import time
from datetime import datetime
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
    "wait": "Stay still briefly.",
    "left": "Walk left.",
    "right": "Walk right.",
    "jump_left": "Jump or double jump while moving left.",
    "jump_right": "Jump or double jump while moving right.",
    "dash_left": "Dash left if the cooldown permits.",
    "dash_right": "Dash right if the cooldown permits.",
    "attack_left": "Face left and attack with the sword.",
    "attack_right": "Face right and attack with the sword.",
    "attack_up": "Attack upward.",
    "attack_down": "Attack downward while airborne.",
}


def request_for(observation: dict[str, Any], frames: int) -> dict[str, Any]:
    """Keep the complete observation and action set in the existing decision wire format."""
    return {
        "id": observation["id"],
        "parts": {
            "essential": [
                "Play a Hollow Knight imitation platformer. Reach the level exit, stay alive, "
                "and fight nearby enemies. Positive x is right and positive y is up. "
                "floorLeft/floorRight report ground within two units below a point 1.2 units "
                "to that side; false indicates a gap, which may lead to an exit below you. "
                "Exits are rectangles with world coordinates and sizes. wallLeft/wallRight "
                f"are distances capped at two units. Each input is requested for {frames} "
                "rendered frames at 30 fps. The game pauses while you decide. "
                "Inputs only succeed when the game's own rules permit them.",
                json.dumps(observation, separators=(",", ":"), allow_nan=False),
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


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    """Publish a complete command with a same-directory rename."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def executable(path: Path) -> Path:
    """Accept a Mac .app or an explicit player executable."""
    if path.is_dir() and path.suffix == ".app":
        with (path / "Contents/Info.plist").open("rb") as handle:
            name = plistlib.load(handle)["CFBundleExecutable"]
        path = path / "Contents/MacOS" / name
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"Game executable not found or not executable: {path}")
    return path.resolve()


def wait_observation(session: Path, previous: int, process: subprocess.Popen, timeout: float) -> dict[str, Any]:
    """Wait for a new sequence number, reporting a dead game or a stalled bridge."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Unity exited with code {process.returncode}; see {session / 'player.log'}")
        try:
            observation = json.loads((session / "observation.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.02)
            continue
        if not isinstance(observation, dict) or type(observation.get("id")) is not int:
            raise ValueError("Bridge observation must have an integer id")
        if observation["id"] > previous:
            return observation
        time.sleep(0.02)
    raise TimeoutError(f"No new game observation within {timeout:g}s; see {session / 'player.log'}")


def run_episode(args: argparse.Namespace) -> Path:
    """Own the game process and retain every model request, answer and game observation."""
    client = DecisionClient(args.decision_url, args.timeout)
    health = client.health()
    binary = executable(args.game)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("%Y%m%d-%H%M%S-"), dir=args.output_dir))
    (session / "responses").mkdir()
    atomic_json(session / "model.json", health)
    environment = {
        **os.environ,
        "HONING_SESSION_DIR": str(session.resolve()),
        "HONING_CAPTURE": "1" if args.capture else "0",
        "HONING_SCENE": args.scene,
        "HONING_MODEL_TIMEOUT_SECONDS": str(args.timeout + 30),
    }
    command = [str(binary), "-logFile", str((session / "player.log").resolve())]
    if args.headless:
        command += ["-batchmode", "-nographics"]
    else:
        command += ["-screen-fullscreen", "0", "-screen-width", "960", "-screen-height", "540"]
    previous, completed = 0, 0
    print(f"Session: {session}", flush=True)
    with (session / "stdout.log").open("w") as output:
        process = subprocess.Popen(command, env=environment, stdout=output, stderr=subprocess.STDOUT)
        try:
            for index in range(args.decisions + 1):
                observation = wait_observation(session, previous, process, args.observation_timeout)
                previous = observation["id"]
                if index == args.decisions:
                    atomic_json(session / "action.json", {"id": previous, "action": "stop"})
                    if process.wait(timeout=15) != 0:
                        raise RuntimeError("Unity did not exit cleanly after the final action")
                    break
                request = request_for(observation, args.frames)
                raw_file = f"responses/{previous:06d}.raw"
                record: dict[str, Any] = {"request": request, "responseRawFile": raw_file}
                try:
                    response = client.decide(request, raw_path=session / raw_file)
                    record["response"] = response
                    action, probability = checked_action(response, request)
                except Exception as error:
                    record["error"] = f"{type(error).__name__}: {error}"
                    raise
                finally:
                    with (session / "decisions.jsonl").open("a") as log:
                        log.write(json.dumps(record, allow_nan=False) + "\n")
                atomic_json(
                    session / "action.json",
                    {
                        "id": previous,
                        "action": action,
                        "frames": args.frames,
                        "model": "Honing decision server",
                        "probability": probability,
                    },
                )
                completed += 1
                print(f"{completed}: frame={observation['frame']} action={action} p={probability:.4f}", flush=True)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    atomic_json(
        session / "result.json",
        {"decisions": completed, "capture": args.capture, "headless": args.headless, "heuristicFallback": False},
    )
    return session


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", type=Path, required=True, help="Isolated app produced by build.py")
    parser.add_argument("--decision-url", required=True, help="Existing Honing decision server")
    parser.add_argument("--output-dir", type=Path, default=Path("work/hollow-knight/runs"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--headless", action="store_true")
    mode.add_argument("--capture", action="store_true", help="Export game render frames for encode.py")
    parser.add_argument("--scene", choices=["Spawn", "Level1", "Level2"], default="Level1")
    parser.add_argument("--decisions", type=int, default=50)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=180, help="Model HTTP timeout in seconds")
    parser.add_argument("--observation-timeout", type=float, default=60)
    args = parser.parse_args()
    if args.decisions < 1 or not 1 <= args.frames <= 120:
        parser.error("decisions must be positive and frames must be between 1 and 120")
    if not math.isfinite(args.observation_timeout) or args.observation_timeout <= 0:
        parser.error("observation-timeout must be positive and finite")
    run_episode(args)


if __name__ == "__main__":
    main()
