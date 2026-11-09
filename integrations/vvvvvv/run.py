"""Run a real VVVVVV process with an existing Honing decision server."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from decision import ACTIONS, DecisionClient, checked_action


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class NativeClient:
    """Send one sequence-checked command while the original engine is paused."""

    def __init__(self, session: Path, process: subprocess.Popen) -> None:
        self.session, self.process = session, process
        self.calls = 0

    def observe(self, previous: int = 0, timeout: float = 60) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"VVVVVV exited {self.process.returncode}; see player.log")
            try:
                state = json.loads((self.session / "observation.json").read_text())
                if state["id"] > previous:
                    return state
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            time.sleep(0.01)
        raise TimeoutError("Native VVVVVV observation stalled")

    def command(self, state: dict[str, Any], action: str, ticks: int) -> None:
        if action not in {*ACTIONS, "stop", "reset"}:
            raise ValueError("Unknown native action")
        if type(ticks) is not int or (ticks != 0 if action in {"stop", "reset"} else not 2 <= ticks <= 120):
            raise ValueError("Invalid native tick window")
        temporary = self.session / "command.tmp"
        temporary.write_text(f"{state['id']} {action} {ticks}\n")
        temporary.replace(self.session / "command.txt")
        self.calls += 1

    def step(self, state: dict[str, Any], action: str, ticks: int) -> dict[str, Any]:
        self.command(state, action, ticks)
        after = self.observe(state["id"])
        if (
            after["id"] != state["id"] + 1
            or after["tick"] != state["tick"] + ticks
            or after.get("lastAction") != action
            or after.get("lastActionTicks") != ticks
            or after.get("engineFrame") != state["engineFrame"] + ticks
        ):
            raise ValueError("Native acknowledgement has the wrong sequence or tick count")
        return after


def request_for(state: dict[str, Any], ticks: int) -> dict[str, Any]:
    return {
        "id": state["id"],
        "parts": {
            "essential": [
                "Play the original VVVVVV. Move through Space Station 1, surviving spikes. "
                "Positive x is right and y down, in pixels; a tile is 8 pixels. "
                "Flip reverses gravity only on the ground/roof. Each action holds movement "
                f"for {ticks} native ticks; flip is a single edge at tick 2. The game pauses while deciding. "
                "gravity=0 down,1 up. local rows: # solid, ! static spike, . clear, ? outside room; "
                "rows start at localOriginTiles, each is 9 cells. entities are visible room objects "
                "[native type,x,y,width,height,harmful]. Head toward the next room; avoid dying.",
                json.dumps(state, separators=(",", ":"), allow_nan=False),
            ]
        },
        "questions": {"action": {"type": "choice", "instructions": "Choose the next input.", "criteria": ACTIONS}},
    }


def decide_and_step(
    client: DecisionClient, native: NativeClient, state: dict[str, Any], ticks: int, output: Path
) -> dict[str, Any]:
    """Capture raw bytes and validation failure before the native action boundary."""
    request = request_for(state, ticks)
    prefix = output / f"{state['id']:06d}"
    write(prefix.with_suffix(".request.json"), request)
    raw = prefix.with_suffix(".response.raw")
    try:
        response = client.decide(request, raw_path=raw)
        action, probability = checked_action(response, request)
    except Exception as error:
        write(
            prefix.with_suffix(".rejected.json"),
            {
                "request": request,
                "reason": f"{type(error).__name__}: {error}",
                "responseRawFile": raw.name,
                "nativeCalls": native.calls,
                "pausedState": state,
            },
        )
        raise
    after = native.step(state, action, ticks)
    write(
        prefix.with_suffix(".accepted.json"),
        {
            "before": state,
            "after": after,
            "action": action,
            "ticks": ticks,
            "probability": probability,
            "responseSha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        },
    )
    return after


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True, help="Caller-supplied original desktop data.zip")
    parser.add_argument("--decision-url", required=True)
    parser.add_argument("--output", type=Path, required=True, help="New local private run directory")
    parser.add_argument("--decisions", type=int, default=20)
    parser.add_argument("--ticks", type=int, default=15)
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    if not 2 <= args.ticks <= 120 or not 1 <= args.decisions <= 1000:
        parser.error("ticks must be 2..120; decisions 1..1000")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "frames").mkdir()
    (args.output / "user").mkdir()
    client = DecisionClient(args.decision_url)
    write(args.output / "model.json", client.health())
    environment = {**os.environ, "HONING_SESSION_DIR": str(args.output)}
    if args.capture:
        environment["HONING_CAPTURE"] = "1"
    command = [str(args.game.resolve()), "-assets", str(args.assets.resolve()), "-basedir", str(args.output / "user")]
    started = time.monotonic()
    completed = 0
    with (args.output / "player.log").open("wb") as log:
        process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        write(
            args.output / "launch.json",
            {
                "pid": process.pid,
                "command": command,
                "procStartTicks": Path(f"/proc/{process.pid}/stat").read_text().split()[21],
            },
        )
        native = NativeClient(args.output, process)
        try:
            state = native.observe()
            initial = state
            for _ in range(args.decisions):
                state = decide_and_step(client, native, state, args.ticks, args.output)
                completed += 1
                print(
                    f"decision={completed} tick={state['tick']} room={state['room']} player={state['player']}",
                    flush=True,
                )
            native.command(state, "stop", 0)
            if process.wait(timeout=15) != 0:
                raise RuntimeError("Native stop failed")
            write(
                args.output / "result.json",
                {
                    "decisions": completed,
                    "initial": initial,
                    "final": state,
                    "wallSeconds": time.monotonic() - started,
                    "nativeCalls": native.calls,
                    "capture": args.capture,
                    "timeline": "Native action ticks only; model waiting omitted",
                    "heuristicFallback": False,
                    "training": False,
                },
            )
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            write(
                args.output / "cleanup.json",
                {"ownedPid": process.pid, "returncode": process.returncode, "sharedModelChanged": False},
            )


if __name__ == "__main__":
    main()
