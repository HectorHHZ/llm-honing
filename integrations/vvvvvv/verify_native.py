"""Fixed-input native lifecycle probe; fixture rejection checks use no actual model."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from decision import ACTIONS
from run import NativeClient, decide_and_step, write


def probe_watchdog(game: Path, assets: Path, output: Path) -> dict:
    """A continuous invalid-command flood must still hit the idle deadline."""
    output.mkdir()
    (output / "user").mkdir()
    environment = {**os.environ, "HONING_SESSION_DIR": str(output), "HONING_IDLE_TIMEOUT_MS": "500"}
    environment.pop("HONING_CAPTURE", None)
    with (output / "player.log").open("wb") as log:
        process = subprocess.Popen(
            [str(game.resolve()), "-assets", str(assets.resolve()), "-basedir", str(output / "user")],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            state = NativeClient(output, process).observe()
            started = time.monotonic()
            sent = 0
            while process.poll() is None and time.monotonic() - started < 2:
                temporary = output / "command.tmp"
                temporary.write_text("-1 invented 0\n")
                temporary.replace(output / "command.txt")
                sent += 1
                time.sleep(0.02)
            assert process.wait(timeout=2) == 3
            assert json.loads((output / "observation.json").read_text()) == state
            assert sent >= 5
            return {
                "timeoutMilliseconds": 500,
                "invalidCommandsSent": sent,
                "exitCode": 3,
                "pausedStateUnchanged": True,
                "ownedPid": process.pid,
            }
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "user").mkdir()
    environment = {**os.environ, "HONING_SESSION_DIR": str(output)}
    environment.pop("HONING_CAPTURE", None)
    records = []
    with (output / "player.log").open("wb") as log:
        process = subprocess.Popen(
            [str(args.game.resolve()), "-assets", str(args.assets.resolve()), "-basedir", str(output / "user")],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        native = NativeClient(output, process)
        try:
            state = native.observe()
            initial = state
            time.sleep(0.1)
            assert native.observe() == state
            for action in ACTIONS:
                before = state
                state = native.step(state, action, 15)
                if "left" in action:
                    assert state["player"]["x"] < before["player"]["x"]
                if "right" in action:
                    assert state["player"]["x"] > before["player"]["x"]
                if action.startswith("flip"):
                    assert state["flips"] == before["flips"] + 1
                records.append({"action": action, "before": before, "after": state})
            # Rejected native commands must leave the entire paused state unchanged.
            for command in (
                f"{state['id'] - 1} right 15",
                f"{state['id']} right 0",
                f"{state['id']} invented 15",
                f"{state['id']} right 121",
                f"{state['id']} right 15 trailing",
            ):
                (output / "command.txt").write_text(command + "\n")
                time.sleep(0.1)
                assert native.observe() == state
                assert not (output / "command.txt").exists()
            native.command(state, "reset", 0)
            state = native.observe(state["id"])
            assert state["epoch"] == 1 and state["tick"] == 0
            for field in ("room", "player", "gravity", "checkpoint", "deaths", "flips"):
                assert state[field] == initial[field], (field, state[field], initial[field])

            # This is a deliberately invalid portable response, against a real paused native game.
            class InvalidClient:
                def decide(self, request, *, raw_path):
                    raw = b'{"id":-1,"answers":{}}'
                    raw_path.write_bytes(raw)
                    return json.loads(raw)

            before_calls = native.calls
            try:
                decide_and_step(InvalidClient(), native, state, 15, output)
                raise AssertionError("Invalid model response was accepted")
            except ValueError:
                pass
            time.sleep(0.1)
            assert native.calls == before_calls and native.observe() == state
            native.command(state, "stop", 0)
            assert process.wait(timeout=10) == 0
            write(
                output / "result.json",
                {
                    "allActions": records,
                    "initial": initial,
                    "reset": state,
                    "nativeRejectedCommands": 5,
                    "fixtureRejectedResponseNativeCalls": 0,
                    "pausedStateUnchangedAfterFixtureRejection": True,
                    "ownedPid": process.pid,
                    "exitCode": process.returncode,
                    "sharedModelChanged": False,
                    "scope": "Fixed-input native test plus invalid response fixture; no model performance evidence",
                },
            )
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
    result = json.loads((output / "result.json").read_text())
    result["watchdogInvalidCommandFlood"] = probe_watchdog(args.game, args.assets, output / "watchdog")
    write(output / "result.json", result)


if __name__ == "__main__":
    main()
