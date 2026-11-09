"""Exercise fixed native controls, resets, rejection and deadline behavior separately from model play."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from run import ACTIONS, Native, decide_and_step, request_for, write_json


def unchanged(before: dict, after: dict) -> None:
    assert {k: v for k, v in before.items() if k != "rejected"} == {k: v for k, v in after.items() if k != "rejected"}


def run_checks(game: Path, output: Path, library_path: str) -> None:
    output.mkdir(parents=True, exist_ok=False)
    native = Native(game, output / "controls", library_path=library_path)
    pairs = []
    try:
        state = native.start()
        initial = state
        time.sleep(0.15)
        assert state == native.state(), "paused update or draw advanced"
        moved = native.step(state, "right", 12)
        assert moved["player"]["x"] > state["player"]["x"]
        pairs.append({"action": "right", "before": state, "after": moved})
        native.send_raw(f"reset {moved['id']} 0 0\n".encode())
        state = native.wait(moved["id"])
        for key in ("player", "room", "deaths", "clockFrame", "tiles", "entities", "freezeTicks"):
            assert state[key] == initial[key], key
        assert state["ticks"] == moved["ticks"] and state["episodeTicks"] == 0
        for action in ACTIONS:
            native.send_raw(f"reset {state['id']} 0 0\n".encode())
            state = native.wait(state["id"])
            before = state
            state = native.step(before, action, 1)
            player = state["player"]
            if action.startswith("dash"):
                assert player["dashes"] == 0 and state["freezeTicks"] == 2, action
                if "left" in action:
                    assert player["vx"] < 0, action
                if "right" in action:
                    assert player["vx"] > 0, action
                if "up" in action:
                    assert player["vy"] < 0, action
                if "down" in action:
                    assert player["vy"] > 0, action
            elif action.startswith("jump"):
                assert player["vy"] < 0, action
            elif action == "left":
                assert player["vx"] < 0
            elif action == "right":
                assert player["vx"] > 0
            elif action == "wait":
                assert player["vx"] == player["vy"] == 0
            pairs.append({"action": action, "before": before, "after": state})
        bad = [
            b"junk",
            b"step -1 2 12",
            b"step 9999999999999999999999999999999 0 1",
            f"step {state['id'] - 1} 2 12".encode(),
            f"step {state['id'] + 1} 2 12".encode(),
            f"step {state['id']} 3 12".encode(),
            f"step {state['id']} 12 12".encode(),
            f"step {state['id']} 2 0".encode(),
            f"step {state['id']} 2 31".encode(),
            f"stop {state['id']} 2 1".encode(),
            f"step {state['id']} 2 12 extra".encode(),
            b"x" * 300,
            f"step {state['id']} 2 12\0".encode(),
        ]
        for command in bad:
            before = state
            native.send_raw(command)
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and native.state()["rejected"] == before["rejected"]:
                time.sleep(0.01)
            state = native.state()
            assert state["rejected"] == before["rejected"] + 1
            unchanged(before, state)
        before = state

        class BadModel:
            def decide(self, request: dict, *, raw_path: Path) -> dict:
                response = {"id": request["id"], "stateText": "missing required current state"}
                raw_path.write_text(json.dumps(response))
                return response

        try:
            decide_and_step(native, BadModel(), state, 12)
        except ValueError:
            pass
        else:
            raise AssertionError("bad model fixture reached native action")
        time.sleep(0.1)
        assert native.state() == before
        assert request_for(state, 12)["id"] == before["id"]
        write_json(
            output / "controls.json",
            {
                "fixedActions": len(pairs),
                "badNativeCommands": len(bad),
                "pausedStable": True,
                "resetSnapshotRestored": True,
                "badModelZeroActions": True,
                "final": native.state(),
                "pairs": pairs,
            },
        )
    finally:
        native.close()
    flood = Native(game, output / "watchdog-flood", library_path=library_path, watchdog=2)
    try:
        initial = flood.start()
        start = time.monotonic()
        while flood.player.poll() is None and time.monotonic() - start < 6:
            flood.send_raw(b"step 0 2 12\n")
            time.sleep(0.01)
        assert flood.player.poll() is not None, "invalid commands bypassed watchdog"
        final = flood.state()
        assert final["status"] == "watchdog" and final["ticks"] == 0
        assert final["nativeUpdates"] == initial["nativeUpdates"] == final["nativeDraws"]
        write_json(output / "watchdog.json", {"seconds": time.monotonic() - start, "initial": initial, "final": final})
    finally:
        flood.close()
    print(json.dumps({"controls": "pass", "watchdogFlood": "pass", "output": str(output)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library-path", default="")
    args = parser.parse_args()
    run_checks(args.game, args.output, args.library_path)


if __name__ == "__main__":
    main()
