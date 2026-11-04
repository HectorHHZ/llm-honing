"""Check the supplied native bridge with explicit test inputs, without a decision model."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from run import ACTIONS, atomic_json, executable, wait_observation


@contextmanager
def native_player(game: Path, output: Path, *, watchdog: float = 10):
    """Start and clean up only the player owned by this verification case."""
    output.mkdir()
    environment = {
        **os.environ,
        "HONING_SESSION_DIR": str(output.resolve()),
        "HONING_SCENE": "Spawn",
        "HONING_CAPTURE": "0",
        "HONING_MODEL_TIMEOUT_SECONDS": str(watchdog),
    }
    with (output / "stdout.log").open("w") as log:
        process = subprocess.Popen(
            [str(game), "-batchmode", "-nographics", "-logFile", str((output / "player.log").resolve())],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            yield process
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            atomic_json(output / "process.json", {"pid": process.pid, "exited": True, "exitCode": process.returncode})


def verify(game: Path, output: Path) -> dict:
    """Verify frame advancement, two-way movement, command rejection and native watchdog exit."""
    game = executable(game)
    output.mkdir(parents=True, exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix="native-", dir=output)).resolve()
    sequence = [
        "wait",
        "right",
        "left",
        "jump_right",
        "attack_left",
        "dash_right",
        "attack_up",
        "attack_down",
        "jump_left",
        "dash_left",
        "attack_right",
    ]
    assert set(sequence) == set(ACTIONS)
    control = session / "control"
    steps = []
    with native_player(game, control) as process:
        before = wait_observation(control, 0, process, 30)
        for action in sequence:
            command = {"id": before["id"], "action": action, "frames": 12, "model": "SCRIPTED BRIDGE TEST"}
            atomic_json(control / "action.json", command)
            after = wait_observation(control, before["id"], process, 10)
            assert after["id"] == before["id"] + 1
            assert after["frame"] - before["frame"] == 12
            assert after["fixedSteps"] > before["fixedSteps"]
            assert after["simulationTime"] > before["simulationTime"]
            if action == "right":
                assert after["x"] > before["x"]
            if action == "left":
                assert after["x"] < before["x"]
            # Leave the already-consumed command in place. It must not execute twice.
            time.sleep(0.15)
            assert json.loads((control / "observation.json").read_text()) == after
            steps.append({"command": command, "before": before, "after": after})
            before = after
        atomic_json(control / "action.json", {"id": before["id"], "action": "stop"})
        assert process.wait(timeout=10) == 0
    atomic_json(control / "steps.json", {"modelUsed": False, "steps": steps})

    cases = {
        "stale": {"id": 0, "action": "left", "frames": 12},
        "future": {"id": 999, "action": "left", "frames": 12},
        "missing_frames": {"id": 1, "action": "left"},
        "zero_frames": {"id": 1, "action": "left", "frames": 0},
        "too_many_frames": {"id": 1, "action": "left", "frames": 121},
        "unknown_action": {"id": 1, "action": "teleport", "frames": 12},
        "malformed_json": None,
    }
    rejected = []
    for name, command in cases.items():
        case = session / name
        with native_player(game, case, watchdog=0.5) as process:
            first = wait_observation(case, 0, process, 30)
            assert first["id"] == 1
            if command is None:
                (case / "action.json").write_text("{broken JSON")
            else:
                atomic_json(case / "action.json", command)
            process.wait(timeout=5)
            assert json.loads((case / "observation.json").read_text()) == first
            assert "timed out waiting for a model response" in (case / "player.log").read_text()
            rejected.append({"case": name, "command": command, "framesAdvanced": 0, "watchdogExited": True})
    result = {
        "kind": "scripted-native-bridge-verification",
        "modelUsed": False,
        "scene": "Spawn",
        "framesPerAction": 12,
        "actions": sequence,
        "renderFramesAdvanced": steps[-1]["after"]["frame"],
        "allPhysicsAndTimeAdvanced": True,
        "movementBothDirections": True,
        "duplicateCommandsDidNotAdvance": True,
        "rejections": rejected,
        "ownedPlayersExited": True,
    }
    atomic_json(session / "result.json", result)
    print(session)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("work/hollow-knight/native-checks"))
    args = parser.parse_args()
    print(json.dumps(verify(args.game, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
