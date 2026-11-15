"""Exercise Brogue's input boundaries and native recording contract."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from integrations.brogue_ce.engine import BrogueSession, Observation, _Display


def test_ai_observation_uses_map_coordinates() -> None:
    rows = [""] * 34
    rows[0] = "@: You (lit)         You hear a sound."
    rows[1] = "       Health"
    rows[3] = " " * 21 + " #@"
    rows[33] = "  -- Depth: 2 --        Explore        Rest (z)       Search (s)        Menu         Inventory"
    state = _Display().observe_for_ai(Observation(tuple(rows), 3, 15), seed=2**64 - 1, keys=2)
    assert state["seed"] == "18446744073709551615"
    assert state["depth"] == 2
    assert state["deepest_level"] == 3
    assert state["map"]["rows"][0] == " #@".ljust(79)
    assert len(state["map"]["rows"]) == 29
    assert state["player"]["position"] == {"x": 2, "y": 0}
    assert state["messages"] == ["You hear a sound."]


@pytest.mark.integration
def test_native_recording(tmp_path: Path) -> None:
    engine_path = os.environ.get("BROGUE_CE_ENGINE")
    if not engine_path:
        pytest.skip("set BROGUE_CE_ENGINE to the pinned WebBrogue executable")
    engine = Path(engine_path).resolve()
    recording = tmp_path / "match.broguerec"
    with BrogueSession(engine, 7) as first, BrogueSession(engine, 7) as second:
        assert first.observe() == second.observe()
        for key in ("x", "x", "x", "Q", "y"):
            assert first.step(key) == second.step(key)
        assert first.observe().outcome == "quit"
        first.save_recording(recording)
        with pytest.raises(FileExistsError):
            first.save_recording(recording)
    subprocess.run([engine, "-vn", recording], cwd=tmp_path, check=True, timeout=15)


@pytest.mark.integration
def test_agent_observes_native_play_and_prompts() -> None:
    engine_path = os.environ.get("BROGUE_CE_ENGINE")
    if not engine_path:
        pytest.skip("set BROGUE_CE_ENGINE to the pinned WebBrogue executable")
    requests = [
        {"op": "observe"},
        {"op": "step", "action": "invalid key"},
        {"op": "observe"},
        {"op": "step", "action": "h"},
        {"op": "step", "action": "i"},
        {"op": "step", "action": "\x1b"},
        {"op": "step", "action": "Q"},
        {"op": "step", "action": "y"},
        {"op": "step", "action": "h"},
    ]
    result = subprocess.run(
        [sys.executable, "-m", "integrations.brogue_ce", "agent", "--engine", engine_path],
        input="".join(json.dumps(request) + "\n" for request in requests),
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    start, invalid, again, move, inventory, closed, confirm, quit_game, after = map(
        json.loads, result.stdout.splitlines()
    )
    assert start == again
    assert invalid["ok"] is False
    initial = start["observation"]
    position = initial["player"]["position"]
    assert initial["map"]["rows"][position["y"]][position["x"]] == "@"
    assert len(initial["player"]["bars"]["health"]["background"]) == 20
    assert move["observation"]["clock"] == {"unit": "key", "value": 1}
    assert move["observation"]["player"]["position"] == {"x": position["x"] - 1, "y": position["y"]}
    assert inventory["observation"]["status"] == "prompt"
    assert inventory["observation"]["map"] is None
    assert any("dagger" in row.lower() for row in inventory["observation"]["display"])
    assert closed["observation"]["status"] == "playing"
    assert confirm["observation"]["status"] == "prompt"
    assert quit_game["observation"]["outcome"] == "quit"
    assert quit_game["observation"]["action_space"] is None
    assert after["ok"] is False
