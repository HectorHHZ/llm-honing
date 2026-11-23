"""Check the AI observation against the public engine state and wire protocol."""

import json
import subprocess
import sys

import pytest

from integrations.rea_issac.agent import observe_for_ai, step_for_ai
from integrations.rea_issac.engine import Engine


@pytest.mark.unit
def test_ai_observation_uses_public_room_data() -> None:
    engine = Engine(7)
    before = engine.digest()
    public = engine.observe()
    state = observe_for_ai(engine)
    assert state == observe_for_ai(engine)
    assert "seed" not in state
    assert state["map"]["known_rooms"] == public["map"]
    assert len(state["map"]["known_rooms"]) < public["total_rooms"]
    assert state["map"]["obstacles"] == public["room"]["rocks"]
    assert state["map"]["doors"] == public["doors"]
    assert state["player"] == public["player"]
    state["map"]["known_rooms"].clear()
    state["player"]["health"] = 0
    assert engine.digest() == before

    result = step_for_ai(engine, {"move_x": 1, "shoot_x": 1, "bomb": True}, frames=3)
    assert result["clock"]["value"] == 3
    assert result["player"]["x"] > public["player"]["x"]
    assert result["projectiles"] == engine.observe()["bullets"]
    assert result["bombs"] == engine.observe()["bombs"]
    assert len(result["projectiles"]) == len(result["bombs"]) == 1
    assert result["action_space"]["fire_ready"] is False
    assert result["action_space"]["bomb_ready"] is False


@pytest.mark.integration
def test_agent_cli_returns_one_snapshot_per_valid_request() -> None:
    requests = (
        '{"op":"observe"}',
        "{",
        '{"op":"step","action":{"move_x":2}}',
        '{"op":"step","action":{},"frames":301}',
        '{"op":"observe"}',
        '{"op":"step","action":{"move_x":1},"frames":30}',
        '{"op":"step","action":{}}',
        '{"op":"observe"}',
    )
    process = subprocess.run(
        [sys.executable, "-m", "integrations.rea_issac.agent", "--max-ticks", "3"],
        input="\n".join(requests) + "\n",
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
    )
    start, bad_json, bad_action, bad_frames, again, ended, after, final = map(json.loads, process.stdout.splitlines())
    assert start["ok"] is True
    assert start["observation"]["schema_version"] == 1
    assert start["observation"]["game"] == "rea-issac"
    assert all("seed" not in reply["observation"] for reply in (start, again, ended, final))
    assert all(reply["ok"] is False for reply in (bad_json, bad_action, bad_frames, after))
    assert start == again
    assert ended == final
    assert ended["observation"]["clock"] == {"unit": "tick", "value": 3, "ticks_per_second": 30, "limit": 3}
    assert ended["observation"]["status"] == "truncated"
    assert ended["observation"]["done"] is True
    assert ended["observation"]["action_space"] is None
