"""Expose REA-Issac's public game state to a headless JSON Lines controller."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from integrations.rea_issac.engine import Action, Engine


def _observation(engine: Engine, public: dict[str, Any]) -> dict[str, Any]:
    room, player, tick = public["room"], public["player"], public["tick"]
    done = public["status"] != "running"
    action_space = None
    if not done:
        fields: dict[str, dict[str, Any]] = {
            name: {"type": "integer", "enum": [-1, 0, 1]} for name in ("move_x", "move_y", "shoot_x", "shoot_y")
        }
        fields["bomb"] = {"type": "boolean"}
        action_space = {
            "kind": "tick_input",
            "fields": fields,
            "frames": {"type": "integer", "minimum": 1, "maximum": 300, "default": 1},
            "fire_ready": tick + 1 >= player["next_shot"],
            "bomb_ready": player["bombs"] > 0 and not engine.bomb_latch,
        }
    return {
        "schema_version": 1,
        "game": "rea-issac",
        "status": public["status"],
        "done": done,
        "clock": {"unit": "tick", "value": tick, "ticks_per_second": public["fps"], "limit": public["max_ticks"]},
        "map": {
            "kind": "room_layout",
            "units": "pixels",
            "width": public["width"],
            "height": public["height"],
            "current_room": {name: room[name] for name in ("id", "position", "name", "kind", "cleared")},
            "known_rooms": public["map"],
            "obstacles": room["rocks"],
            "doors": public["doors"],
        },
        "player": player,
        "enemies": room["enemies"],
        "projectiles": public["bullets"],
        "bombs": public["bombs"],
        "pickups": room["pickups"],
        "events": public["events"],
        "metrics": public["metrics"],
        "action_space": action_space,
    }


def observe_for_ai(engine: Engine) -> dict[str, Any]:
    """Read a detached observation of the current room and discovered map."""
    return _observation(engine, engine.observe())


def step_for_ai(engine: Engine, action: dict[str, Any] | Action, frames: int = 1) -> dict[str, Any]:
    """Apply the native action for up to `frames` ticks and read the new observation."""
    if engine.status != "running":
        raise ValueError("the game has ended")
    return _observation(engine, engine.step(action, frames))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-ticks", type=int, default=18000)
    args = parser.parse_args(argv)
    try:
        engine = Engine(args.seed, args.max_ticks)
    except ValueError as error:
        parser.error(str(error))

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            if not isinstance(request, dict) or request.get("op") not in ("observe", "step"):
                raise ValueError("expected an object with op 'observe' or 'step'")
            if request["op"] == "observe":
                observation = observe_for_ai(engine)
            else:
                action = request.get("action")
                if not isinstance(action, dict):
                    raise ValueError("expected an action object")
                observation = step_for_ai(engine, action, request.get("frames", 1))
        except ValueError as error:
            reply = {"ok": False, "error": str(error)}
        else:
            reply = {"ok": True, "observation": observation}
        print(json.dumps(reply, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
