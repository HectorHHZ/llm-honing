"""Expose Craftax's visibility-masked state to a headless JSON Lines controller."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from integrations.craftax.catalog import ACHIEVEMENTS, ACTIONS, DEFAULT_GOALS
from integrations.craftax.engine import CraftaxEngine


@dataclass(frozen=True)
class _Settings:
    seed: int
    max_steps: int
    goals: tuple[str, ...]


class AgentSession:
    """Start a game and reuse its latest public frame between actions."""

    def __init__(self, engine: CraftaxEngine) -> None:
        self._engine = engine
        self._frame = engine.reset()

    def observe_for_ai(self) -> dict[str, Any]:
        """Read a detached view of the local map and character state."""
        frame = self._frame
        public = frame["observation"]
        tiles = public["tiles"]
        y, x = public["player_cell"]
        done = frame["status"] != "running"
        scenario = self._engine.scenario
        return deepcopy(
            {
                "schema_version": 1,
                "game": "craftax",
                "status": frame["status"],
                "done": done,
                "clock": {"unit": "turn", "value": frame["step"], "limit": scenario.max_steps},
                "map": {
                    "kind": "tile_grid",
                    "width": len(tiles[0]),
                    "height": len(tiles),
                    "tiles": tiles,
                    "items": public["items"],
                },
                "player": {
                    **public["player"],
                    "x": x,
                    "y": y,
                    "sleeping": public["sleeping"],
                    "resting": public["resting"],
                },
                "inventory": public["inventory"],
                "equipment": public["equipment"],
                "progress": self._engine.public_progress(),
                "entities": [
                    {
                        "x": entity["column"],
                        "y": entity["row"],
                        "class": public["entity_classes"][entity["class"]],
                        "type": entity["type"],
                    }
                    for entity in public["entities"]
                ],
                "goals": list(scenario.goals),
                "achievements": frame["achievements"],
                "reward": frame["reward"],
                "terminated": frame["terminated"],
                "truncated": frame["truncated"],
                "action_space": None if done else {"kind": "enum", "choices": list(ACTIONS)},
            }
        )

    def step_for_ai(self, action: str) -> dict[str, Any]:
        """Apply one native action and return the next observation."""
        if self._frame["status"] != "running":
            raise ValueError("the game has ended")
        self._frame = self._engine.step(action)
        return self.observe_for_ai()


def _serve(session: AgentSession) -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            if not isinstance(request, dict) or request.get("op") not in ("observe", "step"):
                raise ValueError("expected an object with op 'observe' or 'step'")
            if request["op"] == "observe":
                observation = session.observe_for_ai()
            else:
                action = request.get("action")
                if not isinstance(action, str):
                    raise ValueError("expected a Craftax action name")
                observation = session.step_for_ai(action)
        except ValueError as error:
            reply = {"ok": False, "error": str(error)}
        else:
            reply = {"ok": True, "observation": observation}
        print(json.dumps(reply, allow_nan=False), flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--goals", default=",".join(DEFAULT_GOALS), metavar="NAME[,NAME...]")
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be a uint32 integer")
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    goals = tuple(args.goals.split(","))
    if len(set(goals)) != len(goals) or not set(goals) <= ACHIEVEMENTS:
        parser.error("--goals must contain distinct Craftax achievement names")

    settings = _Settings(args.seed, args.max_steps, goals)
    with CraftaxEngine(settings) as engine:
        _serve(AgentSession(engine))


if __name__ == "__main__":
    main()
