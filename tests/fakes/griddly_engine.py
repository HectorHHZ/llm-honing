"""A deterministic stand-in for the native Spiders engine, so matches, audits and replays run without Griddly.

:class:`FakeSpidersEngine` has :class:`integrations.griddly.engine.GriddlyEngine`'s
interface and frame layout, frame hash included, on a fixed 4x4 map: the gnome
starts at (0, 3) facing UP (``NONE``, as Griddly starts it), the gem is at
(3, 0) and a spider sits still at (3, 3). FORWARD and BACKWARD move one cell
along the facing when that cell is on the map, the turns rotate, WAIT does
nothing. Reaching the gem wins with reward 1; reaching the spider loses with
reward -1. On level 1 the gnome is stunned every third tick and can only WAIT,
which stages a forced action. It is never evidence of the real rules.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from integrations.griddly.catalog import ACTIONS, GAME_SHA256
from integrations.griddly.engine import EngineScenario, canonical, digest

HEADINGS = ("UP", "RIGHT", "DOWN", "LEFT")
STEPS = {"UP": (0, -1), "RIGHT": (1, 0), "DOWN": (0, 1), "LEFT": (-1, 0)}
SIZE = 4
GEM = (3, 0)
SPIDER = (3, 3)


def fake_versions() -> dict[str, str]:
    return {"griddly": "stand-in", "python": "stand-in"}


class FakeSpidersEngine:
    #: Every engine constructed, newest last, for a test to inspect.
    opened: ClassVar[list[FakeSpidersEngine]] = []

    def __init__(self, scenario: EngineScenario, *, render: bool = False) -> None:
        self.scenario = scenario
        self.game_sha256 = GAME_SHA256
        self.render = render
        self.closed = False
        self.started = False
        self.ended = False
        self.steps = 0
        self.gnome: tuple[int, int] | None = (0, 3)
        self.heading = "NONE"
        self.gem = True
        FakeSpidersEngine.opened.append(self)

    def __enter__(self) -> FakeSpidersEngine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.closed = True

    def reset(self) -> dict[str, Any]:
        if self.closed:
            raise RuntimeError("the engine is closed")
        self.steps, self.gnome, self.heading, self.gem = 0, (0, 3), "NONE", True
        self.started, self.ended = True, False
        return self._frame(0.0, False)

    def _target(self, action: str) -> tuple[int, int] | None:
        assert self.gnome is not None
        dx, dy = STEPS["UP" if self.heading == "NONE" else self.heading]
        sign = {"FORWARD": 1, "BACKWARD": -1}.get(action)
        if sign is None:
            return None
        x, y = self.gnome[0] + sign * dx, self.gnome[1] + sign * dy
        return (x, y) if 0 <= x < SIZE and 0 <= y < SIZE else None

    def _legal(self) -> list[str]:
        if self.scenario.level_id == 1 and self.steps % 3 == 2:
            return ["WAIT"]
        moves = {"FORWARD", "BACKWARD"}
        return [name for name in ACTIONS if name not in moves or self._target(name) is not None]

    def _frame(self, reward: float, done: bool) -> dict[str, Any]:
        objects = [{"Name": "gem", "Location": list(GEM), "Orientation": "NONE", "PlayerId": 0, "Variables": {}}]
        objects = objects if self.gem else []
        objects.append({"Name": "spider", "Location": list(SPIDER), "Orientation": "LEFT", "PlayerId": 0})
        if self.gnome is not None:
            objects.append({"Name": "gnome", "Location": list(self.gnome), "Orientation": self.heading, "PlayerId": 1})
        objects.sort(key=canonical)
        alive = self.gnome is not None
        status = (
            "won"
            if done and alive and not self.gem
            else "lost" if not alive else "step_limit" if self.steps >= self.scenario.max_steps else "running"
        )
        self.ended = status != "running"
        native = {"Grid": {"Width": SIZE, "Height": SIZE}, "GameTicks": self.steps, "Objects": objects}
        frame = {
            "step": self.steps,
            "observation": {
                "grid": native["Grid"],
                "objects": objects,
                "tick": self.steps,
                "npc_period_ticks": 3,
                "coordinates": "[x,y]; right/down positive; orientation is absolute; actions are relative",
            },
            "legal_actions": [] if self.ended else self._legal(),
            "reward": reward,
            "status": status,
            "terminated": done,
            "truncated": status == "step_limit",
            "native_state": native,
            "events": [],
        }
        frame["frame_hash"] = digest(frame)
        return json.loads(canonical(frame))

    def step(self, action: str) -> dict[str, Any]:
        if self.closed or not self.started or self.ended:
            raise RuntimeError("reset an open, nonterminal engine before stepping")
        if action not in self._legal():
            raise ValueError(f"action is not currently legal: {action}")
        self.steps += 1
        heading = HEADINGS.index("UP" if self.heading == "NONE" else self.heading)
        if action in ("TURN_LEFT", "TURN_RIGHT"):
            self.heading = HEADINGS[(heading + (1 if action == "TURN_RIGHT" else -1)) % 4]
        target = self._target(action)
        if target is not None:
            self.gnome = target
        if self.gnome == GEM:
            self.gem = False
            return self._frame(1.0, True)
        if self.gnome == SPIDER:
            self.gnome = None
            return self._frame(-1.0, True)
        return self._frame(0.0, False)

    def pixels(self) -> list[list[int]]:
        if not self.render or self.closed or not self.started:
            raise RuntimeError("vector rendering must be enabled before reset")
        cells = [[0] * SIZE for _ in range(SIZE)]
        if self.gnome is not None:
            cells[self.gnome[1]][self.gnome[0]] = 1
        return cells
