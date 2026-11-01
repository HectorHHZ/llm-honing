"""A deterministic stand-in used only to check Craftax integration contracts."""

from __future__ import annotations

from typing import Any

from honing.backends.craftax.engine import frame_hash
from honing.backends.craftax.scenarios import CraftaxScenario


class FakeCraftaxEngine:
    def __init__(self, scenario: CraftaxScenario) -> None:
        self.scenario = scenario
        self.step_count = 0
        self.achievements: list[str] = []

    def __enter__(self) -> FakeCraftaxEngine:
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def reset(self) -> dict[str, Any]:
        self.step_count, self.achievements = 0, []
        return self.frame(0.0)

    def frame(self, reward: float) -> dict[str, Any]:
        complete = set(self.scenario.goals) <= set(self.achievements)
        status = (
            "goal_complete" if complete else "step_limit" if self.step_count >= self.scenario.max_steps else "running"
        )
        frame = {
            "step": self.step_count,
            "observation": {
                "player": {"health": 9, "level": 0},
                "tiles": [[None, "TREE"], ["GRASS", "GRASS"]],
                "inventory": {"wood": len(self.achievements)},
            },
            "achievements": sorted(self.achievements),
            "reward": reward,
            "status": status,
            "terminated": False,
            "truncated": status != "running",
            "native_state_hash": "stand-in",
        }
        frame["frame_hash"] = frame_hash(frame)
        return frame

    def step(self, action: str) -> dict[str, Any]:
        self.step_count += 1
        if action == "DO" and len(self.achievements) < len(self.scenario.goals):
            self.achievements.append(self.scenario.goals[len(self.achievements)])
            return self.frame(1.0)
        return self.frame(0.0)


def fake_versions() -> dict[str, str]:
    return {"engine": "stand-in"}
