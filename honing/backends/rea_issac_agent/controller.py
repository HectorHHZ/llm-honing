"""REA-Issac controller that gets a model into combat using public observations."""

from __future__ import annotations

import math
from typing import Any

from integrations.rea_issac.policy import Baseline

NAVIGATE_EMPTY_ROOMS = "navigate-empty-rooms-v1"
MODEL_ASSISTED_COMBAT = "model-assisted-combat-v1"
MODEL_ASSISTED_COMBAT_V2 = "model-assisted-combat-v2"
MODEL_ASSISTED_COMBAT_V3 = "model-assisted-combat-v3"
CONTROLLER_MODES = (NAVIGATE_EMPTY_ROOMS, MODEL_ASSISTED_COMBAT, MODEL_ASSISTED_COMBAT_V2, MODEL_ASSISTED_COMBAT_V3)


def _similar_direction(model: tuple[int, int], rule: tuple[int, int], minimum_cosine: float) -> bool:
    if model == (0, 0) or rule == (0, 0):
        return model == rule
    cosine = (model[0] * rule[0] + model[1] * rule[1]) / (math.hypot(*model) * math.hypot(*rule))
    return cosine >= minimum_cosine - 1e-9


def _door_entry_nudge(observation: dict[str, Any], action: dict[str, Any]) -> dict[str, Any] | None:
    """Take a smaller diagonal step when a cardinal step skips a door's narrow trigger band."""
    if not observation["room"]["cleared"]:
        return None
    open_doors = {door["direction"] for door in observation["doors"] if door["open"]}
    player = observation["player"]
    x, y = player["x"], player["y"]
    width, height = observation["width"], observation["height"]
    center_x, center_y = width / 2, height / 2
    step = player["speed"] / observation["fps"]
    diagonal = step / math.sqrt(2)
    radius = player["radius"]
    move = (action["move_x"], action["move_y"])
    if (
        move == (1, 0)
        and "east" in open_doors
        and abs(y - center_y) < 28
        and x < width - 46 <= x + diagonal <= width - 32 - radius < x + step
    ):
        return {**action, "move_y": -1 if y >= center_y else 1}
    if (
        move == (-1, 0)
        and "west" in open_doors
        and abs(y - center_y) < 28
        and x > 46 >= x - diagonal >= 32 + radius > x - step
    ):
        return {**action, "move_y": -1 if y >= center_y else 1}
    if (
        move == (0, -1)
        and "north" in open_doors
        and abs(x - center_x) < 28
        and y > 54 >= y - diagonal >= 40 + radius > y - step
    ):
        return {**action, "move_x": -1 if x >= center_x else 1}
    if (
        move == (0, 1)
        and "south" in open_doors
        and abs(x - center_x) < 28
        and y < height - 54 <= y + diagonal <= height - 40 - radius < y + step
    ):
        return {**action, "move_x": -1 if x >= center_x else 1}
    return None


def _cleanup_alignment(observation: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any] | None:
    """Keep aligning with a distant, one-shot enemy when the rule reverses into the side wall."""
    room, player = observation["room"], observation["player"]
    enemies = room["enemies"]
    if room["kind"] != "combat" or len(enemies) != 1:
        return None
    enemy = enemies[0]
    dx, dy = enemy["x"] - player["x"], enemy["y"] - player["y"]
    near_side_wall = player["x"] < 80 or player["x"] > observation["width"] - 80
    toward_y = 1 if dy > 0 else -1
    if (
        enemy["health"] <= player["damage"]
        and near_side_wall
        and abs(dx) > 80
        and abs(dy) > 5
        and rule["move_x"] == 0
        and rule["move_y"] == -toward_y
    ):
        return {**rule, "move_y": toward_y}
    return None


class AgentController:
    """Combine NanoJev choices with the existing public-state rule baseline.

    Both modes query the model at every decision. The assisted mode accepts
    model movement and firing only when their directions are close to the
    rule suggestion; the recorded reason shows which choices were accepted.
    """

    def __init__(self, mode: str = NAVIGATE_EMPTY_ROOMS) -> None:
        if mode not in CONTROLLER_MODES:
            raise ValueError(f"unsupported REA-Issac agent controller: {mode}")
        self.mode = mode
        self.navigator = Baseline()

    def apply(self, observation: dict[str, Any], selected: dict[str, Any]) -> tuple[dict[str, Any], str]:
        if observation["room"]["enemies"] and self.mode == NAVIGATE_EMPTY_ROOMS:
            return selected, "model_combat"
        rule = self.navigator.act(observation)
        if not observation["room"]["enemies"]:
            if self.mode in (MODEL_ASSISTED_COMBAT_V2, MODEL_ASSISTED_COMBAT_V3) and (
                nudge := _door_entry_nudge(observation, rule)
            ):
                return nudge, "door_entry_nudge"
            return rule, "empty_room_navigation"
        if self.mode == MODEL_ASSISTED_COMBAT_V3 and (alignment := _cleanup_alignment(observation, rule)):
            return alignment, "cleanup_alignment"
        action = rule.copy()
        model_move = (selected["move_x"], selected["move_y"])
        rule_move = (rule["move_x"], rule["move_y"])
        use_move = _similar_direction(model_move, rule_move, 0.7)
        if use_move:
            action["move_x"], action["move_y"] = model_move
        model_shoot = (selected["shoot_x"], selected["shoot_y"])
        rule_shoot = (rule["shoot_x"], rule["shoot_y"])
        use_shoot = model_shoot != (0, 0) and _similar_direction(model_shoot, rule_shoot, 0.7)
        if use_shoot:
            action["shoot_x"], action["shoot_y"] = model_shoot
        return action, f"assisted_combat_move_{int(use_move)}_shoot_{int(use_shoot)}"
