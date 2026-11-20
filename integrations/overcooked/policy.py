"""Public-state rule baselines. No native planner or future simulator is consulted."""

from __future__ import annotations

from collections import deque
from typing import Any

from integrations.overcooked.engine import ACTIONS


def park(frame: dict[str, Any], player: int) -> str:
    """Return to a starting square while waiting, so the pot approach stays clear."""
    players = frame["state"]["players"]
    start = tuple(players[player]["position"])
    goal = tuple(frame["map"]["start_positions"][player])
    floor = {tuple(p) for p in frame["map"]["walkable"]} - {tuple(players[1 - player]["position"])}
    queue = deque([(start, "STAY")])
    seen = {start}
    while queue:
        pos, first = queue.popleft()
        if pos == goal:
            if player == 1 and first != "STAY":
                dx, dy = ACTIONS[first]
                nxt = (start[0] + dx, start[1] + dy)
                peer = players[0]["position"]
                if abs(nxt[0] - peer[0]) + abs(nxt[1] - peer[1]) <= 1:
                    return "STAY"
            return first
        for name in ("NORTH", "SOUTH", "EAST", "WEST"):
            dx, dy = ACTIONS[name]
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt in floor and nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, name if pos == start else first))
    return "STAY"


def toward(frame: dict[str, Any], player: int, targets: list[tuple[int, int]]) -> str | None:
    """Shortest path to a floor square facing a target; current teammate blocks paths."""
    players = frame["state"]["players"]
    own = players[player]
    start = tuple(own["position"])
    blocked = tuple(players[1 - player]["position"])
    floor = {tuple(p) for p in frame["map"]["walkable"]} - {blocked}
    queue = deque([(start, None)])
    seen = {start}
    directions = [(name, delta) for name, delta in ACTIONS.items() if name not in ("STAY", "INTERACT")]
    while queue:
        pos, first = queue.popleft()
        for target in targets:
            delta = (target[0] - pos[0], target[1] - pos[1])
            for name, vector in directions:
                if delta == vector:
                    if first is not None:
                        return first
                    return "INTERACT" if tuple(own["orientation"]) == vector else name
        for name, (dx, dy) in directions:
            nxt = (pos[0] + dx, pos[1] + dy)
            if nxt in floor and nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, first or name))
    return None


def baseline(frame: dict[str, Any], player: int, style: str = "role") -> str:
    """Onion-soup baseline; roles help on simple kitchens, not a general solved partner."""
    if style == "stay" or frame["status"] != "running":
        return "STAY"
    if style == "solo_slow":
        return baseline(frame, player, "solo") if frame["step"] % 3 == 0 else "STAY"
    state = frame["state"]
    own, other = state["players"][player], state["players"][1 - player]
    held = own["held_object"]
    name = held["name"] if held else None
    terrain = frame["map"]["terrain"]
    objects = {tuple(obj["position"]): obj for obj in state["objects"]}

    def locations(tile: str) -> list[tuple[int, int]]:
        return [(x, y) for y, row in enumerate(terrain) for x, v in enumerate(row) if v == tile]

    pots = locations("P")
    ready = [p for p in pots if objects.get(p, {}).get("is_ready")]
    idle = [p for p in pots if objects.get(p, {}).get("is_idle")]
    fillable = [
        p
        for p in pots
        if p not in objects
        or (objects[p].get("is_idle") and len(objects[p]["_ingredients"]) < frame["rules"]["max_ingredients"])
    ]
    cookable = [p for p in idle if len(objects[p]["_ingredients"]) >= frame["rules"]["max_ingredients"]]
    active_pot = any(objects.get(p, {}).get("is_cooking") for p in pots)
    if style == "role" and player == 1 and not held and not ready and not active_pot and not cookable:
        return park(frame, player)
    targets: list[tuple[int, int]] = []
    if name == "soup":
        targets = locations("S")
    elif name in ("onion", "tomato"):
        targets = fillable
    elif name == "dish":
        targets = ready
        if not ready and active_pot:
            return park(frame, player)
    elif cookable and not frame["rules"]["old_dynamics"]:
        targets = cookable
    else:
        other_name = other["held_object"]["name"] if other["held_object"] else None
        # Player 0 supplies ingredients; player 1 serves when a pot is active.
        should_serve = bool(ready) or (
            style == "role" and player == 1 and any(objects.get(p, {}).get("is_cooking") for p in pots)
        )
        if should_serve and other_name != "dish":
            counters = [p for p, obj in objects.items() if obj["name"] == "dish" and terrain[p[1]][p[0]] == "X"]
            targets = counters or locations("D")
        elif fillable:
            counters = [p for p, obj in objects.items() if obj["name"] == "onion" and terrain[p[1]][p[0]] == "X"]
            targets = counters or locations("O")
    action = toward(frame, player, targets)
    if action:
        return action
    # Put an unwanted held object down if no useful target is reachable.
    if held:
        free = [p for p in locations("X") if p not in objects]
        action = toward(frame, player, free)
        if action:
            return action
    if style == "role" and player == 0 and not held and active_pot:
        action = toward(frame, player, locations("O"))
        return action if action and action != "INTERACT" else "STAY"
    return "STAY"


def joint_baseline(frame: dict[str, Any], controlled_player: int = 0, partner: str = "role") -> list[str]:
    return [baseline(frame, i, "role" if i == controlled_player else partner) for i in range(2)]
