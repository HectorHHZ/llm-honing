"""Experiment-only public-state wording; no planner, simulator or action selection."""

from __future__ import annotations

from typing import Any

from integrations.overcooked.engine import ACTIONS

FACING = {(0, -1): "north", (0, 1): "south", (1, 0): "east", (-1, 0): "west"}


def object_text(obj: dict[str, Any] | None) -> str:
    if obj is None:
        return "empty hands"
    if obj["name"] != "soup":
        return obj["name"]
    ingredients = [item["name"] for item in obj.get("_ingredients", [])]
    status = "ready" if obj.get("is_ready") else "cooking" if obj.get("is_cooking") else "idle"
    return f"{status} soup, ingredients={ingredients}, cooking tick={obj.get('_cooking_tick')}/{obj.get('cook_time')}"


def readable_context(frame: dict[str, Any], player: int, history: list[dict[str, Any]], notes: str) -> str:
    kitchen, state, rules = frame["map"], frame["state"], frame["rules"]
    lines = [
        f"You control chef {player}. Maximize shared soup delivery points in {frame['max_steps']} ticks.",
        "Your fixed teammate independently collects, cooks and serves. Both act from the same public state; "
        "its next action is hidden. Thinking does not advance game time.",
        f"Tick {frame['step']}; remaining {frame['max_steps']-frame['step']}; "
        f"team score {frame['score']}; delivered {frame['deliveries']}.",
        f"Full map {kitchen['layout']}, width {kitchen['width']}, height {kitchen['height']}. "
        "Coordinates (x,y): origin upper-left, x right, y down.",
        "Map legend: .=floor, X=counter, O=onion supply, T=tomato supply, D=dish supply, P=pot, S=serve.",
        *[f"row y={y}: {''.join(row).replace(' ', '.')}" for y, row in enumerate(kitchen["terrain"])],
    ]
    for index, chef in enumerate(state["players"]):
        lines.append(
            f"Chef {index} ({'YOU' if index == player else 'teammate'}): at {chef['position']}, "
            f"facing {FACING[tuple(chef['orientation'])]}, holding {object_text(chef['held_object'])}."
        )
    lines.append("Objects on counters and in pots (all other stations/counters are empty):")
    lines.extend(
        f"At {obj['position']}: {object_text(obj)}."
        for obj in sorted(state["objects"], key=lambda obj: obj["position"])
    )
    lines.append(f"Pot capacity: {rules['max_ingredients']} ingredients.")
    lines.extend(
        f"Allowed soup: {'+'.join(recipe['ingredients'])}; cook {recipe['cook_ticks']} ticks; "
        f"delivery reward {recipe['value']}."
        for recipe in rules["recipes"]
    )
    lines.append(f"Bonus orders: {state['bonus_orders']}; order bonus {rules['order_bonus']}.")
    lines.extend(rules["instructions"])
    lines.append("Recent joint actions (chef 0, chef 1) and reward:")
    lines.extend(f"{row['actions']}; reward {row['reward']}" for row in history)
    if notes:
        lines.append(notes)
    return "\n".join(lines)


def action_descriptions(frame: dict[str, Any], player: int) -> dict[str, str]:
    """Describe local native effects from the current public state; keep all six options."""
    state, kitchen = frame["state"], frame["map"]
    chef = state["players"][player]
    x, y = chef["position"]
    objects = {tuple(obj["position"]): obj for obj in state["objects"]}

    def tile_at(pos: tuple[int, int]) -> str:
        px, py = pos
        if not 0 <= px < kitchen["width"] or not 0 <= py < kitchen["height"]:
            return "outside the kitchen"
        return kitchen["legend"][kitchen["terrain"][py][px]]

    result = {}
    for name in ("NORTH", "SOUTH", "EAST", "WEST"):
        dx, dy = ACTIONS[name]
        target = (x + dx, y + dy)
        tile = tile_at(target)
        if list(target) == state["players"][1 - player]["position"]:
            effect = "teammate currently occupies the destination; movement depends on both actions"
        elif tile == "floor":
            effect = "move one floor tile"
        else:
            effect = f"turn toward the {tile} without moving; interact on a later tick"
        result[name] = f"Face {name.lower()} toward {list(target)}: {effect}."
    result["STAY"] = "Wait one tick without moving or interacting; cooking progresses."
    dx, dy = chef["orientation"]
    target = (x + dx, y + dy)
    tile, held = tile_at(target), chef["held_object"]
    obj = objects.get(target)
    effect = "no useful interaction at the faced tile"
    if held is None:
        if tile in ("onion dispenser", "tomato dispenser", "dish dispenser"):
            effect = f"take an object from the {tile}"
        elif tile == "counter" and obj:
            effect = f"pick up {obj['name']} from the counter"
        elif (
            tile == "pot"
            and obj
            and obj.get("is_idle")
            and obj.get("_ingredients")
            and not frame["rules"]["old_dynamics"]
        ):
            effect = "start cooking the ingredients in the faced pot"
    elif tile == "counter" and obj is None:
        effect = f"put held {held['name']} on the empty counter"
    elif tile == "pot":
        if held["name"] in ("onion", "tomato") and (
            obj is None or (obj.get("is_idle") and len(obj["_ingredients"]) < frame["rules"]["max_ingredients"])
        ):
            effect = f"add held {held['name']} to the pot"
        elif held["name"] == "dish" and obj and obj.get("is_ready"):
            effect = "collect ready soup using the held dish"
    elif tile == "serving counter" and held["name"] == "soup":
        effect = "deliver the held soup for the allowed recipe's team reward"
    result["INTERACT"] = f"Interact with {list(target)}: {effect}."
    return result
