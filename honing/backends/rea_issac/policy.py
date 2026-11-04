"""A transparent rule baseline that sees only the public observation."""

import math
from collections import deque
from typing import Any

from honing.backends.rea_issac.engine import CX, CY, FPS, HEIGHT, WIDTH, overlaps


class Baseline:
    def __init__(self) -> None:
        self.graph: dict[str, set[str]] = {}

    def waypoint(self, room: dict[str, Any], player: dict[str, Any], goal: tuple[float, float]) -> tuple[float, float]:
        def valid(node: tuple[int, int]) -> bool:
            x, y = node[0] * 20, node[1] * 20
            return 3 <= node[0] <= 29 and 3 <= node[1] <= 17 and not any(overlaps(x, y, 14, r) for r in room["rocks"])

        candidates = [(x, y) for x in range(3, 30) for y in range(3, 18) if valid((x, y))]
        start = min(candidates, key=lambda n: math.hypot(n[0] * 20 - player["x"], n[1] * 20 - player["y"]))
        end = min(candidates, key=lambda n: math.hypot(n[0] * 20 - goal[0], n[1] * 20 - goal[1]))
        if start == end:
            return goal
        queue = deque([start])
        parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        while queue:
            node = queue.popleft()
            if node == end:
                break
            for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                nxt = node[0] + dx, node[1] + dy
                if nxt not in parent and valid(nxt):
                    parent[nxt] = node
                    queue.append(nxt)
        if end not in parent:
            return goal
        path = [end]
        while (previous := parent[path[-1]]) is not None:
            path.append(previous)
        path.reverse()
        for node in path[1:]:
            if math.hypot(node[0] * 20 - player["x"], node[1] * 20 - player["y"]) > 16:
                return node[0] * 20, node[1] * 20
        return goal

    def act(self, obs: dict) -> dict:
        room, player = obs["room"], obs["player"]
        key = room["id"]
        self.graph[key] = {door["destination"] for door in obs["doors"]}
        visited = {r["id"] for r in obs["map"] if r["visited"]}
        shoot_x = shoot_y = 0
        if room["enemies"]:
            enemy = min(room["enemies"], key=lambda e: math.hypot(e["x"] - player["x"], e["y"] - player["y"]))
            dx, dy = enemy["x"] - player["x"], enemy["y"] - player["y"]
            if abs(dx) > abs(dy):
                shoot_x = 1 if dx > 0 else -1
                goal = (player["x"], enemy["y"])
            else:
                shoot_y = 1 if dy > 0 else -1
                goal = (enemy["x"], player["y"])
            if math.hypot(dx, dy) < 125:
                length = max(math.hypot(dx, dy), 1)
                goal = (
                    player["x"] - dx / length * 110 - dy / length * 25,
                    player["y"] - dy / length * 110 + dx / length * 25,
                )
            else:
                goal = ((goal[0] + player["x"]) / 2, (goal[1] + player["y"]) / 2)
            firing_origin = (player["x"], enemy["y"]) if shoot_x else (enemy["x"], player["y"])
            blocked = any(
                overlaps(
                    firing_origin[0] + (enemy["x"] - firing_origin[0]) * i / 24,
                    firing_origin[1] + (enemy["y"] - firing_origin[1]) * i / 24,
                    4,
                    r,
                )
                for r in room["rocks"]
                for i in range(1, 24)
            )
            if blocked:
                lanes = [(enemy["x"] + offset, enemy["y"]) for offset in (-180, 180)] + [
                    (enemy["x"], enemy["y"] + offset) for offset in (-145, 145)
                ]
                lanes = [
                    point
                    for point in lanes
                    if 58 < point[0] < WIDTH - 58
                    and 60 < point[1] < HEIGHT - 60
                    and not any(overlaps(*point, 15, r) for r in room["rocks"])
                    and not any(
                        overlaps(
                            point[0] + (enemy["x"] - point[0]) * i / 24,
                            point[1] + (enemy["y"] - point[1]) * i / 24,
                            4,
                            r,
                        )
                        for r in room["rocks"]
                        for i in range(1, 24)
                    )
                ]
                if lanes:
                    goal = min(lanes, key=lambda point: math.hypot(point[0] - player["x"], point[1] - player["y"]))
        else:
            pickups = [
                p
                for p in room["pickups"]
                if (p["kind"] != "heart" or player["health"] < player["max_health"])
                and p.get("price", 0) <= player["coins"]
            ]
            if pickups:
                target = min(pickups, key=lambda p: math.hypot(p["x"] - player["x"], p["y"] - player["y"]))
                goal = (target["x"], target["y"])
            else:
                queue: deque[tuple[str, list[str]]] = deque([(key, [])])
                seen = {key}
                route = []
                while queue:
                    node, path = queue.popleft()
                    if node not in visited:
                        route = path
                        break
                    for neighbor in sorted(self.graph.get(node, set())):
                        if neighbor not in seen:
                            seen.add(neighbor)
                            queue.append((neighbor, [*path, neighbor]))
                doors = [d for d in obs["doors"] if route and d["destination"] == route[0]] or obs["doors"]
                door = next(
                    (d for d in doors if not d["visited"] and d["style"] == "treasure"), doors[0] if doors else None
                )
                if not door:
                    return {}
                # Aim beyond the collision boundary so a held direction enters the next room.
                goal = (
                    0 if door["direction"] == "west" else WIDTH if door["direction"] == "east" else CX,
                    0 if door["direction"] == "north" else HEIGHT if door["direction"] == "south" else CY,
                )
        goal = self.waypoint(room, player, goal)
        candidates = [(x, y) for x in (-1, 0, 1) for y in (-1, 0, 1)]

        def cost(move: tuple[int, int]) -> float:
            mx, my = move
            norm = max(1, math.hypot(mx, my))
            nx = player["x"] + mx / norm * player["speed"] / FPS * 3
            ny = player["y"] + my / norm * player["speed"] / FPS * 3
            # Project against walls without penalizing the final door approach.
            nx, ny = max(42, min(WIDTH - 42, nx)), max(50, min(HEIGHT - 50, ny))
            score = math.hypot(nx - goal[0], ny - goal[1])
            if not mx and not my and score > 10:
                score += 4
            if any(overlaps(nx, ny, 14, rect) for rect in room["rocks"]):
                score += 500
            for enemy in room["enemies"]:
                d = math.hypot(nx - enemy["x"], ny - enemy["y"]) - enemy["radius"]
                score += max(0, 90 - d) * 5
            for bullet in obs["bullets"]:
                if bullet["hostile"]:
                    for t in (0.1, 0.25, 0.4):
                        d = math.hypot(nx - bullet["x"] - bullet["vx"] * t, ny - bullet["y"] - bullet["vy"] * t)
                        score += max(0, 29 - d) * 3
            return score

        mx, my = min(candidates, key=cost)
        return {"move_x": mx, "move_y": my, "shoot_x": shoot_x, "shoot_y": shoot_y}
