"""A deterministic, CPU-only room-combat scenario with original simulation rules."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

VERSION = "rea-issac-1.0.0"
SOURCE = json.loads((Path(__file__).resolve().parent / "source_facts.json").read_text())
WIDTH, HEIGHT, FPS = 640, 416, 30
CX, CY = WIDTH / 2, HEIGHT / 2
ITEMS = {
    "glass": ("Glass lens", "+0.8 damage"),
    "ribbon": ("Blue ribbon", "Faster tears"),
    "boots": ("Worn boots", "+18 movement speed"),
    "vessel": ("Clay vessel", "+1 heart container"),
}
DIRECTIONS = {"north": (0, -1), "east": (1, 0), "south": (0, 1), "west": (-1, 0)}


@dataclass(frozen=True)
class Action:
    move_x: int = 0
    move_y: int = 0
    shoot_x: int = 0
    shoot_y: int = 0
    bomb: bool = False

    @classmethod
    def parse(cls, value: Any) -> Action:
        if not isinstance(value, dict) or set(value) - set(cls.__dataclass_fields__):
            raise ValueError("action must contain only move_x, move_y, shoot_x, shoot_y and bomb")
        for name in ("move_x", "move_y", "shoot_x", "shoot_y"):
            v = value.get(name, 0)
            if type(v) is not int or v not in (-1, 0, 1):
                raise ValueError(f"{name} must be -1, 0 or 1")
        if type(value.get("bomb", False)) is not bool:
            raise ValueError("bomb must be boolean")
        return cls(**value)


def distance(a: dict, b: dict) -> float:
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def overlaps(x: float, y: float, radius: float, rect: list[float]) -> bool:
    rx, ry, rw, rh = rect
    return math.hypot(x - max(rx, min(rx + rw, x)), y - max(ry, min(ry + rh, y))) < radius


def segment_hits(x: float, y: float, nx: float, ny: float, target: dict, radius: float) -> bool:
    dx, dy = nx - x, ny - y
    t = max(0.0, min(1.0, ((target["x"] - x) * dx + (target["y"] - y) * dy) / max(dx * dx + dy * dy, 1e-9)))
    return math.hypot(x + dx * t - target["x"], y + dy * t - target["y"]) <= radius


class Engine:
    """Explicit reset/step simulation. Rendering and wall time never advance it."""

    def __init__(self, seed: int = 7, max_ticks: int = 18000) -> None:
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError("seed must be a uint32")
        if type(max_ticks) is not int or not 1 <= max_ticks <= 108000:
            raise ValueError("max_ticks must be in [1, 108000]")
        self.seed, self.max_ticks = seed, max_ticks
        self.rng = random.Random(seed)
        self.tick, self.serial, self.status = 0, 0, "running"
        self.current = "0,0"
        self.events: list[dict] = []
        self.bullets: list[dict] = []
        self.bombs: list[dict] = []
        self.metrics = {"kills": 0, "shots": 0, "hits": 0, "damage_taken": 0, "coins_collected": 0}
        self.player = {
            "id": "player",
            "x": CX,
            "y": CY,
            "vx": 0.0,
            "vy": 0.0,
            "radius": SOURCE["player"]["radius"],
            "health": SOURCE["player"]["health"],
            "max_health": SOURCE["player"]["health"],
            "damage": 3.5,
            "speed": 138.0,
            "shot_speed": 250.0,
            "shot_interval": 8,
            "next_shot": 0,
            "invulnerable_until": 0,
            "coins": 0,
            "bombs": SOURCE["player"]["bombs"],
            "facing_x": 0,
            "facing_y": 1,
            "items": [],
        }
        self.bomb_latch = False
        self.rooms: dict[str, dict] = {}
        self._generate()
        self.rooms[self.current]["visited"] = True

    def _id(self, prefix: str) -> str:
        self.serial += 1
        return f"{prefix}{self.serial}"

    def _event(self, kind: str, x: float, y: float, **extra: Any) -> None:
        self.events.append(
            {"id": self._id("event"), "tick": self.tick, "room": self.current, "kind": kind, "x": x, "y": y, **extra}
        )
        self.events = self.events[-50:]

    def _enemy(self, kind: str, x: float, y: float) -> dict:
        facts = SOURCE["entities"][kind]
        hp, radius = float(facts["baseHP"]), float(facts["collisionRadius"])
        speed = {"chaser": 48, "spitter": 0, "fly": 57, "boss": 28}[kind]
        if kind == "boss":
            hp *= SOURCE["simulation_choices"]["boss_hp_scale"]
            radius *= SOURCE["simulation_choices"]["boss_radius_scale"]
        return {
            "id": self._id("enemy"),
            "kind": kind,
            "x": float(x),
            "y": float(y),
            "health": hp,
            "max_health": hp,
            "speed": speed,
            "radius": radius,
            "source_type_id": int(facts["id"]),
            "contact_damage": int(facts["collisionDamage"]),
            "next_shot": 45 + self.rng.randrange(30),
            "phase": self.rng.random() * math.tau,
            "hit_until": 0,
            "age": 0,
        }

    def _drop(self, room: dict, kind: str, x: float, y: float, **values: Any) -> None:
        room["pickups"].append({"id": self._id("pickup"), "kind": kind, "x": x, "y": y, **values})

    def _generate(self) -> None:
        corners = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
        self.rng.shuffle(corners)
        boss, treasure, shop = corners[:3]
        coordinates = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), *corners[:3]]
        names = ["Grit Hall", "Bone Passage", "Dust Chapel", "The Spill"]
        self.rng.shuffle(names)
        for index, (x, y) in enumerate(coordinates):
            kind = (
                "start"
                if (x, y) == (0, 0)
                else (
                    "boss"
                    if (x, y) == boss
                    else "treasure" if (x, y) == treasure else "shop" if (x, y) == shop else "combat"
                )
            )
            room: dict[str, Any] = {
                "id": f"{x},{y}",
                "position": [x, y],
                "kind": kind,
                "visited": False,
                "cleared": kind in ("start", "treasure", "shop"),
                "rewarded": kind != "combat",
                "name": {
                    "start": "Threshold",
                    "boss": "The Warden",
                    "treasure": "The Offering",
                    "shop": "Wanderer's Cache",
                }.get(kind, names[index % 4]),
                "rocks": [],
                "enemies": [],
                "pickups": [],
                "stains": [],
                "decoration_seed": self.rng.randrange(2**31),
            }
            if kind == "combat":
                pattern = self.rng.choice((0, 1, 2))
                rocks = (
                    [[150, 106, 34, 34], [456, 276, 34, 34]]
                    if pattern == 0
                    else (
                        [[170, 276, 34, 34], [436, 106, 34, 34]]
                        if pattern == 1
                        else [[124, 110, 34, 34], [482, 110, 34, 34], [124, 272, 34, 34], [482, 272, 34, 34]]
                    )
                )
                room["rocks"] = rocks
                points = [(240, 112), (405, 288), (190, 240)]
                for ex, ey in points[: self.rng.choice((2, 3, 3))]:
                    room["enemies"].append(
                        self._enemy(self.rng.choice(("chaser", "chaser", "spitter", "fly")), ex, ey)
                    )
            elif kind == "boss":
                room["enemies"].append(self._enemy("boss", CX, 140))
            elif kind in ("treasure", "shop"):
                self._drop(
                    room, "upgrade", CX, CY, item=self.rng.choice(list(ITEMS)), price=3 if kind == "shop" else 0
                )
            self.rooms[room["id"]] = room

    @property
    def room(self) -> dict:
        return self.rooms[self.current]

    def blocked(self, x: float, y: float, radius: float, flying: bool = False) -> bool:
        return (
            x < 32 + radius
            or x > WIDTH - 32 - radius
            or y < 40 + radius
            or y > HEIGHT - 40 - radius
            or (not flying and any(overlaps(x, y, radius, rect) for rect in self.room["rocks"]))
        )

    def _move(self, body: dict, vx: float, vy: float, flying: bool = False) -> None:
        nx, ny = body["x"] + vx / FPS, body["y"] + vy / FPS
        if not self.blocked(nx, body["y"], body["radius"], flying):
            body["x"] = nx
        if not self.blocked(body["x"], ny, body["radius"], flying):
            body["y"] = ny

    def _hurt(self, damage: int) -> None:
        if self.tick >= self.player["invulnerable_until"]:
            self.player["health"] = max(0, self.player["health"] - damage)
            self.player["invulnerable_until"] = self.tick + 32
            self.metrics["damage_taken"] += damage
            self._event("hurt", self.player["x"], self.player["y"])

    def _shoot(self, body: dict, angle: float, hostile: bool = False, damage: float = 1, speed: float = 110) -> None:
        self.bullets.append(
            {
                "id": self._id("shot"),
                "x": body["x"] + math.cos(angle) * (body["radius"] + 5),
                "y": body["y"] + math.sin(angle) * (body["radius"] + 5),
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed,
                "radius": 4 if hostile else 3.7,
                "hostile": hostile,
                "damage": damage,
                "life": 165 if hostile else 60,
            }
        )

    def _enemies(self) -> None:
        for enemy in self.room["enemies"]:
            enemy["age"] += 1
            angle = math.atan2(self.player["y"] - enemy["y"], self.player["x"] - enemy["x"])
            speed = enemy["speed"]
            if enemy["kind"] == "fly":
                angle += math.sin(enemy["age"] / 18 + enemy["phase"]) * 0.75
            if speed:
                self._move(enemy, math.cos(angle) * speed, math.sin(angle) * speed, enemy["kind"] == "fly")
            if enemy["contact_damage"] and distance(enemy, self.player) < enemy["radius"] + self.player["radius"]:
                self._hurt(enemy["contact_damage"])
            if enemy["kind"] != "chaser" and enemy["age"] >= enemy["next_shot"]:
                offsets = [-0.28, 0, 0.28] if enemy["kind"] == "boss" else [0]
                for offset in offsets:
                    self._shoot(enemy, angle + offset, True, speed=105 if enemy["kind"] != "boss" else 125)
                if enemy["kind"] == "boss" and enemy["health"] < enemy["max_health"] / 2:
                    for index in range(10):
                        self._shoot(enemy, enemy["age"] / 50 + index * math.tau / 10, True, speed=84)
                enemy["next_shot"] += 75 if enemy["kind"] == "spitter" else 105 if enemy["kind"] == "fly" else 90

    def _projectiles(self) -> None:
        survivors = []
        for bullet in self.bullets:
            x, y = bullet["x"], bullet["y"]
            nx, ny = x + bullet["vx"] / FPS, y + bullet["vy"] / FPS
            bullet["life"] -= 1
            if self.blocked(nx, ny, bullet["radius"]) or bullet["life"] <= 0:
                self._event("splash", nx, ny, hostile=bullet["hostile"])
                continue
            targets = [self.player] if bullet["hostile"] else self.room["enemies"]
            hit = next(
                (
                    target
                    for target in targets
                    if target.get("health", 0) > 0
                    and segment_hits(x, y, nx, ny, target, target["radius"] + bullet["radius"])
                ),
                None,
            )
            if hit:
                if bullet["hostile"]:
                    self._hurt(int(bullet["damage"]))
                else:
                    hit["health"] -= bullet["damage"]
                    hit["hit_until"] = self.tick + 4
                    self.metrics["hits"] += 1
                self._event("splash", nx, ny, hostile=bullet["hostile"])
                continue
            bullet["x"], bullet["y"] = nx, ny
            survivors.append(bullet)
        self.bullets = survivors

    def _explosions(self) -> None:
        pending = []
        for bomb in self.bombs:
            if self.tick < bomb["explode_at"]:
                pending.append(bomb)
                continue
            self._event("explosion", bomb["x"], bomb["y"])
            for enemy in self.room["enemies"]:
                if distance(enemy, bomb) < 100 + enemy["radius"]:
                    enemy["health"] -= 20
                    enemy["hit_until"] = self.tick + 5
            if distance(self.player, bomb) < 100:
                self._hurt(2)
            self.room["rocks"] = [rect for rect in self.room["rocks"] if not overlaps(bomb["x"], bomb["y"], 100, rect)]
        self.bombs = pending

    def _deaths_and_rewards(self) -> None:
        alive = []
        for enemy in self.room["enemies"]:
            if enemy["health"] > 0:
                alive.append(enemy)
                continue
            self.metrics["kills"] += 1
            self.room["stains"].append([enemy["x"], enemy["y"], enemy["radius"], self.serial])
            self._event("defeat", enemy["x"], enemy["y"])
            if self.rng.random() < 0.55 and enemy["kind"] != "boss":
                self._drop(self.room, "coin", enemy["x"], enemy["y"])
        self.room["enemies"] = alive
        if not alive and not self.room["cleared"]:
            self.room["cleared"] = True
            self._event("clear", CX, CY, text="Chamber cleared")
            if self.room["kind"] == "boss" and self.player["health"] > 0:
                self.status = "won"
            elif not self.room["rewarded"]:
                self.room["rewarded"] = True
                self._drop(self.room, "heart" if self.rng.random() < 0.65 else "coin", CX, CY)
                self._drop(self.room, "coin", CX + 27, CY)

    def _pickups(self) -> None:
        remaining = []
        for pickup in self.room["pickups"]:
            if distance(pickup, self.player) > 21:
                remaining.append(pickup)
                continue
            kind = pickup["kind"]
            if kind == "heart":
                if self.player["health"] == self.player["max_health"]:
                    remaining.append(pickup)
                    continue
                self.player["health"] = min(self.player["max_health"], self.player["health"] + 2)
                text = "A little more life"
            elif kind == "coin":
                self.player["coins"] += 1
                self.metrics["coins_collected"] += 1
                text = "+1 coin"
            else:
                price = pickup.get("price", 0)
                if self.player["coins"] < price:
                    remaining.append(pickup)
                    continue
                self.player["coins"] -= price
                item = pickup["item"]
                self.player["items"].append(item)
                if item == "glass":
                    self.player["damage"] += 0.8
                elif item == "ribbon":
                    self.player["shot_interval"] = max(3, self.player["shot_interval"] - 2)
                elif item == "boots":
                    self.player["speed"] += 18
                else:
                    self.player["max_health"] += 2
                    self.player["health"] = min(self.player["max_health"], self.player["health"] + 3)
                text = ITEMS[item][0]
            self._event("pickup", pickup["x"], pickup["y"], text=text)
        self.room["pickups"] = remaining

    def _transition(self, action: Action) -> None:
        if not self.room["cleared"]:
            return
        p = self.player
        direction = (
            "west"
            if action.move_x < 0 and p["x"] <= 46 and abs(p["y"] - CY) < 28
            else (
                "east"
                if action.move_x > 0 and p["x"] >= WIDTH - 46 and abs(p["y"] - CY) < 28
                else (
                    "north"
                    if action.move_y < 0 and p["y"] <= 54 and abs(p["x"] - CX) < 28
                    else "south" if action.move_y > 0 and p["y"] >= HEIGHT - 54 and abs(p["x"] - CX) < 28 else None
                )
            )
        )
        if not direction:
            return
        dx, dy = DIRECTIONS[direction]
        x, y = self.room["position"]
        key = f"{x+dx},{y+dy}"
        if key not in self.rooms:
            return
        self.current = key
        self.room["visited"] = True
        p["x"] = 60 if dx > 0 else WIDTH - 60 if dx < 0 else CX
        p["y"] = 65 if dy > 0 else HEIGHT - 65 if dy < 0 else CY
        p["invulnerable_until"] = max(p["invulnerable_until"], self.tick + 22)
        self.bullets, self.bombs = [], []
        self._event("enter", p["x"], p["y"], text=self.room["name"])

    def step(self, action: dict | Action, frames: int = 1) -> dict:
        if type(frames) is not int or not 1 <= frames <= 300:
            raise ValueError("frames must be in [1, 300]")
        action = Action.parse(asdict(action) if isinstance(action, Action) else action)
        for _ in range(frames):
            if self.status != "running":
                break
            self.tick += 1
            p = self.player
            norm = max(1, math.hypot(action.move_x, action.move_y))
            p["vx"], p["vy"] = action.move_x * p["speed"] / norm, action.move_y * p["speed"] / norm
            self._move(p, p["vx"], p["vy"])
            if action.shoot_x or action.shoot_y:
                p["facing_x"], p["facing_y"] = action.shoot_x, action.shoot_y
                if self.tick >= p["next_shot"]:
                    self._shoot(
                        p, math.atan2(action.shoot_y, action.shoot_x), damage=p["damage"], speed=p["shot_speed"]
                    )
                    p["next_shot"] = self.tick + p["shot_interval"]
                    self.metrics["shots"] += 1
            if action.bomb and not self.bomb_latch and p["bombs"]:
                p["bombs"] -= 1
                self.bombs.append({"id": self._id("bomb"), "x": p["x"], "y": p["y"], "explode_at": self.tick + 55})
            self.bomb_latch = action.bomb
            self._enemies()
            self._projectiles()
            self._explosions()
            self._deaths_and_rewards()
            self._pickups()
            if p["health"] <= 0:
                self.status = "lost"
            if self.status == "running":
                self._transition(action)
            if self.tick >= self.max_ticks and self.status == "running":
                self.status = "truncated"
        return self.observe()

    def observe(self) -> dict:
        x, y = self.room["position"]
        doors = []
        for direction, (dx, dy) in DIRECTIONS.items():
            destination = f"{x+dx},{y+dy}"
            if destination in self.rooms:
                other = self.rooms[destination]
                doors.append(
                    {
                        "direction": direction,
                        "destination": destination,
                        "open": self.room["cleared"],
                        "visited": other["visited"],
                        "style": other["kind"] if other["kind"] in ("boss", "treasure", "shop") else "normal",
                        "x": 38 if dx < 0 else WIDTH - 38 if dx > 0 else CX,
                        "y": 46 if dy < 0 else HEIGHT - 46 if dy > 0 else CY,
                    }
                )
        known = {self.current}
        for room in self.rooms.values():
            if room["visited"]:
                known.add(room["id"])
                rx, ry = room["position"]
                known.update(f"{rx+dx},{ry+dy}" for dx, dy in DIRECTIONS.values() if f"{rx+dx},{ry+dy}" in self.rooms)
        public = {
            "engine_version": VERSION,
            "seed": self.seed,
            "tick": self.tick,
            "fps": FPS,
            "status": self.status,
            "width": WIDTH,
            "height": HEIGHT,
            "max_ticks": self.max_ticks,
            "player": self.player,
            "bullets": self.bullets,
            "bombs": self.bombs,
            "events": self.events,
            "room": {k: v for k, v in self.room.items() if k not in ("visited", "rewarded")},
            "doors": doors,
            "map": [
                {
                    "id": key,
                    "position": self.rooms[key]["position"],
                    "visited": self.rooms[key]["visited"],
                    "kind": self.rooms[key]["kind"] if self.rooms[key]["visited"] else "unknown",
                    "cleared": self.rooms[key]["cleared"] if self.rooms[key]["visited"] else False,
                }
                for key in sorted(known)
            ],
            "metrics": self.metrics,
            "rooms_visited": sum(r["visited"] for r in self.rooms.values()),
            "rooms_cleared": sum(r["visited"] and r["cleared"] for r in self.rooms.values()),
            "total_rooms": len(self.rooms),
            "items": ITEMS,
        }
        return json.loads(json.dumps(public, allow_nan=False))

    def digest(self) -> str:
        state = {
            "version": VERSION,
            "seed": self.seed,
            "max_ticks": self.max_ticks,
            "tick": self.tick,
            "status": self.status,
            "serial": self.serial,
            "current": self.current,
            "player": self.player,
            "rooms": self.rooms,
            "bullets": self.bullets,
            "bombs": self.bombs,
            "events": self.events,
            "metrics": self.metrics,
            "bomb_latch": self.bomb_latch,
            "rng": self.rng.getstate(),
        }
        return hashlib.sha256(
            json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()

    def result(self) -> dict:
        return {
            "engine_version": VERSION,
            "seed": self.seed,
            "tick": self.tick,
            "status": self.status,
            "success": self.status == "won",
            "health": self.player["health"],
            "rooms_visited": sum(r["visited"] for r in self.rooms.values()),
            "rooms_cleared": sum(r["visited"] and r["cleared"] for r in self.rooms.values()),
            **self.metrics,
            "final_hash": self.digest(),
        }


def replay(record: dict) -> Engine:
    if not isinstance(record, dict) or record.get("engine_version") != VERSION:
        raise ValueError("unsupported replay engine version")
    engine = Engine(record["seed"], record["max_ticks"])
    actions = record.get("actions")
    if not isinstance(actions, list) or len(actions) > 108000:
        raise ValueError("invalid replay action count")
    frames = 0
    for entry in actions:
        if not isinstance(entry, dict) or type(entry.get("frames")) is not int:
            raise ValueError("invalid replay action entry")
        frames += entry["frames"]
        if frames > engine.max_ticks + 300:
            raise ValueError("replay exceeds the run budget")
        engine.step(entry["action"], entry["frames"])
    if engine.digest() != record.get("final_hash"):
        raise ValueError("replay hash mismatch")
    return engine
