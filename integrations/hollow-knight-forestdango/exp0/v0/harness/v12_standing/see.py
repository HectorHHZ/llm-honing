"""v9_keys perception: the visible observation (schema honing.visible.v1, adapter options view=window, units=game,
kinds=raw) turned into world-unit rectangles. Nothing but observation.json is read: what is inside the game window,
the HUD, and the knight's own box. World units are the game's (the knight's body is 0.5 x 1.28); x right, y up.

Information tool only: nothing in here chooses or presses anything.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

CELL = 0.5            # the bridge rasterises terrain at 0.5 world units; the memory grid uses the same cells
BODY_W, BODY_H = 0.5, 1.2813


@dataclass
class Box:
    x0: float
    x1: float
    y0: float          # bottom
    y1: float          # top
    kind: str = ""     # terrain | wall | door | spikes | acid | hazard | enemy
    oid: str = ""

    @property
    def cx(self):
        return (self.x0 + self.x1) / 2

    @property
    def cy(self):
        return (self.y0 + self.y1) / 2

    @property
    def w(self):
        return self.x1 - self.x0

    @property
    def h(self):
        return self.y1 - self.y0

    def overlaps(self, x0, x1, y0, y1) -> bool:
        return self.x0 < x1 and self.x1 > x0 and self.y0 < y1 and self.y1 > y0


class View:
    """One observation. `x`, `y` = the knight's feet centre."""

    def __init__(self, obs: dict):
        self.obs = obs
        self.frame_id = obs["frame"]["id"]
        self.time = float(obs["frame"]["simulation_time"])
        m = obs["view"]["world_from_screen"]
        self.m = m
        w, h = obs["view"]["screen_rect"][2], obs["view"]["screen_rect"][3]
        win = (obs["view"].get("window") or {}).get("rect") or [0, 0, w, h]
        cx0, cy1 = self.world(win[0], win[1])
        cx1, cy0 = self.world(win[0] + win[2], win[1] + win[3])
        self.cam = Box(cx0, cx1, cy0, cy1, "camera")
        flow = obs.get("flow") or {}
        self.segment, self.can_act = flow.get("segment", "play"), bool(flow.get("can_act", True))
        hud = obs.get("hud") or {}
        self.hp = (hud.get("health") or {}).get("value")
        self.hp_max = (hud.get("health_max") or {}).get("value")
        self.soul = (hud.get("soul") or {}).get("fraction")
        self.geo = (hud.get("geo") or {}).get("value")
        self.texts = [t.get("text", "") for t in hud.get("text", []) if isinstance(t, dict)]
        me = obs.get("self") or {}
        self.facing = me.get("facing") or "right"
        sid = me.get("object_id")
        self.room = sid.split("/")[0] if sid else ""
        self.terrain: list[Box] = []     # raster rectangles of solid ground
        self.walls: list[Box] = []       # other solid boxes (may be breakable; nothing says so)
        self.doors: list[Box] = []       # solid boxes that look like doors
        self.hazards: list[Box] = []
        self.enemies: list[Box] = []
        self.me = None
        for o in obs.get("objects", []):
            b = o["bounds"]
            x0, y1 = self.world(b[0], b[1])
            x1, y0 = self.world(b[0] + b[2], b[1] + b[3])
            kind, look, oid = o["kind"], o.get("look", ""), o["id"]
            if oid == sid:
                continue
            if not self.room:
                self.room = oid.split("/")[0]
            if kind == "wall" and look == "terrain":
                self.terrain.append(Box(x0, x1, y0, y1, "terrain", oid))
            elif kind == "wall":
                self.walls.append(Box(x0, x1, y0, y1, "wall", oid))
            elif kind == "interactable":
                self.doors.append(Box(x0, x1, y0, y1, "door", oid))
            elif kind == "visible_hazard":
                self.hazards.append(Box(x0, x1, y0, y1, look if look in ("spikes", "acid") else "hazard", oid))
            elif kind == "enemy":
                self.enemies.append(Box(x0, x1, y0, y1, "enemy", oid))
        sb = me.get("screen_box")
        if sb:
            x0, y1 = self.world(sb[0], sb[1])
            x1, y0 = self.world(sb[0] + sb[2], sb[1] + sb[3])
            self.me = Box(x0, x1, y0, y1, "knight", sid or "")
        self.x = self.me.cx if self.me else math.nan
        self.y = self.me.y0 if self.me else math.nan

    def world(self, sx: float, sy: float) -> tuple[float, float]:
        m = self.m
        return m[0] * sx + m[2] * sy + m[4], m[1] * sx + m[3] * sy + m[5]

    @property
    def solids(self) -> list[Box]:
        return self.walls + self.doors

    def on_screen(self, x: float, y: float) -> bool:
        return self.cam.x0 <= x <= self.cam.x1 and self.cam.y0 <= y <= self.cam.y1
