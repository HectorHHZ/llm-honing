"""v9_keys: geometric reading of the room around the knight (information tool only).

Surfaces = tops of solid ground with room for the knight to stand (three free 0.5-cells above), cut from the
remembered grid inside a window around the knight; plus the tops of solid boxes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .see import CELL, BODY_H


@dataclass
class Surface:
    x0: float
    x1: float
    y: float

    @property
    def w(self):
        return self.x1 - self.x0


def surfaces(room, boxes, x: float, y: float, rx: float = 15.0, up: float = 9.0, down: float = 9.0) -> list[Surface]:
    out = []
    ix0, ix1 = math.floor((x - rx) / CELL), math.floor((x + rx) / CELL)
    iy0, iy1 = math.floor((y - down) / CELL), math.floor((y + up) / CELL)
    solid, seen = room.solid, room.seen
    for iy in range(iy0, iy1 + 1):
        start = None
        for ix in range(ix0, ix1 + 2):
            ok = (ix <= ix1 and (ix, iy) in solid and all((ix, iy + k) in seen and (ix, iy + k) not in solid for k in (1, 2, 3)))
            if ok and start is None:
                start = ix
            elif not ok and start is not None:
                out.append(Surface(start * CELL, ix * CELL, (iy + 1) * CELL))
                start = None
    for b in boxes:
        if abs(b.cx - x) <= rx and -down <= b.y1 - y <= up and b.w >= 0.4:
            out.append(Surface(b.x0, b.x1, b.y1))
    return out


def surface_at(surfs: list[Surface], x: float, y: float, tol: float = 0.3):
    best = None
    for s in surfs:
        if s.x0 - 0.3 <= x <= s.x1 + 0.3 and abs(s.y - y) <= tol:
            if best is None or abs(s.y - y) < abs(best.y - y):
                best = s
    return best


def side_probe(world, x: float, y: float, direction: int, reach: float = 12.0):
    """Walking from (x, y) in `direction` on the ground: (distance, what) with what = wall | edge | far | unseen;
    for a wall also its height above the feet, for an edge the depth of the drop (None = no floor seen)."""
    hw = 0.25
    d = 0.0
    while d < reach:
        d += CELL / 2
        nx = x + direction * d
        if world.unknown(nx, y + 0.3):
            return d, "unseen", None
        if world.blocked(nx - hw, nx + hw, y + 0.05, y + BODY_H - 0.02):
            # height of the obstacle above the feet
            h = 0.0
            while h < 12 and world.blocked(nx - hw, nx + hw, y + h + 0.05, y + h + 0.45):
                h += CELL
            return d - CELL / 2, "wall", h
        if not world.blocked(nx - hw + 0.02, nx + hw - 0.02, y - 0.3, y - 0.002):
            drop = 0.0
            while drop < 20 and not world.blocked(nx - hw + 0.02, nx + hw - 0.02, y - drop - 0.5, y - drop):
                drop += CELL
                if world.unknown(nx, y - drop):
                    return d - CELL / 2, "edge", None
            return d - CELL / 2, "edge", drop if drop < 20 else None
    return reach, "far", None
