# fmt: off
# isort: skip_file
# ruff: noqa
"""Rule-based parsing of the *visible* observation (schema honing.visible.v1) into geometry.

Input: only what the game writes to observation.json — the 48x27 semantic grid (20 px cells, bit
unions), the visible object bounds, the HUD health and the camera transform. No debug state.

Screen convention: x right, y down, pixels. The camera is rigidly attached to the knight (checked on
every frame against the knight's own sprite), so the knight's body is always the same screen box and
the world scrolls under it. World coordinates (camera transform) are used only to remember things
across frames (where a platform was, how far we moved).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BG, WALL, PLAT, HAZ, ACTOR, INTER, UNK, SCEN = 1, 2, 4, 8, 16, 32, 64, 128
SOLID = WALL | PLAT
CELL, COLS, ROWS = 20, 48, 27
W, H = COLS * CELL, ROWS * CELL

# Knight body on screen (collider 0.66 x 1.32 world units at ~54 px per unit), measured by probe_01.
KX, KY = 480.0, 378.0
HALF_W, HALF_H = 17.8, 35.6
FEET_Y = KY + HALF_H          # 413.6
HEAD_Y = KY - HALF_H          # 342.4


@dataclass
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float
    kind: str = ""
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


@dataclass
class Seg:
    """A top surface something could stand on. Screen px; y is the visible top edge of the sprite."""
    x0: float
    x1: float
    y: float
    kind: str            # platform | wall
    oid: str
    thick: bool          # sprite taller than 40 px (floors have a decorative lip above the walkable top)
    safe: bool = True    # no hazard cell on or just above it
    clipped_left: bool = False   # continues beyond the screen edge
    clipped_right: bool = False
    clipped_top: bool = False    # the sprite is cut by the top of the screen: its real top is not visible

    @property
    def cx(self):
        return (self.x0 + self.x1) / 2

    @property
    def w(self):
        return self.x1 - self.x0


class Scene:
    def __init__(self, obs: dict):
        self.obs = obs
        self.frame_id = obs["frame"]["id"]
        self.tick = obs["frame"]["tick"]
        self.time = obs["frame"]["simulation_time"]
        rows = obs["terrain"]["rows"]
        if len(rows) != ROWS or any(len(r) != COLS for r in rows) or obs["terrain"]["cell_size"] != [CELL, CELL]:
            raise ValueError("unexpected terrain grid")
        self.masks = np.array([[ord(c) - 0x100 for c in row] for row in rows], dtype=np.int16)
        m = obs["view"]["world_from_screen"]
        self.m = m
        self.ppu = 1.0 / m[0]
        self.objects = [Rect(o["bounds"][0], o["bounds"][1], o["bounds"][0] + o["bounds"][2],
                             o["bounds"][1] + o["bounds"][3], o["kind"], o["id"]) for o in obs["objects"]]
        sid = obs.get("self", {}).get("object_id")
        self.me = next((o for o in self.objects if o.oid == sid), None)
        # The knight's sprite must sit where the rigid camera puts it; otherwise say so (do not guess).
        self.self_ok = self.me is not None and abs(self.me.cx - KX) < 30 and abs(self.me.cy - KY) < 30
        health = obs.get("hud", {}).get("health")
        self.hp = health["value"] if isinstance(health, dict) else None
        self.enemies = [o for o in self.objects if o.kind == "enemy"]
        self.hazards = [o for o in self.objects if o.kind == "visible_hazard"]
        self.interactables = [o for o in self.objects if o.kind == "interactable"]
        self.walls = [o for o in self.objects if o.kind == "wall"]
        self.platforms = [o for o in self.objects if o.kind == "platform"]
        self.segs = self._segments()

    # ---- coordinates -----------------------------------------------------------------------
    def to_world(self, sx: float, sy: float) -> tuple[float, float]:
        m = self.m
        return m[0] * sx + m[2] * sy + m[4], m[1] * sx + m[3] * sy + m[5]

    def to_screen(self, wx: float, wy: float) -> tuple[float, float]:
        m = self.m
        return (wx - m[4]) / m[0], (wy - m[5]) / m[3]

    @property
    def knight_world(self) -> tuple[float, float]:
        """World position of the knight's feet centre (from the camera, i.e. from the screen scroll)."""
        return self.to_world(KX, FEET_Y)

    # ---- grid queries ----------------------------------------------------------------------
    def cells(self, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
        """Masks of all cells overlapping the screen rectangle (clipped to the screen)."""
        c0, c1 = max(0, int(x0 // CELL)), min(COLS - 1, int((x1 - 1e-6) // CELL))
        r0, r1 = max(0, int(y0 // CELL)), min(ROWS - 1, int((y1 - 1e-6) // CELL))
        if c0 > c1 or r0 > r1:
            return np.zeros((0, 0), dtype=np.int16)
        return self.masks[r0:r1 + 1, c0:c1 + 1]

    def any(self, x0, y0, x1, y1, bits: int) -> bool:
        block = self.cells(x0, y0, x1, y1)
        return bool(block.size) and bool((block & bits).any())

    def offscreen(self, x: float, y: float) -> bool:
        return not (0 <= x < W and 0 <= y < H)

    # ---- surfaces --------------------------------------------------------------------------
    def _segments(self) -> list[Seg]:
        segs: list[Seg] = []
        for o in self.platforms + self.walls:
            bit = PLAT if o.kind == "platform" else WALL
            if o.w < 6 or o.h < 4:
                continue
            top_row = int(o.y0 // CELL)
            c0, c1 = int(o.x0 // CELL), min(COLS - 1, int((o.x1 - 1) // CELL))
            # Columns where the object's top row (or the row just below, for a top edge on a cell
            # boundary) really shows this class and nothing solid of the same object sits above.
            run_start = None
            for c in range(c0, c1 + 2):
                ok = False
                if c <= c1:
                    # the top band may be hidden behind foreground scenery (grass, lamps): accept the
                    # class anywhere in the first rows of the sprite
                    # ... or behind a wall sprite drawn in front of the platform's end (a fence): inside the
                    # sprite's own bounds any solid class counts
                    r_hi = min(ROWS - 1, max(top_row, int((min(o.y1, o.y0 + 70) - 1) // CELL)))
                    ok = bool((self.masks[top_row:r_hi + 1, c] & SOLID).any()) if top_row < ROWS else False
                if ok and run_start is None:
                    run_start = c
                elif not ok and run_start is not None:
                    x0 = max(o.x0, run_start * CELL)
                    x1 = min(o.x1, c * CELL)
                    if x1 - x0 >= 8:
                        seg = Seg(x0, x1, o.y0, o.kind, o.oid, thick=o.h > 40,
                                  clipped_left=x0 <= 0.5, clipped_right=x1 >= W - 0.5, clipped_top=o.y0 <= 0.5)
                        seg.safe = self._seg_safe(seg)
                        segs.append(seg)
                    run_start = None
        return segs

    def _seg_safe(self, seg: Seg) -> bool:
        """Safe to stand on: no hazard in the cells on/above the top, and head room for the knight."""
        if self.any(seg.x0 + 2, seg.y - 2 * HALF_H, seg.x1 - 2, seg.y + 6, HAZ):
            return False
        return True

    def headroom(self, x0: float, x1: float, y: float, own: str = "") -> bool:
        """No other solid in the knight-high band above a surface span."""
        for o in self.platforms + self.walls:
            if o.oid == own:
                continue
            if o.x1 > x0 + 2 and o.x0 < x1 - 2 and o.y1 > y - 2 * HALF_H + 4 and o.y0 < y - 6:
                return False
        return True

    def support(self) -> Seg | None:
        """The surface under the knight's feet.

        Thick floor sprites have a decorative lip: the knight's feet sit about 12 px below the sprite's
        visible top; on thin platforms the feet sit about 3 px above it. Walls are a last resort.
        """
        best, best_score = None, 1e9
        for seg in self.segs:
            if seg.clipped_top or not (seg.x0 - HALF_W <= KX <= seg.x1 + HALF_W):
                continue
            d = seg.y - FEET_Y
            if not -26 <= d <= 10:
                continue
            score = abs(d - (-12 if seg.thick else 3)) + (10 if seg.kind == "wall" else 0)
            if score < best_score:
                best, best_score = seg, score
        return best

    # ---- things ----------------------------------------------------------------------------
    def nearest_enemy(self) -> Rect | None:
        if not self.enemies:
            return None
        return min(self.enemies, key=lambda e: abs(e.cx - KX) + 0.5 * abs(e.cy - KY))

    def ascii(self) -> str:
        """Debug print for humans (never sent to the model)."""
        out = []
        for r in range(ROWS):
            line = ""
            for c in range(COLS):
                m = int(self.masks[r, c])
                ch = "."
                for bit, g in ((HAZ, "!"), (ACTOR, "@"), (INTER, "+"), (WALL, "#"), (PLAT, "="), (SCEN, ":"), (UNK, "?")):
                    if m & bit:
                        ch = g
                        break
                line += ch
            out.append(f"{r:2d} {line}")
        return "\n".join(out)
