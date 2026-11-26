# fmt: off
# isort: skip_file
# ruff: noqa
"""Memory of what the knight has seen, in world coordinates (built only from visible observations).

The screen shows 8.9 units to each side but only 2.3 units below the feet, so a floor the knight just
jumped up from is already out of view. A player remembers it; so does this map: every platform, wall
and hazard sprite that appeared on screen is kept as a world rectangle (union of all partial views)
until it is seen to be gone (its place is on screen again and it is not there).
"""
from __future__ import annotations

from dataclasses import dataclass

from . import physics as ph
import math

from .perception import CELL, COLS, HAZ, ROWS, UNK, H, W, Scene


@dataclass
class Surf:
    oid: str
    kind: str                 # platform | wall | hazard
    x0: float
    x1: float
    y0: float
    y1: float                 # sprite top as far as seen
    thick: bool = False
    lip: float = 0.0
    open_left: bool = False   # cut by the screen edge when seen: the real extent is unknown
    open_right: bool = False
    open_top: bool = False
    seen: int = 0
    moving: bool = False      # v8: a platform that goes back and forth (kept where it is now, not as a union of views)

    @property
    def top(self) -> float:
        return self.y1 - self.lip

    def wrect(self) -> ph.WRect:
        return ph.WRect(self.x0, self.x1, self.y0, self.y1, self.kind, self.oid, self.lip)


class WorldMap:
    def __init__(self):
        self.solids: dict[str, Surf] = {}
        self.hazards: dict[str, Surf] = {}
        self.things: dict[str, Surf] = {}     # interactables seen (world rect), for the language layer
        self.visited: set[str] = set()        # surfaces the knight has stood on
        self.inert: set[str] = set()          # interactables that did not react to a sword strike
        self.climbed: set[str] = set()        # walls the knight has clung to
        self.no_hold: set[str] = set()        # wall-like things that gave no hold (cannot be climbed)
        self.fruitless: set[str] = set()      # walls climbed as far as possible without finding a ledge
        self.collapsing: set[tuple] = set()   # (width, height) of platforms that vanished after the knight stood on them
        # Where danger was seen: world cells of GRID units marked from the hazard class of the screen grid.
        # A grid keeps the true shape of long sloped spike strips (a box around them would swallow the
        # platforms above), survives partial views, and follows moving hazards (cells are cleared when
        # their place is on screen without danger in it).
        self.danger: dict[tuple, tuple] = {}  # cell -> tight box (x0, x1, y0, y1) of the danger seen in it
        self._danger_rects = None

    def reset(self) -> None:
        self.solids.clear()
        self.hazards.clear()
        self.things.clear()
        self.visited.clear()
        self.inert.clear()
        self.climbed.clear()
        self.no_hold.clear()
        self.fruitless.clear()
        self.danger.clear()
        self._danger_rects = None

    def update(self, sc: Scene, standing_oid: str | None = None, motion=None) -> None:
        vx0, vy1 = sc.to_world(0, 0)
        vx1, vy0 = sc.to_world(W, H)
        fx, fy = sc.knight_world
        seen = set()
        for o in sc.platforms + sc.walls + sc.hazards + sc.interactables:
            x0, y1 = sc.to_world(o.x0, o.y0)
            x1, y0 = sc.to_world(o.x1, o.y1)
            kind = {"visible_hazard": "hazard"}.get(o.kind, o.kind)
            table = self.hazards if kind == "hazard" else self.things if kind == "interactable" else self.solids
            ol, orr, ot = o.x0 <= 0.5, o.x1 >= W - 0.5, o.y0 <= 0.5
            seen.add(o.oid)
            old = table.get(o.oid)
            mt = motion.get(o.oid) if motion is not None else None
            if kind == "platform" and mt is not None and mt.moving and mt.swings and mt.w is not None:
                # v8: a moving platform is remembered where it is now (its full size, also when the screen
                # edge cuts it), never as the union of the places it has been seen at
                bx0, bx1, by0, by1 = mt.rect(sc.time)
                if old is None:
                    old = table[o.oid] = Surf(o.oid, kind, bx0, bx1, by0, by1, False, 0.0, False, False, False, sc.tick)
                old.x0, old.x1, old.y0, old.y1 = bx0, bx1, by0, by1
                old.open_left = old.open_right = old.open_top = False
                old.moving, old.seen = True, sc.tick
                continue
            if old is None:
                thick = kind == "platform" and (o.h > 40)
                table[o.oid] = Surf(o.oid, kind, x0, x1, y0, y1, thick, ph.LIP_DEFAULT if thick else 0.0, ol, orr, ot, sc.tick)
                continue
            if kind == "hazard":
                # Hazards are kept as last seen, never as a union of views: they can move (falling
                # spikes, saws), and a long sloped strip of spikes seen piece by piece would otherwise
                # grow into one huge box that swallows the platforms above it.
                old.x0, old.x1, old.y0, old.y1 = x0, x1, y0, y1
                old.open_left, old.open_right, old.open_top = ol, orr, ot
                old.seen = sc.tick
                continue
            if x0 < old.x0 - 0.01:
                old.x0, old.open_left = x0, ol
            elif abs(x0 - old.x0) <= 0.01:
                old.open_left = old.open_left and ol
            if x1 > old.x1 + 0.01:
                old.x1, old.open_right = x1, orr
            elif abs(x1 - old.x1) <= 0.01:
                old.open_right = old.open_right and orr
            if y1 > old.y1 + 0.01:
                old.y1, old.open_top = y1, ot
            elif abs(y1 - old.y1) <= 0.01:
                old.open_top = old.open_top and ot
            old.y0 = min(old.y0, y0)
            if kind == "platform" and (old.y1 - old.y0) * sc.ppu > 40 and not old.thick:
                old.thick, old.lip = True, ph.LIP_DEFAULT
            old.seen = sc.tick
        # measured lip of the surface under the feet: its walkable top is exactly where the feet are
        if standing_oid is not None and standing_oid in self.solids:
            s = self.solids[standing_oid]
            s.lip = min(0.6, max(0.0, s.y1 - fy))
            self.visited.add(standing_oid)
        # v8: a moving platform that is off the picture keeps moving: it is remembered where the motion
        # memory says it is now (first seen wrong in laya_01: a jump aimed at where one had been 10 s before)
        if motion is not None:
            for oid, s in self.solids.items():
                if s.moving and oid not in seen:
                    mt = motion.get(oid)
                    if mt is not None and mt.swings and mt.w is not None and mt.cx is not None:
                        s.x0, s.x1, s.y0, s.y1 = mt.rect(sc.time)
        self._update_danger(sc, motion)
        # forget things whose place is fully on screen now but which are no longer drawn
        hud_y = sc.to_world(0, 115)[1]
        for table in (self.solids, self.hazards, self.things):
            for oid in [k for k in table if k not in seen]:
                s = table[oid]
                inside = s.x0 > vx0 + 0.3 and s.x1 < vx1 - 0.3 and s.y0 > vy0 + 0.3 and s.y1 < min(vy1 - 0.3, hud_y)
                if inside:
                    if table is self.solids and s.kind == "platform" and oid in self.visited and not s.moving:
                        # a platform the knight stood on is gone: platforms of this look do not hold
                        self.collapsing.add(self.look(s))
                    del table[oid]

    GRID = 0.5

    def _update_danger(self, sc: Scene, motion=None) -> None:
        """Mark world cells where the screen grid shows the hazard class.

        Each cell keeps the tight box of the danger seen inside it: the screen cell (20 px) is clipped
        to the pixel-accurate bounds of the hazard sprites of this frame, so that a spike pit does not
        'leak' half a unit onto the floor edge next to it.
        v8: hazards known to roll back and forth or to fly across (saws, projectiles) are left out: a
        place they pass is not dangerous for good; they are handled by prediction (control.Ctl.dyn).
        """
        g = self.GRID
        haz = (sc.masks & HAZ) != 0
        boxes = []
        movers = []
        for o in sc.hazards:
            x0, y1 = sc.to_world(o.x0, o.y0)
            x1, y0 = sc.to_world(o.x1, o.y1)
            mt = motion.get(o.oid) if motion is not None else None
            if mt is not None and mt.predictable:
                movers.append((x0 - 0.02, x1 + 0.02, y0 - 0.02, y1 + 0.02))
                continue
            boxes.append((x0, x1, y0, y1))
        seen = {}
        for r in range(ROWS):
            row = haz[r]
            if not row.any():
                continue
            for c in range(COLS):
                if not row[c]:
                    continue
                cx0, cy1 = sc.to_world(c * CELL, r * CELL)
                cx1, cy0 = sc.to_world((c + 1) * CELL, (r + 1) * CELL)
                pieces = [(max(cx0, b[0]), min(cx1, b[1]), max(cy0, b[2]), min(cy1, b[3])) for b in boxes]
                pieces = [q for q in pieces if q[1] > q[0] and q[3] > q[2]]
                if not pieces:
                    if any(cx1 > m[0] and cx0 < m[1] and cy1 > m[2] and cy0 < m[3] for m in movers):
                        continue                            # the danger in this cell is a moving thing
                    pieces = [(cx0, cx1, cy0, cy1)]
                for px0, px1, py0, py1 in pieces:
                    for ix in range(math.floor(px0 / g), math.floor((px1 - 1e-6) / g) + 1):
                        for iy in range(math.floor(py0 / g), math.floor((py1 - 1e-6) / g) + 1):
                            q = (max(px0, ix * g), min(px1, (ix + 1) * g), max(py0, iy * g), min(py1, (iy + 1) * g))
                            old = seen.get((ix, iy))
                            seen[(ix, iy)] = q if old is None else (min(old[0], q[0]), max(old[1], q[1]), min(old[2], q[2]), max(old[3], q[3]))
        vx0, vy1 = sc.to_world(0, 0)
        vx1, vy0 = sc.to_world(W, H)
        gone = []
        for cell in self.danger:
            if cell in seen:
                continue
            cx0, cy0 = cell[0] * g, cell[1] * g
            if cx0 < vx0 or cx0 + g > vx1 or cy0 < vy0 or cy0 + g > vy1:
                continue                                    # not (fully) on screen: keep the memory
            sx0, sy0 = sc.to_screen(cx0, cy0 + g)
            sx1, sy1 = sc.to_screen(cx0 + g, cy0)
            block = sc.cells(sx0 + 1, sy0 + 1, sx1 - 1, sy1 - 1)
            if block.size and not ((block & UNK) != 0).all():   # visible (not behind the HUD) and no danger there
                gone.append(cell)
        # v8: a hazard now known to move may have been marked as standing danger at its first sightings
        # (before two looks showed that it moves, or while the screen edge cut it). Wherever any part of
        # it was ever drawn, danger that is not confirmed by a standing hazard in this very frame is dropped.
        if motion is not None and self.danger:
            for tr in motion.hazards():
                if tr.swept is None:
                    continue
                mx0, mx1, my0, my1 = tr.swept
                for cell in self.danger:
                    if cell in seen or cell in gone:
                        continue
                    cx0, cy0 = cell[0] * g, cell[1] * g
                    if cx0 + g > mx0 - 0.05 and cx0 < mx1 + 0.05 and cy0 + g > my0 - 0.05 and cy0 < my1 + 0.05:
                        gone.append(cell)
        changed = bool(gone)
        for cell in gone:
            del self.danger[cell]
        for cell, q in seen.items():
            old = self.danger.get(cell)
            new = q if old is None else (min(old[0], q[0]), max(old[1], q[1]), min(old[2], q[2]), max(old[3], q[3]))
            if new != old:
                self.danger[cell] = new
                changed = True
        if changed:
            self._danger_rects = None

    def danger_rects(self) -> list:
        """The danger grid merged into a few world rectangles (cached until the grid changes)."""
        if self._danger_rects is None:
            rows = {}
            for (ix, iy) in self.danger:
                rows.setdefault(iy, []).append(ix)
            runs = {}                                       # (ix0, ix1) -> list of (iy, x0, x1, y0, y1)
            for iy, xs in rows.items():
                xs.sort()
                start = prev = xs[0]
                for ix in xs[1:] + [None]:
                    if ix is None or ix != prev + 1:
                        cells = [self.danger[(k, iy)] for k in range(start, prev + 1)]
                        runs.setdefault((start, prev), []).append(
                            (iy, cells[0][0], cells[-1][1], min(q[2] for q in cells), max(q[3] for q in cells)))
                        start = ix
                    prev = ix if ix is not None else prev
            rects = []
            for key, items in runs.items():
                items.sort()
                group = [items[0]]
                for it in items[1:] + [None]:
                    if it is None or it[0] != group[-1][0] + 1:
                        rects.append(ph.WRect(min(q[1] for q in group) - 0.08, max(q[2] for q in group) + 0.08,
                                              min(q[3] for q in group) - 0.05, max(q[4] for q in group) + 0.12, "hazard", "danger"))
                        group = []
                    if it is not None:
                        group.append(it)
            self._danger_rects = rects
        return self._danger_rects

    @staticmethod
    def look(s: Surf) -> tuple:
        return (round(s.x1 - s.x0, 1), round(s.y1 - s.y0, 1))

    def is_collapsing(self, s: Surf | None) -> bool:
        return s is not None and not (s.open_left or s.open_right) and self.look(s) in self.collapsing

    def rects(self, x: float, y: float, radius: float = 22.0):
        """Standing-still solids and dangers around (x, y). Moving platforms are not in here: the planner
        gets them, with where they will be, from control.Ctl.dyn()."""
        solids = [s.wrect() for s in self.solids.values() if s.x1 > x - radius and s.x0 < x + radius and not s.moving]
        hazards = [r for r in self.danger_rects() if r.x1 > x - radius and r.x0 < x + radius
                   and r.y1 > y - 14 and r.y0 < y + 12]
        return solids, hazards

    def safe_spans(self, surf: Surf, x: float | None = None, y: float | None = None) -> list[tuple[float, float]]:
        """Parts of a surface one can stand on: not under/next to a hazard, not under a low ceiling.

        If the knight is above the surface (y given), parts shadowed by another solid in between are
        cut out as well (one cannot arrive there from above).
        """
        cuts = []
        top = surf.top
        for hz in self.danger_rects():
            if hz.x1 > surf.x0 - 1 and hz.x0 < surf.x1 + 1 and hz.y1 > top - 0.2 and hz.y0 < top + ph.BODY_H + 0.2:
                cuts.append((hz.x0 - 0.45, hz.x1 + 0.45))
        for o in self.solids.values():
            if o.oid == surf.oid or o.x1 <= surf.x0 or o.x0 >= surf.x1:
                continue
            if o.y0 < top + ph.BODY_H + 0.05 and o.y1 > top + 0.35:
                cuts.append((o.x0 - ph.BODY_HW - 0.05, o.x1 + ph.BODY_HW + 0.05))       # low ceiling / block on it
            elif y is not None and y > top + 0.5 and o.y0 >= top + ph.BODY_H and o.top <= y + 2.6:
                cuts.append((o.x0 - ph.BODY_HW - 0.3, o.x1 + ph.BODY_HW + 0.3))         # shadow from above
        spans = [(surf.x0, surf.x1)]
        for c0, c1 in cuts:
            nxt = []
            for a, b in spans:
                if c1 <= a or c0 >= b:
                    nxt.append((a, b))
                else:
                    if c0 > a:
                        nxt.append((a, c0))
                    if c1 < b:
                        nxt.append((c1, b))
            spans = nxt
        return [(a, b) for a, b in spans if b - a >= 0.9]
