"""v9_keys memory: what the knight has seen and done, as a player would remember it. Built only from the visible
observations of this run. Information tool only.

  * per room, a grid of 0.5-unit cells: solid ground / free / never on screen (the union of every camera view);
  * solid boxes, doors, hazards last seen at their place (a box that is gone when its place is on screen is forgotten);
  * where the knight has stood (2 x 2-unit places, with counts);
  * the last decisions with what came of them, and damage taken.
"""
from __future__ import annotations

import math

from .see import CELL, Box, View

PLACE = 2.0           # side of a remembered place, world units


def cell(v: float) -> int:
    return math.floor(v / CELL + 1e-9)


class Room:
    def __init__(self, code: str):
        self.code = code
        self.solid: set[tuple[int, int]] = set()
        self.seen: set[tuple[int, int]] = set()       # every cell that has been inside the camera (solid or free)
        self.boxes: dict[str, Box] = {}               # walls, doors (solid), by id
        self.hazards: dict[str, Box] = {}             # hazards that do not move, by id
        self.places: dict[tuple[int, int], int] = {}  # place -> decisions that ended with the knight standing there
        self.stood: dict[tuple[int, int], int] = {}   # 0.5-cell -> decisions that ended with the feet standing in it
        self.walked: set[tuple[int, int]] = set()     # every 0.5-cell the feet have stood in or walked through

    def update(self, view: View) -> None:
        cam = view.cam
        ix0, ix1 = cell(cam.x0 + CELL / 2), cell(cam.x1 - CELL / 2)
        iy0, iy1 = cell(cam.y0 + CELL / 2), cell(cam.y1 - CELL / 2)
        now = set()
        for t in view.terrain:
            # a cell belongs to a rectangle when its centre is inside (rectangles are cut on the 0.5 raster; the
            # pixel rounding of the bounds is below 0.04)
            for ix in range(max(ix0, math.ceil((t.x0 - CELL / 2) / CELL - 1e-6)), min(ix1, math.floor((t.x1 - CELL / 2) / CELL - 1e-6)) + 1):
                for iy in range(max(iy0, math.ceil((t.y0 - CELL / 2) / CELL - 1e-6)), min(iy1, math.floor((t.y1 - CELL / 2) / CELL - 1e-6)) + 1):
                    now.add((ix, iy))
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                c = (ix, iy)
                self.seen.add(c)
                if c in now:
                    self.solid.add(c)
                else:
                    self.solid.discard(c)
        ids = set()
        for b in view.solids:
            self.boxes[b.oid] = self._whole(self.boxes.get(b.oid), b, cam)
            ids.add(b.oid)
        for oid, b in list(self.boxes.items()):
            if oid not in ids and view.on_screen(b.cx, b.cy):
                del self.boxes[oid]                   # it was there, now its place shows nothing: broken or gone
        ids = set()
        for hz in view.hazards:
            self.hazards[hz.oid] = self._whole(self.hazards.get(hz.oid), hz, cam)
            ids.add(hz.oid)
        for oid, b in list(self.hazards.items()):
            if oid not in ids and view.on_screen(b.cx, b.cy):
                del self.hazards[oid]

    @staticmethod
    def _whole(old, new, cam):
        """A box cut by the border of the screen is only partly seen: keep what was seen of it before as well
        (same size and place only; a box that moved is taken as it is now)."""
        if old is None:
            return new
        cut = (new.x0 <= cam.x0 + 0.05 or new.x1 >= cam.x1 - 0.05 or new.y0 <= cam.y0 + 0.05 or new.y1 >= cam.y1 - 0.05)
        if not cut or not old.overlaps(new.x0 - 0.05, new.x1 + 0.05, new.y0 - 0.05, new.y1 + 0.05):
            return new
        return Box(min(old.x0, new.x0), max(old.x1, new.x1), min(old.y0, new.y0), max(old.y1, new.y1), new.kind, new.oid)

    # ---- queries -----------------------------------------------------------------------------------
    def is_solid(self, ix: int, iy: int) -> bool:
        return (ix, iy) in self.solid

    def known(self, ix: int, iy: int) -> bool:
        return (ix, iy) in self.seen

    def place(self, x: float, y: float) -> tuple[int, int]:
        return math.floor(x / PLACE), math.floor((y + 0.1) / PLACE)

    def visits(self, x: float, y: float) -> int:
        """How often the knight has stood near this point: 0 when its feet were never within 1 unit of it at
        this height; else the number of decisions that ended there (at least 1)."""
        ix, iy = cell(x), cell(y + 0.1)
        near = [(ix + dx, iy + dy) for dx in (-2, -1, 0, 1, 2) for dy in (-1, 0, 1)]
        if not any(c in self.walked for c in near):
            return 0
        return max(1, sum(self.stood.get(c, 0) for c in near))


class Memory:
    def __init__(self):
        self.rooms: dict[str, Room] = {}
        self.room_order: list[str] = []
        self.recent: list[dict] = []          # last decisions: {"move","attack","result"}
        self.damage: list[dict] = []          # {"decision", "what", "lost"}
        self.tried: dict[tuple, int] = {}     # ((room, place), move option) -> times chosen while standing there
        self.kills = 0
        self.deaths = 0
        self.decisions = 0

    def room(self, code: str) -> Room:
        if code not in self.rooms:
            self.rooms[code] = Room(code)
            self.room_order.append(code)
        return self.rooms[code]

    def see(self, view: View) -> Room:
        room = self.room(view.room)
        if view.segment == "play":
            room.update(view)
        return room

    def stand(self, view: View, grounded: bool, before: View | None = None, was_grounded: bool = False) -> None:
        """After a decision: remember where the knight stands, and the ground it walked over to get there."""
        if not grounded or view.me is None:
            return
        room = self.room(view.room)
        p = room.place(view.x, view.y)
        room.places[p] = room.places.get(p, 0) + 1
        c = (cell(view.x), cell(view.y + 0.1))
        room.stood[c] = room.stood.get(c, 0) + 1
        room.walked.add(c)
        if before is not None and was_grounded and before.room == view.room and abs(before.y - view.y) < 0.3:
            for ix in range(min(cell(before.x), c[0]), max(cell(before.x), c[0]) + 1):
                room.walked.add((ix, c[1]))
