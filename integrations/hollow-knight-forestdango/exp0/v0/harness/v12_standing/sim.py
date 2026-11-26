"""v10_sword_cadence (from v9_keys, physics unchanged; `run` takes the time already played, `t0`, so that moving
dangers are placed correctly in a continuation) forecast physics: where the keys of an option take the knight. Information tool only: it never presses
anything; its results are written into the text the model reads, and every forecast is compared with what then
happened (agent.py logs both).

The numbers were measured on this game by the designer's probes (probes/probe_keys.py, runs/probe01_jump of
experiments/hk-forestdango-demo/20261125_0017_v9_keys_harness; the same values as harness/v8f's physics.py in world
units): physics steps of 0.02 s, one or two per 1/30 s frame; walking and air steering 6.5 units/s while a
direction is held, 0 otherwise; a jump press sets the vertical speed to 18 at the start of the first two steps;
gravity 23.542; letting go of the jump key while rising stops the rise at once; the fall speed is capped at 15
before gravity is added. One jump, none in the air.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .see import CELL, BODY_H

DT = 0.02
FDT = 1.0 / 30.0
G = 0.79 * 29.8
WALK = 6.5
JUMP_V = 18.0
MAX_FALL = 15.0
LAUNCH = 2
HW = 0.25
SKIN = 0.0172          # the feet rest this far above the top of the ground
EPS = 0.02


def nsteps(t0: float, t1: float) -> int:
    return int(math.floor(t1 / DT + 1e-3) - math.floor(t0 / DT + 1e-3))


class World:
    """Solid ground for the forecast: the remembered grid of the room plus the solid boxes."""

    def __init__(self, room, boxes):
        self.room = room
        self.boxes = list(boxes)

    def blocked(self, x0: float, x1: float, y0: float, y1: float) -> bool:
        for ix in range(math.floor(x0 / CELL), math.floor((x1 - 1e-9) / CELL) + 1):
            for iy in range(math.floor(y0 / CELL), math.floor((y1 - 1e-9) / CELL) + 1):
                if (ix, iy) in self.room.solid:
                    return True
        for b in self.boxes:
            if b.x0 < x1 and b.x1 > x0 and b.y0 < y1 and b.y1 > y0:
                return True
        return False

    def top_under(self, x0: float, x1: float, y_hi: float, y_lo: float):
        """Highest solid top in the band [y_lo, y_hi] under the span, or None."""
        best = None
        for ix in range(math.floor(x0 / CELL), math.floor((x1 - 1e-9) / CELL) + 1):
            for iy in range(math.floor(y_hi / CELL), math.floor(y_lo / CELL) - 1, -1):
                if (ix, iy) in self.room.solid:
                    top = (iy + 1) * CELL
                    if best is None or top > best:
                        best = top
                    break
        for b in self.boxes:
            if b.x0 < x1 and b.x1 > x0 and y_lo <= b.y1 <= y_hi + CELL:
                if best is None or b.y1 > best:
                    best = b.y1
        return best

    def unknown(self, x: float, y: float) -> bool:
        return not self.room.known(math.floor(x / CELL), math.floor(y / CELL))


@dataclass
class State:
    x: float
    y: float
    vy: float = 0.0
    grounded: bool = True
    jumping: bool = False      # rising from a jump whose key is still down
    launch: int = 0
    t: float = 0.0


@dataclass
class Trace:
    end: State = None                      # state after the window
    path: list = field(default_factory=list)   # (frame, x, y) after every frame
    hit: tuple | None = None               # (frame, kind) first hazard / enemy touched
    landed: tuple | None = None            # (frame, x, y) first landing after leaving the ground
    unknown: tuple | None = None           # (frame, x, y) first entry into a part of the room never seen
    blocked_x: bool = False                # a wall stopped the sideways move
    bumped: bool = False                   # the head hit a ceiling


def frame(world: World, s: State, direction: int, jump_down: bool, fresh: bool, tr: Trace | None = None) -> None:
    """Advance one 1/30 s frame in place. `fresh` = the jump key goes down in this frame."""
    steps = nsteps(s.t, s.t + FDT)
    s.t += FDT
    if fresh and s.grounded:
        s.launch, s.jumping, s.grounded = LAUNCH, True, False
    for _ in range(steps):
        if s.launch > 0:
            s.vy = JUMP_V
            s.launch -= 1
        elif s.jumping and not jump_down and s.vy > 0:
            s.vy, s.jumping = 0.0, False
        if s.grounded:
            s.vy = 0.0
        else:
            if s.vy < -MAX_FALL:
                s.vy = -MAX_FALL
            s.vy -= G * DT
        if direction:
            dx = direction * WALK * DT
            for k in range(4):
                nx = s.x + dx / 4
                if world.blocked(nx - HW, nx + HW, s.y + 0.05, s.y + BODY_H - 0.02):
                    if tr is not None:
                        tr.blocked_x = True
                    break
                s.x = nx
        if s.grounded:
            if not world.blocked(s.x - HW + EPS, s.x + HW - EPS, s.y - 0.12, s.y - 0.002):
                s.grounded = False             # walked off an edge
            continue
        ny = s.y + s.vy * DT
        if s.vy <= 0:
            if world.blocked(s.x - HW + EPS, s.x + HW - EPS, ny - SKIN - 0.003, s.y):
                top = world.top_under(s.x - HW + EPS, s.x + HW - EPS, s.y, ny - CELL)
                s.y = (top if top is not None else ny) + SKIN
                s.vy, s.grounded, s.jumping, s.launch = 0.0, True, False, 0
            else:
                s.y = ny
        else:
            if world.blocked(s.x - HW + EPS, s.x + HW - EPS, s.y + BODY_H, ny + BODY_H):
                s.vy, s.jumping, s.launch = 0.0, False, 0
                if tr is not None:
                    tr.bumped = True
            else:
                s.y = ny
            if s.vy <= 0:
                s.jumping = False


def run(world: World, start: State, plan, frames: int, dangers=(), stop_on_land: bool = False, t0: float = 0.0) -> Trace:
    """Play `frames` frames. plan(k) -> (direction, jump_down, fresh) for frame k. dangers: [(Box, vx, vy)], each box
    as it is at the moment of the decision; `t0` = seconds of the decision already played before this run."""
    s = State(start.x, start.y, start.vy, start.grounded, start.jumping, start.launch, start.t)
    tr = Trace()
    was_air = not s.grounded
    for k in range(frames):
        direction, jump_down, fresh = plan(k)
        frame(world, s, direction, jump_down, fresh, tr)
        tr.path.append((k + 1, s.x, s.y))
        if not s.grounded:
            was_air = True
        if tr.hit is None:
            tsec = t0 + (k + 1) * FDT
            for b, vx, vy in dangers:
                if (b.x0 + vx * tsec < s.x + HW and b.x1 + vx * tsec > s.x - HW
                        and b.y0 + vy * tsec < s.y + BODY_H and b.y1 + vy * tsec > s.y):
                    tr.hit = (k + 1, b.kind)
                    break
        if tr.unknown is None and world.unknown(s.x, s.y + 0.3):
            tr.unknown = (k + 1, s.x, s.y)
        if s.grounded and was_air and tr.landed is None:
            tr.landed = (k + 1, s.x, s.y)
            if stop_on_land:
                break
    tr.end = s
    return tr
