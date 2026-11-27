# fmt: off
# isort: skip_file
# ruff: noqa
"""Knight physics model + jump planner (pure computation, no game access).

Constants were measured with probes/probe_physics.py (probe_01_physics) — the designer measuring the
arcs once, as the brief suggests — and the model below reproduces the probe trace frame by frame:
  * physics runs in fixed 0.02 s steps; a rendered simulation frame is 1/30 s, so it contains 1 or 2
    fixed steps; the count follows from the simulation time shown in the visible observation;
  * walking / air steering: horizontal speed is exactly +-5 units/s while a left/right axis is held and
    0 otherwise (no inertia);
  * a jump press sets vertical speed to 7 units/s; gravity 9.81; a second press in the air does the
    same once (double jump). There is no short hop (no jump-release edge in the bridge).
Units here are world units (y up). 1 unit = 54 px on screen.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

G = 9.81
DT = 0.02
FDT = 1.0 / 30.0
WALK = 5.0
JUMP_V = 7.0
BODY_HW = 0.33
BODY_H = 1.32
LIP_DEFAULT = 0.22      # thick floor sprites: walkable top is this far below the sprite's visible top


def nsteps(t0: float, t1: float) -> int:
    """Fixed physics steps between two simulation times (matches Unity's accumulation)."""
    return int(math.floor(t1 / DT + 1e-3) - math.floor(t0 / DT + 1e-3))


@dataclass
class WRect:
    """World rectangle, y up. For a solid, `top` is the walkable top (sprite top minus lip)."""
    x0: float
    x1: float
    y0: float
    y1: float
    kind: str = "platform"
    oid: str = ""
    lip: float = 0.0

    @property
    def top(self) -> float:
        return self.y1 - self.lip


@dataclass
class Plan:
    ok: bool
    reason: str = ""
    press1: bool = False
    delay: int = 0                 # simulation frames before horizontal motion starts
    t2: int | None = None          # frame boundary of the second jump press
    dirn: int = 1
    x_land: float = 0.0
    land_frame: int = 0
    land_x: float = 0.0
    clearance: float = 0.0
    path: list = field(default_factory=list)   # per frame boundary: (x, y, vy, axis)
    landed_on: str = ""
    gap: float = 9.0               # v8: smallest distance to a moving hazard along the path
    wait: int = 0                  # v8: frames to wait before the press (timed jumps)
    dash: int | None = None        # v8: frame boundary of a dash press toward dirn

    def score(self) -> tuple:
        # prefer keeping the second jump, then a comfortable clearance, then a quick landing
        return (0 if self.t2 is None else 1, -min(self.clearance, 0.8), self.land_frame)


UNSURE = "its motion is not known that far yet"
UNSEEN = "it moves and is out of sight now"     # a remembered moving platform that cannot be watched from here
HEAD_SKIN = 0.015       # the head stops this far below the underside of a solid (measured, probe_01_saw)
HAZ_MARGIN = 0.12       # keep this far from a predicted moving hazard
DASH_V = 10.0           # dash: horizontal speed ...
DASH_FRAMES = 5         # ... during the 5 frames after the press (0.15 s), steering ignored (measured, probe_04_zero);
                        # the executor ends a window exactly there, so that steering takes over at once
POGO_VY = 4.0           # vertical speed after a downward strike that hit something (measured, probe_03_pogo)
STRIKE_REACH = 1.44     # a downward strike reaches this far below the feet (0.6 radius swept 1.5 from the centre)


class Dyn:
    """Predicted moving things for one planning call (v8), indexed by frame boundary k = 0..n from the
    time of the current observation. Built by control.Ctl.dyn() from the motion memory.

      haz: per moving hazard a list of shapes, ('disc', cx, cy, r) or ('box', x0, x1, y0, y1);
      sol: per moving platform (rect, x0, x1, dx) where rect is a WRect that is shifted in place to
           x0 + dx[k], x1 + dx[k] while a simulation runs.
    """

    def __init__(self, n: int):
        self.n = n
        self.haz: list = []
        self.sol: list = []
        self.haz_ids: list = []
        self.sure: dict = {}          # moving platform id -> last boundary up to which its motion is certain

    def place(self, k: int) -> None:
        k = min(max(k, 0), self.n)
        for rect, x0, x1, dx in self.sol:
            rect.x0, rect.x1 = x0 + dx[k], x1 + dx[k]

    def shift(self, oid: str, k: int) -> float:
        """How far the moving platform `oid` has moved at boundary k (0 for anything that stands still)."""
        k = min(max(k, 0), self.n)
        for rect, x0, x1, dx in self.sol:
            if rect.oid == oid:
                return dx[k]
        return 0.0

    def clear(self, k: int, x: float, y: float, margin: float = HAZ_MARGIN) -> bool:
        """Is the knight's body at (x, y feet) free of every moving hazard at boundary k?"""
        k = min(max(k, 0), self.n)
        for shapes in self.haz:
            if hits(shapes[k], x, y, margin):
                return False
        return True

    def gap(self, k: int, x: float, y: float) -> float:
        """Smallest distance between the body and any moving hazard at boundary k (9 when none)."""
        k = min(max(k, 0), self.n)
        best = 9.0
        for shapes in self.haz:
            best = min(best, distance(shapes[k], x, y))
        return best


def distance(shape: tuple, x: float, y: float) -> float:
    """Distance between the knight's body box (centre x, feet y) and a hazard shape (negative inside)."""
    if shape[0] == "disc":
        _, cx, cy, r = shape
        dx = abs(x - cx) - BODY_HW
        if dx < 0:
            dx = 0.0
        if cy < y:
            dy = y - cy
        elif cy > y + BODY_H:
            dy = cy - y - BODY_H
        else:
            dy = 0.0
        return math.hypot(dx, dy) - r
    _, x0, x1, y0, y1 = shape
    dx = max(x0 - (x + BODY_HW), (x - BODY_HW) - x1, 0.0)
    dy = max(y0 - (y + BODY_H), y - y1, 0.0)
    if dx == 0.0 and dy == 0.0:
        return -min(x + BODY_HW - x0, x1 - (x - BODY_HW), y + BODY_H - y0, y1 - y)
    return math.hypot(dx, dy)


def hits(shape: tuple, x: float, y: float, margin: float = HAZ_MARGIN) -> bool:
    return distance(shape, x, y) < margin


def simulate(x: float, y: float, vy: float, t0: float, jumps: int, press1: bool, delay: int, t2: int | None,
             dirn: int, x_land: float, target: WRect, land_lo: float, land_hi: float,
             solids: list[WRect], hazards: list[WRect], max_frames: int = 170,
             clear_need: float = 0.0, dyn: Dyn | None = None, k0: int = 0, bump: bool = False,
             dash: int | None = None, safe: int = 0) -> Plan:
    """Roll the model forward under one parameterised jump plan.

    Start: feet centre (x, y), vertical speed vy, simulation time t0, `jumps` presses still available.
    press1: press jump at boundary 0 (from the ground, or an air jump right now).
    Horizontal: hold `dirn` from frame `delay` until the centre reaches x_land; a press frame always
    moves one frame toward `dirn` (the bridge has no neutral jump).
    v8: `dyn` adds the predicted moving things (frame boundary k0 = the start of this simulation):
    moving hazards must be kept clear of, moving platforms are solids that are where they will be, and a
    landing zone on a moving platform moves with it. With `bump`, hitting the underside of a solid from
    below does not fail the plan: the head stops there and the knight falls (measured in probe_01_saw).
    `dash`: frame boundary of a dash press toward `dirn` (measured in probe_04_zero: vertical speed is set to 0,
    then 8 physics steps at 10 units/s during which steering is ignored; gravity keeps acting).
    `safe`: for this many frames from the start nothing dangerous can hurt (the knight was just hit and is
    untouchable for 1.5 s): hazards are ignored on that part of the path.
    """
    plan = Plan(False, press1=press1, delay=delay, t2=t2, dirn=dirn, x_land=x_land, dash=dash)
    dash_left = 0
    if dyn is not None and dyn.sol:
        solids = solids + [m[0] for m in dyn.sol]
    gap = 9.0
    if press1:
        if jumps <= 0:
            plan.reason = "no jump left"
            return plan
        vy = JUMP_V
        jumps -= 1
    forced = 1 if press1 else 0          # frames of forced horizontal axis after a press
    clearance = 9.0
    path = [(x, y, vy, 0)]
    t = t0
    for frame in range(1, max_frames + 1):
        boundary = frame - 1
        if t2 is not None and boundary == t2 and boundary > 0:
            if jumps <= 0:
                plan.reason = "no second jump"
                return plan
            vy = JUMP_V
            jumps -= 1
            forced = 1
        if dash is not None and boundary == dash:
            vy, dash_left = 0.0, DASH_FRAMES + 1
        shift = 0.0
        if dyn is not None:
            kb = k0 + frame                      # the boundary at the end of this frame
            dyn.place(kb)
            shift = dyn.shift(target.oid, kb)
        remaining = (x_land + shift - x) * dirn
        axis = 0
        if forced > 0:
            axis = dirn
            forced -= 1
        elif boundary >= delay and remaining > 0.05:
            axis = dirn
        n = nsteps(t, t + FDT)
        t += FDT
        if dash_left > 0:
            dash_left -= 1
        for _ in range(n):
            vy -= G * DT
            if dash_left > 0:
                nx = x + dirn * DASH_V * DT
            else:
                nx = x + axis * WALK * DT
            ny = y + vy * DT
            # landing / collisions against solids
            for r in solids:
                if nx + BODY_HW <= r.x0 + 0.02 or nx - BODY_HW >= r.x1 - 0.02:
                    continue
                top = r.top
                if vy <= 0 and y >= top - 1e-6 and ny <= top:
                    # lands on r
                    plan.land_frame, plan.land_x, plan.landed_on = frame, nx, r.oid
                    plan.clearance = clearance
                    plan.gap = gap
                    plan.path = path + [(nx, top, 0.0, axis)]
                    if dyn is not None and kb > dyn.sure.get(r.oid, 1 << 30):
                        plan.reason = UNSURE          # a moving platform, and where it is by then is not certain
                        return plan
                    if r.oid == target.oid and land_lo + shift <= nx <= land_hi + shift:
                        if clearance >= clear_need:
                            plan.ok = True
                        else:
                            plan.reason = "edge clearance too small"
                    else:
                        plan.reason = "lands elsewhere" if r.oid != target.oid else "lands outside the safe zone"
                    return plan
                if ny < top - 0.02 and ny + BODY_H > r.y0 + 0.02:
                    if (bump and vy > 0 and y + BODY_H <= r.y0 + 0.03
                            and x + BODY_HW > r.x0 + 0.02 and x - BODY_HW < r.x1 - 0.02):
                        ny = r.y0 - BODY_H - HEAD_SKIN       # head bump: stop under it and fall
                        vy = 0.0
                        continue
                    plan.reason = f"hits {r.kind}"
                    plan.path = path
                    return plan
            if frame > safe:
                for hz in hazards:
                    if nx + BODY_HW > hz.x0 and nx - BODY_HW < hz.x1 and ny < hz.y1 and ny + BODY_H > hz.y0:
                        plan.reason = "touches hazard"
                        plan.path = path
                        return plan
            x, y = nx, ny
        if dyn is not None and dyn.haz and frame > safe:
            g = min(dyn.gap(kb, x, y), dyn.gap(kb - 1, x, y))
            if g < HAZ_MARGIN:
                plan.reason = "touches a moving hazard"
                plan.path = path
                return plan
            gap = min(gap, g)
        # clearance above the target's top while the body straddles one of its edges (the moment a
        # too-low approach would bump into the platform's side)
        if (x - BODY_HW < target.x0 < x + BODY_HW) or (x - BODY_HW < target.x1 < x + BODY_HW):
            clearance = min(clearance, y - target.top)
        path.append((x, y, vy, axis))
        if y < target.top - 6.0:
            break
    plan.reason = "never lands"
    plan.path = path
    return plan


def plan_jump(x: float, y: float, vy: float, t0: float, jumps: int, grounded: bool, target: WRect,
              solids: list[WRect], hazards: list[WRect], margin: float = 0.3, clear_need: float = 0.25,
              allow_second: bool = True, dyn: Dyn | None = None, k0: int = 0, bump: bool = False,
              delays=None, t2s=None, safe: int = 0) -> Plan:
    """Search the small plan family and return the best valid plan (or the most informative failure).

    v8: `dyn`/`k0`/`bump` as in simulate(); `delays` / `t2s` restrict the family (used when the same search
    is repeated for many start times)."""
    lo, hi = target.x0 + margin, target.x1 - margin
    if hi < lo:
        lo = hi = (target.x0 + target.x1) / 2
    if lo <= x <= hi:
        x_land, dirn = x, (1 if (target.x0 + target.x1) / 2 >= x else -1)
    elif x < lo:
        dirn = 1
        x_land = min(lo + 0.35, (lo + hi) / 2)
    else:
        dirn = -1
        x_land = max(hi - 0.35, (lo + hi) / 2)
    best, fail = None, None
    press_options = [True] if grounded else [False, True]
    for press1 in press_options:
        left = jumps - (1 if press1 else 0)
        if left < 0:
            continue
        t2_options = [None]
        if allow_second and left >= 1:
            t2_options += list(range(3, 46, 3)) if t2s is None else list(t2s)
        for t2 in t2_options:
            for delay in (range(0, 46, 3) if delays is None else delays):
                p = simulate(x, y, vy, t0, jumps, press1, delay, t2, dirn, x_land, target, lo - 0.05, hi + 0.05,
                             solids, hazards, clear_need=clear_need, dyn=dyn, k0=k0, bump=bump, safe=safe)
                if p.ok:
                    if best is None or p.score() < best.score():
                        best = p
                elif fail is None or len(p.path) > len(fail.path):
                    fail = p
            if best is not None and t2 is None:
                break       # a single-jump plan exists; do not search double jumps
        if best is not None:
            break
    return best if best is not None else (fail or Plan(False, reason="no plan"))


@dataclass
class Cross:
    """A hop over a moving hazard on the knight's own floor (v8)."""
    ok: bool
    reason: str = ""
    walk: int = 0                  # frames to walk toward it before the press
    plan: Plan | None = None       # the jump from the takeoff point
    gap: float = 9.0               # smallest distance to a moving hazard over walk-in, jump and run-out
    x_takeoff: float = 0.0


def plan_cross(x: float, y: float, t0: float, dirn: int, floor: WRect, solids: list[WRect], hazards: list[WRect],
               dyn: Dyn, block: int, walk_room: float, max_walk: int = 30, runout: int = 20,
               need_gap: float = HAZ_MARGIN, dashes=(None,)) -> Cross:
    """Can the knight get past moving hazard number `block` of `dyn` by hopping over it right now?

    The manoeuvre: walk `walk` frames toward it, press jump and keep steering (under a low ceiling the head
    bumps and the hop stays low), land on the same floor behind the hazard and keep walking. Every frame is
    checked against where all moving hazards are predicted to be then. Among the feasible walk-in lengths the
    one that leaves the largest gap wins.
    """
    best, why = None, "no hop clears it now"
    xx, tt, walk_gap = x, t0, 9.0
    for w in range(0, max_walk + 1):
        if w > 0:
            n = nsteps(tt, tt + FDT)
            tt += FDT
            xx += dirn * n * WALK * DT
            g = dyn.gap(w, xx, y)
            if g < HAZ_MARGIN or abs(xx - x) > walk_room + 1e-6:
                break                              # walking further in only gets worse
            walk_gap = min(walk_gap, g)
        if w % 2:
            continue
        for dash in dashes:
            c = _cross_variant(xx, y, tt, dirn, floor, solids, hazards, dyn, block, w, dash, walk_gap, runout)
            if isinstance(c, str):
                why = c
            elif c.gap >= need_gap and (best is None or c.gap > best.gap + 0.02):
                best = c
    return best or Cross(False, reason=why)


def _cross_variant(xx, y, tt, dirn, floor, solids, hazards, dyn, block, w, dash, walk_gap, runout):
    """One hop of plan_cross: Cross, or the reason it fails."""
    if True:
        why = ""
        p = simulate(xx, y, 0.0, tt, 2, True, 0, None, dirn, xx + dirn * 40.0, floor, -1e9, 1e9, solids, hazards,
                     dyn=dyn, k0=w, bump=True, dash=dash)
        if not p.ok:
            return p.reason or "no hop clears it now"
        land = w + p.land_frame
        shape = dyn.haz[block][min(land, dyn.n)]
        centre = shape[1] if shape[0] == "disc" else (shape[1] + shape[2]) / 2
        half = shape[3] if shape[0] == "disc" else (shape[2] - shape[1]) / 2
        if (p.land_x - centre) * dirn < half + BODY_HW + 0.2:
            return "would land in front of it"
        gap = min(walk_gap, p.gap)
        xl, tl, ok = p.land_x, tt + p.land_frame * FDT, True
        for k in range(1, runout + 1):             # keep walking after the landing: nothing may catch up
            n = nsteps(tl, tl + FDT)
            tl += FDT
            xl += dirn * n * WALK * DT
            g = dyn.gap(land + k, xl, floor.top)
            if g < HAZ_MARGIN + 0.1:
                ok = False
                break
            gap = min(gap, g)
        if not ok:
            return "it would catch up after the landing"
        return Cross(True, walk=w, plan=p, gap=gap, x_takeoff=xx)


def velocity_after(dy: float, n: int) -> float:
    """Vertical speed at the end of a window of n fixed steps with free flight and net rise dy."""
    if n <= 0:
        return 0.0
    v0 = (dy / DT + G * DT * n * (n + 1) / 2) / n
    return v0 - n * G * DT


# =================================================================================================
# Wall mechanics (measured with probes/probe_wall.py, probe_03_wall):
#   * flying into a wall while airborne -> the knight clings: slides down at exactly 2 units/s, no gravity,
#     and has one jump again;
#   * a jump press while clinging is a wall kick: vertical speed 8, horizontal speed away from the wall
#     3 (press toward the wall) or 13 (press away) for 0.2 s = 6 frames during which input is ignored;
#     gravity is back at once; one air jump is still available afterwards;
#   * kick toward + steer back = +2.2 units per 0.33 s cycle on the same wall;
#   * jumping from the ground while already touching the wall does not start a cling.
# =================================================================================================
KICK_VY = 8.0
KICK_VX_TOWARD = 3.0
KICK_VX_AWAY = 13.0
KICK_LOCK = 6
SLIDE_V = 2.0
WALL_GAP = 0.015


@dataclass
class Start:
    x: float
    y: float
    vy: float
    t0: float
    jumps: int
    mode: str = "ground"      # ground | air | climb
    side: int = 0             # climb: +1 wall on the right of the knight, -1 wall on the left


@dataclass
class Move:
    first: str | None         # None | jump | kick_toward | kick_away
    dirn: int                 # steering direction after `delay`
    delay: int
    t2: int | None
    x_goal: float | None      # stop steering when the centre reached it; None = steer until contact


@dataclass
class Outcome:
    kind: str                 # land | wall | fail
    oid: str = ""
    side: int = 0
    x: float = 0.0
    y: float = 0.0
    frame: int = 0
    reason: str = ""
    clearance: float = 9.0
    path: list = field(default_factory=list)       # per frame boundary: (x, y, vy, axis)
    presses: dict = field(default_factory=dict)    # boundary -> (action name, window frames or None)
    move: Move | None = None


def simulate2(st: Start, mv: Move, solids: list[WRect], hazards: list[WRect], target: WRect | None = None,
              max_frames: int = 170) -> Outcome:
    x, y, vy, t, jumps = st.x, st.y, st.vy, st.t0, st.jumps
    out = Outcome("fail", move=mv)
    lock, kick_vx, lock_axis, forced = 0, 0.0, 0, 0
    nograv = 0
    jname = {1: "jump_right", -1: "jump_left"}
    if mv.first == "jump":
        if jumps <= 0:
            out.reason = "no jump left"
            return out
        vy, jumps, forced = JUMP_V, jumps - 1, 1
        out.presses[0] = (jname[mv.dirn], None)
    elif mv.first in ("kick_toward", "kick_away"):
        if st.mode != "climb" or st.side == 0:
            out.reason = "not on a wall"
            return out
        away = -st.side
        toward = mv.first == "kick_toward"
        kick_vx = away * (KICK_VX_TOWARD if toward else KICK_VX_AWAY)
        lock_axis = st.side if toward else away
        vy, lock, jumps = KICK_VY, KICK_LOCK, 1
        nograv = 2            # gravity returns two fixed steps after the kick (measured)
        out.presses[0] = (jname[lock_axis], KICK_LOCK)
    clearance = 9.0
    path = [(x, y, vy, 0)]
    for frame in range(1, max_frames + 1):
        boundary = frame - 1
        if mv.t2 is not None and boundary == mv.t2 and boundary > 0 and lock == 0:
            if jumps <= 0:
                out.reason, out.path = "no second jump", path
                return out
            vy, jumps, forced = JUMP_V, jumps - 1, 1
            out.presses[boundary] = (jname[mv.dirn], None)
        if lock > 0:
            axis, vx = lock_axis, kick_vx
        elif forced > 0:
            axis, forced = mv.dirn, forced - 1
            vx = axis * WALK
        elif boundary >= mv.delay and (mv.x_goal is None or (mv.x_goal - x) * mv.dirn > 0.05):
            axis = mv.dirn
            vx = axis * WALK
        else:
            axis, vx = 0, 0.0
        n = nsteps(t, t + FDT)
        t += FDT
        for _ in range(n):
            if nograv > 0:
                nograv -= 1
            else:
                vy -= G * DT
            nx = x + vx * DT
            ny = y + vy * DT
            for r in solids:
                if nx + BODY_HW <= r.x0 + 0.02 or nx - BODY_HW >= r.x1 - 0.02:
                    continue
                top = r.top
                if vy <= 0 and y >= top - 1e-6 and ny <= top:
                    out.kind, out.oid, out.x, out.y, out.frame = "land", r.oid, nx, top, frame
                    out.clearance, out.path = clearance, path + [(nx, top, 0.0, axis)]
                    return out
                if ny < top - 0.02 and ny + BODY_H > r.y0 + 0.02:
                    from_side = x + BODY_HW <= r.x0 + 0.02 or x - BODY_HW >= r.x1 - 0.02
                    if from_side and r.kind == "wall":
                        side = 1 if nx > x else -1
                        cx = (r.x0 - BODY_HW - WALL_GAP) if side > 0 else (r.x1 + BODY_HW + WALL_GAP)
                        out.kind, out.oid, out.side, out.x, out.y, out.frame = "wall", r.oid, side, cx, ny, frame
                        out.clearance, out.path = clearance, path + [(cx, ny, -SLIDE_V, axis)]
                        return out
                    out.reason, out.path = f"hits {r.kind}", path
                    return out
            for hz in hazards:
                if nx + BODY_HW > hz.x0 and nx - BODY_HW < hz.x1 and ny < hz.y1 and ny + BODY_H > hz.y0:
                    out.reason, out.path = "touches hazard", path
                    return out
            x, y = nx, ny
        if lock > 0:
            lock -= 1
        if target is not None and ((x - BODY_HW < target.x0 < x + BODY_HW) or (x - BODY_HW < target.x1 < x + BODY_HW)):
            clearance = min(clearance, y - target.top)
        path.append((x, y, vy, axis))
        if y < st.y - 14.0:
            break
    out.reason, out.path = "never lands", path
    return out


def plan_to_wall(st: Start, wall: WRect, side: int, h_lo: float, h_hi: float, solids, hazards) -> Outcome:
    """Reach a wall face in the air (to cling to it) with the feet between h_lo and h_hi; highest wins."""
    best, fail = None, None
    if st.mode == "climb":
        firsts, delays = ["kick_toward", "kick_away"], range(6, 43, 3)
    elif st.mode == "ground":
        firsts, delays = ["jump"], range(0, 43, 3)
    else:
        firsts, delays = [None, "jump"] if st.jumps > 0 else [None], range(0, 43, 3)
    for first in firsts:
        jumps_after = st.jumps - (1 if first == "jump" else 0)
        if st.mode == "climb":
            jumps_after = 1
        t2s = [None] + (list(range(3, 43, 3)) if jumps_after >= 1 else [])
        for t2 in t2s:
            for delay in delays:
                o = simulate2(st, Move(first, side, delay, t2, None), solids, hazards)
                if o.kind == "wall" and o.oid == wall.oid and h_lo <= o.y <= h_hi:
                    key = (round(o.y, 1), 0 if t2 is None else -1, -o.frame)
                    if best is None or key > best[0]:
                        best = (key, o)
                elif fail is None or len(o.path) > len(fail.path):
                    fail = o
    if best is not None:
        return best[1]
    out = fail or Outcome("fail", reason="no plan")
    if out.kind != "fail":
        out = Outcome("fail", reason=f"would end as {out.kind} at {out.y:.1f}", path=out.path)
    return out


def plan_from_wall_to_land(st: Start, zone: WRect, solids, hazards, margin: float = 0.3, clear_need: float = 0.2) -> Outcome:
    """From a cling: kick (toward or away), optional air jump, steer, land inside `zone`."""
    lo, hi = zone.x0 + margin, zone.x1 - margin
    if hi < lo:
        lo = hi = (zone.x0 + zone.x1) / 2
    if st.x < lo:
        dirn, x_land = 1, min(lo + 0.35, (lo + hi) / 2)
    elif st.x > hi:
        dirn, x_land = -1, max(hi - 0.35, (lo + hi) / 2)
    else:
        dirn, x_land = (1 if (lo + hi) / 2 >= st.x else -1), st.x
    best, fail = None, None
    for first in ("kick_toward", "kick_away"):
        for t2 in [None] + list(range(6, 43, 3)):
            for delay in range(6, 43, 3):
                o = simulate2(st, Move(first, dirn, delay, t2, x_land), solids, hazards, target=zone)
                if o.kind == "land" and o.oid == zone.oid and lo - 0.05 <= o.x <= hi + 0.05 and o.clearance >= clear_need:
                    key = (0 if t2 is None else 1, -min(o.clearance, 0.8), o.frame)
                    if best is None or key < best[0]:
                        best = (key, o)
                elif fail is None or len(o.path) > len(fail.path):
                    fail = o
        if best is not None:
            break
    if best is not None:
        return best[1]
    out = fail or Outcome("fail", reason="no plan")
    if out.kind != "fail":
        out = Outcome("fail", reason=f"would end as {out.kind} on {out.oid}", path=out.path)
    return out
