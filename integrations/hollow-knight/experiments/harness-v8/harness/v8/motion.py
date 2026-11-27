# fmt: off
# isort: skip_file
# ruff: noqa
"""Motion memory (new in v8): how hazards and platforms move, learned by watching.

Input is only what a player sees: the same object drawn at a different place a moment later (the
visible observation keeps an object's id while it exists). From successive sightings this module
estimates each thing's velocity and tells three kinds of motion apart:

  * still                       (spikes, ordinary platforms),
  * back and forth along x      (rolling saw, moving platform): constant speed, turns around at two
                                fixed points; a turning point is known once the thing was seen to
                                turn there (the two straight runs before and after are intersected,
                                so the turn itself may happen between two looks or just off screen);
  * straight flight             (projectile): constant velocity, never turns.

What is followed is a real edge of the thing: its left edge and its top edge as long as the screen does
not cut them (a platform ahead of the knight is usually cut by the right screen edge and still shows
its left edge). Telling real motion from apparent motion: a thing partly hidden behind something fixed
on the screen (the HUD, a caption box, the screen edge) seems to move when the knight - and with it
the camera - moves; such an edge moves exactly as the knight does. So a run of sightings counts as
motion when the thing was seen whole with an unchanged size throughout, or when its edge moved
differently from the knight.

`Track.predict(t)` says where the centre will be at simulation time t; `Track.certain(t)` says whether
that is safe to rely on: a back-and-forth mover whose turning point ahead has not been seen yet is
only certain to keep going as far as it was already seen to go.

Nothing here reads the debug state.
"""
from __future__ import annotations

import math

from .perception import H, UNK, W, Scene

MOVE_EPS = 0.25          # net displacement (units) within one straight run that makes a thing a mover
LINE_TOL = 0.07          # deviation from the current straight run that ends the run (plus a velocity term)
PAUSE = 1.0 / 30.0       # a back-and-forth mover stands still for about one frame when it turns
FRAME = 1.0 / 30.0
STEP = 0.02              # fixed physics step
EDGE = 24.0              # px: a box side this close to the screen edge counts as cut by it (a strip of spikes
                         # cut between two teeth ends well inside the picture)
SCREEN_LOCK = 0.15       # an edge that moved within this much of the knight's own movement may be a screen artefact
FULL_TOL = 0.03          # a cut hazard shows its full height / width when the visible one is within this of the exact one


class Track:
    def __init__(self, oid: str, kind: str):
        self.oid = oid
        self.kind = kind                    # hazard | platform
        self.w = None                       # size: exact once seen whole twice with the same size,
        self.h = None                       # otherwise the largest visible size so far
        self.w_exact = False
        self.h_exact = False
        self.t = None                       # time of the latest exact position
        self._px = None                     # latest exact position: left edge x, top edge y (world)
        self._py = None
        self.box = None                     # latest visible box (x0, x1, y0, y1), world; may be cut
        self.clip = (False, False, False, False)
        self.seen = None                    # time last drawn at all
        self.first = None
        self.seg: list = []                 # samples of the current straight run: (t, px, py, knight x, knight y, whole, visible w, top edge measured)
        self.vx = 0.0
        self.vy = 0.0
        self.moving = False
        self._lo = None                     # turning points of the left edge along x
        self._hi = None
        self._xmin = None                   # extreme left-edge positions seen (it certainly goes that far)
        self._xmax = None
        self.turns = 0
        self.flew = False                   # was seen moving vertically too (flyer / falling thing)
        self._sizes: list = []
        self.speed = None                   # speed of a back-and-forth mover, from a long straight run
        self.swept = None                   # (x0, x1, y0, y1): every place any part of it was ever drawn
        self.unrest = None                  # time a real (not screen-cut) side edge was last seen at a new place
        self._edges = (None, None)

    # ---- the public view is in centre coordinates ----------------------------------------------
    @property
    def cx(self):
        return None if self._px is None else self._px + (self.w or 0.0) / 2

    @property
    def cy(self):
        return None if self._py is None else self._py - (self.h or 0.0) / 2

    def _c(self, v):
        return None if v is None else v + (self.w or 0.0) / 2

    @property
    def lo(self):
        return self._c(self._lo)

    @property
    def hi(self):
        return self._c(self._hi)

    @property
    def xmin(self):
        return self._c(self._xmin)

    @property
    def xmax(self):
        return self._c(self._xmax)

    # ---- learning ---------------------------------------------------------------------------
    def update(self, t: float, box: tuple, clip: tuple, knight: tuple = (0.0, 0.0)) -> None:
        """One sighting. `clip` = (left, right, top, bottom): that side of the box is cut by the screen
        edge (or by the caption box), so it is not the thing's real edge. `knight` = the knight's world
        position at this sighting (the camera is fixed to it)."""
        x0, x1, y0, y1 = box
        left, right, top, bottom = clip
        self.seen = t
        if self.first is None:
            self.first = t
        self.box, self.clip = box, clip
        # has a real side edge moved since the last sighting? (a platform that stands still keeps its edges)
        edges = (None if left else x0, None if right else x1)
        if self.unrest is None or any(a is not None and b is not None and abs(a - b) > 0.05 for a, b in zip(edges, self._edges)):
            self.unrest = t
        self._edges = edges
        sw = self.swept
        self.swept = box if sw is None else (min(sw[0], x0), max(sw[1], x1), min(sw[2], y0), max(sw[3], y1))
        w, h = x1 - x0, y1 - y0
        whole_x, whole_y = not (left or right), not (top or bottom)
        # size: exact after two whole sightings in a row that agree (a bigger whole sighting later corrects
        # it); until then the largest visible size
        if whole_x and whole_y:
            self._sizes.append((w, h))
            del self._sizes[:-2]
            if len(self._sizes) == 2 and abs(self._sizes[0][0] - w) + abs(self._sizes[0][1] - h) <= 0.08:
                if not self.w_exact or w > self.w + 0.08 or h > self.h + 0.08:
                    self.w, self.h = (self._sizes[0][0] + w) / 2, (self._sizes[0][1] + h) / 2
                    self.w_exact = self.h_exact = True
        else:
            self._sizes = []
        if not self.w_exact:
            self.w = w if self.w is None else max(self.w, w)
        if not self.h_exact:
            self.h = h if self.h is None else max(self.h, h)
        # position: the left edge / the top edge where they are real; behind a cut edge only with the exact size
        px = x0 if not left else (x1 - self.w if (not right and self.w_exact) else None)
        py = y1 if not top else (y0 + self.h if (not bottom and self.h_exact) else None)
        # smaller than its known size although no screen edge cuts it: something in front hides part of it,
        # and which edge is the real one cannot be told
        if (self.w_exact and whole_x and w < self.w - 0.08) or (self.h_exact and whole_y and h < self.h - 0.08):
            px = None
        # A hazard need not be a rectangle (a saw blade is a disc): cut on one side it shows less of the other
        # dimension too - the top of the sliver of a disc that sticks into the picture is lower than the top
        # of the disc. So the top edge of a hazard cut left/right is not a measurement (the track keeps the
        # height it expects), and the left edge of a hazard cut above/below is none either.
        hz = self.kind == "hazard"
        if hz and not whole_y and not (self.w_exact and abs(w - self.w) <= FULL_TOL):
            px = None
        yok = not hz or whole_x or (self.h_exact and abs(h - self.h) <= FULL_TOL)
        if px is None or py is None:
            return                          # no exact position this time
        pred = self._extrapolate(t, with_v=True, raw=True) if self._px is not None else None
        if not yok and pred is not None:
            py = max(pred[1], y1)           # the top of the part that shows is never above the real top
        whole = whole_x and whole_y
        p = (t, px, py, knight[0], knight[1], whole, w, yok)
        seg = self.seg
        if not seg:
            seg.append(p)
        elif t > seg[-1][0] + 1e-6:
            t1 = seg[-1][0]
            dur = max(seg[-1][0] - seg[0][0], FRAME)
            tol = LINE_TOL + 0.04 * (t - t1) / dur
            if len(seg) >= 2 and (abs(px - pred[0]) > tol or (yok and abs(py - pred[1]) > tol)):
                self._break(p)
            elif pred[2] * self.vx < 0:
                # it turned at a turning point already known, as predicted: a new straight run starts
                self.seg = [p]
                self.vx = pred[2]
                self.turns += 1
            else:
                seg.append(p)
                if len(seg) > 60:
                    del seg[1:len(seg) - 40]          # keep the first sample as the long baseline
                self._fit()
        self.t, self._px, self._py = t, px, py
        self._xmin = px if self._xmin is None else min(self._xmin, px)
        self._xmax = px if self._xmax is None else max(self._xmax, px)

    def clock(self, t: float) -> float:
        """The time that drives this thing. Flyers (projectiles, falling things) are bodies moved by the
        fixed physics steps of 0.02 s, one or two of which fall into a frame; back-and-forth movers
        advance with the frame time."""
        return math.floor(t / STEP + 1e-3) * STEP if self.flew else t

    def _fit(self) -> None:
        seg = self.seg
        if len(seg) < 2:
            return
        a, b = seg[0], seg[-1]
        dx = b[1] - a[1]
        # vertical motion only from the sightings whose top edge was a measurement
        ok = [s for s in seg if s[7]]
        ya, yb = (ok[0], ok[-1]) if len(ok) >= 2 else (a, a)
        dy = yb[2] - ya[2]
        # real motion, or the screen sliding over a thing that is partly hidden behind something fixed on it?
        rigid = all(s[5] for s in seg) and max(abs(s[6] - a[6]) for s in seg) <= 0.08
        steady = len(seg) >= 3                   # evidence from one edge alone needs a run of three sightings
        real_x = abs(dx) >= MOVE_EPS and (rigid or (steady and abs(dx - (b[3] - a[3])) >= SCREEN_LOCK))
        real_y = abs(dy) >= MOVE_EPS and (rigid or (len(ok) >= 3 and abs(dy - (yb[4] - ya[4])) >= SCREEN_LOCK))
        if real_y:
            self.flew = True
        dt = self.clock(b[0]) - self.clock(a[0])
        if dt <= 1e-6:
            return
        dty = self.clock(yb[0]) - self.clock(ya[0])
        self.vx, self.vy = dx / dt, (dy / dty if dty > 1e-6 else 0.0)
        if real_x or real_y:
            self.moving = True
        if self.moving and not self.flew:
            if dt >= 0.5:
                self.speed = abs(self.vx)           # measured over a long straight run
            elif self.speed is not None and abs(self.vx) > 0.3 * self.speed:
                self.vx = math.copysign(self.speed, self.vx)   # short run: direction from the run, speed from before

    def _break(self, p: tuple) -> None:
        """The newest sighting is off the current straight run: a turn-around, or a new motion."""
        t, px, py = p[0], p[1], p[2]
        t1, xs, ys = self.seg[-1][0], self.seg[-1][1], self.seg[-1][2]
        v = self.vx
        run = self.seg[-1][0] - self.seg[0][0]
        if (self.moving and not self.flew and abs(v) > 0.5 and abs(self.vy) < 0.2 and abs(py - ys) < 0.1
                and len(self.seg) >= 4 and run >= 0.2 and t - t1 < 1.2):
            # same speed, opposite direction: the turn is where the two straight runs meet
            tau = (px - xs + v * (t1 + t)) / (2 * v)
            if t1 - 1e-6 <= tau <= t + 1e-6:
                turn = xs + v * (tau - t1) - (PAUSE / 2) * v
                if v > 0:
                    self._hi = turn if self._hi is None else 0.5 * (self._hi + turn)
                else:
                    self._lo = turn if self._lo is None else 0.5 * (self._lo + turn)
                self.turns += 1
                self.seg = [p]
                self.vx, self.vy = -v, 0.0
                return
        if t - t1 < 0.25:
            self.seg = [self.seg[-1], p]
            self._fit()
        else:
            self.seg = [p]
            self.vx = self.vy = 0.0

    # ---- prediction -------------------------------------------------------------------------
    @property
    def swings(self) -> bool:
        """A back-and-forth mover (moves along x only)."""
        return self.moving and not self.flew and abs(self.vy) < 0.6

    def quiet(self, t: float) -> float:
        """For how long (seconds) its real side edges have been seen at the same place."""
        return t - self.unrest if self.unrest is not None else 0.0

    @property
    def predictable(self) -> bool:
        """A moving hazard that is handled by prediction instead of the danger memory: one that rolls
        back and forth, or a small one that flies across (a projectile). A big thing coming down (a
        falling trap) stays with the danger memory and its own dodge reflex, as in v7."""
        return self.moving and self._px is not None and (not self.flew or ((self.w or 9) <= 0.9 and (self.h or 9) <= 0.9))

    def _extrapolate(self, t: float, with_v: bool = False, raw: bool = False) -> tuple:
        """Position at time t (centre; with raw=True the left/top edge that is tracked); with_v also
        returns the horizontal velocity it has then."""
        if self._px is None:
            return (0.0, 0.0, 0.0) if with_v else (0.0, 0.0)
        ox, oy = (0.0, 0.0) if raw else ((self.w or 0.0) / 2, -(self.h or 0.0) / 2)
        dt = self.clock(t) - self.clock(self.t)
        x, y, v = self._px, self._py, self.vx
        if dt <= 0 or (not self.moving and len(self.seg) < 2):
            return (x + ox, y + oy, v) if with_v else (x + ox, y + oy)
        if self.swings and abs(v) > 1e-6:
            left = dt
            for _ in range(8):
                bound = self._hi if v > 0 else self._lo
                if bound is None:
                    break
                need = (bound - x) / v
                if need >= left:
                    break
                x, v, left = bound, -v, left - max(need, 0.0) - PAUSE
                if left <= 0:
                    return (x + ox, y + oy, v) if with_v else (x + ox, y + oy)
            x += v * left
            return (x + ox, y + oy, v) if with_v else (x + ox, y + oy)
        return (x + v * dt + ox, y + self.vy * dt + oy, v) if with_v else (x + v * dt + ox, y + self.vy * dt + oy)

    def predict(self, t: float) -> tuple:
        """Centre (x, y) at simulation time t (the latest sighting when t is not in the future)."""
        return self._extrapolate(t)

    def certain(self, t: float) -> bool:
        """Can the prediction for time t be relied on?

        A back-and-forth mover turns around somewhere; until that turning point has been seen, the only
        certain thing is that it goes on as far as it was already seen to go.
        """
        if self._px is None:
            return False
        if self.moving and len(self.seg) < 2:
            return False                    # just changed its motion: direction not known yet
        if not self.swings:
            return True
        dt = t - self.t
        if dt <= 0:
            return True
        x, v, left = self._px, self.vx, dt
        if abs(v) < 1e-6:
            return True
        for _ in range(8):
            bound = self._hi if v > 0 else self._lo
            if bound is None:
                end = x + v * left
                return (end <= self._xmax + 1e-6) if v > 0 else (end >= self._xmin - 1e-6)
            need = (bound - x) / v
            if need >= left:
                return True
            x, v, left = bound, -v, left - max(need, 0.0) - PAUSE
            if left <= 0:
                return True
        return True

    def velocity(self, t: float) -> float:
        """Horizontal velocity at time t (sign flips at the known turning points)."""
        return self._extrapolate(t, with_v=True)[2]

    # ---- shape ------------------------------------------------------------------------------
    @property
    def round(self) -> bool:
        """Moving hazards are taken to be round (a saw blade, a ball): the observation only gives a
        box, and a blade rolling along a floor shows just its upper part. Designer knowledge, flagged
        in the CHANGELOG."""
        return self.kind == "hazard" and self.w is not None and self.h is not None and self.h <= 1.2 * self.w

    def shape(self, t: float) -> tuple:
        """('disc', cx, cy, r) or ('box', x0, x1, y0, y1) at time t."""
        cx, cy = self._extrapolate(t)
        w, h = self.w or 0.0, self.h or 0.0
        if self.round:
            r = w / 2
            return ("disc", cx, cy + h / 2 - r, r)
        return ("box", cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2)

    def rect(self, t: float) -> tuple:
        cx, cy = self._extrapolate(t)
        w, h = self.w or 0.0, self.h or 0.0
        return (cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2)


class MotionMemory:
    def __init__(self):
        self.tracks: dict[str, Track] = {}

    def reset(self) -> None:
        self.tracks.clear()

    def update(self, sc: Scene) -> None:
        t = sc.time
        knight = sc.knight_world
        # Cells that show nothing but "unknown" are things drawn over the picture (HUD hearts, the caption
        # box of recorded runs) or a fading sprite: whatever ends right next to them is hidden there.
        hidden = sc.masks == UNK
        rows, cols = hidden.shape

        def cut(c0, c1, r0, r1):
            c0, c1, r0, r1 = max(0, c0), min(cols - 1, c1), max(0, r0), min(rows - 1, r1)
            return c0 <= c1 and r0 <= r1 and bool(hidden[r0:r1 + 1, c0:c1 + 1].any())

        for o in sc.hazards + sc.platforms:
            kind = "hazard" if o.kind == "visible_hazard" else "platform"
            tr = self.tracks.get(o.oid)
            if tr is None:
                tr = self.tracks[o.oid] = Track(o.oid, kind)
            x0, y1 = sc.to_world(o.x0, o.y0)
            x1, y0 = sc.to_world(o.x1, o.y1)
            ca, cb = int(o.x0 // 20), int((o.x1 - 1) // 20)
            ra, rb = int(o.y0 // 20), int((o.y1 - 1) // 20)
            # "cut by the screen edge" with a tolerance: the outermost pixels of a sprite are often
            # unknown (alpha edges), so a cut box can end a few pixels inside the picture
            left = o.x0 <= EDGE or cut(ca - 2, ca - 1, ra, rb)
            right = o.x1 >= W - EDGE or cut(cb + 1, cb + 2, ra, rb)
            top = o.y0 <= EDGE or cut(ca, cb, ra - 2, ra - 1)
            bottom = o.y1 >= H - EDGE or cut(ca, cb, rb + 1, rb + 2)
            tr.update(t, (x0, x1, y0, y1), (left, right, top, bottom), knight)
        for oid in [k for k, tr in self.tracks.items() if tr.seen < t - 1.0 and not tr.swings]:
            del self.tracks[oid]            # still things and flyers that are gone need no memory

    def get(self, oid: str) -> Track | None:
        return self.tracks.get(oid)

    def is_mover(self, oid: str) -> bool:
        tr = self.tracks.get(oid)
        return tr is not None and tr.moving

    def hazards(self) -> list[Track]:
        """Moving hazards handled by prediction (rolling saws, projectiles)."""
        return [tr for tr in self.tracks.values() if tr.kind == "hazard" and tr.predictable]

    def platforms(self) -> list[Track]:
        """Platforms that go back and forth."""
        return [tr for tr in self.tracks.values() if tr.kind == "platform" and tr.swings and tr._px is not None]
