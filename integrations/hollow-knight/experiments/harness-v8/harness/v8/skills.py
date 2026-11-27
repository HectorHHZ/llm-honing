# fmt: off
# isort: skip_file
# ruff: noqa
"""Behaviour-level skills (L3: behaviours with reflexes). One skill = one model decision.

Each skill is a loop of native windows with stop checks on every new visible observation. Rules are
generic platformer knowledge driven by what is on screen: floor ahead -> walk; edge -> look for the
next safe surface and jump with air steering; hazard or enemy ahead -> stop; damage -> stop.
Nothing here is tied to level coordinates.
"""
from __future__ import annotations

import math

from dataclasses import dataclass, field

from . import physics as ph
from .control import Ctl, SkillStop
from .perception import FEET_Y, HALF_H, HALF_W, HEAD_Y, KX, KY, W, Rect, Scene, Seg

ATTACK_GAP = 1.45        # body-to-body gap (units) at which a sword strike reaches (probe: reach 2.1 from centre)
ATTACK_COOLDOWN = 11     # simulation frames between strikes (0.35 s cooldown = 10.5 frames)
ENEMY_LANDING_KEEPOUT = 3.0   # do not jump to a spot with an enemy standing this close to it (units)


@dataclass
class Result:
    skill: str
    reason: str
    notes: list = field(default_factory=list)


# ---- shared reflex helpers ----------------------------------------------------------------------
def body_gap(sc: Scene, e: Rect) -> float:
    """Horizontal body-to-body gap between knight and enemy, in units (negative = overlapping)."""
    return (abs(e.cx - KX) - e.w / 2 - HALF_W) / sc.ppu


def same_level(sc: Scene, e: Rect, tol_units: float = 1.6) -> bool:
    """The enemy stands on the surface the knight stands on (or its continuation), not across a gap."""
    if abs(e.cy - KY) > tol_units * sc.ppu:
        return False
    sup = sc.support()
    if sup is None:
        return True
    lo, hi = sup.x0, sup.x1
    for dirn in (-1, 1):
        cont = continuation(sc, sup, dirn)
        if cont is not None:
            lo, hi = min(lo, cont.x0), max(hi, cont.x1)
    return lo - 10 <= e.cx <= hi + 10


def enemy_ahead(sc: Scene, dirn: int, reach_units: float = 4.5):
    """Nearest enemy in front, within reach and roughly at the knight's height."""
    best = None
    for e in sc.enemies:
        dx = (e.cx - KX) * dirn
        if -HALF_W <= dx <= reach_units * sc.ppu and same_level(sc, e):
            if best is None or dx < (best.cx - KX) * dirn:
                best = e
    return best


def nearest_enemy(sc: Scene, level_only: bool = True):
    pool = [e for e in sc.enemies if same_level(sc, e)] if level_only else list(sc.enemies)
    return min(pool, key=lambda e: abs(e.cx - KX)) if pool else None


def continuation(sc: Scene, sup: Seg, dirn: int):
    """A neighbouring surface at the same height that simply continues the floor."""
    for seg in sc.segs:
        if seg.oid == sup.oid or not seg.safe or seg.kind != "platform" or seg.clipped_top:
            continue
        if abs(seg.y - sup.y) > 14:
            continue
        if dirn > 0 and seg.x0 <= sup.x1 + 10 and seg.x1 > sup.x1 + 10:
            return seg
        if dirn < 0 and seg.x1 >= sup.x0 - 10 and seg.x0 < sup.x0 - 10:
            return seg
    return None


def walk_limit(sc: Scene, sup: Seg, dirn: int, with_movers: bool = True) -> tuple[float, str]:
    """How many px the knight may walk in `dirn` before an edge, an obstacle or a hazard. (px, why)"""
    limit, why = 1e9, "clear"
    edge = sup.x1 if dirn > 0 else sup.x0
    clipped = sup.clipped_right if dirn > 0 else sup.clipped_left
    cont = continuation(sc, sup, dirn)
    if cont is not None:
        edge = cont.x1 if dirn > 0 else cont.x0
        clipped = cont.clipped_right if dirn > 0 else cont.clipped_left
    if not clipped:
        d = (edge - KX) * dirn - 0.30 * sc.ppu
        if d < limit:
            limit, why = d, "edge"
    for o in sc.platforms + sc.walls:        # something solid at body height
        if o.oid == sup.oid or (cont is not None and o.oid == cont.oid):
            continue
        if o.y1 <= HEAD_Y + 4 or o.y0 >= FEET_Y - 18:
            continue
        face = o.x0 if dirn > 0 else o.x1
        d = (face - KX) * dirn - HALF_W - 3
        if d > -2 * HALF_W and d < limit:
            limit, why = d, "wall" if o.kind == "wall" else "step"
    movers = getattr(sc, "mover_ids", ())
    for hz in sc.hazards:                    # hazard at body or feet height
        if hz.y1 <= HEAD_Y - 4 or hz.y0 >= FEET_Y + 12 or hz.oid in movers:
            continue
        face = hz.x0 if dirn > 0 else hz.x1
        d = (face - KX) * dirn - HALF_W - 0.5 * sc.ppu
        if d > -3 * HALF_W and d < limit:
            limit, why = d, "hazard"
    # v8: a rolling hazard is not a wall at the place where it is drawn; what counts is the stretch it will
    # cover in the next half second (control.Ctl._publish_movers)
    for bx0, by0, bx1, by1, _oid, now in (getattr(sc, "mover_boxes", ()) if with_movers else ()):
        if by1 <= HEAD_Y - 4 or by0 >= FEET_Y + 2 or (now - KX) * dirn <= 0:
            continue                         # passes over the head / under the feet, or it is behind the knight
        face = bx0 if dirn > 0 else bx1
        d = (face - KX) * dirn - HALF_W - 0.5 * sc.ppu
        if d < limit:
            limit, why = d, "mover"
    return limit, why


def jump_candidates(ctl: Ctl, mode: str, dirn: int = 1, max_drop: float = 9.0) -> list[ph.WRect]:
    """Landing zones worth jumping to, from the terrain memory (visible now or remembered).

    mode: 'ahead' (further in `dirn` than the surface under the feet), 'up' (higher), 'down' (lower).
    Each zone is a hazard-free, reachable-from-above span of a platform top, as a world rectangle.
    """
    x, y = ctl.pos
    supw = ctl.support_surf()
    out = []
    rollers = [tr for tr in ctl.motion.hazards() if tr.swings and tr.w is not None]
    for surf in ctl.map.solids.values():
        if surf.kind != "platform" or surf.open_top or (supw is not None and surf.oid == supw.oid):
            continue
        if surf.x1 < x - 14 or surf.x0 > x + 14 or surf.top - y > 9 or y - surf.top > max_drop:
            continue
        rise = surf.top - y
        spans = ctl.map.safe_spans(surf, x, y)
        if rollers:
            # v8: the stretch of this surface that a saw is known to roll along is no place to land
            spans, _swept = _minus_sweeps(surf, spans, rollers)
        for a, b in spans:
            # only spots within the width of the screen: one cannot see who waits further away
            # (v8: not for a platform that moves back and forth - it comes closer and goes again)
            if not surf.moving:
                a, b = max(a, x - 8.6), min(b, x + 8.6)
            if b - a < 0.9:
                continue
            if mode == "ahead":
                if supw is not None:
                    beyond = (b - supw.x1) if dirn > 0 else (supw.x0 - a)
                else:
                    beyond = ((a + b) / 2 - x) * dirn
                ahead = (b - x) if dirn > 0 else (x - a)
                if beyond < 0.5 or ahead < 0.8:
                    continue
                key = (a - x) if dirn > 0 else (x - b)
            elif mode == "up":
                if rise < 0.6:
                    continue
                key = abs((a + b) / 2 - x) - 0.75 * rise
            else:
                if rise > -0.6:
                    continue
                key = abs((a + b) / 2 - x)
            out.append((key, ph.WRect(a, b, surf.y0, surf.y1, surf.kind, surf.oid, surf.lip)))
    out.sort(key=lambda kv: kv[0])
    return [r for _, r in out]


def _minus_sweeps(surf, spans: list, rollers: list) -> tuple[list, float]:
    """Cut the stretches that rolling hazards are known to cover on `surf` out of its landing spans.
    Returns (spans, total width cut on this surface)."""
    cuts = []
    for tr in rollers:
        _cx, half, bot, top = _extent(tr.shape(tr.t))
        if top <= surf.top + 0.05 or bot >= surf.top + ph.BODY_H:
            continue                           # it does not pass at standing height on this surface
        lo = (tr.xmin if tr.lo is None else min(tr.xmin, tr.lo)) - half - ph.BODY_HW - 0.3
        hi = (tr.xmax if tr.hi is None else max(tr.xmax, tr.hi)) + half + ph.BODY_HW + 0.3
        if hi > surf.x0 and lo < surf.x1:
            cuts.append((lo, hi))
    if not cuts:
        return spans, 0.0
    swept = sum(max(0.0, min(c1, surf.x1) - max(c0, surf.x0)) for c0, c1 in cuts)
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
    return [(a, b) for a, b in spans if b - a >= 0.9], swept


def sweep_width(ctl: Ctl, oid: str) -> float:
    """How much of the top of surface `oid` a rolling hazard is known to cover (0 if none)."""
    surf = ctl.map.solids.get(oid)
    rollers = [tr for tr in ctl.motion.hazards() if tr.swings and tr.w is not None]
    if surf is None or not rollers:
        return 0.0
    return _minus_sweeps(surf, [(surf.x0, surf.x1)], rollers)[1]


def is_remembered(ctl: Ctl, target: ph.WRect) -> bool:
    """True when the landing zone is not on the current screen (it comes from memory)."""
    sc = ctl.scene
    sx0, sy = sc.to_screen(target.x0, target.top)
    sx1, _ = sc.to_screen(target.x1, target.top)
    on_screen = sx1 > 0 and sx0 < W and 0 <= sy < 540
    return not (on_screen and any(o.oid == target.oid for o in sc.platforms))


def takeoff_spot(ctl: Ctl, supw, target: ph.WRect) -> float:
    """World x to jump from: near the edge of our surface that faces the target."""
    x, _ = ctl.pos
    if supw is None:
        return x
    if target.x0 >= supw.x1 - 0.1 and not supw.open_right:      # target begins beyond our right edge
        return max(x, supw.x1 - 0.32)
    if target.x1 <= supw.x0 + 0.1 and not supw.open_left:       # beyond our left edge
        return min(x, supw.x0 + 0.32)
    return x


def enemy_near_landing(ctl: Ctl, target: ph.WRect, land_x: float) -> Rect | None:
    """A visible enemy standing on the target surface close to where we would land."""
    sc = ctl.scene
    for e in sc.enemies:
        ex, ebottom = sc.to_world(e.cx, e.y1)
        if abs(ex - land_x) <= ENEMY_LANDING_KEEPOUT and -0.6 <= ebottom - target.top <= 1.6:
            return e
    return None


TIMED_DELAYS = (0, 3, 6, 9, 12, 18)        # plan family used when the same jump is tried for many start times
TIMED_T2S = (6, 12, 18, 24, 30)
MAX_TIMED_WAIT = 120                       # frames: look this far ahead for the moment to jump (4 s)


def _timed_plan(ctl: Ctl, target: ph.WRect, spot: float, y: float, riding, solids, hazards, dyn) -> ph.Plan | None:
    """v8: the first later moment at which a jump from `spot` onto `target` works, given where the moving
    platforms and hazards will be (plan.wait = frames to wait before the press). None if there is none."""
    t0 = ctl.scene.time
    for w in range(3, MAX_TIMED_WAIT + 1, 3):
        if riding and w > dyn.sure.get(riding, 1 << 30):
            break                                                   # where the platform under the feet is by then is not known
        if w > dyn.sure.get(target.oid, 1 << 30):
            break                                                   # ... nor where the target is (no landing can be planned)
        sx = spot + (dyn.shift(riding, w) if riding else 0.0)       # carried along while waiting
        p = ph.plan_jump(sx, y, 0.0, t0 + w * ph.FDT, 2, True, target, solids, hazards, dyn=dyn, k0=w,
                         delays=TIMED_DELAYS, t2s=TIMED_T2S)
        if p.ok:
            p.wait = w
            return p
    return None


def plan_candidates(ctl: Ctl, cands: list[ph.WRect], timed: bool = True):
    """Evaluate jump plans (from the takeoff spot, then from here). Returns (choice | None, report).

    choice = (target, spot, plan). The report lists why candidates were rejected.
    v8: plans keep clear of where moving hazards will be; when the target or the surface under the feet is
    a moving platform (or a moving hazard crosses the way) and no jump works right now, the first later
    moment at which one works is looked for (plan.wait > 0: wait that long, then jump).
    """
    sc = ctl.scene
    supw = ctl.support_surf()
    solids, hazards = ctl.world_rects()
    dyn = ctl.dyn()
    x, y = sc.knight_world
    riding = supw.oid if (supw is not None and supw.moving) else None
    report = []
    for target in cands:
        spots = [x]
        spot = takeoff_spot(ctl, supw, target)
        if abs(spot - x) > 0.1 and not riding:
            spots = [spot, x]
        tsurf = ctl.map.solids.get(target.oid)
        moving = bool(riding) or (tsurf is not None and tsurf.moving)
        # v8: a small platform that has only just come into view may be a moving one that has not shown it
        # yet: do not jump at it before its edges have been seen at the same place for a moment
        mt = ctl.motion.get(target.oid)
        if (tsurf is not None and not tsurf.moving and tsurf.x1 - tsurf.x0 < 4.0
                and target.oid not in ctl.map.visited and (mt is None or mt.quiet(sc.time) < 0.4)):
            report.append((target, ph.UNSURE))
            continue
        for spot in spots:
            plan = ph.plan_jump(spot, y, 0.0, sc.time, 2, True, target, solids, hazards, dyn=dyn)
            if not plan.ok and timed and (moving or dyn.haz):
                later = _timed_plan(ctl, target, spot, y, riding, solids, hazards, dyn)
                if later is not None:
                    plan = later
            if plan.ok:
                if enemy_near_landing(ctl, target, plan.land_x) is not None:
                    report.append((target, "enemy at the landing spot"))
                    break
                return (target, spot, plan), report
            unsure = moving and (dyn.sure.get(target.oid, 1 << 30) < MAX_TIMED_WAIT or
                                 (riding and dyn.sure.get(riding, 1 << 30) < MAX_TIMED_WAIT))
            if (unsure and tsurf is not None and tsurf.moving and dyn.sure.get(target.oid, 1 << 30) < MAX_TIMED_WAIT
                    and not 0 <= sc.to_screen(target.x0, target.top)[1] < 540):
                # where it turns around is not known and it cannot be watched from here (it is below or
                # above the picture): waiting will not help - this is not a jump to make from memory
                report.append((target, ph.UNSEEN))
                break
            report.append((target, ph.UNSURE if unsure else plan.reason))
    return None, report


def try_jump(ctl: Ctl, cands: list[ph.WRect], why: str, regen=None):
    """Go to the takeoff spot of the first feasible candidate and jump. (ok | None, message, target)

    v8: when the plan says "not yet" (plan.wait), wait for the moment (standing, or riding a moving
    platform) and plan again; `regen` rebuilds the candidates after time has passed (things have moved).
    """
    waited = 0
    for _ in range(40):
        choice, report = plan_candidates(ctl, cands)
        if choice is None:
            if any(r == "enemy at the landing spot" for _, r in report):
                return None, "enemy at the landing spot", None
            if any(r == ph.UNSURE for _, r in report) and waited < 300:
                # a moving platform whose turning points have not been seen yet: watch it first
                ctl.step("wait", 4, why + ":watching")
                waited += 4
                if not ctl.tr.standing:
                    return False, "lost the footing while watching", None
                cands = regen() if regen is not None else cands
                continue
            return None, "; ".join(sorted({r for _, r in report}))[:120] or "no surface known there", None
        target, spot, plan = choice
        x, _ = ctl.pos
        if abs(spot - x) > 0.1:
            if not ctl.walk_to(spot, why=why + ":takeoff"):
                return False, "could not reach the takeoff spot", target
            if plan.wait > 0:
                cands = regen() if regen is not None else cands
                continue
        if plan.wait > 0:
            if waited >= 300:
                return None, "no good moment to jump came", target
            n = plan.wait if plan.wait <= 4 else min(4, plan.wait - 2)
            ctl.step("wait", n, why + ":timing")
            waited += n
            if not ctl.tr.standing:
                return False, "lost the footing while waiting", target
            cands = regen() if regen is not None else cands
            continue
        if enemy_near_landing(ctl, target, plan.land_x) is not None:
            return None, "enemy at the landing spot", target
        ok, msg = ctl.jump_to(target, why=why)
        return ok, msg, target
    return None, "no good moment to jump came", None


def _install_stops(ctl: Ctl, stop_on_damage: bool = True, movers: bool = True):
    deaths0 = ctl.tr.deaths
    dmg0 = ctl.tr.damage_events
    area0 = ctl.area

    def check(c: Ctl):
        if c.area > area0:
            return "entered a new area"                    # v8: walked through the level exit
        if c.tr.deaths > deaths0:
            return "died"
        if stop_on_damage and c.tr.damage_events > dmg0:
            return "took damage"
        if c.falling_over_me is not None and c.tr.standing:
            return "something dangerous is falling from above"
        if movers and c.tr.standing and mover_threat(c, frames=12) is not None:
            return "a moving hazard is about to hit"       # v8: a saw rolling at the knight
        return None

    ctl.stop_checks = [check]


def after_death(ctl: Ctl) -> None:
    """After dying the game reloads the scene; wait (no input) until the knight is back and standing."""
    ctl.stop_checks = []
    for _ in range(80):
        ctl.step("wait", 4, "respawn")
        if ctl.tr.hp and ctl.tr.hp > 0 and ctl.tr.standing:
            break
    ctl.tr.jumps = 2


def after_damage(ctl: Ctl) -> None:
    """A hit locks the input for about half a second and throws the knight back.

    Wait the lock out; if the knight is then in the air (knocked off a ledge), use the generic
    air-recovery reflex to reach a safe surface with whatever jump is left.
    """
    ctl.stop_checks = []
    waited = 0
    while waited < 14:
        ctl.step("wait", 2, "hurt-lock")
        waited += 2
        if ctl.tr.hp == 0:
            return
    if not ctl.tr.standing:
        ctl.recover("recover-after-hit")
    elif in_danger(ctl):
        escape_danger(ctl)
    else:
        ctl.settle(20)


def in_danger(ctl: Ctl) -> bool:
    """Does the knight's body overlap remembered standing danger (it was knocked into spikes)?"""
    x, y = ctl.pos
    _, hazards = ctl.world_rects()
    return any(x + ph.BODY_HW > h.x0 and x - ph.BODY_HW < h.x1 and y < h.y1 and y + ph.BODY_H > h.y0 for h in hazards)


def escape_danger(ctl: Ctl) -> bool:
    """v8 reflex: knocked into spikes by a hit. For 1.5 s after a hit nothing can hurt the knight; use what is
    left of that time to jump out onto the nearest safe surface."""
    sc = ctl.scene
    x, y = sc.knight_world
    solids, hazards = ctl.world_rects()
    safe = ctl.safe_frames()
    if safe <= 0:
        return False
    best = None
    for surf in ctl.map.solids.values():
        if surf.kind != "platform" or surf.open_top or (abs(surf.x0 - x) > 12 and abs(surf.x1 - x) > 12):
            continue
        for a, b in ctl._without_enemies(surf, ctl.map.safe_spans(surf, x, None)):
            target = ph.WRect(a, b, surf.y0, surf.y1, surf.kind, surf.oid, surf.lip)
            plan = ph.plan_jump(x, y, 0.0, sc.time, 2, True, target, solids, hazards, margin=0.25, clear_need=0.1,
                                dyn=ctl.dyn(), safe=safe)
            if plan.ok and (best is None or plan.land_frame < best[0].land_frame):
                best = (plan, target)
    if best is None:
        return False
    ok, _ = ctl.jump_to(best[1], why="escape-danger", margin=0.25, clear_need=0.1)
    return ok or ctl.tr.standing


# ---- skills --------------------------------------------------------------------------------------
def jump_past_guard(ctl: Ctl, dirn: int, why: str):
    """An enemy stands where the knight would land: land on the same surface well behind it instead.

    Returns (ok, message) or None when no such jump exists.
    """
    sc = ctl.scene
    x, y = sc.knight_world
    solids, hazards = ctl.world_rects()
    supw = ctl.support_surf()
    for target in jump_candidates(ctl, "ahead", dirn)[:3]:
        guards = []
        for e in sc.enemies:
            ex0, etop = sc.to_world(e.x0, e.y0)
            ex1, ebot = sc.to_world(e.x1, e.y1)
            if -0.6 <= ebot - target.top <= 1.6 and ex1 > target.x0 - 0.5 and ex0 < target.x1 + 0.5:
                guards.append((ex0, ex1, ebot, etop, e.oid))
        if not guards:
            continue
        far = max(g[1] for g in guards) if dirn > 0 else min(g[0] for g in guards)
        lo, hi = (far + 3.0, far + 7.5) if dirn > 0 else (far - 7.5, far - 3.0)
        lo, hi = max(lo, target.x0), min(hi, target.x1)
        if hi - lo < 0.9:
            continue
        zone = ph.WRect(lo, hi, target.y0, target.y1, target.kind, target.oid, target.lip)
        # the enemy follows the knight's shadow while it flies over: widen the keep-out on the far side
        keep = [ph.WRect(g[0] - (0.4 if dirn > 0 else 1.8), g[1] + (1.8 if dirn > 0 else 0.4), g[2] - 0.3, g[3] + 0.4, "enemy", g[4])
                for g in guards]
        spot = takeoff_spot(ctl, supw, target)
        for sp in ([spot, x] if abs(spot - x) > 0.1 else [x]):
            plan = ph.plan_jump(sp, y, 0.0, sc.time, 2, True, zone, solids, hazards + keep, margin=0.2, clear_need=-9)
            if plan.ok:
                if abs(sp - x) > 0.1 and not ctl.walk_to(sp, why=why + ":takeoff"):
                    return False, "could not reach the takeoff spot"
                ok, msg = ctl.jump_to(zone, why=why, margin=0.2, clear_need=-9, extra_hazards=keep)
                return (ok or ctl.tr.standing), msg
    return None


def advance(ctl: Ctl, dirn: int = 1, max_sim: int = 600, max_landings: int = 7, name: str | None = None,
            allow_jumps: bool = True, max_units: float | None = None, max_drop: float = 9.0, stop_when=None) -> Result:
    """Make progress in one direction: walk, hop to the next safe surface, stop at danger.

    Human experience built in: do not linger on a small platform — after landing on one, go on to the
    next surface at once; stop for a new decision on wide floors, at an enemy, at a dead end.
    """
    name = name or ("advance_right" if dirn > 0 else "retreat_left")
    _install_stops(ctl)
    landings = 0
    notes = []
    walked = 0.0
    stalls = 0
    next_check = 1.0
    try:
        while True:
            if ctl.sim_frames_this_skill >= max_sim:
                return Result(name, "time cap", notes=notes)
            if max_units is not None and walked >= max_units - 0.05:
                return Result(name, f"walked {walked:.1f} units", notes=notes)
            sc = ctl.scene
            if not ctl.tr.standing:
                ok, msg = ctl.recover(name + ":air")
                if not ok:
                    return Result(name, "not standing on anything I can see", notes=notes)
                continue
            if stop_when is not None and walked + 3 * landings >= next_check:
                next_check = walked + 3 * landings + 2.5
                why_stop = stop_when(ctl)
                if why_stop:
                    return Result(name, why_stop, notes=notes)
            e = enemy_ahead(sc, dirn)
            if e is not None:
                return Result(name, f"enemy ahead at {max(0.0, body_gap(sc, e)):.1f} units", notes=notes)
            sup = sc.support()
            if sup is None:
                return Result(name, "standing on something that is not a visible platform", notes=notes)
            limit, why = walk_limit(sc, sup, dirn)
            if why == "mover" and limit < 8:
                if blocking_mover(ctl, dirn) is not None:
                    # v8: do not stand and wait in front of a rolling hazard here; how to deal with it is a decision
                    return Result(name, "a moving hazard rolls back and forth across the way ahead", notes=notes)
                limit, why = walk_limit(sc, sup, dirn, with_movers=False)
            if limit >= 8:
                want = min(limit / sc.ppu, 1.0)
                if max_units is not None:
                    want = min(want, max_units - walked)
                x_before = ctl.pos[0]
                moved = ctl.walk(dirn, want, why=name + ":walk")
                if moved > 0:
                    actual = abs(ctl.pos[0] - x_before)
                    walked += actual
                    stalls = stalls + 1 if actual < 0.4 * moved else 0
                    if stalls >= 2:
                        return Result(name, "something I cannot see blocks the way", notes=notes)
                    continue
            if why == "clear":
                return Result(name, "the floor continues past the screen edge", notes=notes)
            if not allow_jumps:
                return Result(name, f"stopped at {why}", notes=notes)
            cands = jump_candidates(ctl, "ahead", dirn, max_drop=max_drop)
            ok, msg, seg = try_jump(ctl, cands, name + ":jump",
                                    regen=lambda: jump_candidates(ctl, "ahead", dirn, max_drop=max_drop))
            if ok is None and msg == "enemy at the landing spot" and ctl.map.is_collapsing(ctl.support_surf()):
                # cannot wait here (this kind of platform gives way): land behind the enemy, or at
                # least on the platform, and let the next decision deal with it
                res = jump_past_guard(ctl, dirn, name + ":past-guard")
                if res is None:
                    globals()["ENEMY_LANDING_KEEPOUT"], keep = -1.0, globals()["ENEMY_LANDING_KEEPOUT"]
                    try:
                        ok, msg, seg = try_jump(ctl, cands, name + ":jump-anyway")
                    finally:
                        globals()["ENEMY_LANDING_KEEPOUT"] = keep
                else:
                    ok, msg = res
                    notes.append("jumped over an enemy that guarded the landing")
            if ok:
                landings += 1
                notes.append("jumped to the next platform")
                supw = ctl.support_surf()
                small = supw is not None and (supw.x1 - supw.x0) < 4.0 and not (supw.open_left or supw.open_right)
                if landings >= max_landings:
                    return Result(name, f"landed on a new platform after {landings} jumps", notes=notes)
                if not small:
                    # On a wide floor: stop for a new decision only if there is something to decide here
                    # (a lever, something not explored yet); otherwise keep going. Every extra stop is a
                    # chance for a pointless detour.
                    if interactable_here(ctl) is not None:
                        return Result(name, f"landed on a wide floor after {landings} jump{'s' if landings > 1 else ''}; a lever is here", notes=notes)
                    if dirn > 0 and stop_when is None:
                        note = unexplored_here(ctl)
                        if note:
                            return Result(name, f"landed on a wide floor after {landings} jump{'s' if landings > 1 else ''}; {note}", notes=notes)
                continue
            if ok is None:
                if msg == "enemy at the landing spot":
                    return Result(name, "an enemy stands at the landing spot of the next platform", notes=notes)
                return Result(name, f"blocked by {why}; no safe surface within jump reach ({msg})", notes=notes)
            return Result(name, f"jump failed: {msg}", notes=notes)
    except SkillStop as stop:
        return Result(name, stop.reason, notes=notes)


def retreat(ctl: Ctl, dirn: int = -1, units: float = 3.0) -> Result:
    """Step back along the current surface (no jumps): gets out of an enemy's sight range."""
    name = "retreat_left" if dirn < 0 else "retreat_right"
    return advance(ctl, dirn, max_sim=60, name=name, allow_jumps=False, max_units=units)


def wait(ctl: Ctl, sim_frames: int = 30) -> Result:
    _install_stops(ctl)
    try:
        if not ctl.tr.standing:
            ctl.recover("wait:air")
        done = 0
        while done < sim_frames:
            ctl.step("wait", 5, "wait")
            done += 5
            e = nearest_enemy(ctl.scene)
            if e is not None and body_gap(ctl.scene, e) < 1.0:
                return Result("wait", "an enemy came close")
        return Result("wait", "one second passed, nothing changed")
    except SkillStop as stop:
        return Result("wait", stop.reason)


def fight(ctl: Ctl, max_sim: int = 300) -> Result:
    """Sword fight with the nearest enemy on the knight's level.

    Reflexes: face it; step in until the strike reaches; strike once per cooldown; step back if it
    gets too close; never walk off the platform or into a hazard while doing so.
    """
    name = "fight_nearest_enemy"
    _install_stops(ctl)
    strikes = 0
    since_attack = 99
    seen = False
    leaps = 0
    try:
        while True:
            if ctl.sim_frames_this_skill >= max_sim:
                return Result(name, f"time cap after {strikes} strikes")
            sc = ctl.scene
            if not ctl.tr.standing:
                ok, _ = ctl.recover(name + ":air")
                if not ok:
                    return Result(name, "not standing on anything I can see")
                continue
            e = nearest_enemy(sc)
            if e is None:
                if seen:
                    if ctl.fade_guard:
                        ctl.step("wait", 1, name + ":done")      # the fade guard runs first (see Ctl.step)
                    return Result(name, f"the enemy is gone after {strikes} strikes")
                return Result(name, "no enemy on my level")
            seen = True
            dirn = 1 if e.cx >= KX else -1
            gap = body_gap(sc, e)
            sup = sc.support()
            if gap > ATTACK_GAP:
                room = 0.0
                if sup is not None:
                    limit, _ = walk_limit(sc, sup, dirn)
                    room = limit / sc.ppu
                step = min(gap - 1.1, 0.7, room)
                if step >= 0.1 and ctl.walk(dirn, step, why=name + ":approach") > 0:
                    since_attack += 2
                    continue
                before = ctl.sim_frames_this_skill
                ctl.step("wait", 2, name + ":let-it-come")
                since_attack += 2
                continue
            if gap < 0.45 and sup is not None:
                limit, _ = walk_limit(sc, sup, -dirn)
                if limit / sc.ppu >= 0.3 and ctl.walk(-dirn, min(0.5, limit / sc.ppu), why=name + ":back-off") > 0:
                    since_attack += 3
                    continue
                if leaps < 2:
                    # cornered (edge or wall behind): get to the open side instead of trading hits
                    leaps += 1
                    res = _leap_over(ctl, name + ":leap-out")
                    since_attack += 20
                    if res is not None:
                        continue
            if since_attack >= ATTACK_COOLDOWN:
                ctl.step("attack_right" if dirn > 0 else "attack_left", 1, name + ":strike")
                strikes += 1
                since_attack = 1
                continue
            ctl.step("wait", 2, name + ":cooldown")
            since_attack += 2
    except SkillStop as stop:
        return Result(name, f"{stop.reason} after {strikes} strikes")


def _leap_over(ctl: Ctl, why: str) -> str | None:
    """Jump over the nearest enemy on our surface and land behind it. Returns a message, None if impossible."""
    sc = ctl.scene
    e = nearest_enemy(sc)
    sup = sc.support()
    supw = ctl.support_surf()
    if e is None or sup is None or supw is None:
        return None
    dirn = 1 if e.cx >= KX else -1
    x, y = sc.knight_world
    ex0, etop = sc.to_world(e.x0, e.y0)
    ex1, ebot = sc.to_world(e.x1, e.y1)
    lo, hi = (ex1 + 1.3, ex1 + 4.5) if dirn > 0 else (ex0 - 4.5, ex0 - 1.3)
    spans = ctl.map.safe_spans(supw)
    zone = None
    for a, b in spans:
        a2, b2 = max(a + 0.3, lo), min(b - 0.3, hi)
        if b2 - a2 >= 0.6:
            zone = ph.WRect(a2, b2, supw.y0, supw.y1, supw.kind, supw.oid, supw.lip)
            break
    if zone is None:
        return None
    solids, hazards = ctl.world_rects()
    # the enemy walks toward the knight while it is in the air: widen the keep-out on our side
    toward, away = (1.3, 0.3) if dirn > 0 else (0.3, 1.3)
    keepout = ph.WRect(ex0 - toward, ex1 + away, ebot - 0.3, etop + 0.35, "enemy", e.oid)
    if keepout.x0 < x + ph.BODY_HW and keepout.x1 > x - ph.BODY_HW:
        # already inside the keep-out (the enemy is on top of us): only its far half counts
        mid = (ex0 + ex1) / 2
        keepout = ph.WRect(mid if dirn > 0 else ex0 - toward, ex1 + away if dirn > 0 else mid, ebot - 0.3, etop + 0.2, "enemy", e.oid)
        if keepout.x0 < x + ph.BODY_HW and keepout.x1 > x - ph.BODY_HW:
            keepout = None
    extra = [keepout] if keepout is not None else []
    plan = ph.plan_jump(x, y, 0.0, sc.time, 2, True, zone, solids, hazards + extra, margin=0.0, clear_need=-9)
    if not plan.ok:
        return None
    ok, msg = ctl.jump_to(zone, why=why, margin=0.0, clear_need=-9, extra_hazards=extra)
    return "landed behind the enemy" if (ok or ctl.tr.standing) else f"jump failed: {msg}"


def jump_over_enemy(ctl: Ctl) -> Result:
    """Leap over the enemy in front and land on the same floor behind it."""
    name = "jump_over_enemy"
    _install_stops(ctl)
    try:
        if not ctl.tr.standing:
            ctl.recover(name + ":air")
        if nearest_enemy(ctl.scene) is None:
            return Result(name, "no enemy on my platform")
        msg = _leap_over(ctl, name)
        if msg is None:
            return Result(name, "cannot clear the enemy from here")
        return Result(name, msg)
    except SkillStop as stop:
        return Result(name, stop.reason)


def _vertical_only(ctl: Ctl, mode: str) -> list[ph.WRect]:
    """Up/down candidates minus the platform that advancing right/left would jump to anyway."""
    sc = ctl.scene
    sup = sc.support()
    taken = set()
    if sup is not None:
        for dirn in (1, -1):
            limit, why = walk_limit(sc, sup, dirn)
            if why != "clear":
                choice, _ = plan_candidates(ctl, jump_candidates(ctl, "ahead", dirn)[:3])
                if choice is not None:
                    taken.add(choice[0].oid)
    return [c for c in jump_candidates(ctl, mode) if c.oid not in taken]


def climb_up(ctl: Ctl, max_landings: int = 1) -> Result:
    """Jump to a reachable higher surface (double jump with air steering)."""
    name = "climb_up"
    _install_stops(ctl)
    try:
        if not ctl.tr.standing:
            ctl.recover(name + ":air")
        cands = _vertical_only(ctl, "up")
        ok, msg, seg = try_jump(ctl, cands, name, regen=lambda: _vertical_only(ctl, "up"))
        if ok:
            return Result(name, "climbed onto a higher platform")
        if ok is None:
            return Result(name, f"no higher surface within jump reach ({msg})")
        return Result(name, f"jump failed: {msg}")
    except SkillStop as stop:
        return Result(name, stop.reason)


def drop_down(ctl: Ctl) -> Result:
    """Jump or step down to a visible lower safe surface."""
    name = "drop_down"
    _install_stops(ctl)
    try:
        if not ctl.tr.standing:
            ctl.recover(name + ":air")
        cands = _vertical_only(ctl, "down")
        ok, msg, seg = try_jump(ctl, cands, name, regen=lambda: _vertical_only(ctl, "down"))
        if ok:
            return Result(name, "dropped onto a lower platform")
        if ok is None:
            return Result(name, f"no lower safe surface within reach ({msg})")
        return Result(name, f"jump failed: {msg}")
    except SkillStop as stop:
        return Result(name, stop.reason)


# ---- wall climbing ------------------------------------------------------------------------------
def exit_zones(ctl: Ctl, ground_y: float, start_oid: str | None, limit: int = 4) -> list[ph.WRect]:
    """Higher platform tops near the knight that a wall kick might reach (highest first)."""
    x, y = ctl.pos
    zones = []
    for surf in ctl.map.solids.values():
        if surf.kind != "platform" or surf.open_top or surf.oid == start_oid:
            continue
        if surf.top < ground_y + 1.5 or surf.top > y + 6.5 or surf.x1 < x - 9 or surf.x0 > x + 9:
            continue
        for a, b in ctl.map.safe_spans(surf):
            zones.append(ph.WRect(a, b, surf.y0, surf.y1, surf.kind, surf.oid, surf.lip))
    zones.sort(key=lambda z: -z.top)
    return zones[:limit]


def wall_options(ctl: Ctl) -> list[dict]:
    """Walls the knight could cling to from where it stands, with the ledge they lead to (if known)."""
    if not ctl.tr.standing:
        return []
    st = ctl.start_state()
    x, y = ctl.pos
    solids, hazards = ctl.world_rects()
    supw = ctl.support_surf()
    out = []
    for w in ctl.map.solids.values():
        if w.kind != "wall" or (w.y1 < y + 2.0 and not w.open_top) or w.oid in ctl.map.no_hold or w.oid in ctl.map.fruitless:
            continue
        if w.x1 <= x - 0.3:
            side, face = -1, w.x1
        elif w.x0 >= x + 0.3:
            side, face = 1, w.x0
        else:
            continue
        if abs(face - x) > 7.5:
            continue
        h_lo, h_hi = max(y + 0.6, w.y0 - ph.BODY_H + 0.25), w.y1 - 0.15
        if h_hi <= h_lo:
            continue
        o = ph.plan_to_wall(st, w.wrect(), side, h_lo, h_hi, solids, hazards)
        if o.kind != "wall":
            continue
        goal = None
        for surf in ctl.map.solids.values():
            if surf.kind != "platform" or (supw is not None and surf.oid == supw.oid):
                continue
            if surf.open_top:
                # only its underside has been seen: a ledge up there whose top is still unknown
                if surf.y0 < y + 1.0:
                    continue
            elif surf.top < y + 1.5 or surf.top > w.y1 + 4.0:
                continue
            gap = max(surf.x0 - face, face - surf.x1, 0.0)
            if gap <= 3.5 and (goal is None or surf.y1 > goal.y1):
                goal = surf
        over_danger = any(h.x0 < w.x1 + 0.3 and h.x1 > w.x0 - 0.3 and h.y1 < w.y0 + 0.5 and h.y1 > w.y0 - 6.0 for h in hazards)
        out.append({"wall": w, "side": side, "plan": o, "goal": goal, "dist": abs(face - x), "over_danger": over_danger,
                    "rise": (goal.top - y) if (goal is not None and not goal.open_top) else None,
                    "goal_visited": goal is not None and goal.oid in ctl.map.visited,
                    "climbed": w.oid in ctl.map.climbed})
    out.sort(key=lambda r: (r["goal"] is None, r["goal_visited"], r["dist"]))
    return out


def climb_wall(ctl: Ctl, side_pref: int | None = None, name: str = "climb_wall") -> Result:
    """Cling to a wall and wall-kick upward; step off onto the first higher ledge that comes within reach."""
    _install_stops(ctl)
    try:
        if not ctl.tr.standing:
            ctl.recover(name + ":air")
        x0, ground_y = ctl.pos
        sup = ctl.scene.support()
        start_oid = sup.oid if sup is not None else None
        opts = [o for o in wall_options(ctl) if side_pref is None or o["side"] == side_pref]
        if not opts:
            return Result(name, "no wall within reach to cling to")
        opt = opts[0]
        wall_oid, side = opt["wall"].oid, opt["side"]
        res = ctl.follow(opt["plan"], name + ":to-wall")
        if res == "died":
            return Result(name, "died")
        if not ctl.confirm_cling(side, name):
            ctl.map.no_hold.add(wall_oid)
            ctl.recover(name + ":no-hold")
            return Result(name, "could not get a hold on the wall")
        ctl.map.climbed.add(wall_oid)
        walls_used = {wall_oid}
        top_reached = ctl.pos[1]
        crossings = 0
        for cycle in range(18):
            if ctl.sim_frames_this_skill > 900:
                break
            st = ctl.start_state()
            x, y = ctl.pos
            top_reached = max(top_reached, y)
            solids, hazards = ctl.world_rects()
            landed = None
            for zone in exit_zones(ctl, ground_y, start_oid):
                o = ph.plan_from_wall_to_land(st, zone, solids, hazards)
                if o.kind != "land":
                    continue
                res = ctl.follow(o, name + ":step-off")
                if res == "died":
                    return Result(name, "died")
                if res == "deviated" and not ctl.tr.standing and ctl.tr.mode != "climb":
                    ctl.recover(name + ":step-off-recover")
                if ctl.tr.mode != "climb":
                    ctl.settle(40)
                landed = zone
                break
            if landed is not None:
                now = ctl.scene.support()
                if ctl.tr.standing and now is not None and now.oid == landed.oid:
                    return Result(name, f"climbed the wall and stepped onto a ledge {ctl.pos[1] - ground_y:.0f} steps up")
                if ctl.tr.mode == "climb":
                    continue
                if ctl.tr.standing:
                    return Result(name, "came off the wall and landed on another surface")
                break
            w = ctl.map.solids.get(wall_oid)
            if w is None:
                break
            o = ph.plan_to_wall(st, w.wrect(), side, y + 0.8, w.y1 - 0.15, solids, hazards)
            new_wall, new_side = wall_oid, side
            if o.kind != "wall" and crossings < 6:
                # no way higher on this wall (its top, or spikes on it): cross over to a wall on the
                # other side of the shaft, arriving higher than here (zigzag climbing)
                best = None
                for w2 in ctl.map.solids.values():
                    if w2.kind != "wall" or w2.oid == wall_oid or w2.oid in ctl.map.no_hold:
                        continue
                    if side > 0 and w2.x1 <= x - 0.5 and x - w2.x1 <= 8.0:
                        side2 = -1
                    elif side < 0 and w2.x0 >= x + 0.5 and w2.x0 - x <= 8.0:
                        side2 = 1
                    else:
                        continue
                    o2 = ph.plan_to_wall(st, w2.wrect(), side2, y + 0.5, w2.y1 - 0.15, solids, hazards)
                    if o2.kind == "wall" and (best is None or o2.y > best[0].y):
                        best = (o2, w2.oid, side2)
                if best is not None:
                    o, new_wall, new_side = best
                    crossings += 1
            if o.kind != "wall":
                break
            res = ctl.follow(o, name + ":kick")
            if res == "died":
                return Result(name, "died")
            if not ctl.confirm_cling(new_side, name):
                break
            wall_oid, side = new_wall, new_side
            ctl.map.climbed.add(wall_oid)
            walls_used.add(wall_oid)
        if not ctl.tr.standing:
            ctl.recover(name + ":down")
        gain = top_reached - ground_y
        back = ctl.scene.support()
        where = "back where it started" if (back is not None and back.oid == start_oid) else "on another surface"
        ctl.map.fruitless |= walls_used          # nothing up there: do not offer these walls again
        return Result(name, f"went {gain:.0f} steps up the wall, found no ledge to step onto, and is {where}")
    except SkillStop as stop:
        return Result(name, stop.reason)


# ---- interactables ------------------------------------------------------------------------------
def interactable_here(ctl: Ctl):
    """Nearest visible interactable standing on the knight's surface that has not proved inert."""
    sc = ctl.scene
    sup = sc.support()
    if sup is None:
        return None
    best = None
    for it in sc.interactables:
        if it.oid in ctl.map.inert:
            continue
        if it.x1 < sup.x0 - 10 or it.x0 > sup.x1 + 10 or abs(it.y1 - FEET_Y) > 1.6 * sc.ppu:
            continue
        if best is None or abs(it.cx - KX) < abs(best.cx - KX):
            best = it
    return best


def _step_away(ctl: Ctl, away: int, units: float, why: str) -> float:
    """Walk up to `units` in one direction as fast as the floor allows; returns the distance covered."""
    done = 0.0
    while done < units - 0.05:
        sup = ctl.scene.support()
        if sup is None:
            break
        limit, _ = walk_limit(ctl.scene, sup, away)
        moved = ctl.walk(away, min(limit / ctl.scene.ppu, units - done, 1.0), why=why) if limit >= 8 else 0.0
        if moved == 0:
            break
        done += moved
    return done


def hit_switch(ctl: Ctl) -> Result:
    """Walk up to the interactable object and strike it with the sword.

    Human experience built in: a switch may set off a trap, so the moment the object reacts the knight
    walks away from it (3.5 steps) before looking at what happened.
    """
    name = "hit_switch"
    deaths0, dmg0 = ctl.tr.deaths, ctl.tr.damage_events

    def check(c: Ctl):              # the falling-hazard stop is handled inside this skill
        if c.tr.deaths > deaths0:
            return "died"
        if c.tr.damage_events > dmg0:
            return "took damage"
        return None

    ctl.stop_checks = [check]
    try:
        if not ctl.tr.standing:
            ctl.recover(name + ":air")
        it = interactable_here(ctl)
        if it is None:
            return Result(name, "nothing to strike on this platform")
        oid = it.oid
        strikes = 0
        dirn = 1
        for _ in range(60):
            sc = ctl.scene
            it = next((o for o in sc.interactables if o.oid == oid), None)
            if it is None:
                if strikes == 0:
                    return Result(name, "lost sight of the object")
                covered = _step_away(ctl, -dirn, 3.5, name + ":step-away")
                for _ in range(10):
                    if ctl.falling_over_me is not None:
                        dodge_falling(ctl)
                    else:
                        ctl.step("wait", 4, name + ":watch")
                return Result(name, f"struck the object; it changed (activated); stepped {covered:.0f} steps away")
            dirn = 1 if it.cx >= KX else -1
            gap = (abs(it.cx - KX) - it.w / 2 - HALF_W) / sc.ppu
            if gap > 1.2:
                sup = sc.support()
                room = walk_limit(sc, sup, dirn)[0] / sc.ppu if sup is not None else 0.0
                step = min(gap - 0.9, 1.0, room)
                if step < 0.1 or ctl.walk(dirn, step, why=name + ":approach") == 0:
                    return Result(name, "cannot get close enough to strike it")
                continue
            if strikes >= 2:
                ctl.map.inert.add(oid)
                return Result(name, "struck the object twice and it did not react; it is only decoration")
            ctl.step("attack_right" if dirn > 0 else "attack_left", 1, name + ":strike")
            strikes += 1
            for _ in range(5):
                if not any(o.oid == oid for o in ctl.scene.interactables):
                    break
                ctl.step("wait", 2, name + ":look")
        return Result(name, "time cap")
    except SkillStop as stop:
        return Result(name, stop.reason)


def dodge_falling(ctl: Ctl) -> bool:
    """Generic reflex: something dangerous is coming down on the knight -> get out from under it.

    Walk toward the nearer edge of the falling thing, as far as the floor allows; once the body is clear
    of it, stay there. Never cross under it to reach the far side.
    """
    hz = ctl.falling_over_me
    if hz is None:
        return False
    oid = hz.oid
    for _ in range(10):
        sc = ctl.scene
        sup = sc.support()
        x, _ = ctl.pos
        cur = ctl.map.hazards.get(oid)
        if cur is None or sup is None:
            return True
        need_r = (cur.x1 + ph.BODY_HW + 0.35) - x          # distance to be clear on the right
        need_l = x - (cur.x0 - ph.BODY_HW - 0.35)
        if need_r <= 0 or need_l <= 0:
            ctl.step("wait", 3, "dodge-falling:clear")
            if ctl.falling_over_me is None:
                return True
            continue
        dirn, need = (1, need_r) if need_r <= need_l else (-1, need_l)
        room = walk_limit(sc, sup, dirn)[0] / sc.ppu
        if room < 0.1 or ctl.walk(dirn, min(need, room, 1.0), why="dodge-falling") == 0:
            ctl.step("wait", 3, "dodge-falling:no-room")
            return (cur.x1 + ph.BODY_HW) < x or x < (cur.x0 - ph.BODY_HW)
    return True


# ---- an enemy guards the landing spot of the next platform ---------------------------------------
ENEMY_SIGHT = 6.3     # enemies come at the knight when it is closer than this horizontally (the enemy
                      # prefab says detectDistance = 6; a player learns it by watching them)


def outwait_enemy(ctl: Ctl, dirn: int = 1, max_sim: int = 420) -> Result:
    """Patient version of the jump to the next platform when an enemy stands where the knight would land.

    The enemy only walks at the knight while it is within sight range, and then it waits at the edge.
    So: stay out of its sight range until it has wandered at least 3 steps away from the landing spot,
    then go to the edge and jump at once.
    """
    name = "outwait_enemy"
    _install_stops(ctl)
    try:
        while ctl.sim_frames_this_skill < max_sim:
            if not ctl.tr.standing:
                ctl.recover(name + ":air")
            sc = ctl.scene
            sup = sc.support()
            supw = ctl.support_surf()
            if sup is None or supw is None:
                return Result(name, "standing on something that is not a visible platform")
            x, y = sc.knight_world
            cands = jump_candidates(ctl, "ahead", dirn)[:3]
            old_keep = globals()["ENEMY_LANDING_KEEPOUT"]
            globals()["ENEMY_LANDING_KEEPOUT"] = -1.0          # plan as if the landing were free
            try:
                choice, _ = plan_candidates(ctl, cands)
            finally:
                globals()["ENEMY_LANDING_KEEPOUT"] = old_keep
            if choice is None:
                return Result(name, "no platform to jump to from here")
            target, spot, plan = choice
            guards = [e for e in sc.enemies if -0.6 <= sc.to_world(e.cx, e.y1)[1] - target.top <= 1.6]
            ex = [sc.to_world(e.cx, 0)[0] for e in guards]
            near_landing = [v for v in ex if abs(v - plan.land_x) <= ENEMY_LANDING_KEEPOUT]
            sees_me = [v for v in ex if abs(v - x) <= ENEMY_SIGHT]
            if not near_landing:
                # the landing is clear right now: go to the edge and jump (checked again at the edge)
                if abs(spot - x) > 0.1:
                    step = min(abs(spot - x), 0.7)
                    if ctl.walk(1 if spot > x else -1, step, why=name + ":to-edge") == 0:
                        ctl.walk_to(spot, why=name + ":to-edge")
                    continue
                ok, msg = ctl.jump_to(target, why=name + ":jump")
                if ok:
                    return Result(name, "the landing was clear; jumped to the next platform")
                return Result(name, f"jump failed: {msg}")
            if sees_me:
                # it has noticed the knight and will wait at the edge: get out of its sight range
                far = min(sees_me) - ENEMY_SIGHT - 0.3 if dirn > 0 else max(sees_me) + ENEMY_SIGHT + 0.3
                limit, _ = walk_limit(sc, sup, -dirn)
                room = limit / sc.ppu
                need = (x - far) * dirn
                if need > 0.1 and room > 0.15:
                    ctl.walk(-dirn, min(need, room, 1.0), why=name + ":out-of-sight")
                    continue
            ctl.step("wait", 5, name + ":watch")
        return Result(name, "the enemy kept guarding the landing spot")
    except SkillStop as stop:
        return Result(name, stop.reason)


def leap_past_enemy(ctl: Ctl, dirn: int = 1) -> Result:
    """Bold alternative to waiting: jump over the enemy that guards the landing and land behind it."""
    name = "leap_past_enemy"
    _install_stops(ctl)
    try:
        if not ctl.tr.standing:
            ctl.recover(name + ":air")
        res = jump_past_guard(ctl, dirn, name)
        if res is None:
            return Result(name, "no way to clear the enemy and land behind it from here")
        ok, msg = res
        return Result(name, "landed behind the enemy" if ok else f"jump failed: {msg}")
    except SkillStop as stop:
        return Result(name, stop.reason)


# ---- going back to look for what was missed --------------------------------------------------------
def unexplored_here(ctl: Ctl) -> str | None:
    """Is a place that has not been explored within reach from where the knight stands?

    A ledge not visited yet (by a plain jump or up a wall), or a climbable wall never tried whose top is
    not known. Used by go_back to stop where exploring becomes possible.
    """
    for opt in wall_options(ctl):
        if opt["goal"] is not None and not opt["goal_visited"]:
            return "a ledge not visited yet is within reach here (up a wall)"
        if opt["goal"] is None and not opt["climbed"]:
            return "a wall not climbed yet is within reach here"
    choice, _ = plan_candidates(ctl, _vertical_only(ctl, "up")[:2])
    if choice is not None and choice[0].oid not in ctl.map.visited:
        return "a ledge not visited yet is within jump reach here"
    return None


def go_back(ctl: Ctl, dirn: int = -1, name: str = "go_back_left") -> Result:
    """Dead end ahead: go back the way the knight came (walk, jump or drop from platform to platform)
    and stop as soon as something not explored yet comes within reach."""
    return advance(ctl, dirn, name=name, max_landings=3, max_sim=600, max_drop=20.0, stop_when=unexplored_here)



# ====================================================================================================
# v8: moving hazards (rolling saws)
# ====================================================================================================
STANDOFF = 0.45          # wait this far (body edge to hazard edge) outside the stretch a rolling hazard covers
NEED_GAP = 0.12          # a hop over a rolling hazard must keep at least this distance in the model
HOP_DASHES = (None, 4, 5, 6, 7, 8)   # hop variants: plain, or with a dash pressed this many frames after the jump


def _extent(shape: tuple) -> tuple:
    """(centre x, half width, bottom, top) of a hazard shape."""
    if shape[0] == "disc":
        return shape[1], shape[3], shape[2] - shape[3], shape[2] + shape[3]
    return (shape[1] + shape[2]) / 2, (shape[2] - shape[1]) / 2, shape[3], shape[4]


def blocking_mover(ctl: Ctl, dirn: int, reach: float = 11.0):
    """The nearest rolling hazard ahead that crosses the knight's way at body height (None if none)."""
    x, y = ctl.pos
    t = ctl.scene.time
    best = None
    for tr in ctl.motion.hazards():
        if not tr.swings:
            continue
        cx, half, bot, top = _extent(tr.shape(t))
        if top <= y + 0.05 or bot >= y + ph.BODY_H - 0.05:
            continue
        ahead = (cx - x) * dirn
        if 0 < ahead <= reach and (best is None or ahead < best[0]):
            best = (ahead, tr)
    return best[1] if best is not None else None


def mover_threat(ctl: Ctl, frames: int = 12, margin: float = 0.22):
    """(frames until contact, track) when a rolling hazard is going to reach the knight where it stands."""
    dyn = ctl.dyn()
    if not dyn.haz:
        return None
    x, y = ctl.pos
    for i, shapes in enumerate(dyn.haz):
        tr = ctl.motion.get(dyn.haz_ids[i])
        if tr is None or not tr.swings:
            continue
        for k in range(0, frames + 1):
            if ph.hits(shapes[k], x, y, margin):
                return k, tr
    return None


def stage_x(tr, dirn: int) -> float:
    """Where to wait for a rolling hazard: just outside the stretch it has been seen to cover."""
    half = (tr.w or 1.0) / 2
    if dirn > 0:
        near = tr.xmin if tr.lo is None else min(tr.xmin, tr.lo)
        return near - half - ph.BODY_HW - STANDOFF
    near = tr.xmax if tr.hi is None else max(tr.xmax, tr.hi)
    return near + half + ph.BODY_HW + STANDOFF


def _fly_hop(ctl: Ctl, p: ph.Plan, dirn: int, why: str) -> str:
    """Execute a hop planned by physics.plan_cross (press, hold the direction, optional dash). Returns
    crossed | deviated | died."""
    act = "right" if dirn > 0 else "left"
    path, last, b = p.path, p.land_frame, 0
    ctl.air_reflex = False       # no sword reflex on this flight: a strike that hits changes the flight (measured),
    try:                         # and right above a saw a lost hit point is the lesser evil
        return _fly_hop_run(ctl, p, dirn, why, act, path, last, b)
    finally:
        ctl.air_reflex = True


def _fly_hop_run(ctl: Ctl, p: ph.Plan, dirn: int, why: str, act: str, path, last: int, b: int) -> str:
    while b < last:
        if p.dash is not None and b == p.dash:
            ctl.press("dash_right" if dirn > 0 else "dash_left", why + ":dash")
            n = min(ph.DASH_FRAMES, last - b)
            ctl.step(act, n, f"{why}:b{b}:dashing", stretch=0)
        else:
            nxt = p.dash if (p.dash is not None and b < p.dash) else last
            n = max(1, min(4, nxt - b))
            ctl.step(("jump_right" if dirn > 0 else "jump_left") if b == 0 else act, n, f"{why}:b{b}",
                     stretch=max(0, nxt - b - n))
        b += ctl.last_n
        if ctl.tr.respawned or ctl.tr.hp == 0:
            return "died"
        if b >= last:
            break
        ex = path[b]
        x, y = ctl.pos
        if abs(x - ex[0]) > 0.14 or abs(y - ex[1]) > 0.14:
            return "deviated"
    ctl.settle(30)
    return "crossed" if ctl.tr.standing else "deviated"


def _hop(ctl: Ctl, tr, dirn: int, why: str) -> str:
    """Hop over rolling hazard `tr` if a safe hop exists right now. Returns crossed | again (walked a
    little toward the takeoff point) | no | deviated | died."""
    sc = ctl.scene
    sup = sc.support()
    supw = ctl.support_surf()
    dyn = ctl.dyn()
    if sup is None or supw is None or tr.oid not in dyn.haz_ids:
        return "no"
    x, y = ctl.pos
    solids, hazards = ctl.world_rects()
    # the floor exactly where the feet are now (a sprite's drawn top can be a little off the walkable top)
    floor = ph.WRect(supw.x0, supw.x1, supw.y0, y, supw.kind, supw.oid, 0.0)
    solids = [r for r in solids if r.oid != supw.oid] + [floor]
    room = walk_limit(sc, sup, dirn, with_movers=False)[0] / sc.ppu
    c = ph.plan_cross(x, y, sc.time, dirn, floor, solids, hazards, dyn, dyn.haz_ids.index(tr.oid),
                      walk_room=max(0.0, room), dashes=HOP_DASHES, need_gap=NEED_GAP)
    if not c.ok:
        return "no"
    if c.walk > 0:
        ctl.step("right" if dirn > 0 else "left", min(c.walk, 2), why + ":run-in")
        return "again"
    return _fly_hop(ctl, c.plan, dirn, why)


def pass_hazard(ctl: Ctl, dirn: int = 1, max_sim: int = 900) -> Result:
    """Get past a hazard that rolls back and forth across the way: wait outside the stretch it covers, watch
    it, and when it comes hop over it (under a low ceiling the hop is helped by a dash); then keep going
    until out of its stretch.

    Human experience built in: do not jump at a moving hazard before having seen where it turns around;
    never stand in its path; once past it, do not linger where it rolls.
    """
    name = "pass_hazard_right" if dirn > 0 else "pass_hazard_left"
    _install_stops(ctl, movers=False)
    hopped = None
    try:
        while ctl.sim_frames_this_skill < max_sim:
            if not ctl.tr.standing:
                ok, _ = ctl.recover(name + ":air")
                if not ok:
                    return Result(name, "not standing on anything I can see")
                continue
            sc = ctl.scene
            sup = sc.support()
            if sup is None:
                return Result(name, "standing on something that is not a visible platform")
            x, y = ctl.pos
            if hopped is not None:
                # past it: walk on until out of the stretch it was seen to cover
                tr = hopped
                far = (tr.xmax + (tr.w or 1) / 2 + ph.BODY_HW + STANDOFF) if dirn > 0 else \
                      (tr.xmin - (tr.w or 1) / 2 - ph.BODY_HW - STANDOFF)
                left = (far - x) * dirn
                e = enemy_ahead(sc, dirn)
                room = walk_limit(sc, sup, dirn)[0] / sc.ppu
                if left <= 0.1 or e is not None or room < 0.15:
                    tail = "" if left <= 0.1 else ("; an enemy is ahead" if e is not None else "; stopped by what is ahead")
                    return Result(name, "hopped over the moving hazard and left its stretch behind" + tail)
                ctl.walk(dirn, min(left, room, 1.0), why=name + ":run-out")
                continue
            mv = blocking_mover(ctl, dirn)
            if mv is None:
                return Result(name, "no moving hazard is in the way now")
            res = _hop(ctl, mv, dirn, name + ":hop")
            if res == "died":
                return Result(name, "died")
            if res == "crossed":
                hopped = mv
                continue
            if res in ("again", "deviated"):
                continue
            # no safe hop right now: keep out of its way and watch
            th = mover_threat(ctl, frames=24, margin=0.3)
            if th is not None:
                away = 1 if x >= th[1].predict(sc.time)[0] else -1
                room = walk_limit(sc, sup, away)[0] / sc.ppu
                if room >= 0.15:
                    ctl.walk(away, min(room, 1.0), why=name + ":keep-clear")
                else:
                    ctl.step("wait", 1, name + ":cornered")
                continue
            # go to the waiting spot only once it is known where the hazard turns around on this side, or
            # while it is rolling away; while it comes closer and may come further still, just watch
            coming = mv.velocity(sc.time) * dirn < 0
            turn_known = (mv.lo if dirn > 0 else mv.hi) is not None
            d = (stage_x(mv, dirn) - x) * dirn
            if d > 0.3 and (turn_known or not coming):
                room = walk_limit(sc, sup, dirn)[0] / sc.ppu
                if min(d, room) >= 0.15 and ctl.walk(dirn, min(d, room, 1.0), why=name + ":to-waiting-spot") > 0:
                    continue
            far = abs(mv.predict(sc.time)[0] - x) > 6.5 or not coming
            ctl.step("wait", 4 if far else 2, name + ":watch")
        return Result(name, "the moving hazard left no safe gap in time")
    except SkillStop as stop:
        return Result(name, stop.reason)


def evade_mover(ctl: Ctl, max_sim: int = 240) -> str:
    """Reflex: a rolling hazard is about to reach the knight -> walk out of its way; cornered -> hop over it."""
    ctl.stop_checks = []
    start = ctl.sim_frames_this_skill
    for _ in range(120):
        if ctl.sim_frames_this_skill - start > max_sim or ctl.tr.hp == 0:
            break
        if not ctl.tr.standing:
            ctl.recover("evade-mover:air")
            if not ctl.tr.standing:
                break
            continue
        th = mover_threat(ctl, frames=45, margin=0.3)
        if th is None:
            return "stepped out of its way"
        tr = th[1]
        sc = ctl.scene
        sup = sc.support()
        if sup is None:
            break
        x, _ = ctl.pos
        away = 1 if x >= tr.predict(sc.time)[0] else -1
        room = walk_limit(sc, sup, away)[0] / sc.ppu
        if room >= 0.15 and ctl.walk(away, min(room, 1.0), why="evade-mover") > 0:
            continue
        res = _hop(ctl, tr, -away, "evade-mover:hop")
        if res == "died":
            break
        if res == "no":
            ctl.step("wait", 1, "evade-mover:cornered")
    return "could not get out of its way"


# ====================================================================================================
# v8: a hazard field too wide to jump (spikes on the floor) -> bounce across with downward strikes
# ====================================================================================================
# Game mechanic (measured in probe_03_pogo; a Hollow Knight player knows it as the "pogo"): a downward
# sword strike in the air that hits something dangerous (spikes, a saw) throws the knight up again
# (vertical speed 4). The strike reaches about 1.4 units below the feet and can be repeated every 0.35 s,
# so the knight can cross any width of spikes by striking down each time it comes down toward them.
STRIKE_BAND = 0.95       # strike when the feet are no more than this above the visible top of the hazard
STRIKE_FLOOR = 0.25      # emergency height: this close above the top and no strike possible -> air jump
STRIKE_COOLDOWN = 11     # simulation frames between sword strikes (0.35 s)


def hazard_field(ctl: Ctl, dirn: int = 1):
    """A strip of standing-still danger on the floor ahead that is too wide to jump over but can be flown
    over low. Returns {"x0","x1","top","dist"} (world; top = its visible top) or None."""
    if not ctl.tr.standing:
        return None
    x, y = ctl.pos
    solids, hazards = ctl.world_rects()
    # ... standing on the floor: its tips reach above the level of the feet. Spikes at the bottom of a pit
    # (tips below the feet) are not bounced on: a pit is crossed on what leads over it (designer judgement).
    strip = [h for h in hazards if h.y0 <= y + 1.0 and y + 0.07 <= h.y1 <= y + 2.15
             and (h.x1 - x) * dirn > 0.3 and abs((h.x0 if dirn > 0 else h.x1) - x) <= 8.5]
    if not strip:
        return None
    strip.sort(key=lambda h: (h.x0 - x) * dirn if dirn > 0 else (x - h.x1))
    lo, hi, top = strip[0].x0, strip[0].x1, strip[0].y1
    for h in strip[1:]:                              # join what continues the first piece
        if h.x0 <= hi + 0.6 and h.x1 >= lo - 0.6:
            lo, hi, top = min(lo, h.x0), max(hi, h.x1), max(top, h.y1)
    top -= 0.12                                      # danger rectangles are padded by this much
    near = lo if dirn > 0 else hi
    if hi - lo < 2.5 or abs(near - x) > 6.0:
        return None
    # the air above the strip must be open for a body flying at strike height
    a, b = (lo - 0.5, min(hi, lo + 8.0)) if dirn > 0 else (max(lo, hi - 8.0), hi + 0.5)
    for s in solids:
        if s.x1 > a and s.x0 < b and s.y0 < top + 1.9 and s.y1 > top + 0.15:
            return None
    return {"x0": lo, "x1": hi, "top": top, "dist": abs(near - x)}


def _takeoff_delay(ctl: Ctl, field: dict, dirn: int):
    """Frames to rise straight up after the jump press before steering over the strip, so that the body
    clears the strip's near edge (None if no such jump exists from here)."""
    x, y = ctl.pos
    solids, hazards = ctl.world_rects()
    t0 = ctl.scene.time
    for delay in range(0, 17):
        xx, yy, vy, t, ok = x, y, ph.JUMP_V, t0, True
        for frame in range(1, 40):
            n = ph.nsteps(t, t + ph.FDT)
            t += ph.FDT
            for _ in range(n):
                vy -= ph.G * ph.DT
                if frame > delay:
                    xx += dirn * ph.WALK * ph.DT
                yy += vy * ph.DT
                if any(xx + ph.BODY_HW > h.x0 and xx - ph.BODY_HW < h.x1 and yy < h.y1 and yy + ph.BODY_H > h.y0 for h in hazards):
                    ok = False
                    break
                if any(xx + ph.BODY_HW > s.x0 + 0.02 and xx - ph.BODY_HW < s.x1 - 0.02 and yy < s.top - 0.02
                       and yy + ph.BODY_H > s.y0 + 0.02 for s in solids):
                    ok = False
                    break
            if not ok or vy <= 0:
                break
        over = (xx + ph.BODY_HW > field["x0"] + 0.3) if dirn > 0 else (xx - ph.BODY_HW < field["x1"] - 0.3)
        if ok and over and yy - field["top"] >= 0.3:
            return delay
    return None


def _hazard_below(ctl: Ctl, reach: float = STRIKE_BAND):
    """Visible top of standing-still danger under the knight within strike reach (None if nothing)."""
    x, y = ctl.pos
    _, hazards = ctl.world_rects()
    best = None
    for h in hazards:
        if h.x0 < x + 0.45 and h.x1 > x - 0.45:
            top = h.y1 - 0.12
            if top < y + 0.3 and y - top <= reach + 0.6 and (best is None or top > best):
                best = top
    return best


def bounce_across(ctl: Ctl, dirn: int = 1, max_sim: int = 900) -> Result:
    """Cross a wide field of spikes: jump up over its edge, then every time the knight comes down toward
    the spikes strike down with the sword to bounce up again, steering across, until something safe to
    land on comes within reach.

    Reflexes: strike only on the way down and inside sword reach; hanging on a chain or wall met on the way
    (it catches the knight) -> keep steering, slide off its lower end and strike at once; no bounce and too
    low -> use the air jump; a landing spot within reach -> stop bouncing and land.
    """
    name = "cross_spikes_right" if dirn > 0 else "cross_spikes_left"
    _install_stops(ctl, movers=False)
    act = "right" if dirn > 0 else "left"
    bounces = 0
    try:
        if not ctl.tr.standing:
            ctl.recover(name + ":air")
        field = hazard_field(ctl, dirn)
        if field is None:
            return Result(name, "no field of spikes to bounce across here")
        # walk up to the edge of the field
        for _ in range(12):
            sup = ctl.scene.support()
            if sup is None:
                break
            limit, why = walk_limit(ctl.scene, sup, dirn)
            if limit < 8 or ctl.walk(dirn, min(limit / ctl.scene.ppu, 1.0), why=name + ":to-edge") == 0:
                break
        delay = _takeoff_delay(ctl, field, dirn)
        if delay is None:
            return Result(name, "cannot get over the edge of the spikes from here")
        x_start = ctl.pos[0]
        ctl.air_reflex = False       # the sword is needed for the downward strikes (one strike per 0.35 s)
        ctl.press("jump_right" if dirn > 0 else "jump_left", name + ":jump")
        if delay > 0:
            ctl.step("wait", delay, name + ":rise")
        last_strike, tries = -99, 0
        prev_x, steered, caught_x = ctl.pos[0], False, None
        while ctl.sim_frames_this_skill < max_sim:
            tr = ctl.tr
            if tr.standing:
                break
            x, y = ctl.pos
            # caught by a chain / wall on the way: it holds the knight (steady slide, no headway while
            # steering) until the head has slid past its lower end; the sword does nothing while hanging
            stuck = steered and abs(x - prev_x) < 0.02
            prev_x, steered = x, True
            if tr.mode == "climb" or stuck:
                caught_x = x
                ctl.step(act, 1, name + ":caught")
                continue
            # just slid off its lower end: going up right there would touch it again and hang again, so
            # first get the body clear of it sideways (unless the spikes are too close to wait)
            under = caught_x is not None and abs(x - caught_x) < 2 * ph.BODY_HW + 0.1
            if not under:
                caught_x = None
            if abs(x - x_start) > 45.0:
                return Result(name, f"bounced {bounces} times and found nothing to land on")
            # a place to land within reach? (ballistic, or with the air jump)
            target = None if under else _landing_ahead(ctl, dirn)
            if target is not None:
                ok, msg = ctl.jump_to(target, why=name + ":land", margin=0.3, clear_need=0.1)
                if ok or ctl.tr.standing:
                    break
                continue
            top = _hazard_below(ctl)
            since = ctl.sim_total - last_strike
            if under and (top is None or y - top > STRIKE_FLOOR):
                ctl.step(act, 1, name + ":clear-of-it")
                continue
            if top is not None and tr.vy <= 0.6 and y - top <= STRIKE_BAND and since >= STRIKE_COOLDOWN:
                ctl.press("attack_down", name + ":strike")
                last_strike = ctl.sim_total
                steered = False                        # the bounce check below is not a headway check
                ctl.step(act, 1, name + ":bounce?")
                if ctl.tr.vy > 1.5:
                    bounces += 1
                    tries = 0
                else:
                    tries += 1
                continue
            if top is not None and tr.vy < 0 and y - top <= STRIKE_FLOOR and tr.jumps > 0:
                ctl.step("jump_right" if dirn > 0 else "jump_left", 1, name + ":air-jump")   # too low and no strike ready
                continue
            ctl.step(act, 2 if tr.vy > 1.6 else 1, name + ":fly")
        ctl.settle(30)
        if ctl.tr.standing:
            moved = abs(ctl.pos[0] - x_start)
            return Result(name, f"bounced across the spikes with {bounces} downward strikes and landed {moved:.0f} steps on")
        return Result(name, f"bounced {bounces} times and did not find a landing")
    except SkillStop as stop:
        return Result(name, f"{stop.reason} after {bounces} bounces")
    finally:
        ctl.air_reflex = True


def _landing_ahead(ctl: Ctl, dirn: int):
    """A safe surface ahead that the knight, airborne now, can land on by steering on (with the air jump if
    it is still there). Returns the landing zone or None."""
    x, y = ctl.pos
    cands = jump_candidates(ctl, "ahead", dirn)[:3]
    if not cands:
        return None
    solids, hazards = ctl.world_rects()
    dyn = ctl.dyn()
    for target in cands:
        plan = ph.plan_jump(x, y, ctl.tr.vy, ctl.scene.time, ctl.tr.jumps, False, target, solids, hazards,
                            margin=0.3, clear_need=0.1, dyn=dyn, delays=(0,), t2s=(3, 6, 9, 12, 15, 18, 21, 24))
        if plan.ok and enemy_near_landing(ctl, target, plan.land_x) is None:
            return target
    return None


# ====================================================================================================
# v8: an edge with nothing known beyond it -> step off and look, keeping the jumps to come back
# ====================================================================================================
PEEK_DEPTH = 2.6         # fall at most this far below the ledge while looking (a double jump gains 4.85)


def can_peek(ctl: Ctl, dirn: int = 1) -> bool:
    """Standing at (or near) an edge in `dirn` with no surface known beyond it and nothing dangerous seen
    right below it."""
    if not ctl.tr.standing:
        return False
    sc = ctl.scene
    sup = sc.support()
    supw = ctl.support_surf()
    if sup is None or supw is None:
        return False
    limit, why = walk_limit(sc, sup, dirn)
    if why != "edge" or limit / sc.ppu > 6.0:
        return False
    x, y = ctl.pos
    edge = supw.x1 if dirn > 0 else supw.x0
    _, hazards = ctl.world_rects()
    for h in hazards:                               # known danger just beyond the edge, within the fall
        # (the look stops PEEK_DEPTH below the ledge; an air jump turns the fall around at once, 0.6 is the margin)
        if h.y1 > y - PEEK_DEPTH - 0.6 and h.y0 < y + 0.5 and (h.x1 - edge) * dirn > -0.3 and (h.x0 - edge) * dirn < 3.5:
            return False
    return True


def _chase_target(ctl: Ctl, home_oid: str, dirn: int):
    """Airborne: a moving platform ahead (in direction `dirn`) that is on the picture now, not above the feet,
    and close enough to be reached by steering (with the air jumps that are left). Returns its id or None."""
    sc = ctl.scene
    x, y = ctl.pos
    seen = {o.oid for o in sc.platforms}
    best = None
    for surf in ctl.map.solids.values():
        if not surf.moving or surf.oid == home_oid or surf.oid not in seen or surf.top > y + 0.3:
            continue
        if ((surf.x1 if dirn > 0 else surf.x0) - x) * dirn < 0.3:
            continue                                   # behind the knight
        if not _in_view(ctl, surf.x0, surf.x1, surf.top):
            continue
        dist = max(surf.x0 + 0.3 - x, x - (surf.x1 - 0.3), 0.0)
        drop = max(y - surf.top, 0.0)
        t_fall = (ctl.tr.vy + math.sqrt(ctl.tr.vy ** 2 + 2 * ph.G * drop)) / ph.G     # until the feet are down at its top
        reach = ph.WALK * t_fall + 4.0 * ctl.tr.jumps - 0.5
        if dist <= reach and (best is None or dist < best[0]):
            best = (dist, surf.oid)
    return None if best is None else best[1]


def peek_below(ctl: Ctl, dirn: int = 1) -> Result:
    """Step off the edge to see what is below (the screen shows only 2 steps below the feet), keeping both
    jumps: if safe ground comes into view within reach, land on it; otherwise jump back up to the ledge.

    Human experience built in: the ground one came up from usually continues below a ledge, but nobody
    drops blind further than they can jump back. A platform that moves below is landed on by steering
    after it (Ctl.land_on); the ledge itself may be a moving platform too.
    """
    name = "drop_and_look_right" if dirn > 0 else "drop_and_look_left"
    _install_stops(ctl)
    act = "right" if dirn > 0 else "left"
    try:
        if not ctl.tr.standing:
            ctl.recover(name + ":air")
        supw = ctl.support_surf()
        sup = ctl.scene.support()
        if supw is None or sup is None:
            return Result(name, "standing on something that is not a visible platform")
        home_oid, home_moves = supw.oid, supw.moving
        y_home = ctl.pos[1]
        # to the edge
        for _ in range(12):
            sup = ctl.scene.support()
            if sup is None:
                break
            limit, why = walk_limit(ctl.scene, sup, dirn)
            if why != "edge":
                return Result(name, f"stopped at {why} before the edge")
            if limit < 8 or ctl.walk(dirn, min(limit / ctl.scene.ppu, 1.0), why=name + ":to-edge") == 0:
                break
        # off the edge: keep walking until the feet leave it
        for _ in range(8):
            ctl.step(act, 1, name + ":step-off")
            if not ctl.tr.standing:
                break
        if ctl.tr.standing:
            return Result(name, "could not step off the edge")
        ctl.tr.jumps = 2                              # walking off a ledge uses no jump
        for _ in range(60):
            if ctl.tr.standing:
                break
            x, y = ctl.pos
            solids, hazards = ctl.world_rects()
            dyn = ctl.dyn()
            # safe ground in view and within reach (not the ledge itself)?
            best = None
            for surf in ctl.map.solids.values():
                if surf.kind != "platform" or surf.open_top or surf.oid == home_oid or surf.top > y + 0.3 or surf.moving:
                    continue
                if abs(surf.x0 - x) > 12 and abs(surf.x1 - x) > 12:
                    continue
                for a, b in ctl._without_enemies(surf, ctl.map.safe_spans(surf, x, y)):
                    a, b = max(a, x - 8.6), min(b, x + 8.6)
                    if b - a < 0.9 or not _in_view(ctl, a, b, surf.top):
                        continue
                    target = ph.WRect(a, b, surf.y0, surf.y1, surf.kind, surf.oid, surf.lip)
                    plan = ph.plan_jump(x, y, ctl.tr.vy, ctl.scene.time, ctl.tr.jumps, False, target, solids, hazards,
                                        margin=0.3, clear_need=0.1, dyn=dyn, delays=(0, 3, 6, 9), t2s=(3, 9, 15, 21))
                    if plan.ok and (best is None or plan.score() < best[0].score()):
                        best = (plan, target)
            if best is not None:
                ok, msg = ctl.jump_to(best[1], why=name + ":land", margin=0.3, clear_need=0.1)
                if ok or ctl.tr.standing:
                    drop = y_home - ctl.pos[1]
                    return Result(name, f"stepped off the edge and landed on ground {drop:.0f} steps below")
                continue
            chase = _chase_target(ctl, home_oid, dirn)
            if chase is not None:
                ok, msg = ctl.land_on(chase, why=name + ":onto-moving")
                if ok:
                    drop = y_home - ctl.pos[1]
                    return Result(name, f"stepped off the edge and landed on a moving platform {drop:.0f} steps below")
                if ctl.tr.standing:
                    return Result(name, "stepped off the edge; ended on another surface")
                break
            if y_home - y >= PEEK_DEPTH or ctl.tr.vy < -7.5:
                break                                  # deep enough: nothing safe showed up
            # fall just clear of the ledge's side (it may be moving)
            home = ctl.map.solids.get(home_oid)
            a = "wait"
            if home is not None:
                want = (home.x1 if dirn > 0 else home.x0) + dirn * (ph.BODY_HW + 0.25)
                if (want - x) * dirn > 0.08:
                    a = act
                elif (x - want) * dirn > 0.6:
                    a = "left" if dirn > 0 else "right"
            ctl.step(a, 1, name + ":falling")
        if ctl.tr.standing:
            return Result(name, "stepped off the edge and landed")
        home = ctl.map.solids.get(home_oid)
        ok = False
        if home is not None and not home_moves:
            ok, msg = ctl.jump_to(ph.WRect(home.x0, home.x1, home.y0, home.y1, home.kind, home.oid, home.lip),
                                  why=name + ":back-up", margin=0.25, clear_need=0.1)
        if not ok and not ctl.tr.standing and home is not None:
            ok, msg = ctl.land_on(home_oid, why=name + ":back-up")
        if ok:
            return Result(name, "looked below the edge: nothing safe to land on in reach; came back up")
        ctl.recover(name + ":recover")
        return Result(name, "looked below the edge and could not get back to the ledge" if not ctl.tr.standing
                      else "looked below the edge; ended on another surface")
    except SkillStop as stop:
        return Result(name, stop.reason)


def _in_view(ctl: Ctl, a: float, b: float, top: float) -> bool:
    """Is (part of) the span a..b of a surface top on the current screen?"""
    sc = ctl.scene
    sx0, sy = sc.to_screen(a, top)
    sx1, _ = sc.to_screen(b, top)
    return sx1 > 0 and sx0 < W and 0 <= sy < 540
