# fmt: off
# isort: skip_file
# ruff: noqa
"""Executor primitives: send native windows, re-perceive after each one, keep a small self-state.

Everything here reads only the visible observation (through perception.Scene). The self-state
(jumps left, vertical speed, standing or not) is inferred from the camera scroll and our own inputs,
the way a player knows it, not read from the game's debug state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import physics as ph
from .motion import EDGE, MotionMemory
from .perception import FEET_Y, HALF_H, HALF_W, HEAD_Y, KX, KY, HAZ, SOLID, H, W, Scene, Seg
from .worldmap import WorldMap

FADE_FRAMES = 57         # an enemy killed by the sword is fully faded this many simulation frames later (P37)
TRAP_FADE_FRAMES = 19    # a fallen trap likewise
SWORD_REACH = 1.5        # sword: a 0.6-radius circle swept this far from the knight's centre
SWORD_RADIUS = 0.6


class SkillStop(Exception):
    """Raised inside primitives when a stop condition of the running skill fires."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class Tracker:
    jumps: int = 2
    standing: bool = False
    vy: float = 0.0
    hp: int | None = None
    deaths: int = 0
    damage_events: int = 0
    still_windows: int = 0
    last_damage_tick: int = -10**9
    respawned: bool = False
    mode: str = "air"          # ground | air | climb (clinging to a wall, sliding down)
    wall_side: int = 0         # climb: +1 wall on the right, -1 wall on the left
    wall_oid: str = ""


class Ctl:
    def __init__(self, env, log=None):
        self.env = env
        self.scene = Scene(env.obs)
        self.tr = Tracker(hp=self.scene.hp)
        self.map = WorldMap()          # remembered terrain (world coordinates, from visible observations only)
        self.motion = MotionMemory()   # v8: how hazards and platforms move (from successive visible observations)
        self.motion.update(self.scene)
        self._dyn_cache = None
        self._publish_movers(self.scene)
        self.map.update(self.scene, motion=self.motion)
        self.label = "harness"
        self.prob = 0.0
        self.decision = 0
        self.log = log or (lambda *a, **k: None)
        self.windows_this_skill = 0
        self.sim_frames_this_skill = 0
        self.stop_checks = []          # callables(ctl) -> reason | None, installed by the running skill
        self.falling_over_me = None    # hazard seen moving down above the knight in the latest window
        self._hazard_tops = {}
        self._falling = set()          # hazard ids seen moving down recently
        # Fade guard (works around a freeze of the game, see step()): number of simulation frames the
        # next window must cover without a pause.
        self.fade_guard = 0
        self.events = []               # (tick, text) notable things for the language layer
        # v8 ------------------------------------------------------------------------------------
        self.sim_total = 0             # simulation frames played so far (window boundaries are counted in these)
        self.no_pause = []             # [lo, hi] ranges of sim_total in which no window may end (P37, see _guard)
        self.guard_log = []            # what the fade guard did (for the run log)
        self.reflex = True             # strike projectiles that are about to hit while standing
        self.strikes = 0               # reflex strikes made
        self.air_strikes = 0           # ... of which in the air
        self.air_reflex = True         # strike them in the air too (switched off where a strike would spoil a
                                       # tight manoeuvre over something deadly: the saw hop, the spike bounce)
        self.last_strike = -99         # sim_total of the latest sword strike of any kind (cooldown)
        self.last_dash = -99
        self.last_n = 0                # simulation frames really played by the latest step() (the fade guard may change them)
        self.hurt_total = -999         # sim_total at the start of the window in which the knight was last hit
        self._pressed = {}             # kind of zero-time press -> sim_total at which it was sent
        self._pressed_any = False
        self._dyn_cache = None
        self.area = 0                  # counts scene changes seen (the whole picture jumps)
        self._area_sim = -999          # sim_total at the latest scene change
        self.last_flyer = -99.0        # simulation time at which a projectile was last on screen

    # ---- world helpers ---------------------------------------------------------------------
    @property
    def pos(self):
        return self.scene.knight_world

    def u(self, px: float) -> float:
        return px / self.scene.ppu

    def world_rects(self):
        """Remembered + visible solids and hazards around the knight, as world rectangles for the planner."""
        x, y = self.pos
        return self.map.rects(x, y)

    def support_surf(self):
        """Map entry of the surface under the feet (None when standing on nothing recognisable)."""
        sup = self.scene.support()
        return self.map.solids.get(sup.oid) if sup is not None else None

    # ---- stepping --------------------------------------------------------------------------
    def step(self, action: str, sim_frames: int, why: str = "", stretch: int = 1) -> Scene:
        """One native window of `sim_frames` simulated frames (render frames = sim_frames + 1).

        `stretch`: by how many frames the fade guard may make this window longer while the knight is in the
        air (a caller that follows a planned path passes how long the same input stays right, and reads
        `last_n` afterwards to know how many frames were really played).

        v8 puts two things in front of the window itself:
          * projectile reflex (only while standing): when a flying hazard is about to hit and comes within
            sword reach at some frame of this window, the window is cut there and a zero-time sword strike
            is sent (the strike destroys the projectile; a zero-time window moves nothing);
          * fade guard (plumbing around the game freeze P37, see _guard).
        """
        sim_frames = max(1, min(119, int(sim_frames)))
        played = 0
        if self.reflex and action in ("wait", "left", "right") and (
                self.tr.standing or (self.air_reflex and self.tr.mode == "air")):
            for _ in range(3):
                hit = self.strike_plan(action, sim_frames)
                if hit is None:
                    break
                before, attack = hit
                if before > 0:
                    m = self._guard(action, before, 0)
                    self._window(action, m, why + "|until-strike")
                    played += m
                    sim_frames -= before
                airborne = not self.tr.standing
                s0 = self.sim_total, self.last_strike
                self.press(attack, "reflex: strike the projectile")
                if self.last_strike != s0[1] or self.sim_total != s0[0]:
                    self.strikes += 1
                    self.air_strikes += 1 if airborne else 0
                    self.events.append((self.scene.tick, "struck a projectile out of the air"))
                    if airborne:
                        # a strike that hits sets the knight's velocity (measured: up -> (0, -1), forward -> (-/+1, 0))
                        self.tr.vy = -1.0 if attack == "attack_up" else 0.0
                if sim_frames <= 0:
                    self.last_n = played
                    return self.scene
        n = self._guard(action, sim_frames, stretch)
        self.last_n = played + n
        return self._window(action, n, why)

    def press(self, action: str, why: str = "") -> Scene:
        """A zero-time window (one render frame): the game takes the input edge (jump press, sword strike,
        dash) and no game time passes (measured in probe_04_zero). This relies on a quirk of the bridge
        (an N-frame window advances N-1 frames, Honing P37 item 2); if the interface is fixed, these
        presses become one-frame windows and the timing here has to be revisited.

        It is used only for what a player can do within one game frame: press one button while going on
        with whatever the hands are doing (strike while standing or walking, strike down while steering,
        jump without steering, dash while steering). Never for anything a player could not do:
          * the same kind of press is never sent twice without game time in between;
          * the sword's cooldown (0.35 s) is kept by the harness itself. The game enforces it with a timer on
            game time, so several zero-time strikes in a row would all land (probe_05_zero_cooldown: three
            zero-time strikes took an enemy from 3 health to 0 at the same instant). The harness never does that.
        A press that would break these rules is not sent.
        """
        kind = action.split("_")[0]
        if self._pressed.get(kind) == self.sim_total and self._pressed_any:
            return self.scene
        if kind == "attack" and self.sim_total - self.last_strike < 11:
            return self.scene
        if kind == "dash" and self.sim_total - self.last_dash < 17:
            return self.scene                              # dash: 0.15 s + 0.4 s before the next one
        self._pressed[kind] = self.sim_total
        self._pressed_any = True
        if kind == "dash":
            self.last_dash = self.sim_total
        return self._window(action, 0, why)

    def _guard(self, action: str, n: int, stretch: int = 1) -> int:
        """Fade guard. The game fades a dead enemy / a fallen trap with a coroutine that never yields once
        the sprite's alpha is 0; if the frame after that is a paused one (every window starts with one), the
        loop never ends and the game freezes (P37; dev_07 of round 1, and the Level2 first look, where an
        enemy that walked into a saw off screen froze the game). So no window may END inside the few frames
        in which a dying thing seen by the harness reaches alpha 0 (`no_pause`, in simulation frames).

        Standing: one wait window is put in front that spans those frames. In the air a wait is not an
        option: the window is made longer by at most `stretch` frames, or else cut to end before the
        frames in question (the next window then has to span them); if neither is possible it goes out as
        asked and the event is logged as unguarded. Deaths the harness cannot see (off screen) cannot be
        guarded against at all. This is plumbing around a bridge/game bug, not a tactic.
        Returns the number of frames to play.
        """
        for _ in range(4):
            s0 = self.sim_total
            self.no_pause = [z for z in self.no_pause if z[1] >= s0]
            zone = next((z for z in self.no_pause if z[0] <= s0 + n <= z[1] and s0 < z[0]), None)
            if zone is None:
                return n
            span = zone[1] + 1 - s0
            if span > 119:
                return n
            hold = not action.startswith(("jump_", "dash_", "attack_"))
            if self.tr.standing and hold:
                self.guard_log.append({"tick": self.scene.tick, "zone": list(zone), "how": f"wait {span} first"})
                self._window("wait", span, "fade-guard: no pause while the dying thing vanishes")
                continue
            if span - n <= stretch:
                self.guard_log.append({"tick": self.scene.tick, "zone": list(zone), "how": f"{action} {n}->{span}"})
                return span
            if zone[0] - 1 - s0 >= 1:
                self.guard_log.append({"tick": self.scene.tick, "zone": list(zone), "how": f"{action} {n}->{zone[0] - 1 - s0} (cut)"})
                return zone[0] - 1 - s0
            if hold and span - n <= 6:
                # in the air, right in front of the frames in question: holding the same input a few frames
                # longer than asked is the lesser evil (a frozen game ends the run; first_02_v8_L2 froze here)
                self.guard_log.append({"tick": self.scene.tick, "zone": list(zone), "how": f"{action} {n}->{span} (forced)"})
                return span
            self.guard_log.append({"tick": self.scene.tick, "zone": list(zone), "how": f"{action} {n} unguarded"})
            return n
        return n

    def _window(self, action: str, sim_frames: int, why: str = "") -> Scene:
        prev = self.scene
        px, py = prev.knight_world
        s_prev = self.sim_total
        obs = self.env.step(action, sim_frames + 1, label=self.label, probability=self.prob,
                            meta={"decision": self.decision, "why": why, "sim_frames": sim_frames})
        sc = Scene(obs)
        self.scene = sc
        self.windows_this_skill += 1
        self.sim_frames_this_skill += sim_frames
        self.sim_total += sim_frames
        x, y = sc.knight_world
        n = ph.nsteps(prev.time, sc.time)
        tr = self.tr
        dy = y - py
        if action.startswith("attack_"):
            self.last_strike = s_prev
        if action.startswith("jump_"):
            # a press while clinging is a wall kick: it does not use up the air jump
            tr.jumps = 1 if tr.mode == "climb" else max(0, tr.jumps - 1)
        if sim_frames == 0:
            # zero-time press: nothing has moved yet; only what the press itself changes is known
            if action.startswith("jump_"):
                tr.vy, tr.standing, tr.mode, tr.still_windows = ph.JUMP_V, False, "air", 0
            elif action.startswith("dash_"):
                tr.vy = 0.0
            sup = sc.support()
        else:
            if abs(dy) < 0.004 and abs(x - px) < 20:
                tr.still_windows += 1
            else:
                tr.still_windows = 0
            tr.vy = ph.velocity_after(dy, n) if n > 0 else tr.vy
            sup = sc.support()
            # standing: no vertical motion in this window and a surface under the feet (or no motion twice)
            tr.standing = (abs(dy) < 0.004 and sup is not None and not action.startswith("jump_")) or tr.still_windows >= 3
            if tr.standing:
                tr.vy = 0.0
                tr.mode, tr.wall_side, tr.wall_oid = "ground", 0, ""
                if sup is not None and sup.kind == "platform":
                    tr.jumps = 2
            else:
                # clinging to a wall shows as: a wall right next to the body and a steady 2 units/s slide
                adj = self.adjacent_wall(sc)
                sliding = n > 0 and abs(dy + ph.SLIDE_V * ph.DT * n) < 0.012 + 0.004 * n and abs(x - px) < 0.03
                if adj is not None and sliding and not action.startswith("jump_"):
                    tr.mode, tr.wall_side, tr.wall_oid = "climb", adj[0], adj[1].oid
                    tr.jumps = 1
                    tr.vy = -ph.SLIDE_V
                else:
                    tr.mode, tr.wall_side, tr.wall_oid = "air", 0, ""
        # health / death bookkeeping from the HUD
        tr.respawned = False
        jumped = abs(x - px) > 25 or abs(y - py) > 25
        hud_blank = False
        if sc.hp is not None:
            if tr.hp is not None and tr.hp > 0 and sc.hp == 0 and (jumped or self.sim_total - self._area_sim <= 30):
                # v8: the whole picture has just jumped (a new scene was loaded) and no hearts are drawn in this
                # one picture: the HUD of the new scene is not up yet. A death shows the knight dying where it
                # stood, and the jump back to the level start comes two seconds later (random_s1_v8_L1 took the
                # level exit for a death: the picture after the jump showed 0 hearts, the next one 5).
                hud_blank = True
            elif tr.hp is not None and sc.hp < tr.hp:
                tr.damage_events += 1
                tr.last_damage_tick = sc.tick
                self.hurt_total = s_prev               # the hit landed somewhere in this window
                self.events.append((sc.tick, f"lost {tr.hp - sc.hp} health"))
                if sc.hp == 0:
                    tr.deaths += 1
                    self.events.append((sc.tick, "died"))
            if tr.hp == 0 and sc.hp > 0:
                tr.respawned = True
                self.events.append((sc.tick, "respawned at the level start"))
            if not hud_blank:
                tr.hp = sc.hp
        # v8: the whole picture jumped without a death: a new scene was loaded (the level exit)
        new_area = not tr.respawned and jumped and (tr.hp != 0 or hud_blank)
        if new_area:
            self.area += 1
            self._area_sim = self.sim_total
            self.events.append((sc.tick, "entered a new area"))
            tr.jumps, tr.standing, tr.still_windows, tr.mode = 2, False, 0, "air"
        if tr.respawned or new_area:
            self.map.reset()           # the scene was reloaded: object ids are new, start the memory over
            self.motion.reset()
            self._hazard_tops = {}
            self._falling = set()
            self.no_pause = []
            self.fade_guard = 0
        self.motion.update(sc)
        self._dyn_cache = None
        self._publish_movers(sc)
        self.map.update(sc, sup.oid if (tr.standing and sup is not None) else None, motion=self.motion)
        # a hazard that moved down since the last look and hangs over the knight. v8: boxes that are cut by
        # the screen edge, and things known to roll sideways or to fly across (saws, projectiles), are not
        # falling traps (v7 mistook the upper saw and the gunner's shots for one).
        self.falling_over_me = None
        tops = {}
        for hz in sc.hazards:
            mt = self.motion.get(hz.oid)
            if mt is not None and mt.moving and (mt.swings or abs(mt.vx) > 1.0):
                continue
            if hz.x0 <= EDGE or hz.x1 >= W - EDGE or hz.y0 <= EDGE or hz.y1 >= H - EDGE or (mt is not None and mt.clip[2]):
                continue
            hx0, hy1 = sc.to_world(hz.x0, hz.y0)
            hx1, hy0 = sc.to_world(hz.x1, hz.y1)
            if hx1 - hx0 <= 0.9 and hy1 - hy0 <= 0.9:
                continue                                   # a small thing coming down is a shot, not a falling trap
            hcx = (hx0 + hx1) / 2
            before = self._hazard_tops.get(hz.oid)
            if before is not None and abs(hcx - before[1]) > 0.1:
                continue                                   # it moves sideways too: a flyer, not a falling trap
            tops[hz.oid] = (hy0, hcx)
            if before is not None and hy0 < before[0] - 0.05 and hy0 > y + ph.BODY_H - 0.2 and hx0 - 0.6 < x < hx1 + 0.6:
                self.falling_over_me = self.map.hazards.get(hz.oid)
        for oid, (top, _cx) in tops.items():
            before = self._hazard_tops.get(oid)
            if before is not None and top < before[0] - 0.05:
                self._falling.add(oid)
        now_haz = {hz.oid for hz in sc.hazards}
        gone = [oid for oid in self._falling if oid not in now_haz]
        if gone:                       # a falling thing has hit the ground and is fading out (1 s)
            self.no_pause.append([s_prev + TRAP_FADE_FRAMES - 2, self.sim_total + TRAP_FADE_FRAMES + 2])
            self._falling -= set(gone)
        self._hazard_tops = tops
        # an enemy that was in full view and is not drawn any more is dying: it fades for about 1.9 s
        # (killed by the sword, or by walking into a saw / spikes). No pause when the fade ends (_guard).
        if not (tr.respawned or new_area) and prev.enemies:
            now_ids = {e.oid for e in sc.enemies}
            for e in prev.enemies:
                if e.oid in now_ids:
                    continue
                if action.startswith("attack_") and abs(e.cx - KX) < 3.2 * prev.ppu:
                    self.no_pause.append([s_prev + FADE_FRAMES - 1, s_prev + FADE_FRAMES + 1])
                elif EDGE < e.x0 and e.x1 < W - EDGE and e.y0 > 60 and e.y1 < H - EDGE:
                    self.no_pause.append([s_prev + FADE_FRAMES - 2, self.sim_total + FADE_FRAMES + 1])
        for check in self.stop_checks:
            reason = check(self)
            if reason:
                raise SkillStop(reason)
        return sc

    UNTOUCHABLE = 45        # after a hit nothing can hurt the knight for 1.5 s (it passes through enemies and hazards)

    def safe_frames(self) -> int:
        """For how many frames from now the knight certainly cannot be hurt (0 when not just hit)."""
        return max(0, self.UNTOUCHABLE - 8 - (self.sim_total - self.hurt_total))

    # ---- v8: moving things ---------------------------------------------------------------------
    SWEEP_FRAMES = 14       # walking never goes closer than half a step to where a rolling hazard gets within this time

    def _publish_movers(self, sc: Scene) -> None:
        """Tell the scene which hazards are moving ones and where the rolling ones will be shortly:
        `sc.mover_ids` (left out of the still-hazard rules) and `sc.mover_boxes` (screen boxes swept over
        the next SWEEP_FRAMES frames; skills.walk_limit treats them as something not to walk into)."""
        sc.mover_ids = {tr.oid for tr in self.motion.hazards()}
        sc.mover_boxes = []
        if not sc.mover_ids:
            return
        if any(tr.flew and tr.seen >= sc.time - 1e-6 for tr in self.motion.hazards()):
            self.last_flyer = sc.time
        t0 = sc.time
        for tr in self.motion.hazards():
            if not tr.swings:
                continue
            x0 = y0 = 1e9
            x1 = y1 = -1e9
            for k in range(0, self.SWEEP_FRAMES + 1, 2):
                sh = self._shape(tr, t0 + k * ph.FDT)
                if sh[0] == "disc":
                    bx0, bx1, by0, by1 = sh[1] - sh[3], sh[1] + sh[3], sh[2] - sh[3], sh[2] + sh[3]
                else:
                    bx0, bx1, by0, by1 = sh[1], sh[2], sh[3], sh[4]
                x0, x1, y0, y1 = min(x0, bx0), max(x1, bx1), min(y0, by0), max(y1, by1)
            sx0, sy0 = sc.to_screen(x0, y1)
            sx1, sy1 = sc.to_screen(x1, y0)
            now = sc.to_screen(tr.predict(t0)[0], 0.0)[0]
            sc.mover_boxes.append((sx0, sy0, sx1, sy1, tr.oid, now))

    def dyn(self, n: int = 200, flyers: bool = True) -> ph.Dyn:
        """Where the moving hazards and platforms near the knight will be over the next n frames (from
        the motion memory; cached until the next observation). flyers=False leaves the projectiles out:
        for a path that has no way around one - it is struck out of the air when it comes within reach."""
        key = (self.scene.frame_id, n, flyers)
        if self._dyn_cache is not None and self._dyn_cache[0] == key:
            return self._dyn_cache[1]
        x, y = self.pos
        t0 = self.scene.time
        d = ph.Dyn(n)
        for tr in self.motion.hazards():
            if abs(tr.cx - x) > 24 or abs(tr.cy - y) > 14:
                continue
            if tr.flew and not flyers:
                continue
            if tr.flew and tr.seen < t0 - 0.1:
                continue                                   # a flyer that has left the picture
            d.haz.append([self._shape(tr, t0 + k * ph.FDT) for k in range(n + 1)])
            d.haz_ids.append(tr.oid)
        for tr in self.motion.platforms():
            surf = self.map.solids.get(tr.oid)
            if surf is None or abs(tr.cx - x) > 24:
                continue
            base = tr.predict(t0)[0]
            dx = [tr.predict(t0 + k * ph.FDT)[0] - base for k in range(n + 1)]
            rect = surf.wrect()
            d.sol.append((rect, rect.x0, rect.x1, dx))
            # up to which frame is it certain where the platform is (see motion.Track.certain)?
            sure = n
            for k in range(0, n + 1, 3):
                if not tr.certain(t0 + k * ph.FDT):
                    sure = max(0, k - 3)
                    break
            d.sure[tr.oid] = sure
        self._dyn_cache = (key, d)
        return d

    def _shape(self, tr, t: float) -> tuple:
        """Shape of a moving hazard at time t. Where the prediction is not certain (a back-and-forth mover
        beyond where it was seen to go: it may have turned around at any moment since), the shape is the
        box of every place it could be. A thing that is not drawn in the picture now is not in the picture:
        it can only come in from the side it lies on, at its speed (without this bound a saw glimpsed once
        at the screen edge 'could be anywhere' a minute later and blocked every move, random_s1_v8_L2)."""
        if tr.certain(t):
            return tr.shape(t)
        cx, cy = tr.predict(t)
        w, h = tr.w or 1.0, tr.h or 1.0
        if not tr.swings or len(tr.seg) < 2 or abs(tr.vx) < 1e-6:
            span = (tr.speed or 2.0) * max(t - tr.t, 0.0) + 0.2
            lo, hi = tr.cx - span, tr.cx + span
        else:
            v = tr.vx
            edge = tr.xmax if v > 0 else tr.xmin
            t_edge = tr.t + (edge - tr.cx) / v
            back = edge - v * (t - t_edge)                 # had it turned right there
            lo, hi = min(cx, back), max(cx, back)
        sc = self.scene
        if tr.seen < sc.time - 1e-6:
            vx0, vy1 = sc.to_world(0, 0)
            vx1, vy0 = sc.to_world(W, H)
            if cy + h / 2 > vy0 and cy - h / 2 < vy1:      # at a height that the picture shows
                d = (tr.speed or 2.0) * max(t - sc.time, 0.0)
                right = (tr.cx + (0.01 if tr.vx >= 0 else -0.01)) >= sc.knight_world[0]
                if right:
                    lo = max(lo, vx1 + w / 2 - d)
                    hi = max(hi, lo)
                else:
                    hi = min(hi, vx0 - w / 2 + d)
                    lo = min(lo, hi)
        return ("box", lo - w / 2, hi + w / 2, cy - h / 2, cy + h / 2)

    def strike_plan(self, action: str, n: int):
        """(frames to play first, attack action) when a flying hazard is going to hit the knight and is
        within sword reach at some frame boundary of the next n frames; None otherwise.

        The sword hits whatever a circle of radius 0.6 swept 1.5 units from the knight's centre touches
        (forward or up); striking a projectile destroys it.
        """
        if self.sim_total - self.last_strike < 11:
            return None                                    # sword cooldown (0.35 s)
        t0 = self.scene.time
        flyers = [tr for tr in self.motion.hazards()
                  if tr.flew and (tr.w or 9) <= 0.9 and (tr.h or 9) <= 0.9 and tr.seen >= t0 - 1e-6 and len(tr.seg) >= 2]
        if not flyers:
            return None
        x, y = self.pos
        vx = {"left": -ph.WALK, "right": ph.WALK}.get(action, 0.0)
        air = not self.tr.standing                         # v8: in the air too - the body keeps flying meanwhile
        # standing on a moving platform the knight is carried along (2 units/s: 0.7 units in 10 frames)
        ride = None
        if not air:
            supw = self.support_surf()
            ride = self.motion.get(supw.oid) if (supw is not None and supw.moving) else None
            if ride is not None and not ride.swings:
                ride = None
        horizon = n + 7
        best = None
        for tr in flyers:
            r = min(tr.w, tr.h) / 2
            kx, ky, kvy, t = x, y, (self.tr.vy if air else 0.0), t0
            reach_at, hit = None, False
            for k in range(0, horizon + 1):
                if k > 0:
                    steps = ph.nsteps(t, t + ph.FDT)
                    if k <= n or air:
                        kx += vx * steps * ph.DT
                    if ride is not None:
                        kx += ride.predict(t + ph.FDT)[0] - ride.predict(t)[0]
                    if air:
                        for _ in range(steps):
                            kvy -= ph.G * ph.DT
                            ky += kvy * ph.DT
                    t += ph.FDT
                px, py = tr.predict(t)
                if ph.distance(("disc", px, py, r), kx, ky) < 0.03:
                    hit = True
                    break
                if reach_at is None and k <= n:
                    ax, ay = px - kx, py - (ky + ph.BODY_H / 2)
                    reach = SWORD_RADIUS + 0.5 * r - 0.04
                    side = 1 if ax >= 0 else -1
                    fwd = math.hypot(max(0.0, min(abs(ax), SWORD_REACH)) - abs(ax), ay)       # to the forward sweep
                    up = math.hypot(ax, ay - max(0.0, min(ay, SWORD_REACH)))                 # to the upward sweep
                    if up <= reach and ay > 0:
                        reach_at = (k, "attack_up")
                    elif fwd <= reach:
                        reach_at = (k, "attack_right" if side > 0 else "attack_left")
            if hit and reach_at is not None and (best is None or reach_at[0] < best[0]):
                best = reach_at
        return best

    def adjacent_wall(self, sc: Scene | None = None):
        """(side, wall surf) when a wall sprite touches the knight's side (sprites are a little wider
        than what one can touch, so a small overlap is accepted)."""
        sc = sc or self.scene
        x, y = sc.knight_world
        for s in self.map.solids.values():
            if s.kind != "wall" or y >= s.y1 - 0.05 or y + ph.BODY_H <= s.y0 + 0.05:
                continue
            if -0.40 <= (x - ph.BODY_HW) - s.x1 <= 0.08:
                return -1, s
            if -0.40 <= s.x0 - (x + ph.BODY_HW) <= 0.08:
                return 1, s
        return None

    def start_state(self) -> ph.Start:
        x, y = self.pos
        tr = self.tr
        if tr.standing:
            return ph.Start(x, y, 0.0, self.scene.time, 2, "ground", 0)
        if tr.mode == "climb":
            return ph.Start(x, y, -ph.SLIDE_V, self.scene.time, 1, "climb", tr.wall_side)
        return ph.Start(x, y, tr.vy, self.scene.time, tr.jumps, "air", 0)

    def path_hits_hazard(self, path: list, start: int) -> bool:
        """Does the rest of a planned path touch a hazard that is known now (new ones come into view)?"""
        _, hazards = self.world_rects()
        for px, py, *_ in path[start:]:
            for hz in hazards:
                if px + ph.BODY_HW > hz.x0 and px - ph.BODY_HW < hz.x1 and py < hz.y1 and py + ph.BODY_H > hz.y0:
                    return True
        return False

    def follow(self, o: ph.Outcome, why: str, tol: float = 0.14) -> str:
        """Execute a simulated manoeuvre window by window. Returns done | deviated | died."""
        path, presses = o.path, o.presses
        last = len(path) - 1
        b = 0
        while b < last:
            if b in presses:
                action, fixed = presses[b]
                if fixed:
                    length = max(1, min(fixed, last - b))
                else:
                    axis = path[b + 1][3]
                    length = 1
                    while (length < 4 and b + length < last and (b + length) not in presses
                           and path[b + length + 1][3] == axis):
                        length += 1
            else:
                axis = path[b + 1][3]
                action = "wait" if axis == 0 else ("right" if axis > 0 else "left")
                length = 1
                while (length < 4 and b + length < last and (b + length) not in presses
                       and path[b + length + 1][3] == axis):
                    length += 1
            # the same input stays right for `more` further frames (the fade guard may use them)
            more = 0
            if b not in presses or not presses[b][1]:
                while (b + length + more < last and (b + length + more) not in presses
                       and path[b + length + more + 1][3] == path[b + 1][3] and more < 8):
                    more += 1
            sc = self.step(action, length, f"{why}:b{b}", stretch=more)
            b += self.last_n
            if self.tr.respawned or self.tr.hp == 0:
                return "died"
            if b >= last:
                break
            ex = path[b]
            x, y = sc.knight_world
            if abs(x - ex[0]) > tol or abs(y - ex[1]) > tol:
                return "deviated"
            if self.path_hits_hazard(path, b):
                return "deviated"
        return "done"

    def confirm_cling(self, side: int, why: str, frames: int = 5) -> bool:
        """Hold toward the wall for a few frames until the steady slide shows that the knight clings."""
        action = "right" if side > 0 else "left"
        for _ in range(frames):
            if self.tr.mode == "climb":
                return True
            if self.tr.standing:
                return False
            self.step(action, 1, why + ":cling?")
        return self.tr.mode == "climb"

    # ---- primitives ------------------------------------------------------------------------
    def settle(self, max_frames: int = 90) -> bool:
        """Wait (no input) until the knight stands still on something."""
        used = 0
        while used < max_frames:
            if self.tr.standing and self.tr.still_windows >= 1:
                return True
            self.step("wait", 2, "settle")
            used += 2
        return self.tr.standing

    def walk(self, dirn: int, units: float, why: str = "walk") -> float:
        """Walk up to `units` in one window; returns the distance actually commanded."""
        action = "right" if dirn > 0 else "left"
        sc = self.scene
        best = 1
        for k in range(1, 7):
            n = ph.nsteps(sc.time, sc.time + k * ph.FDT)
            if n * ph.WALK * ph.DT <= units + 1e-6:
                best = k
            else:
                break
        n = ph.nsteps(sc.time, sc.time + best * ph.FDT)
        if n * ph.WALK * ph.DT > units + 0.11:
            return 0.0
        self.step(action, best, why)
        return n * ph.WALK * ph.DT

    def walk_to(self, wx: float, tol: float = 0.12, why: str = "walk_to", max_windows: int = 80) -> bool:
        for _ in range(max_windows):
            x, _ = self.pos
            d = wx - x
            if abs(d) <= tol:
                return True
            moved = self.walk(1 if d > 0 else -1, abs(d), why)
            if moved == 0.0:
                return abs(d) <= tol + 0.1
            if not self.tr.standing and self.tr.still_windows == 0 and abs(self.tr.vy) > 1.0:
                return False        # fell off something
        return False

    def jump_to(self, target: ph.WRect, why: str = "jump", margin: float = 0.3, clear_need: float = 0.25,
                extra_hazards: list | None = None, bump: bool = False) -> tuple[bool, str]:
        """Jump (single or double, with air steering) from the current spot onto `target`.

        Closed loop: after every window the observed position is compared with the model; on a
        mismatch the plan is recomputed from where the knight really is.
        v8: the plan keeps clear of where the moving hazards will be, and a target that moves (a moving
        platform) is aimed at where it will be; on a replan the target is taken from where it is now.
        """
        sc = self.scene
        x, y = sc.knight_world
        solids, hazards = self.world_rects()
        hazards = hazards + list(extra_hazards or [])
        grounded = self.tr.standing
        ref = self.map.solids.get(target.oid)
        ref_x0 = ref.x0 if (ref is not None and ref.moving) else None
        safe = self.safe_frames()
        plan = ph.plan_jump(x, y, 0.0 if grounded else self.tr.vy, sc.time, self.tr.jumps if not grounded else 2,
                            grounded, target, solids, hazards, margin=margin, clear_need=clear_need,
                            dyn=self.dyn(), bump=bump, safe=safe)
        if not plan.ok:
            return False, "no jump plan: " + plan.reason
        boundary = 0
        replans = 0
        guard = 0
        no_flyers = False              # the current plan knowingly crosses a projectile's line (see below)
        while guard < 120:
            guard += 1
            path = plan.path
            if boundary >= plan.land_frame:
                break
            press = (boundary == 0 and plan.press1) or (plan.t2 is not None and boundary == plan.t2 and boundary > 0)
            # axis of the next frame according to the plan
            nxt_axis = path[boundary + 1][3] if boundary + 1 < len(path) else 0
            # extend the window while the axis stays the same and no press boundary is crossed
            length = 1
            while (length < 4 and boundary + length < plan.land_frame
                   and not (plan.t2 is not None and boundary + length == plan.t2)
                   and boundary + length + 1 < len(path) and path[boundary + length + 1][3] == nxt_axis):
                length += 1
            if press:
                length = 1 if (boundary + 2 < len(path) and path[boundary + 2][3] != nxt_axis) else length
                action = "jump_right" if plan.dirn > 0 else "jump_left"
            else:
                action = "wait" if nxt_axis == 0 else ("right" if nxt_axis > 0 else "left")
            more = 0
            while (boundary + length + more < plan.land_frame
                   and not (plan.t2 is not None and boundary + length + more == plan.t2)
                   and boundary + length + more + 1 < len(path) and path[boundary + length + more + 1][3] == nxt_axis
                   and more < 8):
                more += 1
            sc = self.step(action, length, f"{why}:b{boundary}", stretch=more)
            boundary += self.last_n
            x, y = sc.knight_world
            if self.tr.respawned or (self.tr.hp == 0):
                return False, "died during the jump"
            exp = path[min(boundary, len(path) - 1)]
            if boundary >= plan.land_frame:
                break
            dyn = self.dyn(flyers=not no_flyers) if no_flyers else self.dyn()
            safe = self.safe_frames()
            threatened = bool(dyn.haz) and any(
                not dyn.clear(k - boundary, path[k][0], path[k][1], ph.HAZ_MARGIN - 0.04)
                for k in range(boundary + safe, min(len(path), boundary + 40)))
            if abs(x - exp[0]) > 0.12 or abs(y - exp[1]) > 0.12 or threatened or (
                    not extra_hazards and self.path_hits_hazard(path, boundary + safe)):
                # the world disagreed with the model (bumped something, got hit, a hazard came into
                # view on the planned path, a moving hazard is not where it was expected, ...): replan
                replans += 1
                if replans > 6:
                    return False, "jump kept deviating from the model"
                no_flyers = False
                dyn = self.dyn()
                solids, hazards = self.world_rects()
                hazards = hazards + list(extra_hazards or [])
                if self.tr.standing:
                    break
                if ref_x0 is not None:
                    cur = self.map.solids.get(target.oid)
                    if cur is not None:
                        d = cur.x0 - ref_x0
                        target = ph.WRect(target.x0 + d, target.x1 + d, target.y0, target.y1, target.kind, target.oid, target.lip)
                        ref_x0 = cur.x0
                plan = ph.plan_jump(x, y, self.tr.vy, sc.time, self.tr.jumps, False, target, solids, hazards,
                                    margin=margin, clear_need=min(clear_need, 0.1), dyn=dyn, bump=bump, safe=safe)
                if not plan.ok and plan.reason == "touches a moving hazard" and self.reflex and any(
                        tr.flew for tr in self.motion.hazards()):
                    # no way around a projectile: keep the line; it is struck out of the air when it comes
                    # within reach (the strike changes the flight, which is then planned again from there)
                    plan = ph.plan_jump(x, y, self.tr.vy, sc.time, self.tr.jumps, False, target, solids, hazards,
                                        margin=margin, clear_need=min(clear_need, 0.1), dyn=self.dyn(flyers=False),
                                        bump=bump, safe=safe)
                    no_flyers = True
                if not plan.ok:
                    return False, "lost the jump plan mid-air: " + plan.reason
                boundary = 0
        # let the landing finish
        self.settle(40)
        sup = self.scene.support()
        if sup is not None and sup.oid == target.oid:
            return True, "landed"
        return False, "landed on something else" if self.tr.standing else "did not land"

    def land_on(self, oid: str, why: str = "land", max_frames: int = 220) -> tuple[bool, str]:
        """Airborne: come down on the surface `oid`, which is in view and may be moving with turning points
        that are not known (new in v8). Closed loop, like a player who drops next to a moving platform and
        steers onto it: while the feet are below its top, stay beside its edge and rise there (an air jump
        when needed); once above, aim every window at where its middle will be when the feet come down to
        it; another air jump when the time is too short. Steering in the air is as fast as walking
        (5 units/s) and a platform moves at 2 units/s, so following it needs no knowledge of where it turns.
        """
        used = 0
        side_gap = ph.BODY_HW + 0.2
        rise = ph.JUMP_V ** 2 / (2 * ph.G) - 0.06          # height an air jump gains
        while used < max_frames:
            if self.tr.standing:
                break
            if self.tr.respawned or self.tr.hp == 0:
                return False, "died"
            surf = self.map.solids.get(oid)
            if surf is None:
                return False, "the landing place is gone"
            sc = self.scene
            x, y = self.pos
            vy = self.tr.vy
            top = surf.top
            x0, x1 = surf.x0, surf.x1
            mt = self.motion.get(oid)
            moving = mt is not None and mt.swings

            def ahead(k: int) -> float:
                """How far the surface will have moved k frames from now (best guess)."""
                return (mt.predict(sc.time + k * ph.FDT)[0] - mt.predict(sc.time)[0]) if moving else 0.0

            if y < top + 0.05:
                # feet not above its top yet: beside it (or under it) - get clear of its side and rise there
                near_right = abs(x - x1) <= abs(x - x0)
                gate = (x1 + side_gap if near_right else x0 - side_gap) + ahead(3)
                clear = (x >= gate - 0.06) if near_right else (x <= gate + 0.06)
                apex = y + (vy * vy / (2 * ph.G) if vy > 0 else 0.0)
                if apex < top + 0.15 and vy <= 0.6 and clear:
                    if self.tr.jumps <= 0:
                        return False, "cannot get up to it: no jump left"
                    if y + rise < top + 0.15:
                        return False, "it is too high to get up to"
                    before = self.sim_total, self.tr.jumps
                    self.press("jump_left" if near_right else "jump_right", why + ":air-jump")
                    if self.tr.jumps == before[1]:             # the press was not sent (one press per frame)
                        self.step("wait", 1, why + ":beside")
                        used += 1
                    continue
                d = gate - x
                act = "wait" if abs(d) < 0.09 else ("right" if d > 0 else "left")
                self.step(act, 1, why + ":beside")
                used += 1
                continue
            # above its top: frames until the feet come down to it, and where it is then
            tau, yy, vv = 1, y, vy
            for k in range(1, 260):
                vv -= ph.G * ph.FDT
                yy += vv * ph.FDT
                if vv <= 0 and yy <= top:
                    tau = k
                    break
            sh = ahead(tau)
            lo, hi = x0 + sh + 0.3, x1 + sh - 0.3
            if hi < lo:
                lo = hi = (x0 + x1) / 2 + sh
            inset = min(0.35, (hi - lo) / 2)
            want = x if lo + inset <= x <= hi - inset else (lo + inset if x < lo + inset else hi - inset)
            need = abs(want - x)
            can = ph.WALK * tau * ph.FDT
            if need > can + 0.2 and self.tr.jumps > 0 and (tau <= 5 or y - top < 0.5):
                # not enough time to get over it: another jump buys it
                before = self.tr.jumps
                self.press("jump_right" if want >= x else "jump_left", why + ":air-jump")
                if self.tr.jumps != before:
                    continue
            if need > can + 0.6 and self.tr.jumps <= 0 and tau <= 1:
                return False, "could not reach it in time"
            d = want - x
            n = 1 if tau <= 3 else 2
            act = "wait" if abs(d) < 0.09 else ("right" if d > 0 else "left")
            if act != "wait" and abs(d) < ph.WALK * ph.FDT * n:
                n = 1
            self.step(act, n, why + f":steer{tau}")
            used += self.last_n
        self.settle(40)
        sup = self.scene.support()
        if sup is not None and sup.oid == oid:
            return True, "landed"
        return False, "landed on something else" if self.tr.standing else "did not land"

    def _without_enemies(self, surf, spans, keep: float = 1.6):
        """Cut the neighbourhood of visible enemies standing on `surf` out of its landing spans."""
        sc = self.scene
        cuts = []
        for e in sc.enemies:
            ex, ebottom = sc.to_world(e.cx, e.y1)
            if -0.6 <= ebottom - surf.top <= 1.6:
                cuts.append((ex - keep, ex + keep))
        out = []
        for a, b in spans:
            parts = [(a, b)]
            for c0, c1 in cuts:
                nxt = []
                for p0, p1 in parts:
                    if c1 <= p0 or c0 >= p1:
                        nxt.append((p0, p1))
                    else:
                        if c0 > p0:
                            nxt.append((p0, c0))
                        if c1 < p1:
                            nxt.append((c1, p1))
                parts = nxt
            out += [(p0, p1) for p0, p1 in parts if p1 - p0 >= 0.9]
        return out or spans      # nowhere else to go: landing next to an enemy beats falling

    def recover(self, why: str = "recover") -> tuple[bool, str]:
        """Airborne and off-plan (knocked back, slipped, failed jump): head for the best safe surface.

        Uses whatever jumps are left. Generic reflex: 'falling and a platform is reachable -> go there'.
        """
        for _ in range(3):
            if self.tr.standing:
                return True, "standing"
            sc = self.scene
            x, y = sc.knight_world
            solids, hazards = self.world_rects()
            best = None
            for surf in self.map.solids.values():
                if surf.kind != "platform" or surf.open_top or abs(surf.x0 - x) > 14 and abs(surf.x1 - x) > 14:
                    continue
                for a, b in self._without_enemies(surf, self.map.safe_spans(surf, x, y)):
                    target = ph.WRect(a, b, surf.y0, surf.y1, surf.kind, surf.oid, surf.lip)
                    plan = ph.plan_jump(x, y, self.tr.vy, sc.time, self.tr.jumps, False, target, solids, hazards,
                                        margin=0.25, clear_need=0.1, dyn=self.dyn(), safe=self.safe_frames())
                    if not plan.ok and plan.reason == "touches a moving hazard" and self.reflex:
                        plan = ph.plan_jump(x, y, self.tr.vy, sc.time, self.tr.jumps, False, target, solids, hazards,
                                            margin=0.25, clear_need=0.1, dyn=self.dyn(flyers=False), safe=self.safe_frames())
                    if plan.ok and (best is None or plan.score() < best[0].score()):
                        best = (plan, target)
            if best is None:
                self.step("wait", 2, why + ":no-safe-surface")
                continue
            ok, msg = self.jump_to(best[1], why=why, margin=0.25, clear_need=0.1)
            if ok:
                return True, "recovered onto a platform"
        self.settle(60)
        return self.tr.standing, "settled" if self.tr.standing else "still falling"
