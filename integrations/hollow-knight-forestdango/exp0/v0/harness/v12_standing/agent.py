"""v12_standing (2026-11-25, from v11_frontier): v11 plus one correction, a knob (`standing`; with "grid" this code
writes v11's requests byte for byte).

  standing still   The knight counts as standing when it has not moved up or down over the last frames of the
                   decision's window (it was not rising, it lost no mask, the game played no frames of its own), even
                   when the harness's 0.5-unit grid shows no ground under its feet. The forecasts then start from
                   that footing. v11 (and v10, v9_keys) asked the grid only: on the corner of a block the game lets
                   the knight stand where the grid has no solid cell below, the harness then said "in the air" for
                   as long as the knight stood there, every forecast was that of a fall (2.5 units off), and a policy
                   that read them never left (runs/e2_v11_reader3_pocket of the brief's experiment: 312 decisions at
                   (79.3, 36.0)).

Everything else is v11_frontier; its notes follow.

v11_frontier (2026-11-25, from v10_sword_cadence): v10 plus two things, each a knob (with both off this code writes
v10's requests byte for byte). Brief: experiments/hk-forestdango-demo/20261125_1642_harness_boundary/BRIEF.md.

  frontier on    When the knight stands on the ground, a sideways option whose own places have all been stood on
                 also says what lies further that way ON THIS FLOOR, from the knight's own memory of the room:
                 "new ground 8 further" (the nearest part of the floor it has not stood on), "not seen beyond 12",
                 or that everything is seen up to a wall or an edge. Both sides get their note by the same rule; no
                 option is hidden, ordered or recommended.
                 Why: every v10 tag speaks about the next 0.4 to 1.3 s only. A model that reads the visit counts
                 (StartLux-Decision 4B: the less-seen side in 38 of 40 dead-end states) stood still for 94 of 125
                 decisions of a new game at a spot where every option within reach said SEEN, with unvisited floor
                 three steps away on both sides (runs/c7_v10_startlux4b_new of the brief's experiment).
  absent wait    When the knight is not on screen (it died and the game has not put it back yet), the harness cannot
                 describe anything: it lets the game run with no key down until the knight is back and logs those
                 windows as its own (label "[adapter] the knight is not on screen"); v10 crashed there
                 (runs/b2_v10_reader2_new, decision 578). No key is pressed for the model.

Everything else is v10_sword_cadence; its notes follow.

v10_sword_cadence (2026-11-25, from v9_keys): the full Hollow Knight build played by raw keys; the harness only
gives information.

The interface is v9_keys' (the user's rules N159-N168, proposal P60): every decision asks two questions, `move`
(9 options) and `attack` (5 options); both are always answered, every option is always offered in the same order
(keys.py), the model's two argmaxes are the keys of the next window and nothing else is pressed. Everything else in
this folder is an information tool whose result reaches the model as text.
None of the knobs hides, orders or recommends options.
"""
from __future__ import annotations

import math
import re

from . import look, sim, text
from .keys import ATTACKS, MOVE, MOVES, bridge_windows, frame_plan
from .mem import Memory
from .see import BODY_H, Box, View
from .text import q

KNOBS = {
    "frames": 13,             # game frames per decision (30 per second); 13 = one sword cooldown
    "tail": 2,                # frames of a decision's last bridge window (speeds are read from it)
    "reach": "on",            # on = an AIR tag names the highest and the lowest place still reachable by holding or
                              # letting go of the option's keys at the next decisions; off = v9: 'then' = same keys held
    "reach_depth": 2,         # how many later decisions the reach forecast lets the keys change in
    "enemy_speed": "frame",   # frame = from the last frames of the window (see _blend); window = over the whole window (v9)
    "strike_facts": "moving", # moving = over the frames the blade is out, with motion; now = at the decision (v9)
    "kills": "strict",        # strict = "an enemy died" needs a hit of ours and a place well inside the screen; v9 = v9's rule
    "split": "one",           # one = a 9-option move question; two = direction (3) and jump (3) as two questions
    "forecast": "option",     # how the per-option move forecasts are shown (they are always computed and logged):
                              # option = a tag in each option text; short = the same tag for this window only (no
                              # "then", no visit count); state = sentences in the state text; off = not shown
    "attack_notes": "option", # where the per-option attack facts are shown: option | state | off
    "screen": "on",           # the "what is on screen" lines
    "memory": "on",           # the memory lines
    "briefing": "full",       # full | short
    "legend": "on",           # on = the briefing explains the tag words used in the option texts (all of them)
    "facing": "on",           # off = the status line does not say which way the knight faces (the attack tags still do)
    "sides": "plain",         # words for sideways distance in the move tags: plain = "WALK 1.9" / "AIR 1.9 across, 4.4 up" (the
                              # option's keys say the side); named = "WALK 1.9 right" (found to pull Laya toward "left")
    "round": 0.1,             # distances in the texts are said to this step (0.1 | 0.5)
    "sword": "on",            # off = no separate "the sword is ready / not ready" line in the state (the attack tags say it)
    "tried": "off",           # on = each move option's tag ends with how often it was chosen at this place before
    "ledges": 4,              # how many nearby ledges are listed
    "recent": 3,              # how many past decisions are listed
    "frontier": "on",         # on = a sideways option that leads only to places stood on before also says what lies
                              # further that way on this floor (new ground / not seen / wall / edge); off = v10
    "absent": "wait",         # wait = while the knight is not on screen the game runs with no key until it is back; off = v10
    "standing": "still",      # still = not moving up or down over the window's last frames is standing, also where
                              # the grid shows no ground below (the corner of a block); grid = v11: the grid decides
}
V11_KNOBS = {"standing": "grid"}
V10_KNOBS = {"frontier": "off", "absent": "off", **V11_KNOBS}
V9_KNOBS = {"frames": 9, "tail": 1, "reach": "off", "enemy_speed": "window", "strike_facts": "now", "kills": "v9", **V10_KNOBS}
FRONTIER_REACH = 60.0     # how far along the floor the frontier note looks (units)
ABSENT_WINDOWS = 80       # at most this many no-key windows while the knight is not on screen
BLADE = (0.033, 0.067, 0.1, 0.133)   # seconds after the press at which the blade is checked (it is out for about 0.1 s)
TURN = 0.033              # a strike to the other side comes out one frame later (the knight turns first)
SPEED_TOL = 1.5           # two speed measures within this (units / s) count as the same motion
EDGE = 1.0                # an enemy closer than this to the screen's border may have walked off
STRIKE_GAP = 2.0          # a side strike hits when the gap between the two bodies is at most this (v8f probe_03)
STRIKE_UP = 1.6           # estimated reach of an up / down strike beyond the body
COOLDOWN = 13             # frames from one strike to the next (0.4 s cooldown: 12 frames is still too early)
FOCUS_SOUL = 0.33
AGREE = 0.15              # forecast check: the forecast agrees when both coordinates are within this


class Agent:
    def __init__(self, env, knobs: dict | None = None):
        self.env = env
        self.knobs = {**KNOBS, **(knobs or {})}
        for k in ("frames", "ledges", "recent", "tail", "reach_depth"):
            self.knobs[k] = int(self.knobs[k])
        if not 1 <= self.knobs["tail"] < self.knobs["frames"] - 4:
            raise ValueError("tail must be at least 1 and leave room for a short jump")
        unknown = set(self.knobs) - set(KNOBS)
        if unknown:
            raise ValueError(f"unknown knob(s): {sorted(unknown)}")
        self.knobs["round"] = float(self.knobs["round"])
        text.STEP = self.knobs["round"]
        self.n = self.knobs["frames"]
        self.mem = Memory()
        self.view: View | None = None
        self.grounded = True
        self.vy = 0.0
        self.rising = False
        self.jump_down = False
        self.frames_played = 0
        self.last_strike = -10 ** 6
        self.focus_run = 0             # decisions in a row with focus held
        self.tracks: dict[str, tuple] = {}   # enemy id -> (cx, cy, time) at the previous decision
        self.speeds: dict[str, tuple] = {}
        self.forecasts: dict = {}
        self.check_log: list[dict] = []
        self.struck_at: dict[str, int] = {}
        self.cruise: dict[str, list] = {}     # enemy id -> largest steady speed seen per axis (how fast it moves when it moves)
        self.last_hit = -10 ** 6              # frames_played when a strike of ours last hit an enemy
        self.text_cuts = 0                    # option texts shortened to fit the model's limit
        self.stats = {"strikes": 0, "not_ready": 0, "hits": 0, "hurt_forecast": [0, 0, 0, 0]}
        self.frontier_stats = {"notes": 0, "cut": 0}      # frontier notes written / left out for the 48-token limit
        self.absent_windows = 0                           # no-key windows played while the knight was not on screen
        self.footing = None                               # (x, y) where the knight stands without ground in the grid
        self.still_decisions = 0                          # decisions that ended on such a footing

    # ---- observing -----------------------------------------------------------------------------------
    def boot(self) -> None:
        self._observe(None, View(self.env.obs))
        self.mem.stand(self.view, self.grounded)

    def _observe(self, before: View | None, now: View, calm: bool = False) -> None:
        """`before` = the observation one frame earlier (same room, no adapter window between), else None."""
        self.view = now
        room = self.mem.see(now)
        world = self._world(room)
        # standing = the feet rest 0.017 above a top (a knight 0.04 above it is still falling: a jump pressed now does
        # not start); on ground that is not on the 0.5 raster: no vertical motion over the last frame, ground just below
        span = (now.x - sim.HW + 0.02, now.x + sim.HW - 0.02) if now.me is not None else (0, 0)
        support = now.me is not None and world.blocked(span[0], span[1], now.y - 0.025, now.y - 0.002)
        loose = now.me is not None and world.blocked(span[0], span[1], now.y - 0.6, now.y - 0.002)
        if before is not None and before.me is not None and now.me is not None:
            steps = max(1, sim.nsteps(before.time, now.time))
            dy = now.y - before.y
            self.grounded = (dy < 0.004 and support) or (abs(dy) < 0.0015 and loose)
            self.footing = None
            if (not self.grounded and self.knobs["standing"] == "still" and calm and steps >= 2 and abs(dy) < 0.0015
                    and not self.rising):
                # no motion up or down over two or more frames, away from any ceiling bump or freeze: it stands
                self.grounded = True
                self.footing = (now.x, now.y)
            v = dy / (steps * sim.DT)
            if not self.grounded and v > -sim.MAX_FALL:
                v -= (steps - 1) / 2 * sim.G * sim.DT     # the speed at the end of the frame, not its mean
            self.vy = 0.0 if self.grounded else v
            self.rising = (not self.grounded) and dy > 0.004
        else:
            self.grounded, self.vy, self.rising = bool(support), 0.0, False
            self.footing = None
        speeds = {}
        tracks = {}
        for e in now.enemies:
            tracks[e.oid] = (e.cx, e.cy, now.time)
            old = self.tracks.get(e.oid)
            win = None
            if old and now.time - old[2] > 1e-3:
                win = ((e.cx - old[0]) / (now.time - old[2]), (e.cy - old[1]) / (now.time - old[2]))
            if self.knobs["enemy_speed"] == "window":
                if win:
                    speeds[e.oid] = win
                continue
            last = None
            if before is not None and now.time - before.time > 1e-3:
                b = next((x for x in before.enemies if x.oid == e.oid), None)
                if b is not None:
                    dt = now.time - before.time
                    last = ((e.cx - b.cx) / dt, (e.cy - b.cy) / dt)
            speeds[e.oid] = self._blend(e.oid, win, last)
        self.tracks, self.speeds = tracks, speeds

    def _blend(self, oid: str, win, last) -> tuple:
        """An enemy's speed now. `win` = over the whole decision (long baseline, exact when the motion was steady),
        `last` = over the window's last frames (short baseline: right about a turn, but one screen pixel is 0.03 units,
        so it is off by up to 0.5 units / s with a 2-frame tail). Steady motion -> `win`; a change inside the window
        -> `last`, and when it is at least half the enemy's own steady speed seen before, that speed with the sign
        of `last` (a crawler walks at one speed)."""
        if last is None:
            return win or (0.0, 0.0)
        if win is None:
            return last
        cruise = self.cruise.setdefault(oid, [0.0, 0.0])
        out = []
        for i in (0, 1):
            if abs(last[i] - win[i]) <= SPEED_TOL:
                out.append(win[i])
                cruise[i] = max(cruise[i], abs(win[i]))
            elif cruise[i] > 1.0 and abs(last[i]) >= 0.5 * cruise[i]:
                out.append(math.copysign(cruise[i], last[i]))
            else:
                out.append(last[i])
        return tuple(out)

    def _world(self, room) -> sim.World:
        boxes = list(room.boxes.values())
        if self.footing is not None:
            # what the knight stands on although the grid has no ground there: a thin top under its feet
            fx, fy = self.footing
            boxes.append(Box(fx - sim.HW, fx + sim.HW, fy - 0.25, fy, "terrain", "footing"))
        return sim.World(room, boxes)

    def _start_state(self) -> sim.State:
        v = self.view
        return sim.State(v.x, v.y, self.vy, self.grounded, self.jump_down and self.rising, 0, v.time)

    # ---- the question --------------------------------------------------------------------------------
    def _wait_for_knight(self) -> None:
        """The knight is not on screen (death, before the game puts it back): nothing can be described or chosen.
        The game runs with no key down, one decision's length at a time, until the knight is there again."""
        waited = 0
        while self.view.me is None and waited < ABSENT_WINDOWS:
            obs = self.env.step_keys(self.n, [], [], [], label="[adapter] the knight is not on screen",
                                     meta={"adapter": "absent"})
            self._observe(None, View(obs))
            waited += 1
        self.absent_windows += waited
        self.jump_down = False
        self.frames_played += waited * self.n
        if self.view.me is None:
            raise RuntimeError(f"the knight is still not on screen after {waited} windows")
        self.mem.stand(self.view, self.grounded)

    def _frontier(self, v: View, room, world, direction: int) -> str | None:
        """What lies further along the floor the knight stands on, from its memory of the room: the nearest part it
        has not stood on, or what ends the floor it knows. None when nothing can be said."""
        hw, d, y = 0.25, 0.0, v.y
        while d < FRONTIER_REACH:
            d += look.CELL / 2
            nx = v.x + direction * d
            if world.unknown(nx, y + 0.3):
                return f"not seen beyond {max(1, round(d))}"
            if world.blocked(nx - hw, nx + hw, y + 0.05, y + BODY_H - 0.02):
                return f"all seen up to a wall {max(1, round(d))} further"
            if not world.blocked(nx - hw + 0.02, nx + hw - 0.02, y - 0.3, y - 0.002):
                return f"all seen up to an edge {max(1, round(d))} further"
            if room.visits(nx, y) == 0:
                return f"new ground {max(1, round(d))} further"
        return None

    def ask(self) -> dict:
        if self.view.me is None and self.knobs["absent"] == "wait":
            self._wait_for_knight()
        v, n = self.view, self.n
        room = self.mem.room(v.room)
        world = self._world(room)
        surfs = look.surfaces(room, room.boxes.values(), v.x, v.y)
        dangers = [(h, 0.0, 0.0) for h in room.hazards.values()]
        for e in v.enemies:
            sx, sy = self.speeds.get(e.oid, (0.0, 0.0))
            dangers.append((e, sx, sy))
        start = self._start_state()
        fc, lines = {}, {}
        for mid, direction, jump in MOVES:
            plan = frame_plan(mid, n, self.jump_down, self.rising)
            tr = sim.run(world, start, plan, n, dangers)
            e = tr.end
            item = {"x": round(e.x, 3), "y": round(e.y, 3), "grounded": e.grounded, "vy": round(e.vy, 2),
                    "hit": tr.hit[1] if tr.hit else None, "blocked": tr.blocked_x, "bumped": tr.bumped}
            tr2 = None
            if not e.grounded and tr.hit is None:
                reach = self.knobs["reach"] == "on"
                # 'then': the same keys stay held. v9 placed moving dangers as at the moment of the decision here.
                tr2 = sim.run(world, e, lambda k, plan=plan: plan(k + n), 150, dangers, stop_on_land=True,
                              t0=n / 30 if reach else 0.0)
                item["then"] = self._outcome(tr2)
                if reach:
                    item["reach"] = self._reach_set(world, room, e, direction, jump, dangers, surfs)
            fc[mid] = item
            lines[mid] = self._forecast_line(mid, direction, jump, start, tr, item, room, surfs)
        self.forecasts = fc
        notes = self._attack_notes(v, room)
        tags = self._attack_tags(v, room)
        knobs = self.knobs
        terse = {m: self._forecast_tag(m, fc[m], room, short=knobs["forecast"] == "short") for m, _, _ in MOVES}
        frontier = {}
        if knobs["frontier"] == "on" and self.grounded and knobs["forecast"] == "option":
            notes_f = {d: self._frontier(v, room, world, d) for d in (-1, 1)}
            for m, direction, _ in MOVES:
                note, tag = notes_f.get(direction), terse[m]
                if not note or "NEW PLACE" in tag or "UNSEEN" in tag or not ("SEEN" in tag):
                    continue
                cand = tag.rstrip(".") + "; " + note + "."
                if text.option_tokens(m, f"{cand} {text.MOVE_TEXT[m]}") <= text.OPTION_TOKENS:
                    terse[m] = cand
                    frontier[m] = note
                    self.frontier_stats["notes"] += 1
                else:
                    self.frontier_stats["cut"] += 1
        if knobs["tried"] == "on" and self.grounded:
            here = (v.room, room.place(v.x, v.y))
            for m, _, _ in MOVES:
                k = self.mem.tried.get((here, m), 0)
                terse[m] += f" Chosen here {k}x." if k else " Not chosen here yet."
        essential = [(text.BRIEFING if knobs["briefing"] == "full" else text.BRIEFING_SHORT).format(secs=text.secs(n), jumps=text.jumps(n))]
        if knobs["legend"] == "on":
            essential[0] += " " + text.LEGEND.format(air=text.AIR_REACH if knobs["reach"] == "on" else text.AIR_HOLD)
            if knobs["frontier"] == "on":
                essential[0] += " " + text.FRONTIER
        essential.append(self._status(v, room))
        if knobs["screen"] == "on":
            essential.append(self._screen(v, room, world, surfs))
        if knobs["forecast"] == "state":
            head = ("What each movement option does (first the next {s} s; 'then' = if you keep holding the same keys, "
                    "with the jump key let go after a short jump):").format(s=text.secs(n))
            essential.append(head + "\n" + "\n".join(f"{text.MOVE_TEXT[m]}: {lines[m]}" for m, _, _ in MOVES))
        if knobs["attack_notes"] == "state":
            essential.append("Attack facts: " + notes["_general"] + " " + " ".join(
                f"{text.ATTACK_TEXT[a]}: {notes[a]}" for a in ATTACKS if notes.get(a)))
        elif knobs["attack_notes"] == "option" and knobs["sword"] == "on":
            essential.append(notes["_general"])
        if knobs["memory"] == "on":
            essential.append(self._memory(v, room))
        cuts = {}
        questions = text.questions(n, terse if knobs["forecast"] in ("option", "short") else None,
                                   tags if knobs["attack_notes"] == "option" else None, knobs["split"],
                                   legacy_fit=knobs["reach"] != "on", cuts=cuts)
        self.text_cuts += len(cuts)
        return {"essential": essential, "questions": questions, "forecasts": fc, "forecast_lines": lines,
                "forecast_tags": terse, "attack_tags": tags, "text_cuts": cuts, "frontier": frontier,
                "facts": {"x": round(v.x, 3), "y": round(v.y, 3), "room": v.room, "grounded": self.grounded, "vy": round(self.vy, 2),
                          "rising": self.rising, "jump_down": self.jump_down, "facing": v.facing, "hp": v.hp, "soul": v.soul,
                          "sword_ready": self._sword_ready(), "enemies": len(v.enemies)}}

    @staticmethod
    def compose(choices: dict) -> tuple[str, str]:
        """The (move, attack) pair of the answers. With split=two the move is the pair of the two key answers."""
        if "move" in choices:
            return choices["move"], choices["attack"]
        d, j = choices["direction"], choices["jump"]
        return text.SPLIT_TO_MOVE[(d, j)], choices["attack"]

    @staticmethod
    def _outcome(tr) -> dict:
        """How a continuation ends: hurt before landing, a landing, out of what was seen, or still falling."""
        if tr.hit and (tr.landed is None or tr.hit[0] <= tr.landed[0]):
            return {"hit": tr.hit[1]}
        if tr.landed:
            return {"land": [round(tr.landed[1], 2), round(tr.landed[2], 2)], "frames": tr.landed[0]}
        if tr.unknown:
            return {"unseen": True}
        return {"falling": True}

    def _reach_set(self, world, room, e, direction: int, jump: str, dangers, surfs) -> dict:
        """Where an option that ends its decision in the air can still land (knob reach=on).

        The continuations: at each of the next `reach_depth` decisions the player holds or lets go of the option's own
        keys (its direction key; its jump key while the knight still rises on it, and a jump key let go stays up: there
        is no second jump); after them the last keys stay as they are until the knight lands. The other direction key
        is not part of it. Every continuation is played with the forecast physics on the remembered room.
        Returns {"lands": the places shown, top to bottom (at most three: the highest, the lowest, and one between),
        "places": all places found, "more": places not shown, "unseen": some continuation leaves the seen part of
        the room, "hurt": continuations that touch a danger first, "hurt_kind", "n": continuations}."""
        n, depth = self.n, self.knobs["reach_depth"]
        dirs = (direction, 0) if direction else (0,)
        releases = range(depth + 1) if (jump == "long" and e.jumping and e.vy > 0) else (0,)
        patterns = [()]
        for _ in range(depth):
            patterns = [p + (d,) for p in patterns for d in dirs]
        lands, unseen, hurt, hurt_kind, count = [], False, 0, None, 0
        for r in releases:
            for pat in patterns:
                def plan(k, r=r, pat=pat):
                    w = k // n
                    if w < depth:
                        return pat[w], w < r, False
                    return pat[-1], r == depth, False
                tr = sim.run(world, e, plan, depth * n + 150, dangers, stop_on_land=True, t0=n / 30)
                count += 1
                o = self._outcome(tr)
                if "land" in o:
                    lands.append(tuple(o["land"]))
                elif "hit" in o:
                    hurt += 1
                    hurt_kind = hurt_kind or o["hit"]
                elif "unseen" in o:
                    unseen = True
        # one entry per place: two landings on the same surface are one place (the landing nearest to the knight
        # stands for it); landings outside the surfaces known around the knight are grouped by where they are
        v = self.view
        places = []
        for x, y in sorted(lands, key=lambda l: (-l[1], abs(l[0] - v.x))):
            s = look.surface_at(surfs, x, y)
            if any((s is not None and s is ps) or (abs(y - py) < 0.3 and abs(x - px) < 1.0) for px, py, ps in places):
                continue
            places.append((x, y, s))
        places = [(x, y) for x, y, _ in places]
        # shown: the highest, the lowest and, when there are more, the one between them that is nearest to the knight
        shown = []
        if places:
            shown.append(places[0])
            if len(places) > 2:
                shown.append(min(places[1:-1], key=lambda l: abs(l[0] - v.x) + abs(l[1] - v.y)))
            if len(places) > 1:
                shown.append(places[-1])
        return {"lands": [list(l) for l in shown], "places": [list(l) for l in places], "more": len(places) - len(shown),
                "unseen": unseen, "hurt": hurt, "hurt_kind": hurt_kind, "n": count}

    def _forecast_tag(self, mid, item, room, short: bool = False) -> str:
        """The forecast of a move option as a short tag: the same fixed words for every option, good or bad."""
        return self._plain(self._forecast_tag_named(mid, item, room, short))

    def _plain(self, tag: str) -> str:
        if self.knobs["sides"] == "plain":
            # the option's keys already name the side; the tag gives the distance only
            if self.knobs["reach"] == "on":
                # also for the whole-unit distances of the reach places; a sideways distance always says "across"
                tag = re.sub(r"(\d+(?:\.\d)?) (left|right)", r"\1 across", tag)
                tag = re.sub(r"^(WALK \d+\.\d) across", r"\1", tag)
            else:
                tag = re.sub(r"(\d+\.\d) (left|right)(, \d+\.\d (up|down))", r"\1 across\3", tag)
                tag = re.sub(r"(\d+\.\d) (left|right)", r"\1", tag)
        return tag

    def _forecast_tag_named(self, mid, item, room, short: bool = False) -> str:
        v = self.view
        dx, dy = item["x"] - v.x, item["y"] - v.y
        hurt = {"spikes": "spikes", "acid": "acid", "hazard": "a hazard", "enemy": "an enemy"}

        def seen(x, y):
            k = room.visits(x, y)
            return "NEW PLACE" if k == 0 else f"SEEN {k}x"
        if short:
            if item.get("hit"):
                return f"HURT: touch {hurt[item['hit']]}."
            if item["grounded"]:
                if abs(dx) < text.small() and abs(dy) < text.small():
                    return "WALL: you do not move." if item.get("blocked") else "STAY: you do not move."
                if abs(dy) < text.small() and self.grounded:
                    return f"WALK {text.rel(dx, 0)}{' to a wall' if item.get('blocked') else ''}."
                return f"LAND {text.rel(dx, dy)}."
            return f"AIR {text.rel(dx, dy, 'same spot')}{', head hits ceiling' if item.get('bumped') else ''}."
        if item.get("hit"):
            return f"HURT: touch {hurt[item['hit']]}."
        if item["grounded"]:
            if abs(dx) < text.small() and abs(dy) < text.small():
                return "WALL: you do not move." if item.get("blocked") else "STAY: you do not move."
            if abs(dy) < text.small() and self.grounded:
                return f"WALK {text.rel(dx, 0)}{' to a wall' if item.get('blocked') else ''}, {seen(item['x'], item['y'])}."
            return f"LAND {text.rel(dx, dy)}, {seen(item['x'], item['y'])}."
        reach = item.get("reach")
        if reach is not None:
            # v10: the tag names where the option can still land (highest, lowest), not where it is in the air
            now = "AIR" + (", head hits ceiling" if item.get("bumped") else "")
            if reach["lands"]:
                # whole units here, and as many of the places as fit into the model's 48 tokens per option: all
                # three; else the highest and the lowest; else the highest. What is left out is counted ("+2 more").
                lands, total = reach["lands"], len(reach["places"])
                picks = [lands, [lands[0], lands[-1]] if len(lands) > 2 else None, lands[:1] if len(lands) > 1 else None]
                for i, pick in enumerate(p for p in picks if p):
                    parts = [f"{text.rel_whole(lx - v.x, ly - v.y)}, {seen(lx, ly)}" for lx, ly in pick]
                    out = f"{now}; then LAND " + "; or ".join(parts)
                    if total > len(pick):
                        out += f"; +{total - len(pick)} more"
                    out += "; or UNSEEN area." if reach["unseen"] else "."
                    plain = self._plain(out)
                    if text.option_tokens(mid, f"{plain} {text.MOVE_TEXT[mid]}") <= text.OPTION_TOKENS:
                        break
                item["reach_shown"] = len(pick)
                return out
            if reach["unseen"]:
                return f"{now}; then UNSEEN area."
            if reach["hurt"]:
                return f"{now}; then HURT: {hurt[reach['hurt_kind']]}."
            return f"{now}; then FALL."
        now = f"AIR {text.rel(dx, dy, 'same spot')}"
        if item.get("bumped"):
            now += ", head hits ceiling"
        then = item.get("then") or {}
        if "hit" in then:
            return f"{now}; then HURT: {hurt[then['hit']]}."
        if "land" in then:
            lx, ly = then["land"]
            return f"{now}; then LAND {text.rel(lx - v.x, ly - v.y, 'here')}, {seen(lx, ly)}."
        if then.get("unseen"):
            return f"{now}; then UNSEEN area."
        return f"{now}; then FALL."

    def _attack_tags(self, v: View, room) -> dict:
        """The attack facts as short tags, one per option, the same fixed words whether the outcome is good or bad."""
        moving = self.knobs["strike_facts"] == "moving"
        reach = self._reach_moving(v, 0, False) if moving else self._reach(v, room)
        ready = self._sword_ready()
        other = "left" if v.facing == "right" else "right"
        tags = {"none": "NO KEY: nothing happens."}

        def side(name, r=None):
            r = reach[name] if r is None else r[name]
            return f"HIT: {r[0]} {name}" if r else f"MISS: nothing in reach {name}"
        if not ready:
            wait = (COOLDOWN - (self.frames_played - self.last_strike)) / 30
            for a in ("forward", "up", "down"):
                tags[a] = f"NOT READY: sword needs {wait:.1f} s more."
        elif moving:
            # the strike's side and the knight's own motion follow the direction key held in the same decision:
            # no key = stands and strikes the way it faces; the facing side's key = walks into the strike;
            # the other key = turns first (one frame later) and walks that way
            sign = 1 if v.facing == "right" else -1
            toward = self._reach_moving(v, sign, False)
            away = self._reach_moving(v, -sign, True)
            fwd = side(v.facing)
            notes = []
            if bool(toward[v.facing]) != bool(reach[v.facing]):
                notes.append(f"{side(v.facing, toward)} with {v.facing} held")
            if away[other]:
                notes.append(f"{side(other, away)} with {other} held")
            tags["forward"] = fwd + (f" ({'; '.join(notes)})" if notes else "") + "."
        else:
            fwd = side(v.facing)
            if reach[other]:
                fwd += f" ({side(other)} with {other} held)"
            tags["forward"] = fwd + "."
        if ready:
            tags["up"] = (f"HIT: {reach['up'][0]} above." if reach["up"] else "MISS: nothing in reach above.")
            if self.grounded:
                tags["down"] = "ON GROUND: comes out forward. " + tags["forward"]
            else:
                tags["down"] = (f"HIT: {reach['down'][0]} below." if reach["down"] else "MISS: nothing in reach below.")
        soul, hp, hp_max = v.soul or 0.0, v.hp or 0, v.hp_max or 0
        if soul < FOCUS_SOUL:
            tags["focus"] = f"NO EFFECT: SOUL {soul:.2f}, a mask needs {FOCUS_SOUL:.2f}."
        elif hp >= hp_max:
            tags["focus"] = "NO EFFECT: health is full."
        elif not self.grounded:
            tags["focus"] = "NO EFFECT: in the air."
        else:
            left = max(0.3, 1.2 - self.focus_run * self.n / 30)
            tags["focus"] = f"HEAL: +1 mask after {left:.1f} s of holding still."
        return tags

    def _sword_ready(self) -> bool:
        return self.frames_played - self.last_strike >= COOLDOWN

    def _forecast_line(self, mid, direction, jump, start, tr, item, room, surfs) -> str:
        v, n = self.view, self.n
        e = tr.end
        dx, dy = e.x - v.x, e.y - v.y
        side = "right" if direction > 0 else "left"
        hurt = {"spikes": "you touch spikes", "acid": "you fall into acid", "hazard": "you touch a hazard", "enemy": "you touch an enemy"}
        if tr.hit:
            return f"{hurt[tr.hit[1]]} after {tr.hit[0] / 30:.1f} s."
        if e.grounded:
            there = text.visited(room.visits(e.x, e.y))
            if start.grounded and tr.landed is None:
                if direction == 0:
                    return "you stay where you are." if jump == "none" else "you stay where you are (the jump does not start)."
                if abs(dx) < 0.15:
                    return f"a wall is in the way; you do not move."
                return f"you walk {abs(dx):.1f} {side}" + (" and a wall stops you" if tr.blocked_x else "") + f"; {there}."
            return f"you land {text.rel(dx, dy)}; {there}."
        now = f"you are {text.rel(dx, dy, 'at the same spot')} and {'rising' if e.vy > 0 else 'falling'}"
        if tr.bumped:
            now += " (head hits the ceiling)"
        if tr.blocked_x and direction:
            now += f" (a wall blocks the way {side})"
        then = item.get("then") or {}
        if "hit" in then:
            return f"{now}; then {hurt[then['hit']]}."
        if "land" in then:
            lx, ly = then["land"]
            s = look.surface_at(surfs, lx, ly)
            what = "" if s is None else (" on a ledge %.1f wide" % s.w if s.w < 6 else " on wide ground")
            return f"{now}; then you land {text.rel(lx - v.x, ly - v.y, 'where you are now')}{what}, {text.visited(room.visits(lx, ly))}."
        if then.get("unseen"):
            return f"{now}; then you leave the part of the room you have seen."
        return f"{now}; then you keep falling."

    def _reach(self, v: View, room) -> dict:
        """What a strike in each direction would touch now: {"left": (kind, gap) | None, "right", "up", "down"}."""
        out = {"left": None, "right": None, "up": None, "down": None}
        things = list(v.enemies) + list(v.doors) + list(v.walls)
        me = v.me
        if me is None:
            return out
        for t in things:
            name = {"enemy": "an enemy", "door": "a door", "wall": "a wall block"}[t.kind]
            if t.y0 < me.y1 + 0.5 and t.y1 > me.y0 - 0.2:
                for side, gap in (("left", me.x0 - t.x1), ("right", t.x0 - me.x1)):
                    if t.overlaps(me.x0, me.x1, me.y0, me.y1):
                        gap = 0.0
                    if -0.6 <= gap <= STRIKE_GAP and (out[side] is None or (t.kind == "enemy" and out[side][0] != "an enemy")):
                        out[side] = (name, max(0.0, gap))
            if t.x0 < me.x1 + 0.6 and t.x1 > me.x0 - 0.6:
                if 0 <= t.y0 - me.y1 <= STRIKE_UP or (t.y0 < me.y1 < t.y1):
                    if out["up"] is None or t.kind == "enemy":
                        out["up"] = (name, max(0.0, t.y0 - me.y1))
                if 0 <= me.y0 - t.y1 <= STRIKE_UP and t.kind == "enemy":
                    out["down"] = (name, me.y0 - t.y1)
        return out

    def _reach_moving(self, v: View, walk: int, turn: bool) -> dict:
        """What a strike pressed now would touch while the blade is out (knob strike_facts=moving): the same result
        as `_reach`, {"left": (kind, gap) | None, "right", "up", "down"}, but checked at the moments of BLADE with
        every enemy moved on by its speed and the knight by its own motion: `walk` = the direction key held in the
        same decision (-1, 0, 1; 6.5 units / s; a solid box stops the walk), falling when in the air. `turn` = the
        strike comes out one frame later because the knight turns first."""
        out = {"left": None, "right": None, "up": None, "down": None}
        me = v.me
        if me is None:
            return out
        things = [(e, *self.speeds.get(e.oid, (0.0, 0.0))) for e in v.enemies]
        things += [(b, 0.0, 0.0) for b in list(v.doors) + list(v.walls)]
        for t in BLADE:
            t += TURN if turn else 0.0
            kx = walk * sim.WALK * t
            ky = 0.0 if self.grounded else max(self.vy * t - 0.5 * sim.G * t * t, -sim.MAX_FALL * t)
            for b, vx, vy in things:
                name = {"enemy": "an enemy", "door": "a door", "wall": "a wall block"}[b.kind]
                x0, x1, y0, y1 = me.x0 + kx, me.x1 + kx, me.y0 + ky, me.y1 + ky
                bx0, bx1, by0, by1 = b.x0 + vx * t, b.x1 + vx * t, b.y0 + vy * t, b.y1 + vy * t
                if b.kind != "enemy" and by0 < y1 and by1 > y0:
                    # a solid box in the way stops the walk at its face
                    if walk > 0 and b.x0 >= me.x1 - 0.05 and x1 > b.x0:
                        x0, x1 = b.x0 - me.w, b.x0
                    if walk < 0 and b.x1 <= me.x0 + 0.05 and x0 < b.x1:
                        x0, x1 = b.x1, b.x1 + me.w
                if by0 < y1 + 0.5 and by1 > y0 - 0.2:
                    for side, gap in (("left", x0 - bx1), ("right", bx0 - x1)):
                        if bx0 < x1 and bx1 > x0 and by0 < y1 and by1 > y0:
                            gap = 0.0
                        if -0.6 <= gap <= STRIKE_GAP and (out[side] is None or (b.kind == "enemy" and out[side][0] != "an enemy")):
                            out[side] = (name, max(0.0, gap))
                if bx0 < x1 + 0.6 and bx1 > x0 - 0.6:
                    if 0 <= by0 - y1 <= STRIKE_UP or (by0 < y1 < by1):
                        if out["up"] is None or b.kind == "enemy":
                            out["up"] = (name, max(0.0, by0 - y1))
                    if 0 <= y0 - by1 <= STRIKE_UP and b.kind == "enemy":
                        out["down"] = (name, y0 - by1)
        return out

    def _attack_notes(self, v: View, room) -> dict:
        reach = self._reach(v, room)
        ready = self._sword_ready()
        wait = (COOLDOWN - (self.frames_played - self.last_strike)) / 30

        def hit(side):
            r = reach[side]
            return f"would hit {r[0]}" if r else "would hit nothing"
        notes = {}
        notes["_general"] = ("The sword is ready." if ready else
                             f"The sword is not ready for another {wait:.1f} s, so a strike now does nothing.")
        notes["none"] = ""
        other = "left" if v.facing == "right" else "right"
        notes["forward"] = (f"{hit(v.facing)} to the {v.facing}; with the {other} key held it turns and {hit(other)} to the {other}.")
        notes["up"] = hit("up") + "."
        notes["down"] = ("you stand on the ground, so it comes out as a forward strike." if self.grounded else hit("down") + ".")
        soul, hp, hp_max = v.soul or 0.0, v.hp or 0, v.hp_max or 0
        if soul < FOCUS_SOUL:
            notes["focus"] = f"does nothing, SOUL is {soul:.2f} and one mask needs {FOCUS_SOUL:.2f}."
        elif hp >= hp_max:
            notes["focus"] = "heals nothing, health is full."
        elif not self.grounded:
            notes["focus"] = "does nothing in the air."
        else:
            left = max(0.0, 1.2 - self.focus_run * self.n / 30)
            notes["focus"] = f"heals one mask after {left:.1f} more s of holding it without moving."
        if not ready:
            for a in ("forward", "up", "down"):
                notes[a] = "does nothing now, the sword is not ready."
        return notes

    def _status(self, v: View, room) -> str:
        hp = f"Health {v.hp} of {v.hp_max} masks." if v.hp is not None else ""
        soul = f" SOUL {v.soul:.2f}." if v.soul is not None else ""
        if self.grounded:
            pose = " You stand on the ground"
        else:
            pose = f" You are in the air, {'rising' if self.vy > 0 else 'falling'}"
        return f"Now: {hp}{soul}{pose}" + (f", facing {v.facing}." if self.knobs["facing"] == "on" else ".")

    def _screen(self, v: View, room, world, surfs) -> str:
        out = []
        if self.grounded:
            sides = []
            for name, d in (("left", -1), ("right", 1)):
                dist, what, size = look.side_probe(world, v.x, v.y, d)
                if what == "wall":
                    sides.append(f"{name}: a wall {q(dist):.1f} away, {q(size):.1f} high" if size < 12 else f"{name}: a high wall {q(dist):.1f} away")
                elif what == "edge":
                    sides.append(f"{name}: the ground ends {q(dist):.1f} away" + (f", drop {q(size):.1f}" if size else ", no floor seen below"))
                elif what == "unseen":
                    sides.append(f"{name}: ground to the edge of what you have seen ({q(dist):.1f})")
                else:
                    sides.append(f"{name}: open ground for {dist:.0f}+")
            out.append("Ground: " + "; ".join(sides) + ".")
        # ceiling
        head = 0.0
        while head < 9 and not world.blocked(v.x - 0.23, v.x + 0.23, v.y + BODY_H + head, v.y + BODY_H + head + 0.5):
            head += 0.5
        if head < 9:
            out.append(f"Ceiling {q(head):.1f} above your head.")
        here = look.surface_at(surfs, v.x, v.y) if self.grounded else None
        cand = []
        for s in surfs:
            if s is here or s.w < 0.5 or (here is not None and abs(s.y - here.y) < 0.3 and s.x0 <= here.x1 + 0.6 and s.x1 >= here.x0 - 0.6):
                continue
            dy = s.y - v.y
            if dy > 7.3:
                continue
            nx = min(max(v.x, s.x0), s.x1)
            cand.append((abs(nx - v.x) + abs(dy), s, nx))
        cand.sort(key=lambda c: c[0])
        ledges = []
        for _, s, nx in cand[: self.knobs["ledges"]]:
            near, far = (s.x0 - v.x, s.x1 - v.x)
            span = (f"{q(abs(near)):.1f} to {q(abs(far)):.1f} right" if near >= 0 else f"{q(abs(far)):.1f} to {q(abs(near)):.1f} left" if far <= 0
                    else f"{q(abs(near)):.1f} left to {q(far):.1f} right")
            dy = s.y - v.y
            level = "same height" if abs(dy) < text.small() else f"{q(abs(dy)):.1f} {'up' if dy > 0 else 'down'}"
            mid = min(max(v.x, s.x0 + 0.3), s.x1 - 0.3)
            ledges.append(f"{span}, {level} ({'new' if room.visits(mid, s.y) == 0 else 'visited'})")
        if ledges:
            out.append("Ledges near you: " + "; ".join(ledges) + ".")
        for e in sorted(v.enemies, key=lambda e: abs(e.cx - v.x) + abs(e.cy - v.y))[:4]:
            sx, sy = self.speeds.get(e.oid, (0.0, 0.0))
            mv = "not moving" if abs(sx) < 0.3 and abs(sy) < 0.3 else "moving " + text.rel(sx, sy).replace(",", " and") + " per second"
            level = ", at your height" if abs(e.y0 - v.y) < text.small() and abs(e.cx - v.x) >= text.small() else ""
            out.append(f"Enemy {text.rel(e.cx - v.x, e.y0 - v.y, 'on you')}{level} ({q(e.w):.1f} wide, {q(e.h):.1f} tall), {mv}.")
        if not v.enemies:
            out.append("No enemy on screen.")
        hz = sorted(v.hazards, key=lambda h: abs(min(max(v.x, h.x0), h.x1) - v.x) + abs(h.cy - v.y))[:3]
        for h in hz:
            near, far = h.x0 - v.x, h.x1 - v.x
            span = (f"{q(near):.1f} to {q(far):.1f} right" if near >= 0 else f"{q(-far):.1f} to {q(-near):.1f} left" if far <= 0 else f"{q(-near):.1f} left to {q(far):.1f} right")
            out.append(f"{h.kind.capitalize()} {span}, {text.rel(0, h.y1 - v.y, 'at your feet height')}.")
        for d in sorted(v.doors, key=lambda d: abs(d.cx - v.x))[:2]:
            out.append(f"A door {text.rel(d.cx - v.x, d.y0 - v.y)}.")
        return "On screen: " + " ".join(out)

    def _memory(self, v: View, room) -> str:
        out = []
        if self.grounded:
            n = room.visits(v.x, v.y)
            out.append(f"You have stood at this spot {n} times before." if n > 1 else "This spot is new to you." if n == 0 else "You have stood at this spot once before.")
        out.append(f"Places stood on in this room: {len(room.places)}." + (f" Rooms seen: {len(self.mem.rooms)}." if len(self.mem.rooms) > 1 else ""))
        k = self.knobs["recent"]
        if self.mem.recent and k:
            out.append("Last decisions: " + " ".join(f"[{r['move']} + {r['attack']}: {r['result']}]" for r in self.mem.recent[-k:]))
        if self.mem.damage:
            d = self.mem.damage[-1]
            ago = self.mem.decisions - d["decision"]
            out.append(f"Last damage: lost {d['lost']} mask to {d['what']} {ago} decisions ago." if ago else f"Just now you lost {d['lost']} mask to {d['what']}.")
        if self.mem.deaths:
            out.append(f"Deaths so far: {self.mem.deaths}.")
        return "Memory: " + " ".join(out)

    # ---- the window ----------------------------------------------------------------------------------
    def act(self, move: str, attack: str, index: int, label: str = "") -> dict:
        """Play the window of the two chosen options. Returns the keys sent, the forecast of the chosen move and
        what really happened."""
        if move not in MOVE or attack not in ATTACKS:
            raise ValueError("not an offered option")
        before = self.view
        fc = self.forecasts.get(move)
        was_grounded, soul0, hp0 = self.grounded, before.soul or 0.0, before.hp
        enemies0 = {e.oid: e for e in before.enemies}
        near0 = [b for b in before.doors + before.walls        # solid boxes within a strike's reach before the window
                 if before.me is not None and b.overlaps(before.me.x0 - 2.6, before.me.x1 + 2.6, before.me.y0 - 2.2, before.me.y1 + 2.2)]
        if self.grounded:
            key = ((before.room, self.mem.room(before.room).place(before.x, before.y)), move)
            self.mem.tried[key] = self.mem.tried.get(key, 0) + 1
        windows, jump_down = bridge_windows(move, attack, self.n, self.jump_down, self.rising, self.knobs["tail"])
        speeds0 = dict(self.speeds)
        auto0 = self.env.auto_frames
        views = []
        for w in windows:
            obs = self.env.step_keys(w["frames"], w["hold"], w["press"], w["first"], label=label,
                                     meta={"decision": index, "move": move, "attack": attack})
            views.append(View(obs))
        auto = self.env.auto_frames - auto0
        struck = attack in ("forward", "up", "down") and self._sword_ready()
        if attack in ("forward", "up", "down"):
            self.stats["strikes" if struck else "not_ready"] += 1
        if struck:
            self.last_strike = self.frames_played
        self.frames_played += self.n
        self.focus_run = self.focus_run + 1 if attack == "focus" else 0
        self.jump_down = jump_down
        now = views[-1]
        prev = views[-2] if len(views) > 1 and not auto and views[-2].room == now.room else None
        if auto or now.room != before.room:
            self.jump_down = False
        calm = (not auto) and (before.hp is None or now.hp is None or now.hp >= before.hp)
        self._observe(prev, now, calm=calm)
        if self.footing is not None:
            self.still_decisions += 1
        self.mem.decisions += 1
        self.mem.stand(now, self.grounded, before, was_grounded)
        # ---- what happened (for the memory lines and the forecast check) ----
        lost = (hp0 or 0) - (now.hp or 0) if hp0 is not None and now.hp is not None else 0
        events = []
        if now.room != before.room:
            events.append("entered another room")
        if lost > 0:
            what = (fc or {}).get("hit") or ((fc or {}).get("then") or {}).get("hit")
            if not what:
                me = before.me
                near = [(abs(t.cx - before.x) + abs(t.cy - before.y), t.kind) for t in list(before.enemies) + list(before.hazards)]
                what = min(near)[1] if near else "something"
            what = {"enemy": "an enemy", "hazard": "a hazard"}.get(what, what)
            self.mem.damage.append({"decision": self.mem.decisions, "what": what, "lost": lost})
            events.append(f"lost {lost} mask to {what}")
            if (now.hp or 0) <= 0:
                self.mem.deaths += 1
                events.append("died")
        gone = [oid for oid in enemies0 if oid not in {e.oid for e in now.enemies} and now.on_screen(enemies0[oid].cx, enemies0[oid].cy)]
        # SOUL rises only when the sword hits an enemy. v9 also required its own count to say that the strike went
        # out; that count is wrong after a hurt (the game refuses a strike during the recoil), and a hit was then
        # missed (laya12_frozen_a, decision 106: the strike that killed the crawler)
        pressed = struck if self.knobs["kills"] == "v9" else attack in ("forward", "up", "down")
        if pressed and (now.soul or 0.0) > soul0 + 0.01:
            events.append("the strike hit an enemy")
            self.last_hit = self.frames_played
            self.stats["hits"] += 1
        if self.knobs["kills"] == "strict":
            # an enemy that is gone died only if a strike of ours hit an enemy in this decision or the one before,
            # and its box, where it was and where its speed would have taken it, lies well inside the screen (an
            # enemy at the border may simply have walked off; v9 counted those)
            cam, span = now.cam, self.n / 30

            def inside(oid):
                b = enemies0[oid]
                sx, sy = speeds0.get(oid, (0.0, 0.0))
                for dt in (0.0, span):
                    if not (cam.x0 + EDGE <= b.x0 + sx * dt and b.x1 + sx * dt <= cam.x1 - EDGE
                            and cam.y0 + EDGE <= b.y0 + sy * dt and b.y1 + sy * dt <= cam.y1 - EDGE):
                        return False
                return True
            gone = [oid for oid in gone if inside(oid)]
            died = bool(gone) and (self.frames_played - self.last_hit) <= 2 * self.n and now.room == before.room
        else:
            died = bool(gone) and (self.frames_played - self.last_strike) <= 3 * self.n and now.room == before.room
        if died:
            self.mem.kills += len(gone)
            events.append("an enemy died")
        still = {b.oid for b in now.doors + now.walls}
        if struck and now.room == before.room and any(b.oid not in still and now.on_screen(b.cx, b.cy) for b in near0):
            events.append("the strike broke something")
        if attack in ("forward", "up", "down") and not struck:
            events.append("the sword was not ready")
        dx, dy = now.x - before.x, now.y - before.y
        if now.room == before.room:
            if self.grounded and not was_grounded:
                events.insert(0, f"landed {text.rel(dx, dy)}")
            elif self.grounded:
                events.insert(0, f"walked {text.rel(dx, 0, 'nowhere')}" if abs(dx) >= 0.15 else "did not move")
            else:
                events.insert(0, f"in the air, {text.rel(dx, dy, 'same spot')}")
        result = ", ".join(events)
        self.mem.recent.append({"move": move, "attack": attack, "result": result})
        del self.mem.recent[:-8]
        check = None
        if fc is not None:
            ex, ey = now.x - fc["x"], now.y - fc["y"]
            check = {"forecast": [fc["x"], fc["y"], fc["grounded"]], "actual": [round(now.x, 3), round(now.y, 3), self.grounded],
                     "error": [round(ex, 3), round(ey, 3)], "agree": abs(ex) <= AGREE and abs(ey) <= AGREE,
                     "hit_forecast": fc.get("hit"), "lost": lost, "adapter_frames": auto, "room_changed": now.room != before.room,
                     "struck": struck}
            self.check_log.append(check)
            if not check["room_changed"] and not auto:
                self.stats["hurt_forecast"][(0 if fc.get("hit") else 2) + (0 if lost > 0 else 1)] += 1
        return {"keys": windows, "result": result, "check": check, "struck": struck, "lost": lost,
                "adapter_frames": auto, "windows": len(windows)}

    def run_log(self) -> dict:
        done = [c for c in self.check_log if not c["room_changed"] and not c["adapter_frames"]]
        return {"decisions": self.mem.decisions, "kills_seen": self.mem.kills, "deaths_seen": self.mem.deaths,
                "damage_events": len(self.mem.damage), "rooms": len(self.mem.rooms),
                "places": sum(len(r.places) for r in self.mem.rooms.values()),
                "forecast_checked": len(done), "forecast_agree": sum(1 for c in done if c["agree"]),
                # v10: strikes that went out / chosen while the sword was not ready / that hit an enemy; the chosen
                # move's own-window danger forecast against what happened: [said hurt & hurt, said hurt & not,
                # said safe & hurt, said safe & not]; option texts shortened to fit the model's limit
                "strikes_out": self.stats["strikes"], "strikes_not_ready": self.stats["not_ready"],
                "strike_hits": self.stats["hits"], "hurt_forecast": self.stats["hurt_forecast"], "text_cuts": self.text_cuts,
                # v11: frontier notes written / left out for the token limit; no-key windows while the knight was absent
                "frontier_notes": self.frontier_stats["notes"], "frontier_cut": self.frontier_stats["cut"],
                "absent_windows": self.absent_windows,
                # v12: decisions that ended standing where the grid shows no ground
                "still_decisions": self.still_decisions}
