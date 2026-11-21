# fmt: off
# isort: skip_file
# ruff: noqa
"""Facts -> short English for the decision model, and the state-dependent option menu.

The decision model never sees the grid. It gets (1) a fixed short briefing, (2) a few sentences that
the rule-based tools computed from the current screen and from the terrain memory, (3) the options
that make sense right now, each with a one-line description that says what the behaviour would do in
this very situation. Distances are in "steps" (1 step = 1 world unit = 54 px, a bit less than the
knight's height).

Loop guard (user's design: steer with language first): when a behaviour was chosen in this same state
and changed nothing, the harness says so in language: the situation text gets "The last choice changed
nothing. Still open here: <other options>.", and the futile option moves to the end of the menu with
the description "Tried Nx here, nothing happened.". Offline tests (offline/try02..04) showed that the
frozen model does not react to a failure statement as such — naming the failed option in the situation
text even raises its score — but it does follow a pointer to the alternatives and the menu order.
Only if the model still repeats the option after NOTE_LIMIT such notes is it withdrawn for that
state. Which mechanism acted is logged with every decision (`loop_guard`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import physics as ph
from . import skills
from .control import Ctl
from .perception import FEET_Y, HALF_W, KX, KY

PROMPT = ("You guide a knight through a 2D side-scrolling platformer level. Reach the exit of the level alive. "
          "Levels like this lead to the right. Spikes kill at once. Enemies hurt on touch; three sword strikes kill one. "
          "You choose a behaviour; reflexes carry it out and stop when something changes.")
INSTRUCTIONS = "Choose the behaviour that makes progress toward the exit while keeping the knight alive."
NOTE_LIMIT = 2          # language notes given before an option that keeps doing nothing is withdrawn


@dataclass
class Memory:
    history: list = field(default_factory=list)
    blocked: dict = field(default_factory=dict)     # surface id -> why going right ended there (dead ends seen)
    recheck: set = field(default_factory=set)       # dead ends seen before a lever was thrown: may be open now
    peeked: set = field(default_factory=set)        # v8: edges looked below from which nothing to land on was found
    kills: int = 0
    levers: int = 0

    def new_area(self) -> None:
        """v8: a new scene was entered through the exit: nothing remembered about dead ends applies any more."""
        self.blocked.clear()
        self.recheck.clear()
        self.peeked.clear()
        self.history.append({"choice": "", "reason": "entered a new area", "moved": [0, 0], "windows": 0, "hp0": None,
                             "hp1": None, "futile": False, "sig": None, "died": True, "token": None})

    def note(self, choice: str, reason: str, surface: str | None, moved_right: float, at_x: float | None = None) -> None:
        """Exploration memory, updated after every behaviour (generic: dead end -> go back and look for
        what was missed; a lever thrown -> the old dead ends are worth another look)."""
        if reason.startswith("died"):
            self.blocked.clear()
            self.recheck.clear()
            self.peeked.clear()
            return
        if choice.startswith("drop_and_look") and surface is not None and reason.startswith("looked below the edge"):
            self.peeked.add(surface)            # v8: looked, and nothing to land on was within reach
        if reason.startswith("the enemy is gone"):
            self.kills += 1
        if choice == "hit_switch" and "activated" in reason:
            self.levers += 1
            self.recheck |= set(self.blocked)
            self.blocked.clear()
            return
        if choice == "advance_right" and surface is not None:
            if reason.startswith("blocked by") or reason.startswith("something I cannot see"):
                # v8: with the place where it ended (one long floor can be blocked at one spot and open further on)
                self.blocked[surface] = (reason, at_x)
            elif moved_right > 1.0:
                self.blocked.pop(surface, None)
                self.recheck.discard(surface)

    def record(self, choice, result, moved, windows, hp0, hp1, sig, token=None):
        dx, dy = moved
        still = abs(dx) < 0.3 and abs(dy) < 0.3
        futile = still and hp0 == hp1 and not result.reason.startswith(("the enemy is gone", "landed", "climbed", "dropped"))
        self.history.append({"choice": choice, "reason": result.reason, "moved": [round(dx, 2), round(dy, 2)],
                             "windows": windows, "hp0": hp0, "hp1": hp1, "futile": futile, "sig": sig,
                             "died": result.reason.startswith("died"), "token": token})

    def futile_count(self, option: str, sig, token=None) -> tuple[int, str]:
        """How often `option` was already chosen in this state without anything coming of it.

        Counts (a) attempts that changed nothing on the spot, and (b) repeats: the same option chosen in
        the same state while nothing new was learned or achieved in between (same progress token) — that
        is how cycles like "walk to the edge / climb the fruitless wall / walk to the edge" show up.
        """
        n, reason = 0, ""
        for h in reversed(self.history):
            if h.get("died"):
                break
            if h["choice"] != option:
                continue
            if h["sig"] != sig:
                # v8: also when other details of the situation flicker (a projectile in the air, an enemy
                # in or out of sight): a behaviour that did nothing at all on this very spot counts
                if (h["futile"] and h["windows"] == 0 and h["sig"] is not None and sig is not None
                        and h["sig"][:2] == sig[:2]):
                    n += 1
                    reason = reason or h["reason"]
                continue
            if h["futile"] or (token is not None and h.get("token") == token):
                n += 1
                reason = reason or h["reason"]
        return n, reason


def progress_token(ctl: Ctl, memory) -> list:
    """Changes whenever something new was learned or achieved (used to tell a cycle from progress)."""
    m = ctl.map
    return [len(m.visited), len(m.inert), len(m.no_hold), len(m.fruitless), len(m.climbed), len(memory.blocked),
            len(memory.recheck), ctl.tr.deaths, ctl.tr.hp, memory.kills, memory.levers]


def signature(ctl: Ctl, facts: dict) -> list:
    x, y = ctl.pos
    place = round(x * 2) / 2
    supw = ctl.support_surf() if ctl.tr.standing else None
    if supw is not None and supw.moving:
        # v8: riding a moving platform, "the same place" is the place on the platform (the world position
        # changes all the time and would hide every repeat from the loop guard, laya_02_v8_L2 decisions 9-20)
        place = f"{supw.oid}:{round((x - supw.x0) * 2) / 2}"
    return [place, round(y * 2) / 2, ctl.tr.hp, len(facts["enemies"]),
            bool(facts["right"]["enemy_at_landing"]), facts["right"]["stop"], facts["right"]["next"] is not None]


def steps(units: float) -> str:
    n = int(round(units))
    if n <= 0:
        return "less than 1 step"
    return f"{n} step" + ("" if n == 1 else "s")


def side_survey(ctl: Ctl, dirn: int) -> dict:
    """What walking / jumping in one direction would meet (current screen + terrain memory)."""
    sc = ctl.scene
    sup = sc.support()
    out = {"walk": 0.0, "stop": "unknown footing", "next": None, "enemy_at_landing": False, "why_no_jump": "", "watching": False}
    if sup is None:
        return out
    limit, why = skills.walk_limit(sc, sup, dirn)
    out["walk"] = max(0.0, min(limit, 960.0) / sc.ppu)
    out["stop"] = why
    if why == "mover":
        return out                 # v8: a rolling hazard ahead; what is beyond it is a matter for after it
    if why != "clear":
        cands = skills.jump_candidates(ctl, "ahead", dirn)[:3]
        choice, report = skills.plan_candidates(ctl, cands)
        if choice is not None:
            target, spot, plan = choice
            x, y = sc.knight_world
            surf = ctl.map.solids[target.oid]
            out["next"] = {"oid": target.oid, "dx": abs((target.x0 if dirn > 0 else target.x1) - x), "rise": target.top - y,
                           "double": plan.t2 is not None, "width": surf.x1 - surf.x0,
                           "open_ended": surf.open_right if dirn > 0 else surf.open_left,
                           "remembered": skills.is_remembered(ctl, target),
                           "moving": surf.moving, "wait": plan.wait}
        else:
            out["enemy_at_landing"] = any(r == "enemy at the landing spot" for _, r in report)
            out["why_no_jump"] = "; ".join(sorted({r for _, r in report}))
            out["watching"] = any(r == ph.UNSURE for _, r in report)     # v8: a moving platform not understood yet
            out["unseen"] = any(r == ph.UNSEEN for _, r in report)       # v8: ... and one that cannot be seen from here
    return out


def vertical_survey(ctl: Ctl, mode: str, exclude: set | None = None) -> dict | None:
    cands = [c for c in skills.jump_candidates(ctl, mode) if not exclude or c.oid not in exclude][:3]
    choice, report = skills.plan_candidates(ctl, cands)
    if choice is None:
        return None
    target, spot, plan = choice
    x, y = ctl.pos
    return {"oid": target.oid, "rise": target.top - y, "dx": (target.x0 + target.x1) / 2 - x, "double": plan.t2 is not None,
            "remembered": skills.is_remembered(ctl, target),
            "saw_on_it": skills.sweep_width(ctl, target.oid) >= 1.5}      # v8: a saw rolls along that ledge


def survey(ctl: Ctl) -> dict:
    sc = ctl.scene
    sup = sc.support()
    facts = {"hp": ctl.tr.hp, "standing": ctl.tr.standing, "right": side_survey(ctl, +1), "left": side_survey(ctl, -1)}
    taken = {s["next"]["oid"] for s in (facts["right"], facts["left"]) if s["next"] is not None}
    facts["up"] = vertical_survey(ctl, "up", taken)
    facts["down"] = vertical_survey(ctl, "down", taken)
    if sup is not None:
        facts["footing"] = "floor" if (sup.w > 6 * sc.ppu or sup.clipped_left or sup.clipped_right) else "small platform"
    else:
        facts["footing"] = "unknown"
    facts["spikes_nearby"] = bool(sc.any(KX - 200, FEET_Y - 10, KX + 200, 540, 8))
    enemies = []
    for e in sc.enemies:
        enemies.append({"side": "right" if e.cx >= KX else "left", "gap": max(0.0, skills.body_gap(sc, e)),
                        "same_level": skills.same_level(sc, e), "dy": (KY - e.cy) / sc.ppu})
    enemies.sort(key=lambda r: r["gap"])
    facts["enemies"] = enemies
    facts["interactables"] = len(sc.interactables)
    facts["remembered_surfaces"] = len(ctl.map.solids)
    walls = {}
    for opt in skills.wall_options(ctl):
        key = "left" if opt["side"] < 0 else "right"
        if key not in walls:
            walls[key] = {"dist": opt["dist"], "rise": opt["rise"], "goal_visited": opt["goal_visited"],
                          "over_danger": opt.get("over_danger", False),
                          "goal": opt["goal"].oid if opt["goal"] is not None else None, "climbed": opt["climbed"],
                          "open_top": opt["wall"].open_top, "contact": opt["plan"].y - ctl.pos[1]}
    facts["walls"] = walls
    facts["surface"] = sup.oid if sup is not None else None
    it = skills.interactable_here(ctl)
    facts["switch"] = None if it is None else {"side": "right" if it.cx >= KX else "left",
                                               "dist": max(0.0, (abs(it.cx - KX) - it.w / 2) / sc.ppu)}
    for key in ("up", "down"):
        if facts[key] is not None:
            facts[key]["visited"] = facts[key]["oid"] in ctl.map.visited
    # v8: moving things and wide spike fields
    x, y = ctl.pos
    facts["mover"] = None
    mv = skills.blocking_mover(ctl, +1, reach=9.5) if ctl.tr.standing else None
    if mv is not None:
        cx = mv.predict(sc.time)[0]
        facts["mover"] = {"oid": mv.oid, "dist": max(0.0, cx - x - (mv.w or 1.0) / 2 - ph.BODY_HW),
                          "coming": mv.velocity(sc.time) < 0, "turn_known": mv.lo is not None}
    facts["mover_left"] = None
    ml = skills.blocking_mover(ctl, -1, reach=9.5) if ctl.tr.standing else None
    if ml is not None:
        facts["mover_left"] = {"oid": ml.oid, "dist": max(0.0, x - ml.predict(sc.time)[0] - (ml.w or 1.0) / 2 - ph.BODY_HW)}
    facts["peek_right"] = skills.can_peek(ctl, +1)
    fld = skills.hazard_field(ctl, +1)
    facts["field"] = None if fld is None else {"dist": fld["dist"], "width": fld["x1"] - fld["x0"],
                                               "past_view": fld["x1"] >= x + 8.3}
    supw = ctl.support_surf()
    facts["riding"] = bool(supw is not None and supw.moving)
    facts["shots"] = sum(1 for tr in ctl.motion.hazards() if tr.flew and tr.seen >= sc.time - 0.1)
    facts["shot_at"] = bool(facts["shots"]) or (sc.time - ctl.last_flyer) < 5.0
    return facts


def side_sentence(side: str, s: dict) -> str:
    if s["stop"] == "clear":
        return f"To the {side} the floor continues as far as you can see."
    if s["stop"] == "mover":
        return f"To the {side} a saw blade rolls back and forth across the floor; its path starts {steps(s['walk'])} from you."
    what = {"edge": "the floor ends", "wall": "a wall blocks the way", "step": "a ledge blocks the way",
            "hazard": "spikes block the way", "unknown footing": "the ground is unclear"}[s["stop"]]
    text = f"To the {side} {what} after {steps(s['walk'])}."
    nxt = s["next"]
    if nxt is not None:
        height = ("at the same height" if abs(nxt["rise"]) < 0.5 else
                  f"{steps(abs(nxt['rise']))} {'higher' if nxt['rise'] > 0 else 'lower'}")
        kind = "more floor" if (nxt["open_ended"] or nxt["width"] > 6) else "a small platform"
        if nxt.get("moving"):
            kind = "a platform that moves back and forth"
        seen = " (remembered from before, not on screen now)" if nxt["remembered"] else ""
        text += f" Beyond it {kind} is within {'double-jump' if nxt['double'] else 'jump'} reach, {height}{seen}."
    elif s.get("watching"):
        text += " Beyond it a platform moves back and forth; where it turns around has not been seen yet."
    elif s["enemy_at_landing"]:
        text += (" The next platform is within jump reach, but an enemy stands right where you would land."
                 " It stays there only while it sees you: keep out of its sight until it leaves the landing spot.")
    elif s.get("unseen"):
        text += " A moving platform seen earlier lies beyond it, lower down and out of sight from here."
    elif s["stop"] in ("edge", "hazard"):
        text += " Nothing safe is known within jump reach beyond it."
    return text


def state_text(facts: dict, memory: Memory, open_names: list | None = None) -> str:
    parts = [f"Health {facts['hp']} of 5."]
    foot = {"floor": "You stand on a stone floor.", "small platform": "You stand on a small platform.",
            "unknown": "You are not on firm ground."}[facts["footing"]]
    if facts.get("riding"):
        foot = "You stand on a platform that moves back and forth."
    if facts["spikes_nearby"]:
        foot += " There are spikes below nearby."
    parts.append(foot)
    parts.append(side_sentence("right", facts["right"]))
    if facts.get("field") is not None and facts["right"]["next"] is None and not facts["right"].get("watching"):
        f = facts["field"]
        wide = "wider than you can see" if f["past_view"] else f"{steps(f['width'])} wide"
        parts.append(f"The spikes ahead cover the floor, {wide}: too wide to jump, but one can bounce across on them with downward sword strikes.")
    parts.append(side_sentence("left", facts["left"]))
    if facts["up"] is not None:
        seen = ", remembered" if facts["up"]["remembered"] else ""
        saw = "; a saw blade rolls along it" if facts["up"].get("saw_on_it") else ""
        parts.append(f"Another ledge above is within jump reach ({steps(facts['up']['rise'])} up{seen}{saw}).")
    if facts["down"] is not None:
        seen = ", remembered from before, not on screen now" if facts["down"]["remembered"] else ""
        side = "right" if facts["down"]["dx"] >= 0 else "left"
        parts.append(f"A lower platform is within reach below to the {side} ({steps(-facts['down']['rise'])} down{seen}).")
    for side, w in facts.get("walls", {}).items():
        if w.get("over_danger") and w["goal"] is None:
            continue                # v8: something to cling to that hangs over spikes and leads to no known ledge
        if w["goal"] is not None:
            new_note = "" if w["goal_visited"] else ", not visited yet"
            height = f"{steps(w['rise'])} up" if w["rise"] is not None else "above"
            parts.append(f"A wall on the {side} can be climbed with wall jumps to a ledge {height}{new_note}.")
        else:
            parts.append(f"A wall on the {side} can be climbed with wall jumps; what is above is not known.")
    if facts.get("switch") is not None:
        sw = facts["switch"]
        parts.append(f"A lever stands {steps(sw['dist'])} to your {sw['side']} on your platform, not struck yet.")
    level = [e for e in facts["enemies"] if e["same_level"]]
    other = [e for e in facts["enemies"] if not e["same_level"]]
    if level:
        e = level[0]
        parts.append(f"An enemy is {steps(e['gap'])} to your {e['side']} on your platform.")
    elif other:
        e = other[0]
        if facts.get("shot_at"):
            parts.append(f"An enemy up to the {e['side']} shoots at you; shots that would hit are struck away while you stand on firm ground.")
        else:
            parts.append(f"An enemy is in sight to the {e['side']} on another platform; it cannot reach you here.")
    else:
        parts.append("No enemy in sight.")
    if facts.get("dead_end"):
        parts.append("The way right from this floor ended in a dead end before: something was missed further back.")
    if facts.get("recheck"):
        parts.append("A lever was thrown since the way was blocked: the blocked place may be open now.")
    if open_names is not None:
        parts.append(f"The last choice changed nothing. Still open here: {', '.join(open_names)}.")
    elif memory.history:
        parts.append(f"Last result: {memory.history[-1]['reason']}.")
    return " ".join(parts)


OPTION_CHAR_BUDGET = 150   # the Laya service rejects an option longer than 48 tokens (measured: ~4.1 chars/token)


def fit(base: str, tail: str = "", note: str = "") -> str:
    """Option text within the budget. The loop-guard note has priority over the situational tail."""
    for text in (f"{base} {tail} {note}", f"{base} {note}", f"{base} {tail}", base):
        text = " ".join(text.split())
        if len(text) <= OPTION_CHAR_BUDGET:
            return text
    return base[:OPTION_CHAR_BUDGET]


# Menu order = the harness's own rule-based ranking (generic platformer experience):
#   danger on my platform first, then a lever within reach, then ledges not visited yet (cheap to check
#   now, costly to come back to), then going right, then climbing into the unknown, then backing off.
# Offline tests (offline/try05) show that the frozen model leans strongly toward the first option, so
# this order acts as the harness's recommendation; the model's argmax is still what gets executed.
RANK = {"fight_nearest_enemy": 10, "jump_over_enemy": 12, "hit_switch": 20, "explore_ledge_left": 25,
        "explore_ledge_right": 25, "explore_ledge_up": 26, "outwait_enemy": 30, "leap_past_enemy": 31, "advance_right": 40,
        "pass_hazard_right": 38, "cross_spikes_right": 41, "drop_and_look_right": 42, "pass_hazard_left": 49,   # v8
        "climb_wall_left": 46, "climb_wall_right": 47, "go_back_left": 48, "climb_up": 55, "drop_down": 60,
        "retreat_left": 70, "wait": 80}


def build_menu(ctl: Ctl, memory: Memory) -> dict:
    facts = survey(ctl)
    # v8: a floor that ends at a rolling saw or at a field of spikes that can be bounced across is not a dead end
    #     nor is an edge with nothing known beyond it that has not been looked below yet
    peek_open = bool(facts.get("peek_right")) and facts.get("surface") not in memory.peeked and (
        facts["right"]["stop"] == "edge" and facts["right"]["next"] is None and not facts["right"]["enemy_at_landing"])
    way_on = (facts["mover"] is not None or facts["right"].get("watching") or peek_open
              or (facts["field"] is not None and facts["right"]["next"] is None))
    blk = memory.blocked.get(facts.get("surface"))
    if blk is not None and blk[1] is not None and ctl.pos[0] > blk[1] + 1.5:
        # v8: the knight stands beyond the spot where this floor was blocked (it got past it): no dead end here
        memory.blocked.pop(facts.get("surface"), None)
        blk = None
    facts["dead_end"] = blk is not None and not way_on
    facts["recheck"] = bool(memory.recheck)
    sig = signature(ctl, facts)
    right, left = facts["right"], facts["left"]
    level = [e for e in facts["enemies"] if e["same_level"]]
    enemy_right = next((e for e in level if e["side"] == "right" and e["gap"] <= 4.5), None)
    enemy_left = next((e for e in level if e["side"] == "left" and e["gap"] <= 4.5), None)
    options: dict[str, tuple[str, str]] = {}          # name -> (base, situational tail)
    withheld: dict[str, str] = {}                     # behaviour that could act here but is kept off the menu -> rule
    rank = dict(RANK)
    can_go_right = right["stop"] == "clear" or right["walk"] >= 0.3 or right["next"] is not None or right.get("watching")
    # Human experience as menu rules (the model kept choosing advance_right in these two situations and
    # wording did not change that, offline/try07..09, run_07):
    #   * nobody walks away from a lever that has not been tried: while one is within reach on the
    #     platform (and no enemy is on it), leaving is not on the menu;
    #   * when an enemy guards the landing of the next platform, "go right" cannot get across: the ways
    #     to deal with the guard are on the menu instead.
    lever_first = facts.get("switch") is not None and not level
    # v8: two more of the same kind (generic, from what is on screen):
    #   * a saw blade rolling across the way: walking on only stops in front of it; the behaviour that waits
    #     for it and hops over it is offered instead (while the blade's path is more than a step away,
    #     walking up to it is still offered);
    #   * spikes on the floor too wide to jump with nothing to land on in reach: bouncing across is offered.
    mover = facts["mover"]
    mover_close = mover is not None and right["stop"] == "mover" and right["walk"] < 1.0
    if mover is not None and enemy_right is None and not lever_first:
        options["pass_hazard_right"] = ("Get past the rolling saw blade: wait out of its way, hop over it when it comes, go on right.", "")
    field_here = (facts["field"] is not None and right["next"] is None and not right.get("watching")
                  and enemy_right is None and not lever_first)
    if field_here:
        options["cross_spikes_right"] = ("Cross the wide spikes to the right: jump over their edge and bounce on them with downward sword strikes.", "")
        if right["walk"] < 1.0:
            rank["cross_spikes_right"] = 39
    #   * an edge with nothing known beyond it (the screen shows only 2 steps below the feet): stepping off to
    #     look, with the jumps kept to come back, is offered;
    #   * a saw blade between the knight and the way back, at a dead end: getting past it leftwards is offered.
    if peek_open and enemy_right is None and not lever_first:
        # wording chosen offline against the model (experiment offline/try01 of round 2): this one is picked
        # first in both test situations; "Step off the right edge to see what is below ..." lost to retreat_left
        options["drop_and_look_right"] = ("Go on right: drop off the edge onto what lies below; the two jumps bring you back if nothing is there.", "")
    elif facts.get("peek_right") and facts.get("surface") in memory.peeked:
        withheld["drop_and_look_right"] = "looked_already: nothing to land on was found below this edge"
    if facts.get("mover_left") is not None and enemy_left is None and not lever_first and (
            right["next"] is None and right["stop"] != "clear" and right["walk"] < 1.0):
        options["pass_hazard_left"] = ("Go back left past the rolling saw blade: wait out of its way, hop over it when it comes.", "")
    # advance_right is offered only when it would do something (walk on, or jump to a next platform)
    if mover_close and enemy_right is None and not lever_first:
        withheld["advance_right"] = "moving_hazard_ahead: it would stop in front of the saw blade; pass_hazard_right is offered"
    elif enemy_right is None and can_go_right and not right["enemy_at_landing"] and not lever_first:
        if right.get("watching") and right["next"] is None:
            tail = "Now: watch the moving platform ahead, then jump onto it."
        elif right["next"] is not None and right["walk"] < 1.0:
            tail = ("Now: a jump onto the moving platform when it is in place." if right["next"].get("moving")
                    else "Now: a jump to the next platform.")
        elif right["stop"] == "clear":
            tail = "Now: the way is open."
        else:
            tail = f"Now: {steps(right['walk'])} to the {'floor end' if right['stop'] == 'edge' else 'obstacle'}."
        if facts["dead_end"] and right["next"] is None:
            tail = "Now: leads to the dead end seen before."
            rank["advance_right"] = 75
        options["advance_right"] = ("Go right toward the exit: walk, jump to the next platform.", tail)
    elif can_go_right or right["enemy_at_landing"]:
        withheld["advance_right"] = ("enemy_ahead: it would stop at once, fight / jump over are offered" if enemy_right is not None
                                     else "guard_at_landing: it cannot get across, outwait / leap are offered" if right["enemy_at_landing"]
                                     else "lever_first: a lever not tried yet is within reach")
    if enemy_right is None and right["enemy_at_landing"]:
        options["outwait_enemy"] = ("Keep out of the enemy's sight until it leaves the landing spot, then jump across.", "")
        options["leap_past_enemy"] = ("Jump over the enemy that guards the landing and land behind it.", "")
    if lever_first:                                        # nobody fiddles with a lever while an enemy is on them
        options["hit_switch"] = (f"Strike the lever on your {facts['switch']['side']} with the sword.", "Levers open the way.")
    elif facts.get("switch") is not None:
        withheld["hit_switch"] = "enemy_on_platform: deal with the enemy first"
    if level:
        e = level[0]
        options["fight_nearest_enemy"] = (f"Fight the enemy on your {e['side']} with the sword until it is dead.", "")
        if e["gap"] <= 4.5:
            options["jump_over_enemy"] = ("Leap over the enemy and land behind it without fighting.", "")
        else:
            rank["fight_nearest_enemy"] = 45              # far away: not urgent
    # going back up to a ledge that was already visited is only on the menu when nothing leads on, or
    # when a thrown lever makes the old dead ends worth another look (fewer pointless detours)
    # (v8: the new ways on - past a saw, across spikes, down over an edge - count as "something leads on" too)
    revisit = not ({"advance_right", "drop_and_look_right", "pass_hazard_right", "cross_spikes_right"} & set(options)) \
        or bool(memory.recheck)
    if facts["up"] is not None:
        if facts["up"].get("saw_on_it") and not facts["up"].get("visited"):
            # v8: a ledge with a saw rolling along it is not a place to go exploring; it stays on the menu,
            # low, only when nothing else leads on
            if revisit:
                options["climb_up"] = (f"Jump up to the higher ledge ({steps(facts['up']['rise'])} up); a saw blade rolls along it.", "")
            else:
                withheld["climb_up"] = "saw_on_ledge: a saw rolls along that ledge and a way on exists"
        elif facts["up"].get("visited"):
            if revisit:
                options["climb_up"] = (f"Jump up to the higher ledge ({steps(facts['up']['rise'])} up), visited before.", "")
                if memory.recheck:
                    rank["climb_up"] = 44     # back up toward the place that may be open now
            else:
                withheld["climb_up"] = "revisit: that ledge was visited and a way on exists"
        else:
            options["explore_ledge_up"] = (f"Explore the ledge not visited yet: jump up, {steps(facts['up']['rise'])} up.", "")
    # wall climbing: toward a known ledge always; into the unknown only when the way right is closed
    for side, w in facts.get("walls", {}).items():
        height = f", {steps(w['rise'])} up" if w["rise"] is not None else ""
        if w["goal"] is not None and not w["goal_visited"]:
            options[f"explore_ledge_{side}"] = (f"Explore the ledge not visited yet: wall-jump up the wall on the {side}{height}.", "")
        elif w["goal"] is not None:
            if revisit:
                options[f"climb_wall_{side}"] = (f"Wall-jump up the wall on the {side} to the ledge above (visited before).", "")
                rank[f"climb_wall_{side}"] = 45 if memory.recheck else 56
            else:
                withheld[f"climb_wall_{side}"] = "revisit: the ledge above was visited and a way on exists"
        elif w.get("over_danger"):
            withheld[f"climb_wall_{side}"] = "over_spikes: it hangs over spikes and no ledge is known above it"
        elif "advance_right" not in options or facts["dead_end"]:
            note = "Climbed before." if w.get("climbed") else ""
            options[f"climb_wall_{side}"] = (f"Wall-jump up the wall on the {side} to see what is above.", note)
            if w.get("climbed"):
                rank[f"climb_wall_{side}"] = 45 if memory.recheck else 57
        else:
            withheld[f"climb_wall_{side}"] = "unknown_wall: climbing into the unknown only when the way right is closed"
    # going down to a place already visited is only offered when nothing leads on
    leads_on = {"advance_right", "outwait_enemy", "leap_past_enemy", "explore_ledge_left", "explore_ledge_right",
                "explore_ledge_up", "climb_up", "climb_wall_left", "climb_wall_right", "hit_switch", "go_back_left"}
    if facts["down"] is not None and (not facts["down"].get("visited") or not (leads_on & set(options))):
        side = "right" if facts["down"]["dx"] >= 0 else "left"
        options["drop_down"] = (f"Jump down to the lower platform on the {side}.", "")
    elif facts["down"] is not None:
        withheld["drop_down"] = "regress: the lower place was visited and something leads on"
    # at a dead end: going back to look for what was missed (not when a thrown lever says "go and look again")
    leads_right = {"advance_right", "drop_and_look_right", "pass_hazard_right", "cross_spikes_right"} & set(options)
    if ((facts["dead_end"] or not leads_right) and not level and not lever_first
            and not right["enemy_at_landing"] and (left["walk"] >= 0.5 or left["next"] is not None)):
        options["go_back_left"] = ("Go back left the way you came and look for what was missed.", "")
        if memory.recheck:
            rank["go_back_left"] = 72
    elif left["walk"] >= 0.5 or left["next"] is not None:
        withheld["go_back_left"] = ("enemy_on_platform" if level else "lever_first" if lever_first
                                    else "guard_at_landing" if right["enemy_at_landing"]
                                    else "no_dead_end: going back is offered only at a dead end")
    if enemy_left is None and left["walk"] >= 0.5:
        options["retreat_left"] = ("Step back left, up to 3 steps, away from the exit.", "")
    elif left["walk"] >= 0.5:
        withheld["retreat_left"] = "enemy_behind: stepping back would walk into it"
    options["wait"] = ("Stand still for one second and watch.", "")
    if len(options) < 3 and "retreat_left" not in options:
        options["retreat_left"] = ("Step back left, up to 3 steps, away from the exit.", "")
    if len(options) < 3 and "advance_right" not in options:
        options["advance_right"] = ("Go right toward the exit: walk, jump to the next platform.", "Now: no way on is known.")
    order = sorted(options, key=lambda name: rank.get(name, 65))
    token = progress_token(ctl, memory)

    # ---- loop guard: language first, withdrawal only after NOTE_LIMIT ignored notes ---------------
    guard = {}
    texts: dict[str, str] = {}
    tail_end: dict[str, str] = {}
    withdrawn = 0
    for name in order:
        base, tail = options[name]
        n, reason = memory.futile_count(name, sig, token)
        if n == 0:
            texts[name] = fit(base, tail)
        elif n > NOTE_LIMIT and len(options) - withdrawn > 3:
            withdrawn += 1
            guard[name] = {"futile": n, "mechanism": "withdrawn", "reason": reason}
            withheld[name] = f"loop_guard: chosen {n}x here and nothing came of it"
        else:
            tail_end[name] = f"Tried {n}x here, nothing came of it."
            guard[name] = {"futile": n, "mechanism": "language", "reason": reason}
    open_names = list(texts) if guard else None
    texts.update(tail_end)                 # futile options go to the end of the menu
    return {"essential": [PROMPT, "Situation: " + state_text(facts, memory, open_names)], "instructions": INSTRUCTIONS,
            "options": texts, "facts": facts, "signature": sig, "loop_guard": guard, "harness_order": list(texts),
            "token": token, "withheld": withheld}


SKILLS = {
    "advance_right": lambda ctl: skills.advance(ctl, +1),
    "retreat_left": lambda ctl: skills.retreat(ctl, -1),
    "wait": lambda ctl: skills.wait(ctl),
    "fight_nearest_enemy": lambda ctl: skills.fight(ctl),
    "jump_over_enemy": lambda ctl: skills.jump_over_enemy(ctl),
    "climb_up": lambda ctl: skills.climb_up(ctl),
    "drop_down": lambda ctl: skills.drop_down(ctl),
    "climb_wall_left": lambda ctl: skills.climb_wall(ctl, -1, "climb_wall_left"),
    "climb_wall_right": lambda ctl: skills.climb_wall(ctl, +1, "climb_wall_right"),
    "explore_ledge_left": lambda ctl: skills.climb_wall(ctl, -1, "explore_ledge_left"),
    "explore_ledge_right": lambda ctl: skills.climb_wall(ctl, +1, "explore_ledge_right"),
    "hit_switch": lambda ctl: skills.hit_switch(ctl),
    "outwait_enemy": lambda ctl: skills.outwait_enemy(ctl),
    "explore_ledge_up": lambda ctl: skills.climb_up(ctl),
    "go_back_left": lambda ctl: skills.go_back(ctl, -1),
    "leap_past_enemy": lambda ctl: skills.leap_past_enemy(ctl),
    "pass_hazard_right": lambda ctl: skills.pass_hazard(ctl, +1),        # v8
    "pass_hazard_left": lambda ctl: skills.pass_hazard(ctl, -1),         # v8
    "drop_and_look_right": lambda ctl: skills.peek_below(ctl, +1),       # v8
    "cross_spikes_right": lambda ctl: skills.bounce_across(ctl, +1),     # v8
}
