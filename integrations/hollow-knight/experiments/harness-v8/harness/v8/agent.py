# fmt: off
# isort: skip_file
# ruff: noqa
"""Harness v8 entry point: boot, build the menu for the decision model, execute the chosen skill."""
from __future__ import annotations

from . import language, skills
from .control import Ctl


class Agent:
    def __init__(self, env):
        self.env = env
        self.ctl = Ctl(env)
        self.memory = language.Memory()
        self.last_sig = None

    def boot(self) -> None:
        """The level starts with the knight in the air: wait until it stands."""
        self.ctl.label = "boot (no decision yet)"
        self.ctl.settle(120)
        self.ctl.tr.jumps = 2

    def run_log(self) -> dict:
        """Harness-side counters for the run log (run_agent.py stores them with every skill row)."""
        c = self.ctl
        return {"reflex_strikes": c.strikes, "air_strikes": c.air_strikes, "fade_guard": len(c.guard_log),
                "sim_total": c.sim_total, "area": c.area}

    def menu(self) -> dict:
        menu = language.build_menu(self.ctl, self.memory)
        self.last_sig = menu["signature"]
        self.last_token = menu["token"]
        return menu

    def execute(self, choice: str, index: int, label: str, probability: float) -> dict:
        ctl = self.ctl
        ctl.label, ctl.prob, ctl.decision = label, probability, index
        ctl.windows_this_skill = 0
        ctl.sim_frames_this_skill = 0
        x0, y0 = ctl.pos
        hp0 = ctl.tr.hp
        deaths0 = ctl.tr.deaths
        area0 = ctl.area
        result = language.SKILLS[choice](ctl)
        ctl.stop_checks = []
        if ctl.area > area0:
            # v8: the level exit was taken and a new scene is on screen: wait until the knight stands,
            # and start the exploration memory over (the terrain memory was reset by the executor)
            ctl.settle(120)
            ctl.tr.jumps = 2
            self.memory.new_area()
            result.reason = "went through the exit into a new area"
        died = ctl.tr.deaths > deaths0 or ctl.tr.hp == 0
        if died:
            skills.after_death(ctl)
        elif result.reason.startswith("took damage"):
            skills.after_damage(ctl)
            if ctl.tr.hp == 0:
                died = True
                skills.after_death(ctl)
        elif result.reason.startswith("a moving hazard is about to hit"):
            result.reason += "; " + skills.evade_mover(ctl)
            if ctl.tr.hp == 0:
                died = True
                skills.after_death(ctl)
        elif result.reason.startswith("something dangerous is falling"):
            dodged = skills.dodge_falling(ctl)
            result.reason += "; stepped out from under it" if dodged else "; could not get out from under it"
            if ctl.tr.hp == 0:
                died = True
                skills.after_death(ctl)
        if not died and not ctl.tr.standing and ctl.tr.hp:
            # v8: never hand a decision to the model while the knight is in the air (a behaviour that gave up
            # mid-jump): land first with the air-recovery reflex
            ctl.stop_checks = []
            ctl.recover("land-before-deciding")
            if ctl.tr.hp == 0:
                died = True
                skills.after_death(ctl)
        x1, y1 = ctl.pos
        if died:
            result.reason = "died (" + result.reason + ") and restarted at the level start" if result.reason != "died" \
                else "died and restarted at the level start"
        self.memory.record(choice, result, (x1 - x0, y1 - y0), ctl.windows_this_skill, hp0, ctl.tr.hp, self.last_sig,
                           self.last_token)
        sup = ctl.scene.support() if ctl.tr.standing else None
        supw = ctl.support_surf() if sup is not None else None
        self.memory.note(choice, result.reason, sup.oid if sup is not None else None, x1 - x0,
                         at_x=None if (supw is None or supw.moving) else x1)
        return {"reason": result.reason, "windows": ctl.windows_this_skill, "sim_frames": ctl.sim_frames_this_skill,
                "notes": result.notes, "moved": [round(x1 - x0, 2), round(y1 - y0, 2)], "hp": ctl.tr.hp,
                "deaths": ctl.tr.deaths, "remembered_surfaces": len(ctl.map.solids)}
