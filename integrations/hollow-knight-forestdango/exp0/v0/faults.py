"""Fault instances of the full-HK Exp 0 v0 dataset under the raw-key harness (v9_keys family), from the #185 catalog.

Each instance is a switch of collect.py (`--condition`, `--fault-instance`, `--fault-form persistent|once`). The
frozen harness snapshot is never changed; a fault acts at exactly one layer:

  harness     between the harness state and the model request: what the decision model is told and offered. The
              clean request is always rendered first by the harness itself (its `ask()`, whose side effects stand);
              the faulted request is rendered from a throw-away copy of the agent (or by editing the rendered text)
              and is what the model sees. The harness keeps perceiving, remembering and executing with the full
              picture. Both requests are kept (`request` and `clean_request`).
  weights     the whole episode is served by the alternate checkpoint (Laya 8812); nothing else differs.
  controller  after the model's selection, between the harness and the game: the harness asks for the keys of the
              chosen options; the controller sends different keys or a different window length. The harness is not
              told (its memory records what it asked for and what it then observed). Every window keeps the keys
              asked for and the keys sent (core/env_keys.py `key_filter`).

Forms (`form`): persistent (default; Tank War's form) = from the trigger decision on, every eligible decision is
altered; once (Griddly's form) = only the trigger decision. The trigger decision is the first one at which the
instance changes something.

Instances (ids are recorded in every manifest; families follow #185):
  harness  omit-enemies-v1    observation-omission        the request is rendered as if no enemy were on screen
                                                          (screen lines, move forecasts, attack tags)
           omit-ground-v1     observation-omission        the "Ground: ..." and "Ledges near you: ..." sentences
                                                          (floor to each side, standable ledges) are left out
           corrupt-jump-v1    option-description-corruption  in the move question each short-jump option carries
                                                          the text of its long-jump counterpart and vice versa
           corrupt-attack-v1  option-description-corruption  in the attack question the texts of `forward` and
                                                          `up` are exchanged
  weights  laya-ra2-v33-v1    alternate-checkpoint        jayzou3773/laya-ra2-v33 on 8812 answers both questions
  controller
           swap-left-right-v1 post-selection-input-remap  LEFT and RIGHT exchanged in every window of a decision
                                                          whose chosen move has a direction
           drop-attack-v1     post-selection-input-drop   the NAIL press (and the UP / DOWN that only aims it) is
                                                          not sent; trigger: the first chosen strike while the
                                                          sword is ready by the controller's own record
           drop-jump-v1       post-selection-input-drop   the JUMP key is not sent (press and hold) in a decision
                                                          whose chosen move has a jump
           extend-window-v1   post-selection-timing       the decision's keys are held EXTEND (9) frames longer:
                                                          the last multi-frame window of each decision is extended
A legal cooldown (the harness's "NOT READY") is native non-execution, not a controller change: the controller
faults only change what is sent.
"""

from __future__ import annotations

import copy
import importlib
import json
import re

HARNESS = ("omit-enemies-v1", "omit-ground-v1", "corrupt-jump-v1", "corrupt-attack-v1")
WEIGHTS = ("laya-ra2-v33-v1",)
CONTROLLER = ("swap-left-right-v1", "drop-attack-v1", "drop-jump-v1", "extend-window-v1")
DEFAULT_INSTANCE = {"harness": "omit-enemies-v1", "weights": "laya-ra2-v33-v1", "controller": "swap-left-right-v1"}
FORMS = ("persistent", "once")
EXTEND = 9  # extra frames of extend-window-v1 (one more 0.3 s window of the same keys)
SWORD_COOLDOWN = 13  # frames between two strikes that both come out (v9_keys COOLDOWN, measured)
MIRROR = {
    "left": "right",
    "right": "left",
    "left_short_jump": "right_short_jump",
    "right_short_jump": "left_short_jump",
    "left_long_jump": "right_long_jump",
    "right_long_jump": "left_long_jump",
    "stay": "stay",
    "short_jump": "short_jump",
    "long_jump": "long_jump",
}
NO_JUMP = {
    "left": "left",
    "left_short_jump": "left",
    "left_long_jump": "left",
    "right": "right",
    "right_short_jump": "right",
    "right_long_jump": "right",
    "stay": "stay",
    "short_jump": "stay",
    "long_jump": "stay",
}
JUMP_PAIRS = (
    ("left_short_jump", "left_long_jump"),
    ("right_short_jump", "right_long_jump"),
    ("short_jump", "long_jump"),
)
STRIKES = ("forward", "up", "down")

DEFINITIONS = {
    "omit-enemies-v1": {
        "family": "observation-omission",
        "instance": "omit-enemies-v1",
        "changed_artifact": "state text and option texts of the model request (both questions)",
        "trigger_rule": "the first model decision at which a visible enemy changes the clean request",
    },
    "omit-ground-v1": {
        "family": "observation-omission",
        "instance": "omit-ground-v1",
        "changed_artifact": "state text of the model request (the ground and ledge sentences)",
        "trigger_rule": "the first model decision whose clean request describes the ground or ledges",
    },
    "corrupt-jump-v1": {
        "family": "option-description-corruption",
        "instance": "corrupt-jump-v1",
        "changed_artifact": "option texts of the move question (short and long jumps exchanged)",
        "trigger_rule": "the first model decision at which the exchange changes the clean request",
    },
    "corrupt-attack-v1": {
        "family": "option-description-corruption",
        "instance": "corrupt-attack-v1",
        "changed_artifact": "option texts of the attack question (forward and up exchanged)",
        "trigger_rule": "the first model decision at which the exchange changes the clean request",
    },
    "laya-ra2-v33-v1": {
        "family": "alternate-checkpoint",
        "instance": "laya-ra2-v33-v1",
        "changed_artifact": "decision checkpoint weights (Laya 8812, jayzou3773/laya-ra2-v33), both questions",
        "trigger_rule": "every decision of the episode",
    },
    "swap-left-right-v1": {
        "family": "post-selection-input-remap",
        "instance": "swap-left-right-v1",
        "changed_artifact": "native direction keys sent to the game (LEFT and RIGHT exchanged)",
        "trigger_rule": "the first model decision whose chosen move holds a direction key",
    },
    "drop-attack-v1": {
        "family": "post-selection-input-drop",
        "instance": "drop-attack-v1",
        "changed_artifact": "native attack keys sent to the game (the NAIL press is not sent)",
        "trigger_rule": "the first model decision whose chosen attack is a strike while the sword is ready",
    },
    "drop-jump-v1": {
        "family": "post-selection-input-drop",
        "instance": "drop-jump-v1",
        "changed_artifact": "native jump keys sent to the game (the JUMP key is not sent)",
        "trigger_rule": "the first model decision whose chosen move has a jump",
    },
    "extend-window-v1": {
        "family": "post-selection-timing",
        "instance": "extend-window-v1",
        "changed_artifact": "native window length (the chosen keys are held 9 frames longer)",
        "trigger_rule": "the first model decision",
    },
}


def definition(instance: str, form: str) -> dict:
    """The plan / manifest form of an instance; the one-shot form is a distinct instance id (`<id>-once`)."""
    d = dict(DEFINITIONS[instance])
    if form == "once":
        d["instance"] = f"{instance}-once"
        d["trigger_rule"] += "; only that decision is altered (one-shot)"
    else:
        d["trigger_rule"] += "; from then on every eligible decision is altered (persistent)"
    return d


def window_text(w: dict) -> str:
    """The canonical native-action string of a keys window (what the contract's windows record)."""
    text = f"keys hold:{'+'.join(w['hold']) or '-'} press:{'+'.join(w['press']) or '-'}"
    if w.get("first"):
        text += f" first:{'+'.join(w['first'])}"
    return text


# ------------------------------------------------------------------------------------------------- base
class NoFault:
    component = None
    instance = None

    def __init__(self, form: str = "persistent"):
        if form not in FORMS:
            raise ValueError(form)
        self.form = form
        self.trigger_decision = None
        self.changed: list[int] = []

    @property
    def once_done(self) -> bool:
        return self.form == "once" and self.trigger_decision is not None

    def render(self, agent, ask: dict, clean_request: dict, index: int, build_request) -> tuple[dict, dict | None]:
        """(the request the model sees, fault record or None) at one decision."""
        return clean_request, None

    def on_selection(self, env, move: str, attack: str, index: int, facts: dict) -> dict | None:
        """Arm the controller layer for this decision's windows; returns what it intends (or None)."""
        return None

    def after_decision(self, env, index: int) -> dict | None:
        """Disarm; returns the controller record of this decision (what was asked for vs sent) or None."""
        return None

    def summary(self) -> dict:
        return {
            "instance": self.instance,
            "component": self.component,
            "form": self.form,
            "triggered": self.trigger_decision is not None,
            "trigger_decision": self.trigger_decision,
            "changed_decisions": list(self.changed),
        }


# ---------------------------------------------------------------------------------------------- harness
class HarnessFault(NoFault):
    component = "harness"

    def __init__(self, instance: str, form: str, package: str):
        super().__init__(form)
        if instance not in HARNESS:
            raise ValueError(instance)
        self.base = instance
        self.instance = definition(instance, form)["instance"]
        self.package = package
        self.see = importlib.import_module(f"{package}.see")

    def _blind_copy(self, agent):
        """A throw-away copy of the agent whose current view shows no enemy (memory and numbers unchanged)."""
        clone = copy.deepcopy(agent, memo={id(agent.env): agent.env})
        view = copy.copy(clone.view)
        view.enemies = []
        view.obs = dict(view.obs)
        view.obs["objects"] = [o for o in view.obs.get("objects", []) if o.get("kind") != "enemy"]
        clone.view = view
        return clone

    def _variant(self, agent, ask: dict, clean_request: dict, build_request) -> dict:
        ident = clean_request["id"]
        if self.base == "omit-enemies-v1":
            clone = self._blind_copy(agent)
            blind = clone.ask()
            return build_request(ident, blind["essential"], blind["questions"])
        request = json.loads(json.dumps(clean_request))
        if self.base == "omit-ground-v1":
            request["parts"]["essential"] = [omit_ground(line) for line in request["parts"]["essential"]]
        elif self.base == "corrupt-jump-v1":
            criteria = request["questions"]["move"]["criteria"]
            for short, long in JUMP_PAIRS:
                criteria[short], criteria[long] = criteria[long], criteria[short]
        elif self.base == "corrupt-attack-v1":
            criteria = request["questions"]["attack"]["criteria"]
            criteria["forward"], criteria["up"] = criteria["up"], criteria["forward"]
        return request

    def render(self, agent, ask, clean_request, index, build_request):
        if self.once_done:
            return clean_request, None
        altered = self._variant(agent, ask, clean_request, build_request)
        changed = json.dumps(altered) != json.dumps(clean_request)
        if not changed and self.trigger_decision is None:
            return clean_request, None  # not triggered yet: no fault record
        if changed:
            if self.trigger_decision is None:
                self.trigger_decision = index
            self.changed.append(index)
        questions = [
            q
            for q in clean_request["questions"]
            if json.dumps(altered["questions"][q]) != json.dumps(clean_request["questions"][q])
        ]
        if altered["parts"] != clean_request["parts"]:
            questions = list(clean_request["questions"])  # the state text both questions read changed
        return altered, {
            "component": "harness",
            "changed": changed,
            "questions": questions,
            "detail": DEFINITIONS[self.base]["changed_artifact"],
        }


SENTENCE = re.compile(r"(?<=\.) (?=[A-Z])")


def omit_ground(line: str) -> str:
    """The "On screen:" line without its "Ground: ..." and "Ledges near you: ..." sentences."""
    prefix = "On screen: "
    if not line.startswith(prefix):
        return line
    items = SENTENCE.split(line[len(prefix) :])
    kept = [item for item in items if not item.startswith(("Ground:", "Ledges near you:"))]
    return prefix + " ".join(kept)


# ------------------------------------------------------------------------------------------- controller
class ControllerFault(NoFault):
    component = "controller"

    def __init__(self, instance: str, form: str):
        super().__init__(form)
        if instance not in CONTROLLER:
            raise ValueError(instance)
        self.base = instance
        self.instance = definition(instance, form)["instance"]
        self.active = None  # this decision's plan: {"index", "move", "attack", "executed": {...}}
        self.windows: list[dict] = []  # this decision's asked/sent windows
        self.last_nail_tick = None  # native tick of the last NAIL press actually sent (the controller's record)
        self.env = None

    def _eligible(self, move: str, attack: str, env) -> bool:
        if self.base == "swap-left-right-v1":
            return MIRROR[move] != move
        if self.base == "drop-attack-v1":
            if attack not in STRIKES:
                return False
            if self.trigger_decision is None:  # the trigger needs a strike that would come out
                return self.last_nail_tick is None or env._tick - self.last_nail_tick >= SWORD_COOLDOWN
            return True
        if self.base == "drop-jump-v1":
            return "jump" in move
        return True  # extend-window-v1

    def on_selection(self, env, move, attack, index, facts):
        armed = (not self.once_done) and self._eligible(move, attack, env)
        executed = {"move": move, "attack": attack}
        if armed:
            if self.base == "swap-left-right-v1":
                executed["move"] = MIRROR[move]
            elif self.base == "drop-attack-v1":
                executed["attack"] = "none"
            elif self.base == "drop-jump-v1":
                executed["move"] = NO_JUMP[move]
        self.active = {"index": index, "move": move, "attack": attack, "armed": armed, "executed": executed}
        self.windows = []
        self.env = env
        env.key_filter = self._filter
        return dict(self.active)

    def _filter(self, window: dict, meta: dict) -> tuple[dict, dict | None]:
        sent = json.loads(json.dumps(window))
        a = self.active or {}
        if a.get("armed"):
            if self.base == "swap-left-right-v1":
                swap = {"LEFT": "RIGHT", "RIGHT": "LEFT"}
                for key in ("hold", "press", "first"):
                    sent[key] = [swap.get(k, k) for k in sent[key]]
            elif self.base == "drop-attack-v1":
                if "NAIL" in sent["press"]:
                    sent["press"] = [k for k in sent["press"] if k != "NAIL"]
                    sent["first"] = []  # `first` only aims a strike (keys.py)
            elif self.base == "drop-jump-v1":
                sent["press"] = [k for k in sent["press"] if k != "JUMP"]
                sent["hold"] = [k for k in sent["hold"] if k != "JUMP"]
            elif self.base == "extend-window-v1":
                if sent["frames"] > 1 and self._is_last_long(window):
                    sent["frames"] = min(120, sent["frames"] + EXTEND)
        self.windows.append({"asked": window, "sent": sent})
        if "NAIL" in sent["press"] and self.env is not None:
            self.last_nail_tick = self.env._tick  # the press goes out in this window's first frame
        change = None
        if sent != window:
            change = {
                "instance": self.instance,
                "asked": window,
                "sent": sent,
                "reason": f"controller fault {self.instance}: {DEFINITIONS[self.base]['changed_artifact']}",
            }
        return sent, change

    def _is_last_long(self, window: dict) -> bool:
        """extend-window-v1 extends the decision's last window that is longer than one frame. The harness sends
        [frames-1, 1] or [lead, rest, 1]; the multi-frame window right before the final 1-frame window is the one."""
        count = len(self.windows)  # windows of this decision already passed through the filter
        plan = self.active.get("plan") if self.active else None
        if plan is None:
            return False
        return count == len(plan) - 2 and plan[count]["frames"] == window["frames"]

    def plan_windows(self, plan: list[dict]) -> None:
        """The harness's windows for this decision (keys.bridge_windows), so the filter knows which one is last."""
        if self.active is not None:
            self.active["plan"] = [{"frames": w["frames"]} for w in plan]

    def after_decision(self, env, index):
        env.key_filter = None
        a, windows = self.active, self.windows
        self.active, self.windows = None, []
        if a is None:
            return None
        changed = any(w["asked"] != w["sent"] for w in windows)
        if changed:
            if self.trigger_decision is None:
                self.trigger_decision = index
            self.changed.append(index)
        if not a["armed"] and not changed:
            return None
        return {
            "component": "controller",
            "changed": changed,
            "armed": a["armed"],
            "selected": {"move": a["move"], "attack": a["attack"]},
            "executed": a["executed"] if changed else {"move": a["move"], "attack": a["attack"]},
            "questions": [q for q in ("move", "attack") if changed and a["executed"][q] != a[q]]
            or (["move", "attack"] if changed and self.base == "extend-window-v1" else []),
            "windows_changed": sum(w["asked"] != w["sent"] for w in windows),
            "detail": DEFINITIONS[self.base]["changed_artifact"],
        }


class WeightsFault(NoFault):
    component = "weights"

    def __init__(self, instance: str, form: str):
        super().__init__(form)
        if instance not in WEIGHTS:
            raise ValueError(instance)
        if form != "persistent":
            raise ValueError("the weights fault serves the whole episode (no one-shot form)")
        self.instance = instance
        self.trigger_decision = 1


def make(condition: str, instance: str | None, form: str, package: str):
    if condition == "no_injection":
        if instance:
            raise ValueError("no_injection takes no fault instance")
        return NoFault(form)
    instance = instance or DEFAULT_INSTANCE[condition]
    if condition == "harness":
        return HarnessFault(instance, form, package)
    if condition == "weights":
        return WeightsFault(instance, form)
    if condition == "controller":
        return ControllerFault(instance, form)
    raise ValueError(condition)
