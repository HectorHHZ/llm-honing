"""A tiny synthetic side-scroller that writes Exp 0 v0 attempts in the contract, in both decision shapes.

It is the executable example of CONTRACT.md and the fixture of the tests: a corridor with pits and
enemies, a clean harness that renders the visible situation into a Laya request, three faults aligned with
Griddly's (harness: the first visible enemy is omitted at the first eligible decision; weights: the
alternate checkpoint serves the episode; controller: native shape replaces the first selected ``right``
with ``wait``, option shape sends the first fight decision's strikes as ``wait``), and the adapter whose
``rerender`` the probes call. It talks to a decision service over HTTP (tests use ``fake_laya.FakeLaya``).
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contract import (
    CONTRACT_VERSION,
    FORMAT,
    ResponseRejected,
    Sampler,
    ServiceClient,
    guard_response,
    write_json,
    write_rows,
)

LENGTH = 22
SIGHT = 6
INTRO = (
    "You guide a runner along a corridor to the exit on the right. Pits kill; enemies hurt on touch; "
    "two strikes kill one."
)
NATIVE_OPTIONS = {
    "right": "Step one cell to the right.",
    "jump_right": "Jump two cells to the right, over a pit.",
    "attack": "Strike the cell to your right.",
    "left": "Step one cell to the left.",
    "wait": "Stand still for one moment.",
}
FAULTS = {
    "native": {
        "harness": {
            "family": "observation_omission",
            "instance": "omit-first-visible-enemy-v1",
            "trigger_rule": "first model decision with a visible enemy",
            "changed_artifact": "situation text of the model request",
        },
        "weights": {
            "family": "alternate_checkpoint",
            "instance": "fake-alternate-v1",
            "trigger_rule": "every decision of the episode",
            "changed_artifact": "decision checkpoint weights",
        },
        "controller": {
            "family": "post_selection_action_override",
            "instance": "first-right-to-wait-v1",
            "trigger_rule": "first model decision that selects right",
            "changed_artifact": "native action sent to the game",
        },
    },
    "option": {
        "harness": {
            "family": "observation_omission",
            "instance": "omit-first-visible-enemy-v1",
            "trigger_rule": "first model decision with a visible enemy",
            "changed_artifact": "situation text and options of the model request",
        },
        "weights": {
            "family": "alternate_checkpoint",
            "instance": "fake-alternate-v1",
            "trigger_rule": "every decision of the episode",
            "changed_artifact": "decision checkpoint weights",
        },
        "controller": {
            "family": "post_selection_input_drop",
            "instance": "drop-first-fight-strikes-v1",
            "trigger_rule": "first model decision that selects fight_nearest_enemy",
            "changed_artifact": "native windows sent to the game",
        },
    },
}
OUTCOMES = {
    "win": ["reached_exit"],
    "loss": ["player_died", "fell_in_pit"],
    "timeout": ["decision_cap"],
    "other_terminal": [],
}


def plan(shape: str, clean_sha: str, alternate_sha: str, groups: int = 14) -> dict[str, Any]:
    splits = ["train", "train", "val", "val"]
    return {
        "schema": "honing-exp0-collection-v0",
        "game_id": f"synthetic-{shape}",
        "dataset_dir": f"synthetic-{shape}/v0/",
        "shape": shape,
        "question_key": "action",
        "observation": "synthetic-visible-v1",
        "per_condition": 10,
        "initial_groups": 10,
        "selection": {"rule": "argmax", "seed": None},
        "clean_weights_sha256": clean_sha,
        "alternate_weights_sha256": alternate_sha,
        "weight_difference_tolerance": 0.01,
        "probe": {"first_decisions": 4, "identity_tolerance": 0.01, "tie": 0.01},
        "diagnosis": {
            "max_tokens": 9000,
            "time": {"unit": "native_tick", "before": 100000, "after": 0},
            "preamble": (
                "The synthetic corridor game counts native ticks; the renderer's time coordinate counts "
                "these ticks. Each executed option is carried out as native input windows."
            ),
            "windows": {"max_runs_listed": 12, "max_mismatches_listed": 6},
            "public_observation_max_chars": 600,
        },
        "outcomes": OUTCOMES,
        "faults": FAULTS[shape],
        "groups": [
            {
                "id": f"s{number:02d}",
                "split": splits[number % 4],
                "scenario": {"seed": 1000 + number, "caps": {"decisions": 30 if shape == "native" else 14}},
            }
            for number in range(groups)
        ],
        "attempt_cap_per_slot": 3,
        "neutral_id_prefix": "e",
        "shuffle_seed": 200,
        "response_tries": 2,
    }


class Game:
    """Native rules. One cell per move; enemies have 2 HP and hit an adjacent runner every other window."""

    def __init__(self, seed: int) -> None:
        rng = random.Random(seed)
        cells = list(range(3, LENGTH - 2))
        rng.shuffle(cells)
        self.pits = set(cells[:2])
        self.enemies = dict.fromkeys(cells[2:4], 2)
        self.x, self.hp, self.tick, self.windows = 0, 3, 0, 0
        self.status: str | None = None
        self.reason: str | None = None

    def visible(self) -> dict[str, Any]:
        ahead = range(self.x + 1, self.x + 1 + SIGHT)
        return {
            "hp": self.hp,
            "enemies_dx": [cell - self.x for cell in sorted(self.enemies) if cell in ahead],
            "pits_dx": [cell - self.x for cell in sorted(self.pits) if cell in ahead],
            "exit_dx": (LENGTH - 1 - self.x) if LENGTH - 1 - self.x <= SIGHT else None,
        }

    def step(self, action: str, ticks: int) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        start = self.tick
        self.tick += ticks
        self.windows += 1

        def event(kind: str, text: str, visible: bool = True) -> None:
            events.append({"tick": self.tick, "kind": kind, "text": text, "visible": visible})

        if action in ("right", "jump_right"):
            target = self.x + (1 if action == "right" else 2)
            if action == "jump_right" and (self.x + 1) in self.enemies:
                target = self.x + 1
            if target in self.enemies:
                self.hp -= 1
                event("damage", f"Ran into an enemy; health {self.hp + 1} -> {self.hp}.")
            elif target in self.pits:
                self.x = target
                self.status, self.reason = "loss", "fell_in_pit"
                event("death", "Fell into a pit.")
            else:
                self.x = min(target, LENGTH - 1)
        elif action == "left":
            self.x = max(0, self.x - 1)
        elif action == "attack" and (self.x + 1) in self.enemies:
            self.enemies[self.x + 1] -= 1
            if self.enemies[self.x + 1] == 0:
                del self.enemies[self.x + 1]
                event("kill", "Killed the enemy to the right.")
        if self.status is None and (self.x + 1) in self.enemies and self.windows % 2 == 0:
            self.hp -= 1
            event("damage", f"An enemy struck; health {self.hp + 1} -> {self.hp}.")
        if self.status is None and self.hp <= 0:
            self.status, self.reason = "loss", "player_died"
            event("death", "The runner died.")
        if self.status is None and self.x >= LENGTH - 1:
            self.status, self.reason = "win", "reached_exit"
            event("goal", "Reached the exit.")
        _ = start
        return events


def situation(obs: dict[str, Any], *, omit_first_enemy: bool = False) -> str:
    enemies = list(obs["enemies_dx"])[1:] if omit_first_enemy else list(obs["enemies_dx"])
    words = [f"Health {obs['hp']} of 3."]
    words += [f"An enemy is {dx} step{'s' if dx > 1 else ''} to your right." for dx in enemies]
    words += [f"A pit is {dx} step{'s' if dx > 1 else ''} to your right." for dx in obs["pits_dx"]]
    if obs["exit_dx"] is not None:
        words.append(f"The exit is {obs['exit_dx']} steps to your right.")
    if not any(dx == 1 for dx in enemies) and 1 not in obs["pits_dx"]:
        words.append("The way is open.")
    return " ".join(words)


def render(shape: str, obs: dict[str, Any], request_id: str, *, omit_first_enemy: bool = False) -> dict[str, Any]:
    """The clean harness (``omit_first_enemy=False``) or its faulted variant."""
    text = situation(obs, omit_first_enemy=omit_first_enemy)
    if shape == "native":
        criteria = dict(NATIVE_OPTIONS)
    else:
        enemies = obs["enemies_dx"][1:] if omit_first_enemy else obs["enemies_dx"]
        criteria = {"advance_right": "Go right toward the exit: walk, jump over pits."}
        if enemies:
            criteria["fight_nearest_enemy"] = "Fight the nearest enemy on your right until it is dead."
        criteria["wait"] = "Stand still for a moment and watch."
    return {
        "id": request_id,
        "parts": {"essential": [INTRO, "Situation: " + text]},
        "questions": {
            "action": {
                "type": "choice",
                "instructions": "Choose what makes progress toward the exit " "while staying alive.",
                "criteria": criteria,
            }
        },
    }


class SyntheticAdapter:
    """The adapter the common probes call: re-render a saved public state with the clean harness."""

    def __init__(self, shape: str = "native") -> None:
        self.shape = shape
        self.game_id = f"synthetic-{shape}"

    def rerender(self, state: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
        return render(self.shape, state["obs"], state["request_id"])


class NativeAdapter(SyntheticAdapter):
    def __init__(self) -> None:
        super().__init__("native")


class OptionAdapter(SyntheticAdapter):
    def __init__(self) -> None:
        super().__init__("option")


def option_windows(game: Game, option: str, wait_sends_nothing: bool = False) -> list[tuple[str, str]]:
    """The harness's execution plan for one option at the current state: (native action, reason) pairs."""
    plan_: list[tuple[str, str]] = []
    if option == "advance_right":
        x = game.x
        for _ in range(4):
            if (x + 1) in game.pits:
                plan_.append(("jump_right", "advance:jump"))
                x += 2
            elif (x + 1) in game.enemies:
                break
            else:
                plan_.append(("right", "advance:walk"))
                x += 1
        return plan_ or [("right", "advance:walk")]
    if option == "fight_nearest_enemy":
        steps = 0
        x = game.x
        while (x + 1) not in game.enemies and steps < 3 and (x + 1) not in game.pits:
            plan_.append(("right", "fight:approach"))
            x += 1
            steps += 1
        if (x + 1) not in game.enemies:
            return plan_  # the behaviour ends here: the enemy cannot be reached (may send no input at all)
        for _ in range(3):
            plan_ += [("attack", "fight:strike"), ("wait", "fight:cooldown")]
        return plan_
    return [] if wait_sends_nothing else [("wait", "wait"), ("wait", "wait")]


def collect(
    root: Path,
    plan_: dict[str, Any],
    group: dict[str, Any],
    condition: str,
    attempt: int,
    endpoint: str,
    *,
    fail_at: int | None = None,
    run: str | None = None,
    request_salt: str = "",
) -> Path:
    shape = plan_["shape"]
    run = run or f"{group['id']}-{condition}-a{attempt}"
    out = root / run
    (out / "responses").mkdir(parents=True)
    client = ServiceClient(endpoint)
    checkpoint = client.checkpoint()
    fault_def = plan_["faults"].get(condition)
    refs = {
        "harness": ("decisions.jsonl:clean_request", "decisions.jsonl:request"),
        "weights": (f"checkpoint:{plan_['clean_weights_sha256']}", f"checkpoint:{plan_['alternate_weights_sha256']}"),
        "controller": ("decisions.jsonl:model_choice", "decisions.jsonl:executed"),
    }
    selection = {key: value for key, value in plan_["selection"].items() if key != "per_group_seed"}
    if plan_["selection"].get("per_group_seed"):
        value: Any = group
        for part in plan_["selection"]["per_group_seed"].split("."):
            value = value[part]
        selection["seed"] = value
    sampler = Sampler(selection["seed"]) if selection["rule"] == "sample" else None
    manifest = {
        "format": FORMAT,
        "contract_version": CONTRACT_VERSION,
        "game_id": plan_["game_id"],
        "shape": shape,
        "run": run,
        "condition": condition,
        "reset_group": group["id"],
        "attempt": {"number": attempt, "slot": f"{group['id']}/{condition}"},
        "scenario": {
            "task": "corridor-to-exit",
            "start": {"x": 0, "hp": 3, "layout_seed": group["scenario"]["seed"]},
            "seed": group["scenario"]["seed"],
            "caps": group["scenario"]["caps"],
        },
        "selection": selection,
        "pins": {
            "game_build": {"id": "synthetic-corridor", "version": "1"},
            "code": {"fingerprint": "synthetic-collector-v1"},
            "harness": {"id": "synthetic-harness", "fingerprint": "synthetic-harness-v1"},
            "controller": {"id": "synthetic-controller", "fingerprint": "synthetic-controller-v1"},
            "observation": {"condition": plan_["observation"], "knobs": {"sight": SIGHT}},
            "checkpoint": checkpoint,
        },
        "clean_weights_sha256": plan_["clean_weights_sha256"],
        "intervention": (
            None
            if condition == "no_injection"
            else {
                "component": condition,
                **{k: fault_def[k] for k in ("family", "instance", "trigger_rule", "changed_artifact")},
                "original_ref": refs[condition][0],
                "altered_ref": refs[condition][1],
            }
        ),
    }
    write_json(out / "manifest.json", manifest)
    game = Game(group["scenario"]["seed"])
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    harness_done = controller_done = False
    ticks = 4 if shape == "native" else 3
    cap = group["scenario"]["caps"]["decisions"]
    # One auto row first: the collector lets the game settle before the first decision (like HK's boot window).
    before = {"tick": game.tick, "frame": game.tick, "sim_seconds": None}
    events = game.step("wait", 2)
    rows.append(
        {
            "index": 0,
            "kind": "auto",
            "native_before": before,
            "public_observation": None,
            "public_state": None,
            "request": None,
            "clean_request": None,
            "response": None,
            "probabilities": None,
            "server_choice": None,
            "model_choice": None,
            "executed": {
                "option": None,
                "windows": [{"native_action": "wait", "ticks": 2, "start_tick": before["tick"], "reason": "settle"}],
            },
            "fault": None,
            "events": events,
            "native_after": {"tick": game.tick, "frame": game.tick, "sim_seconds": None},
            "timing": {"model_seconds": None, "execution_seconds": 0.0, "wall_seconds": 0.0},
        }
    )
    decisions = 0
    try:
        while game.status is None:
            if decisions >= cap:
                game.status, game.reason = "timeout", "decision_cap"
                break
            index = len(rows)
            row_started = time.perf_counter()
            obs = game.visible()
            request_id = f"d{index}-t{game.tick}{request_salt}"
            states.append({"index": index, "state": {"obs": obs, "request_id": request_id}})
            request = render(shape, obs, request_id)
            clean = fault = None
            if condition == "harness" and not harness_done and obs["enemies_dx"]:
                clean, request = request, render(shape, obs, request_id, omit_first_enemy=True)
                fault = {
                    "component": "harness",
                    "changed": request != clean,
                    "detail": "the first visible enemy is left out of the situation",
                }
                harness_done = True
            tries = []
            parsed = None
            for number in range(1, plan_["response_tries"] + 1):
                name = f"responses/{index:04d}.raw" if number == 1 else f"responses/{index:04d}-t{number}.raw"
                raw, seconds = client.decide_raw(request)
                (out / name).write_bytes(raw)
                try:
                    parsed = guard_response(request, raw, plan_["question_key"])
                except ResponseRejected as rejection:
                    tries.append({"file": name, "status": "rejected", "reason": str(rejection), "seconds": seconds})
                    continue
                tries.append({"file": name, "status": "accepted", "reason": None, "seconds": seconds})
                break
            base = {
                "index": index,
                "kind": "model",
                "native_before": {"tick": game.tick, "frame": game.tick, "sim_seconds": None},
                "public_observation": obs,
                "public_state": len(states) - 1,
                "request": request,
                "clean_request": clean,
                "response": {"file": tries[-1]["file"] if parsed else None, "tries": tries},
            }
            if parsed is None:
                rows.append(
                    {
                        **base,
                        "probabilities": None,
                        "server_choice": None,
                        "model_choice": None,
                        "executed": None,
                        "fault": fault,
                        "events": [],
                        "native_after": None,
                        "timing": {
                            "model_seconds": sum(t["seconds"] for t in tries),
                            "execution_seconds": None,
                            "wall_seconds": time.perf_counter() - row_started,
                        },
                    }
                )
                raise ResponseRejected("no accepted response")
            choice = sampler.choose(request, parsed["probabilities"]) if sampler else parsed["server_choice"]
            model_seconds = sum(t["seconds"] for t in tries)
            if fail_at is not None and decisions == fail_at:
                # The reply was accepted, then the bridge failed before any native input: keep the row unexecuted.
                rows.append(
                    {
                        **base,
                        "probabilities": parsed["probabilities"],
                        "server_choice": parsed["server_choice"],
                        "model_choice": choice,
                        "executed": None,
                        "fault": fault,
                        "events": [],
                        "native_after": None,
                        "timing": {
                            "model_seconds": model_seconds,
                            "execution_seconds": None,
                            "wall_seconds": time.perf_counter() - row_started,
                        },
                    }
                )
                raise RuntimeError("simulated bridge timeout")
            executed_option = choice
            plan_windows = (
                [(choice, choice)]
                if shape == "native"
                else option_windows(game, choice, (plan_.get("extra") or {}).get("wait_sends_nothing", False))
            )
            sent: list[tuple[str, str, str | None]] = [(action, reason, None) for action, reason in plan_windows]
            if condition == "controller" and not controller_done:
                if shape == "native" and choice == "right":
                    executed_option, sent = "wait", [("wait", "wait", None)]
                    fault = {"component": "controller", "changed": True, "detail": "selected right; sent wait"}
                    controller_done = True
                elif (
                    shape == "option"
                    and choice == "fight_nearest_enemy"
                    and any(action == "attack" for action, _ in plan_windows)
                ):
                    sent = [
                        (("wait", reason, action) if action == "attack" else (action, reason, None))
                        for action, reason in plan_windows
                    ]
                    fault = {"component": "controller", "changed": True, "detail": "strikes sent as wait"}
                    controller_done = True
            exec_started = time.perf_counter()
            windows, events = [], []
            before_tick = game.tick
            for action, reason, intended in sent:
                window = {"native_action": action, "ticks": ticks, "start_tick": game.tick, "frames": ticks}
                if shape == "option":
                    window["reason"] = reason
                if intended is not None:
                    window["intended_action"] = intended
                windows.append(window)
                events += game.step(action, ticks)
                if game.status is not None:
                    break
            rows.append(
                {
                    **base,
                    "probabilities": parsed["probabilities"],
                    "server_choice": parsed["server_choice"],
                    "model_choice": choice,
                    "executed": {"option": executed_option, "windows": windows},
                    "fault": fault,
                    "events": events,
                    "native_after": {"tick": game.tick, "frame": game.tick, "sim_seconds": None},
                    "timing": {
                        "model_seconds": model_seconds,
                        "execution_seconds": time.perf_counter() - exec_started,
                        "wall_seconds": time.perf_counter() - row_started,
                    },
                }
            )
            _ = before_tick
            decisions += 1
    except (ResponseRejected, RuntimeError) as failure:
        write_rows(out / "decisions.jsonl", rows)
        write_rows(out / "public_states.jsonl", states)
        status = "rejected_response" if isinstance(failure, ResponseRejected) else "bridge_timeout"
        write_json(
            out / "error.json",
            {"status": status, "type": type(failure).__name__, "message": str(failure), "at_index": len(rows) - 1},
        )
        return out
    write_rows(out / "decisions.jsonl", rows)
    write_rows(out / "public_states.jsonl", states)
    changed = [row["index"] for row in rows if row["fault"] and row["fault"]["changed"]]
    counts: dict[str, int] = {}
    for row in rows:
        for event in row["events"]:
            counts[event["kind"]] = counts.get(event["kind"], 0) + 1
    end = client.checkpoint()
    write_json(
        out / "result.json",
        {
            "status": game.status,
            "reason": game.reason,
            "terminal_tick": game.tick,
            "rows": len(rows),
            "model_decisions": sum(row["kind"] == "model" for row in rows),
            "forced_decisions": sum(row["kind"] == "forced" for row in rows),
            "auto_rows": sum(row["kind"] == "auto" for row in rows),
            "windows": sum(len(row["executed"]["windows"]) for row in rows),
            "native_ticks": game.tick,
            "native_frames": game.tick,
            "sim_seconds": None,
            "wall_seconds": time.perf_counter() - started,
            "event_counts": counts,
            "trigger_index": changed[0] if changed else None,
            "fault_changed_rows": changed,
            "checkpoint_end": {"weightsSha256": end["weightsSha256"], "role": end["role"]},
        },
    )
    return out


# ------------------------------------------------------------------------------ two questions (contract 0.2.0)
# The "keys" shape: every decision asks two questions in one request, as Tank War v0 (motion, weapon) and the full
# HK raw-key harness (move, attack) do. One decision is one native window holding both answers' keys
# ("<move>+<attack>"). Faults: harness = the first visible enemy left out at the first decision with one (both
# questions see the changed state); weights = the alternate checkpoint; controller = after the selection the
# first selected strike is not sent (key level: the window goes out as "<move>+none", the executed attack is
# "none"); the `extend` variant holds the first selected move twice as long (the window's length changes).
KEYS = ("move", "attack")
MOVE_OPTIONS = {
    "right": "Hold right: one cell to the right.",
    "jump_right": "Jump right: two cells to the right, over a pit.",
    "left": "Hold left: one cell to the left.",
    "wait": "No direction key: stay.",
}
ATTACK_OPTIONS = {"none": "No attack key.", "strike": "Strike the cell to your right."}
FAULTS["keys"] = {
    "harness": {
        "family": "observation_omission",
        "instance": "omit-first-visible-enemy-v1",
        "trigger_rule": "first model decision with a visible enemy",
        "changed_artifact": "situation text of the model request (both questions)",
    },
    "weights": {
        "family": "alternate_checkpoint",
        "instance": "fake-alternate-v1",
        "trigger_rule": "every decision of the episode",
        "changed_artifact": "decision checkpoint weights",
    },
    "controller": [
        {
            "family": "post_selection_input_drop",
            "instance": "drop-first-strike-v1",
            "trigger_rule": "first model decision whose attack answer is strike",
            "changed_artifact": "native keys sent to the game (the attack key)",
        },
        {
            "family": "post_selection_timing",
            "instance": "extend-first-move-v1",
            "trigger_rule": "first model decision whose move answer is right",
            "changed_artifact": "native window length",
        },
    ],
}


def plan_keys(clean_sha: str, alternate_sha: str, groups: int = 14) -> dict[str, Any]:
    base = plan("native", clean_sha, alternate_sha, groups)
    base.pop("question_key")
    base.update(
        game_id="synthetic-keys", dataset_dir="synthetic-keys/v0/", question_keys=list(KEYS), faults=FAULTS["keys"]
    )
    base["diagnosis"]["preamble"] = (
        "The synthetic corridor game counts native ticks; the renderer's time coordinate "
        "counts these ticks. Each decision answers a move and an attack question; both "
        "answers' keys are held for one native window."
    )
    return base


def render_keys(obs: dict[str, Any], request_id: str, *, omit_first_enemy: bool = False) -> dict[str, Any]:
    text = situation(obs, omit_first_enemy=omit_first_enemy)
    return {
        "id": request_id,
        "parts": {"essential": [INTRO, "Situation: " + text]},
        "questions": {
            "move": {
                "type": "choice",
                "instructions": "Choose the movement keys for this moment.",
                "criteria": dict(MOVE_OPTIONS),
            },
            "attack": {
                "type": "choice",
                "instructions": "Choose the attack key for this moment.",
                "criteria": dict(ATTACK_OPTIONS),
            },
        },
    }


class KeysAdapter:
    game_id = "synthetic-keys"

    def rerender(self, state: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
        return render_keys(state["obs"], state["request_id"])


def step_keys(game: Game, move: str, attack: str, ticks: int) -> list[dict[str, Any]]:
    """One window with both keys: the strike lands first, then the move (native rules of the corridor)."""
    events: list[dict[str, Any]] = []
    if attack == "strike" and (game.x + 1) in game.enemies:
        game.enemies[game.x + 1] -= 1
        if game.enemies[game.x + 1] == 0:
            del game.enemies[game.x + 1]
            events.append(
                {"tick": game.tick + ticks, "kind": "kill", "text": "Killed the enemy to the right.", "visible": True}
            )
    return events + game.step(move, ticks)


def collect_keys(
    root: Path,
    plan_: dict[str, Any],
    group: dict[str, Any],
    condition: str,
    attempt: int,
    endpoint: str,
    *,
    controller_instance: str = "drop-first-strike-v1",
    run: str | None = None,
    request_salt: str = "",
) -> Path:
    from contract import guard_decision

    run = run or f"{group['id']}-{condition}-a{attempt}"
    out = root / run
    (out / "responses").mkdir(parents=True)
    client = ServiceClient(endpoint)
    checkpoint = client.checkpoint()
    definitions = plan_["faults"].get(condition)
    if isinstance(definitions, list):
        fault_def = next(item for item in definitions if item["instance"] == controller_instance)
    else:
        fault_def = definitions
    refs = {
        "harness": ("decisions.jsonl:clean_request", "decisions.jsonl:request"),
        "weights": (f"checkpoint:{plan_['clean_weights_sha256']}", f"checkpoint:{plan_['alternate_weights_sha256']}"),
        "controller": ("decisions.jsonl:model_choice", "decisions.jsonl:executed"),
    }
    keys = list(plan_["question_keys"])
    selection = {key: value for key, value in plan_["selection"].items() if key != "per_group_seed"}
    sampler = Sampler(selection["seed"]) if selection["rule"] == "sample" else None
    manifest = {
        "format": FORMAT,
        "contract_version": CONTRACT_VERSION,
        "game_id": plan_["game_id"],
        "shape": "native",
        "run": run,
        "condition": condition,
        "reset_group": group["id"],
        "attempt": {"number": attempt, "slot": f"{group['id']}/{condition}"},
        "scenario": {
            "task": "corridor-to-exit",
            "start": {"x": 0, "hp": 3, "layout_seed": group["scenario"]["seed"]},
            "seed": group["scenario"]["seed"],
            "caps": group["scenario"]["caps"],
        },
        "selection": selection,
        "pins": {
            "game_build": {"id": "synthetic-corridor", "version": "1"},
            "code": {"fingerprint": "synthetic-collector-v1"},
            "harness": {"id": "synthetic-keys-harness", "fingerprint": "synthetic-keys-harness-v1"},
            "controller": {"id": "synthetic-controller", "fingerprint": "synthetic-controller-v1"},
            "observation": {"condition": plan_["observation"], "knobs": {"sight": SIGHT}},
            "checkpoint": checkpoint,
        },
        "clean_weights_sha256": plan_["clean_weights_sha256"],
        "intervention": (
            None
            if condition == "no_injection"
            else {
                "component": condition,
                **{k: fault_def[k] for k in ("family", "instance", "trigger_rule", "changed_artifact")},
                "original_ref": refs[condition][0],
                "altered_ref": refs[condition][1],
            }
        ),
    }
    write_json(out / "manifest.json", manifest)
    game = Game(group["scenario"]["seed"])
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    harness_done = controller_done = False
    ticks = 4
    cap = group["scenario"]["caps"]["decisions"]
    before = {"tick": game.tick, "frame": game.tick, "sim_seconds": None}
    events = game.step("wait", 2)
    rows.append(
        {
            "index": 0,
            "kind": "auto",
            "native_before": before,
            "public_observation": None,
            "public_state": None,
            "request": None,
            "clean_request": None,
            "response": None,
            "probabilities": None,
            "server_choice": None,
            "model_choice": None,
            "executed": {
                "option": None,
                "windows": [
                    {"native_action": "wait+none", "ticks": 2, "start_tick": before["tick"], "reason": "settle"}
                ],
            },
            "fault": None,
            "events": events,
            "native_after": {"tick": game.tick, "frame": game.tick, "sim_seconds": None},
            "timing": {"model_seconds": None, "execution_seconds": 0.0, "wall_seconds": 0.0},
        }
    )
    decisions = 0
    while game.status is None:
        if decisions >= cap:
            game.status, game.reason = "timeout", "decision_cap"
            break
        index = len(rows)
        row_started = time.perf_counter()
        obs = game.visible()
        request_id = f"d{index}-t{game.tick}{request_salt}"
        states.append({"index": index, "state": {"obs": obs, "request_id": request_id}})
        request = render_keys(obs, request_id)
        clean = fault = None
        if condition == "harness" and not harness_done and obs["enemies_dx"]:
            clean, request = request, render_keys(obs, request_id, omit_first_enemy=True)
            fault = {
                "component": "harness",
                "changed": request != clean,
                "detail": "the first visible enemy is left out of the situation",
                "questions": list(keys),
            }
            harness_done = True
        name = f"responses/{index:04d}.raw"
        raw, seconds = client.decide_raw(request)
        (out / name).write_bytes(raw)
        parsed = guard_decision(request, raw, keys)["questions"]  # the tests' fake service never fails here
        server = {key: parsed[key]["server_choice"] for key in keys}
        probabilities = {key: parsed[key]["probabilities"] for key in keys}
        choice = sampler.choose_all(request, probabilities, keys) if sampler else dict(server)
        move, attack = choice["move"], choice["attack"]
        sent_move, sent_attack, sent_ticks = move, attack, ticks
        if condition == "controller" and not controller_done:
            if controller_instance == "drop-first-strike-v1" and attack == "strike":
                sent_attack = "none"
                fault = {
                    "component": "controller",
                    "changed": True,
                    "detail": "the strike key was not sent",
                    "questions": ["attack"],
                }
                controller_done = True
            elif controller_instance == "extend-first-move-v1" and move == "right":
                sent_ticks = 2 * ticks
                fault = {
                    "component": "controller",
                    "changed": True,
                    "detail": "the window was held twice as long",
                    "questions": ["move"],
                }
                controller_done = True
        window = {
            "native_action": f"{sent_move}+{sent_attack}",
            "ticks": sent_ticks,
            "start_tick": game.tick,
            "frames": sent_ticks,
        }
        if (sent_move, sent_attack) != (move, attack):
            window["intended_action"] = f"{move}+{attack}"
        if sent_ticks != ticks:
            window["intended_ticks"] = ticks
        exec_started = time.perf_counter()
        events = step_keys(game, sent_move, sent_attack, sent_ticks)
        rows.append(
            {
                "index": index,
                "kind": "model",
                "native_before": {"tick": window["start_tick"], "frame": window["start_tick"], "sim_seconds": None},
                "public_observation": obs,
                "public_state": len(states) - 1,
                "request": request,
                "clean_request": clean,
                "response": {
                    "file": name,
                    "tries": [{"file": name, "status": "accepted", "reason": None, "seconds": seconds}],
                },
                "probabilities": probabilities,
                "server_choice": server,
                "model_choice": choice,
                "executed": {"option": {"move": sent_move, "attack": sent_attack}, "windows": [window]},
                "fault": fault,
                "events": events,
                "native_after": {"tick": game.tick, "frame": game.tick, "sim_seconds": None},
                "timing": {
                    "model_seconds": seconds,
                    "execution_seconds": time.perf_counter() - exec_started,
                    "wall_seconds": time.perf_counter() - row_started,
                },
            }
        )
        decisions += 1
    write_rows(out / "decisions.jsonl", rows)
    write_rows(out / "public_states.jsonl", states)
    changed = [row["index"] for row in rows if row["fault"] and row["fault"]["changed"]]
    counts: dict[str, int] = {}
    for row in rows:
        for event in row["events"]:
            counts[event["kind"]] = counts.get(event["kind"], 0) + 1
    end = client.checkpoint()
    write_json(
        out / "result.json",
        {
            "status": game.status,
            "reason": game.reason,
            "terminal_tick": game.tick,
            "rows": len(rows),
            "model_decisions": sum(row["kind"] == "model" for row in rows),
            "forced_decisions": 0,
            "auto_rows": 1,
            "windows": sum(len(row["executed"]["windows"]) for row in rows),
            "native_ticks": game.tick,
            "native_frames": game.tick,
            "sim_seconds": None,
            "wall_seconds": time.perf_counter() - started,
            "event_counts": counts,
            "trigger_index": changed[0] if changed else None,
            "fault_changed_rows": changed,
            "checkpoint_end": {"weightsSha256": end["weightsSha256"], "role": end["role"]},
        },
    )
    return out
