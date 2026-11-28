"""Scripted start prefixes of the full-HK Exp 0 v0 reset groups (2026-11-26, Exp 0 v0 HK collection subagent).

A reset group's start state is reached from a new game by a fixed key script (a "prefix"), played before the first
model decision. The deterministic build (HONING_FAST=1) replays a key sequence exactly, so every attempt of a group
(all four conditions) starts from the identical game and harness state; `verify_starts.py` checks this with two
boots per start.

The prefix is played exactly as the Demo session's runner `code/hk-hitl/run_keys2.py --prefix <file>
--prefix-frames 9` plays it (Fable's starts were made and run that way):
  * each step is one harness decision whose two options are given by the script (no model is asked);
  * the steps play with 9 frames per decision and a 1-frame tail (the cadence the route scripts were written for);
    the harness's own cadence (v12: 13 frames, 2-frame tail) returns with the first model decision;
  * before each step the harness renders its request as for a decision (`ask()`; the request is not sent and not
    kept), because its forecasts are what `act()` uses to name the cause of any damage in its memory;
  * the harness perceives and remembers everything on the way (its memory at the handover includes the prefix:
    visited places, the room picture, its "Last decisions").
The prefix is the reset procedure, not part of the episode: its windows are tagged in windows.jsonl
({"prefix": k}) and the contract rows begin with the first model decision (to_contract.py).

Script syntax (run_keys2.py): whitespace-separated steps "<move>/<attack>[*count]"; move L Ls Ll R Rs Rl N Ns Nl
(left / right / no direction, followed by nothing, s or l for no / short / long jump); attack n f u d h (none,
forward, up, down, focus); "/n" may be left out; '#' starts a comment.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

MOVE_SHORT = {
    "L": "left",
    "Ls": "left_short_jump",
    "Ll": "left_long_jump",
    "R": "right",
    "Rs": "right_short_jump",
    "Rl": "right_long_jump",
    "N": "stay",
    "Ns": "short_jump",
    "Nl": "long_jump",
}
ATTACK_SHORT = {"n": "none", "f": "forward", "u": "up", "d": "down", "h": "focus"}
PREFIX_FRAMES = 9  # the cadence of the route scripts (run_keys2.py --prefix-frames 9)
PREFIX_TAIL = 1


def parse(text: str) -> list[tuple[str, str]]:
    steps = []
    for line in text.splitlines():
        line = line.split("#")[0]
        for tok in line.split():
            tok, _, count = tok.partition("*")
            mv, _, at = tok.partition("/")
            steps += [(MOVE_SHORT[mv], ATTACK_SHORT[at or "n"])] * int(count or 1)
    return steps


def read_script(path: Path) -> list[tuple[str, str]]:
    return parse(Path(path).read_text())


def start_steps(root: Path, start: dict) -> list[tuple[str, str]]:
    """The prefix of a plan start: the script file (relative to `root`), cut to its first `steps` steps if given."""
    if not start.get("prefix"):
        return []
    steps = read_script(Path(root) / start["prefix"])
    if start.get("steps") is not None:
        if int(start["steps"]) > len(steps):
            raise ValueError(f"start asks for {start['steps']} steps; the script has {len(steps)}")
        steps = steps[: int(start["steps"])]
    return steps


def steps_text(steps: list[tuple[str, str]]) -> str:
    """The canonical one-line form of a step list (used for the prefix digest that identifies a start)."""
    inverse_m = {v: k for k, v in MOVE_SHORT.items()}
    inverse_a = {v: k for k, v in ATTACK_SHORT.items()}
    return " ".join(f"{inverse_m[m]}/{inverse_a[a]}" for m, a in steps)


def steps_sha256(steps: list[tuple[str, str]]) -> str:
    return hashlib.sha256(steps_text(steps).encode()).hexdigest()


def play_prefix(
    agent, env, steps: list[tuple[str, str]], frames: int = PREFIX_FRAMES, tail: int = PREFIX_TAIL, on_step=None
) -> list[dict]:
    """Play the prefix through the harness (run_keys2.py semantics) and return one record per step."""
    own_n, own_tail = agent.n, agent.knobs.get("tail")
    out = []
    try:
        for k, (move, attack) in enumerate(steps, start=1):
            if frames:  # None / 0: the prefix plays at the harness's own cadence
                agent.n = frames
                if "tail" in agent.knobs:
                    agent.knobs["tail"] = tail
            env.window_tag = {"prefix": k, "decision": None}
            agent.ask()
            result = agent.act(move, attack, k, f"[start prefix] {move} + {attack}")
            rec = {"step": k, "move": move, "attack": attack, "result": result["result"]}
            out.append(rec)
            if on_step is not None:
                on_step(rec)
    finally:
        env.window_tag = None
        agent.n = own_n
        if "tail" in agent.knobs and own_tail is not None:
            agent.knobs["tail"] = own_tail
    return out
