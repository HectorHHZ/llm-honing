"""Offline check of every fault instance on a recorded clean run (no game, no model): what each would change, when.

  python3 -B offline_faults.py <recorded run> [--max N] [--json OUT]

The pinned harness is driven through the recorded observations with the recorded choices (replay_env.ReplayEnv).
At every decision:
  harness instances   the faulted request is rendered next to the clean one exactly as collect.py would (on a
                      copy of the agent or by editing the rendered text); reported: the trigger decision, how many
                      decisions differ, which parts / options differ at the trigger, and the decision count with an
                      enemy on screen (persistent and one-shot forms);
  controller instances the controller filter transforms the windows the harness asks for (the recording itself is
                      clean, so only the transformation is checked, not its effect on the game): the trigger
                      decision, the decisions changed, the asked and sent keys at the trigger and the executed option.
The clean harness state is checked to stay untouched by the harness faults (the live render after a faulted
render must still equal the logged request).
2026-11-26: runs with a scripted start prefix are replayed with the prefix first (replay_check.prefix_of /
starts.play_prefix) and the faults are applied from the first model decision after it, as collect.py does.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import inspect

import faults
import runtime
import starts
from replay_check import dumps, layout, prefix_of
from replay_env import ReplayEnv, rows

from core.laya_multi import build_request


def diff_summary(clean: dict, altered: dict) -> dict:
    lines_c = [l for l in clean["parts"]["essential"]]  # noqa: C416, E741
    lines_a = [l for l in altered["parts"]["essential"]]  # noqa: C416, E741
    out = {
        "state_lines_changed": [i for i, (a, b) in enumerate(zip(lines_c, lines_a)) if a != b]  # noqa: B905
        + ([f"count {len(lines_c)}->{len(lines_a)}"] if len(lines_c) != len(lines_a) else [])
    }
    for q in clean["questions"]:
        ca, cc = altered["questions"][q]["criteria"], clean["questions"][q]["criteria"]
        out[f"{q}_options_changed"] = [o for o in cc if ca.get(o) != cc[o]]
    return out


def run_harness(run: Path, instance: str, form: str, package: str, limit: int | None) -> dict:
    info = layout(run)
    decisions = rows(info["session"] / "decisions.jsonl")
    agent_mod = runtime.harness_agent(package)  # Laya's tokenizer, or refuse
    env = ReplayEnv(info["session"])
    agent = agent_mod.Agent(env, dict(info["knobs"]))
    agent.boot()
    prefix, frames = prefix_of(info, decisions)
    if prefix:
        starts.play_prefix(agent, env, prefix, frames=frames)
        decisions = [r for r in decisions if r.get("control_policy") != "prefix"]
    fault = faults.make("harness", instance, form, package)
    first = None
    changed, with_enemy, checked = [], 0, 0
    for record in decisions[:limit]:
        k = record["decision"]
        ident = env.obs["frame"]["id"]
        ask = agent.ask()
        clean = build_request(ident, ask["essential"], ask["questions"])
        if dumps(clean) != dumps(record.get("clean_request") or record["request"]):
            raise SystemExit(f"{run.name} d{k}: the clean render differs from the log (harness or recording mismatch)")
        altered, rec = fault.render(agent, ask, clean, k, build_request)
        after = agent.ask()  # the harness state must be untouched by the faulted render
        if dumps(build_request(ident, after["essential"], after["questions"])) != dumps(clean):
            raise SystemExit(f"{run.name} d{k}: the faulted render changed the clean harness state")
        agent.forecasts = ask["forecasts"]
        with_enemy += bool(ask["facts"].get("enemies"))
        if rec and rec["changed"]:
            changed.append(k)
            if first is None:
                first = {
                    "decision": k,
                    "questions": rec["questions"],
                    **diff_summary(clean, altered),
                    "clean_excerpt": [l[:220] for l in clean["parts"]["essential"][2:3]],  # noqa: E741
                    "altered_excerpt": [l[:220] for l in altered["parts"]["essential"][2:3]],  # noqa: E741
                }
                if instance.startswith("corrupt"):
                    q = "move" if "jump" in instance else "attack"
                    first["clean_options"] = clean["questions"][q]["criteria"]
                    first["altered_options"] = altered["questions"][q]["criteria"]
        agent.act(record["choice"]["move"], record["choice"]["attack"], k)
        checked += 1
    return {
        "instance": fault.instance,
        "form": form,
        "decisions": checked,
        "trigger": first,
        "changed_decisions": len(changed),
        "first_changed": changed[:12],
        "decisions_with_enemy_on_screen": with_enemy,
    }


def run_controller(run: Path, instance: str, form: str, package: str, limit: int | None) -> dict:
    info = layout(run)
    decisions = rows(info["session"] / "decisions.jsonl")
    agent_mod = runtime.harness_agent(package)  # Laya's tokenizer, or refuse
    keys_mod = importlib.import_module(f"{package}.keys")
    env = ReplayEnv(info["session"])
    agent = agent_mod.Agent(env, dict(info["knobs"]))
    agent.boot()
    prefix, frames = prefix_of(info, decisions)
    if prefix:
        starts.play_prefix(agent, env, prefix, frames=frames)
        decisions = [r for r in decisions if r.get("control_policy") != "prefix"]
    fault = faults.make("controller", instance, form, package)
    trigger, changed, checked = None, [], 0
    tail = "tail" in inspect.signature(keys_mod.bridge_windows).parameters and "tail" in agent.knobs
    for record in decisions[:limit]:
        k = record["decision"]
        agent.ask()
        move, attack = record["choice"]["move"], record["choice"]["attack"]
        fault.on_selection(env, move, attack, k, {})
        extra = {"tail": agent.knobs["tail"]} if tail else {}
        fault.plan_windows(keys_mod.bridge_windows(move, attack, agent.n, agent.jump_down, agent.rising, **extra)[0])
        start = len(env.sent)
        agent.act(move, attack, k)
        rec = fault.after_decision(env, k)
        sent = env.sent[start:]
        asked = env.asked[start:]
        if rec and rec["changed"]:
            changed.append(k)
            if trigger is None:
                trigger = {
                    "decision": k,
                    "selected": rec["selected"],
                    "executed": rec["executed"],
                    "questions": rec["questions"],
                    "asked": asked,
                    "sent": sent,
                    "asked_text": [faults.window_text(w) + f" x{w['frames']}" for w in asked],
                    "sent_text": [faults.window_text(w) + f" x{w['frames']}" for w in sent],
                }
        checked += 1
    return {
        "instance": fault.instance,
        "form": form,
        "decisions": checked,
        "trigger": trigger,
        "changed_decisions": len(changed),
        "first_changed": changed[:12],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--package", default="harness.v9_keys")
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    report = {"run": a.run.name, "harness": [], "controller": []}
    for instance in faults.HARNESS:
        for form in faults.FORMS:
            report["harness"].append(run_harness(a.run, instance, form, a.package, a.max))
    for instance in faults.CONTROLLER:
        for form in faults.FORMS:
            report["controller"].append(run_controller(a.run, instance, form, a.package, a.max))
    text = json.dumps(report, indent=1)
    if a.json:
        a.json.write_text(text + "\n")
    for part in ("harness", "controller"):
        for item in report[part]:
            t = item["trigger"] or {}
            print(
                f"{item['instance']:24s} {item['form']:10s} trigger d{t.get('decision')} changed {item['changed_decisions']}/"
                f"{item['decisions']} first {item['first_changed'][:8]}"
            )


if __name__ == "__main__":
    main()
