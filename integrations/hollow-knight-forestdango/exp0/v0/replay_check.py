"""Strict harness replay and exact re-render check of one recorded raw-key run (no game, no model, no network).

  python3 -B replay_check.py <run> [--package harness.v9_keys] [--max N] [--json OUT]

<run> is either a Demo-session run_keys.py run (<run>/manifest.json, <run>/session/) or an Exp 0 attempt of this
collector (<run>/extra/manifest.json, <run>/extra/session/). The harness snapshot named by --package (default: the
pinned copy code/exp0/hk_v9/harness/v9_keys) is driven through the recorded observations with the recorded choices
(replay_env.ReplayEnv). Per decision it checks:
  1. live render: the request the harness renders now equals the logged clean request byte for byte
     (json.dumps of both, keys in order);
  2. snapshot re-render: the state saved before the decision (state_codec.snapshot, through a JSON round trip; for
     an Exp 0 attempt the state logged during the run) re-renders the same bytes;
  3. execution: every bridge window the harness asks for equals the logged requested window (strict).
For an Exp 0 attempt it also checks that the logged state equals the snapshot taken during this replay.

Start prefixes (2026-11-26): a run that began with a scripted start prefix (an Exp 0 attempt of a reset group with a
start, or a run_keys2.py run with --prefix) is replayed with the prefix first, at its own cadence, exactly as it was
played (starts.play_prefix); the checks above then cover the model decisions after it. The knobs of a run_keys2.py
prefix run are taken from its `knobs` (its `knobs_effective` was written while the prefix had changed the tail).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import runtime
import starts
import state_codec
from replay_env import ReplayEnv, ReplayMismatch, rows

from core.laya_multi import build_request


def dumps(value) -> bytes:
    """The bytes core.laya.LayaClient.decide sends for a request."""
    return json.dumps(value, allow_nan=False).encode()


def layout(run: Path) -> dict:
    run = Path(run)
    if (run / "extra" / "manifest.json").exists():
        raw = run / "extra"
        manifest = json.loads((raw / "manifest.json").read_text())
        return {
            "kind": "exp0",
            "raw": raw,
            "session": raw / "session",
            "manifest": manifest,
            "knobs": manifest["harness"]["knobs_effective"],
        }
    manifest = json.loads((run / "manifest.json").read_text())
    knobs = (
        (manifest.get("knobs") or {})
        if manifest.get("prefix_steps")
        else (manifest.get("knobs_effective") or manifest.get("knobs") or {})
    )
    return {"kind": "run_keys", "raw": run, "session": run / "session", "manifest": manifest, "knobs": knobs}


def prefix_of(info: dict, decisions: list[dict]) -> tuple[list[tuple[str, str]], int | None]:
    """(steps, frames per step) of the run's start prefix; ([], None) without one."""
    if info["kind"] == "exp0":
        start = info["manifest"].get("start") or {}
        if start.get("script"):
            return starts.parse(start["script"]), int(start.get("prefix_frames") or 0) or None
        return [], None
    steps = [(r["choice"]["move"], r["choice"]["attack"]) for r in decisions if r.get("control_policy") == "prefix"]
    return steps, (int(info["manifest"].get("prefix_frames") or 0) or None)


def check(run: Path, package: str = "harness.v9_keys", max_decisions: int | None = None) -> dict:
    info = layout(run)
    session = info["session"]
    decisions = rows(session / "decisions.jsonl")
    states = {r["decision"]: r["state"] for r in rows(session / "states.jsonl")} if info["kind"] == "exp0" else {}
    agent_mod = runtime.harness_agent(package)  # Laya's tokenizer, or refuse
    env = ReplayEnv(session)
    agent = agent_mod.Agent(env, dict(info["knobs"]))
    agent.boot()
    started = time.perf_counter()
    prefix, prefix_frames = prefix_of(info, decisions)
    report = {
        "run": Path(run).name,
        "kind": info["kind"],
        "package": package,
        "decisions_logged": len(decisions),
        "checked": 0,
        "live_render_exact": 0,
        "snapshot_rerender_exact": 0,
        "logged_state_equal": 0,
        "mismatches": [],
        "windows_replayed": 0,
        "prefix_steps": len(prefix),
    }
    if prefix:
        try:
            starts.play_prefix(agent, env, prefix, frames=prefix_frames)
        except ReplayMismatch as error:
            report["mismatches"].append({"decision": None, "what": f"prefix execution: {error}"})
            report.update(
                windows_replayed=env.k,
                windows_logged=len(env.records),
                ok=False,
                seconds=round(time.perf_counter() - started, 2),
            )
            return report
        decisions = [r for r in decisions if r.get("control_policy") != "prefix"]
    for record in decisions:
        k = record["decision"]
        if max_decisions is not None and report["checked"] >= max_decisions:
            break
        if "choice" not in record:  # a rejected (unexecuted) last decision
            break
        if agent.view.me is None and agent.knobs.get("absent") == "wait":
            agent._wait_for_knight()  # as collect.py does before its snapshot (ask() would run it first)
        ident = env.obs["frame"]["id"]
        snap = state_codec.snapshot(agent, package, package.rsplit(".", 1)[-1], ident)
        snap = json.loads(json.dumps(snap, allow_nan=False))
        ask = agent.ask()
        live = build_request(ident, ask["essential"], ask["questions"])
        logged_clean = record.get("clean_request") or record["request"]
        ok_live = dumps(live) == dumps(logged_clean)
        rerendered = state_codec.rerender(snap, package, build_request)
        ok_snap = dumps(rerendered) == dumps(live)
        if k in states:
            ok_state = json.dumps(states[k], sort_keys=False) == json.dumps(snap, sort_keys=False)
            report["logged_state_equal"] += ok_state
            if not ok_state:
                report["mismatches"].append({"decision": k, "what": "logged state differs from the replay snapshot"})
            logged_rerender = state_codec.rerender(states[k], package, build_request)
            if dumps(logged_rerender) != dumps(logged_clean):
                report["mismatches"].append(
                    {"decision": k, "what": "logged state does not re-render the logged request"}
                )
        report["checked"] += 1
        report["live_render_exact"] += ok_live
        report["snapshot_rerender_exact"] += ok_snap
        if not ok_live:
            report["mismatches"].append({"decision": k, "what": "live render differs from the logged request"})
        if not ok_snap:
            report["mismatches"].append({"decision": k, "what": "snapshot re-render differs from the live render"})
        choice = record["choice"]
        move, attack = (choice["move"], choice["attack"]) if "move" in choice else agent_mod.Agent.compose(choice)
        try:
            agent.act(move, attack, k, record.get("label", ""))
        except ReplayMismatch as error:
            report["mismatches"].append({"decision": k, "what": f"execution: {error}"})
            break
        if len(report["mismatches"]) >= 20:
            break
    report["windows_replayed"] = env.k
    report["windows_logged"] = len(env.records)
    report["seconds"] = round(time.perf_counter() - started, 2)
    report["ok"] = (
        not report["mismatches"]
        and report["checked"] == report["live_render_exact"] == report["snapshot_rerender_exact"]
        and report["checked"] > 0
    )
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--package", default="harness.v9_keys")
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    report = check(a.run, a.package, a.max)
    text = json.dumps(report, indent=1)
    if a.json:
        a.json.write_text(text + "\n")
    print(text)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
