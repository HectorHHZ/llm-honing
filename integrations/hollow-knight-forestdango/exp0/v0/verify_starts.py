"""Reset groups of the full-HK Exp 0 v0: trace the reference route and verify the start states (no model).

  python3 -B verify_starts.py trace --plan plan.json --script starts/route_to_upper_left_ledge.txt --out DIR --display N
      Boot a new game on the plan's build, play the whole script through the pinned harness exactly as a start prefix
      is played (starts.play_prefix: 9 frames per step, 1-frame tail, the harness asks before each step), and write
      DIR/route.json: after every step the knight's debug position (evaluation side), whether it stands, health,
      bridge frame and scene. Used to choose where starts are cut and where their waypoints lie.

  python3 -B verify_starts.py check --plan plan.json --start ID --out DIR --display N
      Boot + the start's prefix, exactly as collect.py does it, then save the harness state snapshot and render the
      clean first request (nothing is sent to any model). DIR/handover.json records the debug row at the handover,
      the SHA-256 of the state snapshot and of the first request, the bridge frame / ids, and the bridge logs'
      digests. Two boots of the same start must give identical records (`compare`).

  python3 -B verify_starts.py compare A/handover.json B/handover.json
      Exit 0 when the two handovers are identical (state, request, debug row and the bridge's logs up to the handover).

Processes: one Xvfb on the given display (:350-:379) and one game process per call, both started and stopped by the
env (core/env_keys.KeysEnv) through their handles. The decision services are not touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "_shared" / "exp0"))
import runtime
import starts
import state_codec

from core.evalwatch import EpisodeWatch
from core.laya_multi import build_request

DISPLAYS = range(350, 380)
DEBUG_KEYS = (
    "frame",
    "scene",
    "x",
    "y",
    "vx",
    "vy",
    "grounded",
    "health",
    "soul",
    "deaths",
    "facing",
    "hazardRespawns",
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_env(plan: dict, out: Path, display: int):
    from core.env_keys import KeysEnv

    game = plan["release"]["game"]
    options = {**game["bridge_options"], "seed": str(plan.get("bridge_seed", 42))}
    if game.get("player"):
        options["player"] = game["player"]
    if display not in DISPLAYS:
        raise SystemExit("display must be within :350-:379")
    return KeysEnv(
        out, scene="new", capture=False, options=options, display=display, bridge_env=game.get("bridge_env") or {}
    )


def harness(plan: dict):
    cfg = plan["release"]["harness"]
    return cfg, runtime.harness_agent(cfg["package"])  # Laya's tokenizer, or refuse


def trace(a) -> None:
    plan = json.loads(a.plan.read_text())
    steps = starts.read_script(a.script)
    cfg, agent_mod = harness(plan)
    out = a.out.resolve()
    env = make_env(plan, out, a.display)
    rows = []
    t0 = time.monotonic()
    try:
        env.start()
        watch = EpisodeWatch(env.session)
        watch.poll()
        agent = agent_mod.Agent(env, dict(cfg.get("knobs") or {}))
        agent.boot()
        first = watch.last or {}
        rows.append({"step": 0, **{k: first.get(k) for k in DEBUG_KEYS}, "harness_grounded": agent.grounded})

        def on_step(rec):
            watch.poll()
            last = watch.last or {}
            rows.append(
                {
                    "step": rec["step"],
                    "move": rec["move"],
                    "attack": rec["attack"],
                    "result": rec["result"],
                    **{k: last.get(k) for k in DEBUG_KEYS},
                    "harness_grounded": agent.grounded,
                }
            )

        starts.play_prefix(agent, env, steps, on_step=on_step)
    finally:
        close = env.close()
    (out / "route.json").write_text(
        json.dumps(
            {
                "script": str(a.script),
                "steps": len(steps),
                "script_sha256": starts.steps_sha256(steps),
                "rows": rows,
                "close": close,
                "seconds": round(time.monotonic() - t0, 1),
            },
            indent=1,
        )
        + "\n"
    )
    for r in rows:
        print(
            f"{r['step']:4d} {r.get('move', ''):17s} x={r['x']:.2f} y={r['y']:.2f} g={r['grounded']} hp={r['health']} f={r['frame']}"
        )


def check(a) -> None:
    plan = json.loads(a.plan.read_text())
    start = plan["starts"][a.start]
    steps = starts.start_steps(HERE, start)
    if start.get("prefix_sha256") and starts.steps_sha256(steps) != start["prefix_sha256"]:
        raise SystemExit("the prefix differs from the plan's digest")
    cfg, agent_mod = harness(plan)
    package, name = cfg["package"], Path(cfg["dir"]).name
    out = a.out.resolve()
    env = make_env(plan, out, a.display)
    t0 = time.monotonic()
    try:
        env.start()
        watch = EpisodeWatch(env.session)
        watch.poll()
        agent = agent_mod.Agent(env, dict(cfg.get("knobs") or {}))
        knobs = dict(agent.knobs)
        agent.boot()
        if steps:
            starts.play_prefix(agent, env, steps, frames=int(start.get("prefix_frames") or starts.PREFIX_FRAMES))
        if agent.knobs != knobs:
            raise RuntimeError("the harness knobs differ after the prefix")
        if agent.view.me is None and agent.knobs.get("absent") == "wait":
            agent._wait_for_knight()
        watch.poll()
        last = watch.last or {}
        ident = env.obs["frame"]["id"]
        state = state_codec.snapshot(agent, package, name, ident)
        ask = agent.ask()
        request = build_request(ident, ask["essential"], ask["questions"])
        session = env.session
        logs = {}
        for fname in ("observations.jsonl", "visible_observations.jsonl", "bridge_windows.jsonl", "frames.jsonl"):
            path = session / fname
            if path.exists():
                data = path.read_bytes()
                logs[fname] = {"lines": data.count(b"\n"), "sha256": sha(data)}
        # windows.jsonl carries wall times; compare it without them
        wpath = session / "windows.jsonl"
        win = (
            [json.loads(l) for l in wpath.read_text().splitlines() if l.strip()]  # noqa: E741
            if wpath.exists()
            else []
        )
        for w in win:
            w.pop("t_wall", None)
        logs["windows.jsonl(without t_wall)"] = {"lines": len(win), "sha256": sha(json.dumps(win).encode())}
        record = {
            "start": a.start,
            "prefix_steps": len(steps),
            "prefix_sha256": starts.steps_sha256(steps),
            "handover_debug": {k: last.get(k) for k in DEBUG_KEYS},
            "handover_ids": {
                "obs_id": ident,
                "tick": env.obs["frame"]["tick"],
                "bridge_obs_id": env.raw["frame"]["id"],
                "bridge_tick": env.raw["frame"]["tick"],
            },
            "harness": {"grounded": agent.grounded, "facts": ask["facts"]},
            "state_sha256": sha(json.dumps(state, allow_nan=False).encode()),
            "first_request_sha256": sha(json.dumps(request, allow_nan=False).encode()),
            "first_request_state_text": request["parts"]["essential"][1:],
            "logs_up_to_handover": logs,
        }
        (out / "state.json").write_text(json.dumps(state, allow_nan=False) + "\n")
        (out / "first_request.json").write_text(json.dumps(request, allow_nan=False) + "\n")
    finally:
        close = env.close()
    record.update(close=close, seconds=round(time.monotonic() - t0, 1))
    (out / "handover.json").write_text(json.dumps(record, indent=1) + "\n")
    print(
        json.dumps(
            {
                k: record[k]
                for k in ("start", "prefix_steps", "handover_debug", "state_sha256", "first_request_sha256", "seconds")
            }
        )
    )


IDENTITY = (
    "prefix_sha256",
    "handover_debug",
    "handover_ids",
    "harness",
    "state_sha256",
    "first_request_sha256",
    "first_request_state_text",
    "logs_up_to_handover",
)


def frames_rows(session: Path) -> list[dict] | None:
    path = session / "frames.jsonl"
    if not path.exists():
        return None
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            row.pop("wall", None)  # wall-clock time
            row.pop("nGo", None)  # count of all game objects: particle recycling (native_replay.py)
            out.append(row)
    return out


def compare(a) -> None:
    """Identical handover: state snapshot, first request, debug row, ids and the bridge's logs up to the handover
    (byte-identical; the per-frame log row by row without its wall-clock field and object count)."""
    pa, pb = Path(a.a), Path(a.b)
    x, y = json.loads(pa.read_text()), json.loads(pb.read_text())
    diff = [k for k in IDENTITY if k != "logs_up_to_handover" and x.get(k) != y.get(k)]
    lx, ly = dict(x["logs_up_to_handover"]), dict(y["logs_up_to_handover"])
    fx, fy = lx.pop("frames.jsonl", None), ly.pop("frames.jsonl", None)
    if lx != ly:
        diff.append("logs_up_to_handover")
    frames = None
    if fx is not None or fy is not None:
        rx, ry = frames_rows(pa.parent / "session"), frames_rows(pb.parent / "session")
        frames = {"rows": [len(rx or []), len(ry or [])], "identical": rx == ry}
        if rx != ry:
            diff.append("frames.jsonl (without wall, nGo)")
    print(json.dumps({"start": x.get("start"), "identical": not diff, "differs": diff, "frames": frames}))
    sys.exit(1 if diff else 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("trace")
    t.add_argument("--plan", type=Path, default=HERE / "plan.json")
    t.add_argument("--script", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--display", type=int, required=True)
    c = sub.add_parser("check")
    c.add_argument("--plan", type=Path, default=HERE / "plan.json")
    c.add_argument("--start", required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--display", type=int, required=True)
    m = sub.add_parser("compare")
    m.add_argument("a")
    m.add_argument("b")
    a = ap.parse_args()
    {"trace": trace, "check": check, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    main()
