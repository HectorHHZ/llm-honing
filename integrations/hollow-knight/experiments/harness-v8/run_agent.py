# fmt: off
# isort: skip_file
# ruff: noqa
"""Run one Hollow Knight episode: harness (perception -> facts -> options) + decision policy + executor.

  python3 run_agent.py --harness v1 --out <run_dir> [--policy laya|fixed:<option>|random|first]
                       [--max-decisions N] [--max-frames F] [--seed S]

policy `laya`   : the frozen Laya service scores the offered options; its argmax is executed exactly
                  (response guard from core/laya.py; a rejection stops the episode, nothing is executed).
policy `fixed:X`: control — always take option X when it is offered, else the first offered option.
policy `random` : control — uniform random among the offered options.
Logs (in <run_dir>/session): decisions.jsonl (request, raw response file, choice, probabilities, facts),
windows.jsonl (every native window: action, frames, decision index, why), skills.jsonl (result per skill).
The debug trace is read only by EpisodeWatch for progress printing and to end the episode at Level2.

Added 2026-11-17 (round 2, additive; the defaults behave as before): `--goal SCENE:X` ends the episode when the
debug trace shows the knight in SCENE at x >= X (Level2 has no exit object), `--through` keeps the episode going
across a scene change (one continuous run Level1 -> Level2). Both are evaluation-side only.

Changed 2026-11-20 (after the runs; nothing else changed): the exit status tells a failed run from a finished one.
0: the episode ended at the goal, at a scene change or at a decision or frame budget. 1: the run failed, either by an
error (for example HONING_HK_GAME is not set) or by a guard rejection. A run stopped by SIGTERM exits with 143, by
Ctrl-C with 130. summary.json is written in every case, and its `status` says how the run ended.
"""
from __future__ import annotations

import argparse
import importlib
import json
import random
import signal
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from core.env import HKEnv, encode_video          # noqa: E402
from core.evalwatch import EpisodeWatch            # noqa: E402
from core.laya import GuardRejection, LayaClient, build_request, checked_choice  # noqa: E402


def exit_status(error: BaseException) -> int:
    """Exit status of a run that `error` stopped: a stop signal keeps its usual status, any other error gives 1."""
    if isinstance(error, SystemExit) and isinstance(error.code, int) and error.code != 0:
        return error.code                            # SIGTERM arrives as SystemExit(143), see main()
    if isinstance(error, KeyboardInterrupt):
        return 130
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", default="v1")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--policy", default="laya")
    ap.add_argument("--max-decisions", type=int, default=40)
    ap.add_argument("--max-frames", type=int, default=9000, help="render-frame budget for the episode")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-capture", action="store_true",
                    help="do not save per-frame PNGs (faster; note the visible grid then has no caption-box unknown cells)")
    ap.add_argument("--menu-order", default="harness",
                    help="control for the model's position bias: harness (as ranked) | reversed | shuffled:<seed>")
    ap.add_argument("--scene", default="Level1")
    ap.add_argument("--goal", default=None,
                    help="evaluation-only end of the episode, SCENE:X (e.g. Level2:124): stop once the debug trace shows the "
                         "knight in SCENE at x >= X. Level2 has no exit, so its end has to be told this way. Never seen by the harness.")
    ap.add_argument("--through", action="store_true",
                    help="do not stop at a scene change (for one continuous run Level1 -> Level2; end it with --goal)")
    args = ap.parse_args()
    goal = None
    if args.goal:
        goal_scene, goal_x = args.goal.split(":")
        goal = (goal_scene, float(goal_x))
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(143))
    rng = random.Random(args.seed)
    order_rng = random.Random(int(args.menu_order.split(":")[1]) if ":" in args.menu_order else 0)

    harness = importlib.import_module(f"harness.{args.harness}.agent")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = HKEnv(out, scene=args.scene, capture=not args.no_capture)
    client = LayaClient() if args.policy == "laya" else None
    manifest = {"harness": args.harness, "policy": args.policy, "menu_order": args.menu_order, "max_decisions": args.max_decisions,
                "capture": not args.no_capture,
                "max_frames": args.max_frames, "seed": args.seed, "scene": args.scene,
                "goal": args.goal, "through": args.through,
                "started": time.strftime("%Y-%m-%d %H:%M:%S %z")}
    if client is not None:
        manifest["model_health"] = client.health()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    # keep the exact harness source used for this run next to its trace
    import shutil
    shutil.copytree(HERE / "harness" / args.harness, out / "harness_snapshot" / args.harness,
                    ignore=shutil.ignore_patterns("__pycache__"))
    status = "running"
    exit_code = 0
    summary = {}
    try:
        env.start()
        session = env.session
        (session / "responses").mkdir()
        watch = EpisodeWatch(session)
        agent = harness.Agent(env)
        agent.boot()
        for index in range(1, args.max_decisions + 1):
            menu = agent.menu()                      # {"essential": [...], "instructions": str, "options": {name: desc}, "facts": {...}}
            options = menu["options"]
            if len(options) < 3:
                raise RuntimeError("harness offered fewer than 3 options")
            # control: same options and texts, different order (the harness order is still logged)
            if args.menu_order == "reversed":
                options = dict(reversed(list(options.items())))
            elif args.menu_order.startswith("shuffled"):
                items = list(options.items())
                order_rng.shuffle(items)
                options = dict(items)
            record = {"decision": index, "obs_id": env.obs["frame"]["id"], "tick": env.obs["frame"]["tick"],
                      "facts": menu.get("facts"), "offered": list(options), "signature": menu.get("signature"),
                      "loop_guard": menu.get("loop_guard"), "harness_order": menu.get("harness_order"),
                      "withheld": menu.get("withheld")}
            request = build_request(env.obs["frame"]["id"], menu["essential"], menu["instructions"], options)
            record["request"] = request
            if args.policy == "laya":
                raw_file = f"responses/{index:04d}.raw"
                record["responseRawFile"] = raw_file
                try:
                    response, seconds = client.decide(request, raw_path=session / raw_file)
                    choice, probability, probabilities = checked_choice(response, request)
                except GuardRejection as error:
                    record["rejected"] = str(error)
                    with (session / "decisions.jsonl").open("a") as log:
                        log.write(json.dumps(record) + "\n")
                    status = f"guard rejection at decision {index}: {error}"
                    exit_code = 1
                    print(status, flush=True)
                    break
                record.update({"choice": choice, "probability": probability, "probabilities": probabilities,
                               "seconds": seconds, "usage": response.get("modelResult", {}).get("usage"),
                               "tokenAudit": response.get("tokenAudit")})
                label = f"Laya chose {choice}"
            else:
                if args.policy.startswith("fixed:"):
                    # priority list "a|b?substring|c": take the first entry that is offered; with
                    # "?substring" only if the offered description contains that text
                    choice = None
                    for entry in args.policy.split(":", 1)[1].split("|"):
                        want, _, needle = entry.partition("?")
                        if want in options and (not needle or needle in options[want]):
                            choice = want
                            break
                    choice = choice or next(iter(options))
                elif args.policy.startswith("script:"):
                    # scripted option names in order ("a,b,c"); an entry that is not offered falls back
                    # to the first offered option; after the script the last entry is repeated
                    seq = args.policy.split(":", 1)[1].split(",")
                    want = seq[min(index - 1, len(seq) - 1)]
                    choice = want if want in options else next(iter(options))
                elif args.policy == "random":
                    choice = rng.choice(list(options))
                elif args.policy == "first":
                    choice = next(iter(options))
                else:
                    raise ValueError("unknown policy")
                probability, probabilities = 0.0, {}
                record.update({"choice": choice, "probability": None, "control_policy": args.policy})
                label = f"[control {args.policy.split(':')[0]}] {choice}"
            with (session / "decisions.jsonl").open("a") as log:
                log.write(json.dumps(record) + "\n")
            result = agent.execute(choice, index, label, probability)
            watch.poll()
            last = watch.last or {}
            row = {"decision": index, "skill": choice, **result,
                   "eval_only": {"x": last.get("x"), "y": last.get("y"), "health": last.get("health"), "scene": last.get("scene")}}
            extra = getattr(agent, "run_log", None)          # harness-side counters (v8+: reflex strikes, fade guard)
            if callable(extra):
                row["harness_log"] = extra()
            with (session / "skills.jsonl").open("a") as log:
                log.write(json.dumps(row) + "\n")
            pr = f"p={probability:.3f}" if args.policy == "laya" else "ctrl"
            print(f"{index:3d} {choice:22s} {pr} -> {result['reason'][:90]} | windows={result['windows']} "
                  f"| eval x={last.get('x', 0):.1f} y={last.get('y', 0):.1f} hp={last.get('health')} {last.get('scene')}", flush=True)
            if goal is not None and last.get("scene") == goal[0] and (last.get("x") or -1e9) >= goal[1]:
                status = f"goal reached: {goal[0]} x >= {goal[1]:g}"
                break
            if last.get("scene") not in (None, args.scene) and not args.through:
                status = f"scene changed to {last.get('scene')}"
                break
            if env.frames >= args.max_frames:
                status = "frame budget reached"
                break
        else:
            status = "decision budget reached"
    except BaseException as error:
        status = f"error: {type(error).__name__}: {error}"
        exit_code = exit_status(error)
        traceback.print_exc()
    finally:
        try:
            watch = EpisodeWatch(env.session)
            summary = watch.summary()
        except Exception:
            pass
        close = env.close()
        summary.update({"status": status, "close": close})
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary), flush=True)
        if not args.no_video and (env.session / "frames").is_dir():
            try:
                encode_video(env.session, out / "gameplay.mp4")
            except Exception as error:
                print("video encoding failed:", error, flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
