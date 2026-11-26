"""Collect ONE complete Exp 0 v0 episode of the full Hollow Knight (ForestDango) under a raw-key harness (v9_keys family).

  python3 -B collect.py --out <new attempt dir> --task kp-smoke-d60 --seed S
                        --condition no_injection|harness|weights|controller [--fault-instance ID] [--fault-form persistent|once]
                        --display N (350-379) [--plan plan.json] [--label TEXT]

An adapted copy of code/hk-hitl/run_keys.py (read-only original, the Demo session owns it; reference copy in
source_ref/, whose absolute local paths are written as <workspace> / <conda-root>). What is the same: the pinned harness snapshot (plan `release.harness`, default harness/v9_keys, a copy of
the frozen v9_keys) builds one request with two questions (`move`, `attack`) per decision; Laya scores both; the
argmax of each question is executed exactly as the keys of the next window; the response guard (core/laya_multi.py)
runs on every question before anything is executed, and a rejection ends the episode with nothing executed; the
game's debug trace is read only by the evaluation side (core/evalwatch.py), never by the harness or the model.
What is new:
  * the episode definition (start, win, loss, caps) comes from the plan's task;
  * one fault can be switched on (faults.py; persistent or one-shot);
  * before each decision the harness state is saved exactly (state_codec.py; session/states.jsonl), so the request
    can be re-rendered byte for byte without the game (probes, audit, replay_check.py);
  * the exact bytes sent to Laya are kept (session/requests/NNNN.json) next to the raw reply bytes (responses/);
  * every window keeps the keys the harness asked for and the keys actually sent (core/env_keys.py);
  * the game build, the bridge options and extra bridge variables are plan settings, so a faster build can be
    dropped in (`release.game`);
  * (2026-11-26, Exp 0 v0 HK collection) `--group ID`: the reset group of the plan names its start (a scripted key
    prefix from a new game, starts.py, played exactly as run_keys2.py --prefix --prefix-frames 9 plays it) and its
    waypoint; the prefix is the reset procedure (its windows are tagged, the contract rows start after it);
    win kind `reach_box` (the knight stands inside the group's waypoint box); before the state snapshot of a
    decision the harness's own no-key wait while the knight is not on screen is run explicitly (tagged), so that the
    snapshot re-renders the request; the frame cap counts from the first model decision; knobs_effective is a copy.
  * (2026-11-26, privacy) no absolute local path in the code: the shared tools are found next to this directory, plan
    paths are relative to the workspace (runtime.py), and Laya's tokenizer (it changes request bytes: option lengths)
    is loaded by runtime.harness_agent from $HONING_LAYA_TOKENIZER or the public checkpoint; without it the collector
    refuses to run.
Raw record in <out>/extra/ (manifest.json, launch.json, harness_snapshot/, session/, result.json, error.json);
afterwards to_contract.py writes the common contract files (integrations/_shared/exp0/CONTRACT.md, 0.2.0) into <out>.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "_shared" / "exp0"))
import faults
import runtime
import starts
import state_codec

from core.evalwatch import EpisodeWatch
from core.laya import GuardRejection, LayaClient
from core.laya_multi import build_request, checked_answers

FORMAT = "hk-forestdango-exp0-v9-episode-v0"
CONDITIONS = ("no_injection", "harness", "weights", "controller")
PLAN = HERE / "plan.json"
DISPLAYS = range(350, 380)  # this collector's Xvfb displays (the Demo session uses :301-:309)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_pins(harness_dir: Path) -> dict:
    from contract import fingerprint_tree

    return {
        "harness": {"dir": str(harness_dir.relative_to(HERE)), "fingerprint": fingerprint_tree(harness_dir)},
        "controller": {
            name: sha256_file(HERE / "core" / name)
            for name in ("env_forestdango.py", "env_keys.py", "env.py", "laya.py", "laya_multi.py")
        },
        "faults_sha256": sha256_file(HERE / "faults.py"),
        "state_codec_sha256": sha256_file(HERE / "state_codec.py"),
        "starts_sha256": sha256_file(HERE / "starts.py"),
        "runtime_sha256": sha256_file(HERE / "runtime.py"),
        "collector_sha256": sha256_file(Path(__file__)),
        "evalwatch_sha256": sha256_file(HERE / "core" / "evalwatch.py"),
    }


def display_free(n: int) -> bool:
    if Path(f"/tmp/.X11-unix/X{n}").exists() or Path(f"/tmp/.X{n}-lock").exists():
        return False
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    return not any(f"Xvfb :{n} " in line + " " for line in out.splitlines())


class Judge:
    """Evaluation only: the task's native terminal events, read from the game's debug trace (observations.jsonl).

    Win kinds: reach_x (x >= min_x in the scene), reach_box (the knight's feet inside the box, standing: debug
    `grounded`; the box is the task's or, with `"box": "group"`, the reset group's waypoint), boss_death. Loss: the
    knight's first death. Leaving the start scene ends the episode (other terminal) when the task says so."""

    def __init__(self, task: dict, first_row: dict, waypoint: dict | None = None):
        self.win = dict(task["win"]) if task.get("win") else None
        if self.win and self.win.get("kind") == "reach_box" and self.win.get("box") == "group":
            if not waypoint:
                raise ValueError("the task's win is the reset group's waypoint, but the group has none")
            self.win["box"] = waypoint
        self.start_scene = first_row.get("scene")
        self.deaths0 = first_row.get("deaths") or 0
        self.leave_ends = bool(task.get("leaving_the_scene_ends", True))
        self.best_x = first_row.get("x")

    def check(self, rows: list[dict]):
        for r in rows:
            if r.get("x") is not None and r.get("scene") == self.start_scene:
                self.best_x = max(self.best_x if self.best_x is not None else r["x"], r["x"])
            if self.win:
                kind = self.win["kind"]
                if kind == "reach_box":
                    b = self.win["box"]
                    x, y = r.get("x"), r.get("y")
                    if (
                        r.get("scene") == b["scene"]
                        and x is not None
                        and y is not None
                        and r.get("grounded")
                        and b["x"][0] <= x <= b["x"][1]
                        and b["y"][0] <= y <= b["y"][1]
                    ):
                        return {
                            "outcome": "win",
                            "event": "reached_place",
                            "frame": r.get("frame"),
                            "detail": f"standing at ({x:.2f}, {y:.2f}) inside the waypoint box x {b['x']} y {b['y']} of {b['scene']}",
                        }
                elif kind == "reach_x":
                    if r.get("scene") == self.win["scene"] and (r.get("x") or -1e9) >= self.win["min_x"]:
                        return {
                            "outcome": "win",
                            "event": "reached_x",
                            "frame": r.get("frame"),
                            "detail": f"x {r.get('x'):.2f} >= {self.win['min_x']} in {r.get('scene')}",
                        }
                elif kind == "boss_death":
                    for e in r.get("enemies") or []:
                        if self.win["enemy"].lower() in str(e.get("name", "")).lower():
                            state = str(e.get("fsm", "")).split(";")[0].split("=")[-1]
                            if e.get("dead") or state in self.win["states"]:
                                return {
                                    "outcome": "win",
                                    "event": "boss_death_sequence",
                                    "frame": r.get("frame"),
                                    "detail": f"{e.get('name')} state {state!r} dead={e.get('dead')}",
                                }
                else:
                    raise ValueError(f"unknown win kind {kind}")
            if (r.get("deaths") or 0) > self.deaths0 or r.get("health") == 0 or (r.get("hero") or {}).get("dead"):
                return {
                    "outcome": "loss",
                    "event": "knight_death",
                    "frame": r.get("frame"),
                    "detail": f"health {r.get('health')}, deaths {r.get('deaths')}, scene {r.get('scene')}",
                }
            if self.leave_ends and r.get("scene") not in (None, self.start_scene):
                return {
                    "outcome": "left_area",
                    "event": "scene_change",
                    "frame": r.get("frame"),
                    "detail": f"{self.start_scene} -> {r.get('scene')}",
                }
        return None


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=1, allow_nan=False) + "\n")


def append(path: Path, row: dict) -> None:
    with path.open("a") as log:
        log.write(json.dumps(row, allow_nan=False) + "\n")


def main(argv: list[str] | None = None, env_factory=None, client_factory=None, display_check=True) -> None:
    """`env_factory` / `client_factory` replace the game and the decision service (offline.py's pre-flight on a
    recorded run only; a dataset attempt always uses core.env_keys.KeysEnv and core.laya.LayaClient)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument(
        "--seed", type=int, default=None, help="HONING_SEED of the bridge (default: the reset group's seed)"
    )
    ap.add_argument("--group", default=None, help="reset group id of the plan: its start (prefix) and waypoint")
    ap.add_argument("--condition", choices=CONDITIONS, required=True)
    ap.add_argument("--fault-instance", default=None)
    ap.add_argument("--fault-form", choices=faults.FORMS, default="persistent")
    ap.add_argument("--plan", type=Path, default=PLAN)
    ap.add_argument("--display", type=int, required=True)
    ap.add_argument("--label", default="", help="free text stored in the manifest (smoke / pilot / dataset)")
    ap.add_argument("--max-decisions", type=int, default=None, help="pilot override of the task's decision cap")
    ap.add_argument("--capture", action="store_true", help="save frames (videos only; off for dataset episodes)")
    ap.add_argument(
        "--endpoint",
        default=None,
        help="one of the plan's endpoints of the condition's role "
        "(clean: endpoints.clean or endpoints.clean_2; default: endpoints.<role>)",
    )
    args = ap.parse_args(argv)

    plan = json.loads(args.plan.read_text())
    task = plan["tasks"][args.task]
    release = plan["release"]
    group = None
    if args.group is not None:
        group = next((g for g in plan.get("groups", []) if g["id"] == args.group), None)
        if group is None:
            raise SystemExit(f"reset group {args.group} is not in the plan")
    seed = args.seed if args.seed is not None else (group or {}).get("seed")
    if seed is None:
        raise SystemExit("--seed or --group is required")
    args.seed = int(seed)
    start = plan["starts"][group["start"]] if group is not None and group.get("start") else None
    prefix = starts.start_steps(HERE, start) if start else []
    if start and start.get("prefix_sha256") and starts.steps_sha256(prefix) != start["prefix_sha256"]:
        raise SystemExit(f"the prefix of start {group['start']} differs from the plan's digest")
    caps = dict(task["caps"])
    if args.max_decisions:
        caps["decisions"] = args.max_decisions
    if args.display not in DISPLAYS:
        raise SystemExit(f"display :{args.display} is not one of this collector's (:350-:379)")
    if display_check and not display_free(args.display):
        raise SystemExit(f"display :{args.display} is in use")
    weights_role = "alternate" if args.condition == "weights" else "clean"
    endpoint = plan["endpoints"][weights_role]
    if args.endpoint:
        allowed = [u for k, u in plan["endpoints"].items() if k.split("_")[0] == weights_role]
        if args.endpoint not in allowed:
            raise SystemExit(f"--endpoint {args.endpoint} is not a {weights_role} endpoint of the plan ({allowed})")
        endpoint = args.endpoint
    expected_sha = plan[f"{weights_role}_weights_sha256"]

    run_dir = args.out.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    out = run_dir / "extra"
    out.mkdir()
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(143))
    harness_cfg = release["harness"]
    harness_dir = HERE / harness_cfg["dir"]
    package = harness_cfg["package"]
    harness_name = harness_dir.name
    fault = faults.make(args.condition, args.fault_instance, args.fault_form, package)
    client = (client_factory or LayaClient)(endpoint)
    health0 = client.health()
    meta0 = health0.get("metadata") or {}
    if meta0.get("weightsSha256") != expected_sha or meta0.get("role") != weights_role:
        write_json(
            out / "error.json",
            {"type": "ModelIdentity", "message": "decision server identity differs from the plan", "health": health0},
        )
        raise SystemExit("decision server identity differs from the plan")
    agent_mod = runtime.harness_agent(package)  # Laya's tokenizer set before any request is rendered
    keys_mod = importlib.import_module(f"{package}.keys")
    from core.env_keys import KeysEnv

    game = release["game"]
    options = {**game["bridge_options"], "seed": str(args.seed)}
    if game.get("player"):
        options["player"] = game["player"]
    env = (env_factory or KeysEnv)(
        out,
        scene=task["scene"],
        capture=args.capture,
        options=options,
        display=args.display,
        bridge_env=game.get("bridge_env") or {},
    )
    manifest = {
        "format": FORMAT,
        "label": args.label,
        "condition": args.condition,
        "task": args.task,
        "task_definition": task,
        "caps": caps,
        "seed": args.seed,
        "fault": (
            None
            if args.condition == "no_injection"
            else {
                "instance": fault.instance,
                "form": args.fault_form,
                "component": args.condition,
                "definition": faults.definition(getattr(fault, "base", fault.instance), args.fault_form),
            }
        ),
        "harness": {
            "name": harness_name,
            "package": package,
            "knobs": harness_cfg.get("knobs") or {},
            "source_record": harness_cfg.get("source_record"),
            "tokenizer": runtime.tokenizer_record(),
        },
        "group": group,
        "start": (
            None
            if start is None
            else {
                "id": group["start"],
                **start,
                "steps": len(prefix),
                "prefix_sha256_computed": starts.steps_sha256(prefix),
                "script": starts.steps_text(prefix),
            }
        ),
        "policy": "laya argmax per question (no sampling)",
        "endpoint": endpoint,
        "weights_role": weights_role,
        "expected_weights_sha256": expected_sha,
        "model_health_before": health0,
        "game": {
            "player": options.get("player", ""),
            "bridge_options": options,
            "bridge_env": game.get("bridge_env") or {},
        },
        "display": args.display,
        "capture": args.capture,
        "pins": code_pins(harness_dir),
        "plan_sha256": sha256_file(args.plan),
        "python": sys.version.split()[0],
        "started": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "started_unix": time.time(),
        "pid": os.getpid(),
    }
    write_json(out / "manifest.json", manifest)
    shutil.copytree(harness_dir, out / "harness_snapshot" / harness_name, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(args.plan, out / "plan.json")

    status, terminal, error = "running", None, None
    started = time.monotonic()
    decisions = 0
    summary, first_row, agent = {}, None, None
    try:
        env.start()
        session = env.session
        for sub in ("responses", "requests"):
            (session / sub).mkdir()
        watch = EpisodeWatch(session)
        watch.poll()
        first_row = watch.rows[0] if watch.rows else {}
        agent = agent_mod.Agent(env, dict(harness_cfg.get("knobs") or {}))
        manifest["harness"]["knobs_effective"] = dict(agent.knobs)
        write_json(out / "manifest.json", manifest)
        agent.boot()
        if prefix:
            frames = int((start or {}).get("prefix_frames", starts.PREFIX_FRAMES))
            starts.play_prefix(agent, env, prefix, frames=frames)
            if agent.knobs != manifest["harness"]["knobs_effective"]:
                raise RuntimeError("the harness knobs differ after the prefix")
        watch.poll()
        handover_row = watch.last or first_row
        judge = Judge(task, handover_row, (group or {}).get("waypoint"))
        frame0 = handover_row.get("frame") or 0
        manifest["handover"] = {
            "prefix_steps": len(prefix),
            "obs_id": env.obs["frame"]["id"],
            "tick": env.obs["frame"]["tick"],
            "bridge_obs_id": env.raw["frame"]["id"],
            "bridge_tick": env.raw["frame"]["tick"],
            "debug": {
                k: handover_row.get(k)
                for k in ("frame", "scene", "x", "y", "grounded", "health", "soul", "deaths", "facing")
            },
        }
        write_json(out / "manifest.json", manifest)
        signature = None
        for index in range(1, caps["decisions"] + 1):
            decisions = index
            t0 = time.time()
            if agent.view.me is None and agent.knobs.get("absent") == "wait":
                # the harness's own no-key wait (agent._wait_for_knight, which ask() would run first), run here so
                # that the state snapshot below is taken after it and re-renders the request
                env.window_tag = {"absent_before": index}
                try:
                    agent._wait_for_knight()
                finally:
                    env.window_tag = None
            ident = env.obs["frame"]["id"]
            state = state_codec.snapshot(agent, package, harness_name, ident)
            append(session / "states.jsonl", {"decision": index, "state": state})
            ask = agent.ask()
            clean_request = build_request(ident, ask["essential"], ask["questions"])
            request, hfault = fault.render(agent, ask, clean_request, index, build_request)
            body = json.dumps(request, allow_nan=False).encode()  # what LayaClient.decide sends
            (session / "requests" / f"{index:04d}.json").write_bytes(body)
            if index == 1:
                clean_body = json.dumps(clean_request, allow_nan=False).encode()
                signature = {
                    "clean_request_sha256": hashlib.sha256(clean_body).hexdigest(),
                    "state_sha256": hashlib.sha256(json.dumps(state, allow_nan=False).encode()).hexdigest(),
                }
                manifest["handover"]["first_decision"] = signature
                write_json(out / "manifest.json", manifest)
            record = {
                "decision": index,
                "obs_id": ident,
                "tick": env.obs["frame"]["tick"],
                "bridge_obs_id": env.raw["frame"]["id"],
                "bridge_tick": env.raw["frame"]["tick"],
                "facts": ask["facts"],
                "request": request,
                "clean_request": clean_request if hfault is not None else None,
                "harness_fault": hfault,
                "requestFile": f"requests/{index:04d}.json",
                "responseRawFile": f"responses/{index:04d}.raw",
                "forecasts": ask["forecasts"],
                "t_ask": round(t0, 4),
                "t_request": round(time.time(), 4),
            }
            try:
                response, seconds = client.decide(request, raw_path=session / record["responseRawFile"])
                answers = checked_answers(response, request)
            except GuardRejection as rejection:
                record["rejected"] = str(rejection)
                append(session / "decisions.jsonl", record)
                terminal = {
                    "outcome": "other",
                    "event": "guard_rejection",
                    "frame": None,
                    "decision": index,
                    "detail": str(rejection),
                }
                status = f"guard rejection at decision {index}: {rejection} (nothing executed)"
                break
            choices = {q: answers[q][0] for q in answers}
            move, attack = agent.compose(choices)
            record.update(
                {
                    "answers": {
                        q: {"choice": v[0], "probability": v[1], "probabilities": v[2]} for q, v in answers.items()
                    },
                    "server_choice": {q: (response.get("answers") or {}).get(q, {}).get("choice") for q in answers},
                    "choice": {"move": move, "attack": attack},
                    "seconds": seconds,
                    "usage": (response.get("modelResult") or {}).get("usage"),
                    "tokenAudit": response.get("tokenAudit"),
                    "t_response": round(time.time(), 4),
                }
            )
            armed = fault.on_selection(env, move, attack, index, ask["facts"])
            if armed is not None:
                extra = (
                    {"tail": agent.knobs["tail"]}
                    if "tail" in inspect.signature(keys_mod.bridge_windows).parameters and "tail" in agent.knobs
                    else {}
                )
                fault.plan_windows(
                    keys_mod.bridge_windows(move, attack, agent.n, agent.jump_down, agent.rising, **extra)[0]
                )
            label = f"Laya: {move} + {attack}"
            result = agent.act(move, attack, index, label)
            controller = fault.after_decision(env, index)
            new_rows = watch.poll()
            last = watch.last or {}
            record.update(
                {
                    "controller": controller,
                    "executed_option": (controller or {}).get("executed") or {"move": move, "attack": attack},
                    "keys": result["keys"],
                    "result": result["result"],
                    "check": result["check"],
                    "adapter_frames": result["adapter_frames"],
                    "t_done": round(time.time(), 4),
                    "eval_only": {
                        "x": last.get("x"),
                        "y": last.get("y"),
                        "health": last.get("health"),
                        "scene": last.get("scene"),
                        "soul": last.get("soul"),
                        "deaths": last.get("deaths"),
                        "frame": last.get("frame"),
                    },
                }
            )
            append(session / "decisions.jsonl", record)
            c = result["check"] or {}
            print(
                f"{index:4d} {move:17s} {attack:8s} p={answers['move'][1]:.2f}/{answers['attack'][1]:.2f} "
                f"{seconds:.1f}s -> {result['result'][:60]:60s} | x={last.get('x', 0):.1f} y={last.get('y', 0):.1f} "
                f"hp={last.get('health')} {'FAULT ' + str((controller or hfault or {}).get('changed')) if (controller or hfault) else ''}",
                flush=True,
            )
            terminal = judge.check(new_rows)
            if terminal is not None:
                terminal["decision"] = index
                status = f"{terminal['outcome']}: {terminal['event']} ({terminal['detail']})"
                break
            if (last.get("frame") or 0) - frame0 >= caps["frames"]:
                terminal = {
                    "outcome": "timeout",
                    "event": "frame_cap",
                    "frame": last.get("frame"),
                    "decision": index,
                    "detail": f"{caps['frames']} bridge frames",
                }
                status = "timeout: frame cap"
                break
            if time.monotonic() - started >= caps["wall_seconds"]:
                terminal = {
                    "outcome": "other",
                    "event": "wall_cap",
                    "frame": last.get("frame"),
                    "decision": index,
                    "detail": f"{caps['wall_seconds']} s wall clock",
                }
                status = "wall-clock cap"
                break
        else:
            terminal = {
                "outcome": "timeout",
                "event": "decision_cap",
                "frame": (watch.last or {}).get("frame"),
                "decision": decisions,
                "detail": f"{caps['decisions']} decisions",
            }
            status = "timeout: decision cap"
    except BaseException as exc:
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "decision": decisions,
        }
        terminal = {
            "outcome": "other",
            "event": "error",
            "frame": None,
            "decision": decisions,
            "detail": f"{type(exc).__name__}: {exc}",
        }
        status = f"error: {type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        try:
            summary = EpisodeWatch(env.session).summary()
        except Exception:
            summary = {}
        close = env.close()
        try:
            health1 = client.health()
        except Exception as exc:
            health1 = {"error": f"{type(exc).__name__}: {exc}"}
        same_model = (health1.get("metadata") or {}).get("weightsSha256") == expected_sha
        result = {
            "format": FORMAT,
            "status": status,
            "terminal": terminal,
            "outcome": (terminal or {}).get("outcome"),
            "decisions": decisions,
            "wall_seconds": round(time.monotonic() - started, 1),
            "fault": fault.summary(),
            "close": close,
            "trace_summary": summary,
            "start_state": first_row,
            "handover": manifest.get("handover"),
            "best_x": getattr(locals().get("judge"), "best_x", None),
            "harness_log": agent.run_log() if agent is not None else None,
            "model_health_after": health1,
            "model_identity_unchanged": same_model,
            "finished": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        }
        if not same_model and (terminal or {}).get("outcome") != "other":
            result["status"] += "; MODEL IDENTITY CHANGED"
            terminal = result["terminal"] = {
                "outcome": "other",
                "event": "model_identity_changed",
                "frame": None,
                "decision": decisions,
                "detail": "weights hash differs after the episode",
            }
            result["outcome"] = "other"
        write_json(out / "result.json", result)
        if error is not None:
            write_json(out / "error.json", error)
        print(json.dumps({k: result[k] for k in ("status", "outcome", "decisions", "wall_seconds")}), flush=True)
        try:
            import to_contract

            to_contract.localize(run_dir)
            print(json.dumps(to_contract.convert(run_dir)), flush=True)
        except Exception as exc:
            (out / "contract_error.txt").write_text(traceback.format_exc())
            print("contract conversion failed:", exc, flush=True)


if __name__ == "__main__":
    main()
