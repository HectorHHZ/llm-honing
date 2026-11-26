"""Write the common Exp 0 v0 contract files (integrations/_shared/exp0/CONTRACT.md, `honing-exp0-episode-v0` 0.2.0) of one
full-HK attempt under the raw-key harness from its raw record, and the collection plan of a collection root.

  python3 -B to_contract.py <run dir> [...]                 (the raw record is <run>/extra/, written by collect.py)
  python3 -B to_contract.py --plan <collection root> --task kp-smoke-d60 --groups 1-2 [--dry-run] [--instances ...]

Per attempt it writes, next to extra/: manifest.json, decisions.jsonl, responses/NNNN.raw (byte copies),
public_states.jsonl, and result.json or error.json. Nothing in extra/ is changed. The conversion is deterministic
(every field comes from the raw record and the plan), so it can be run again at any time.

Rows. Each model decision is one `model` row asking two questions (`move`, `attack`): probabilities, server and
model choice and the executed option are keyed by question (contract 0.2.0). An `auto` row 0 exists only when the
adapter played windows before the first decision (e.g. a scene still loading); none is invented otherwise.
Windows are the bridge commands the game received (session/bridge_windows.jsonl), the keys window of each harness
window as "keys hold:<keys> press:<keys> [first:<keys>]" and the adapter's own windows through segments no player
can act in (marked in `extra`). With a controller fault, a keys window whose keys differ from what the harness asked
for has `intended_action`; one whose length differs has `intended_ticks`. Events come from the visible observation
after each window (HUD health, the death and hazard-respawn segments, the room code of object ids) and from the
game's own enemy health in the debug trace (an enemy's death, which a player sees), plus the task's terminal event
on the last row.
Public state of a decision = the exact harness state saved before it (session/states.jsonl, state_codec.py).

2026-11-26 (Exp 0 v0 HK collection, v12_standing on the deterministic build): the reset group is the plan group of
the attempt (`--group`); the reset procedure (the boot and a group's scripted start prefix, whose windows are tagged
{"prefix": k}) is not part of the attempt: the rows begin with the first model decision, the scenario names the
start (prefix script, steps, cadence, digest) and the handover. The harness's own no-key windows while the knight is
not on screen (tagged {"absent_before": k}) become an `auto` row before decision k. The observation id names the
harness; the plan's collection.json is written from the frozen plan's groups (`write_plan --frozen`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMMON = HERE.parents[2] / "_shared" / "exp0"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(COMMON))
import faults
from contract import canonical, fingerprint_tree

GAME_ID = "hollow-knight-forestdango"
UPSTREAM_SHA = "9d5327f1c5a53b8b134db847952ff6281267c0a6"
OBSERVATION_ID = "hk-forestdango-visible-window-v9keys-v1"


def observation_id(harness_name: str) -> str:
    """The plan / manifest observation condition: the bridge's visible window profile as one harness renders it."""
    return (
        OBSERVATION_ID
        if harness_name == "v9_keys"
        else f"hk-forestdango-visible-window-{harness_name.replace('_', '-')}-v1"
    )


KEYS = ("move", "attack")
OUTCOMES = {
    "win": ["reached_goal", "boss_defeated"],
    "loss": ["player_died"],
    "timeout": ["decision_cap", "frame_cap"],
    "other_terminal": ["left_area"],
}
REASON = {
    "reached_x": "reached_goal",
    "reached_place": "reached_goal",
    "boss_death_sequence": "boss_defeated",
    "knight_death": "player_died",
    "decision_cap": "decision_cap",
    "frame_cap": "frame_cap",
    "scene_change": "left_area",
}
FAILED = ("guard_rejection", "error", "wall_cap", "model_identity_changed")
LOCAL_FIELDS = ("path", "gpuUuid")
LOCAL_ONLY = (
    "extra/launch.json",
    "extra/xvfb.log",
    "extra/home",
    "extra/session/player.log",
    "extra/session/stdout.log",
    "extra/session/frames",
    "extra/gameplay.mp4",
    "extra/session/boot_frames",
    "extra/session/action.json",
    "extra/session/observation.json",
    "extra/session/debug_observation.json",
)
REFS = {
    "harness": ("decisions.jsonl:clean_request", "decisions.jsonl:request"),
    "controller": ("decisions.jsonl:model_choice", "decisions.jsonl:executed"),
}


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def adapter_action(cmd: dict) -> str:
    text = str(cmd.get("action"))
    if cmd.get("press"):
        text += "+press:" + "|".join(cmd["press"])
    if cmd.get("first"):
        text += "+first:" + "|".join(cmd["first"])
    return text


def world_box(obs: dict, bounds: list[float]) -> tuple[float, float, float, float]:
    m = obs["view"]["world_from_screen"]
    x0, y1 = m[0] * bounds[0] + m[2] * bounds[1] + m[4], m[1] * bounds[0] + m[3] * bounds[1] + m[5]
    x1, y0 = (
        m[0] * (bounds[0] + bounds[2]) + m[2] * (bounds[1] + bounds[3]) + m[4],
        m[1] * (bounds[0] + bounds[2]) + m[3] * (bounds[1] + bounds[3]) + m[5],
    )
    return x0, x1, y0, y1


def public_observation(obs: dict) -> dict:
    """What a player sees on the current screen and HUD, compact (<= ~1.4 KB): HUD values, the knight's facing and
    the visible enemies, hazards and doors as boxes relative to the knight's feet (world units; the knight is 0.5 x
    1.3). Terrain is left out (the request describes the ground; the full visible observation is in the raw record)."""
    hud = obs.get("hud") or {}
    me = obs.get("self") or {}
    sb = me.get("screen_box")
    if sb:
        kx0, kx1, ky0, _ = world_box(obs, sb)
        kx, ky = (kx0 + kx1) / 2, ky0
    else:
        kx, ky = None, None
    things = []
    names = {"enemy": "enemy", "visible_hazard": "hazard", "interactable": "door"}
    for o in obs.get("objects", []):
        kind = o.get("kind")
        if kind not in names or o.get("id") == me.get("object_id") or kx is None:
            continue
        x0, x1, y0, y1 = world_box(obs, o["bounds"])
        what = (
            names[kind]
            if kind != "visible_hazard"
            else (o.get("look") if o.get("look") in ("spikes", "acid") else "hazard")
        )
        things.append(
            {
                "what": what,
                "dx": round((x0 + x1) / 2 - kx, 1),
                "dy": round(y0 - ky, 1),
                "w": round(x1 - x0, 1),
                "h": round(y1 - y0, 1),
            }
        )
    things.sort(key=lambda t: abs(t["dx"]) + abs(t["dy"]))
    shown = things[:10]
    flow = obs.get("flow") or {}
    soul = (hud.get("soul") or {}).get("fraction")
    out = {
        "hud": {
            "health": (hud.get("health") or {}).get("value"),
            "health_max": (hud.get("health_max") or {}).get("value"),
            "soul": None if soul is None else round(soul, 2),
            "geo": (hud.get("geo") or {}).get("value"),
        },
        "facing": me.get("facing"),
        "can_act": flow.get("can_act"),
        "segment": flow.get("segment"),
        "text_box": any(t.get("blocking") for t in hud.get("text", []) or []),
        "visible": shown,
    }
    if len(things) > len(shown):
        out["more_not_listed"] = len(things) - len(shown)
    return out


def room_code(obs: dict) -> str | None:
    sid = (obs.get("self") or {}).get("object_id") or ""
    return sid.split("/", 1)[0] if "/" in sid else None


def obs_events(prev: dict, cur: dict) -> list[dict]:
    """Visible native events between two consecutive visible observations (phase 1's rules)."""
    out = []
    tick = cur["frame"]["tick"]
    h0 = ((prev.get("hud") or {}).get("health") or {}).get("value")
    h1 = ((cur.get("hud") or {}).get("health") or {}).get("value")
    if isinstance(h0, int) and isinstance(h1, int) and h1 != h0:
        if h1 < h0:
            out.append(
                {"tick": tick, "kind": "damage", "text": f"The knight was hit: health {h0} -> {h1}.", "visible": True}
            )
        else:
            out.append(
                {"tick": tick, "kind": "heal", "text": f"The knight's health went up: {h0} -> {h1}.", "visible": True}
            )
    s0, s1 = (prev.get("flow") or {}).get("segment"), (cur.get("flow") or {}).get("segment")
    if s1 == "dead" and s0 != "dead":
        out.append({"tick": tick, "kind": "death", "text": "The knight died.", "visible": True})
    if s1 == "hazard_respawn" and s0 != "hazard_respawn":
        out.append(
            {"tick": tick, "kind": "other", "text": "A hazard threw the knight back to a safe spot.", "visible": True}
        )
    r0, r1 = room_code(prev), room_code(cur)
    if r0 and r1 and r0 != r1:
        out.append({"tick": tick, "kind": "room", "text": "The knight entered another room.", "visible": True})
    return out


def enemy_deaths(debug: list[dict]) -> dict[int, list[bool]]:
    """bridge frame -> one entry per enemy whose health reached 0 (or dead) in that debug row (the game's own signal),
    True when the enemy was inside the camera's rectangle then (a player saw it die)."""
    out: dict[int, list[bool]] = {}
    alive: dict[str, bool] = {}
    for r in debug:
        cam = r.get("camera") or [None] * 4
        for e in r.get("enemies") or []:
            name = e.get("name")
            dead = bool(e.get("dead")) or (e.get("health") is not None and e["health"] <= 0)
            if dead and alive.get(name, False):
                seen = (
                    None not in cam and cam[0] <= (e.get("x") or 0) <= cam[2] and cam[1] <= (e.get("y") or 0) <= cam[3]
                )
                out.setdefault(int(r.get("frame") or 0), []).append(bool(seen))
            alive[name] = not dead
    return out


def convert(run: Path) -> dict:
    run = Path(run).resolve()
    raw = run / "extra"
    session = raw / "session"
    hk = json.loads((raw / "manifest.json").read_text())
    res = json.loads((raw / "result.json").read_text()) if (raw / "result.json").exists() else None
    err = json.loads((raw / "error.json").read_text()) if (raw / "error.json").exists() else None
    plan = json.loads((raw / "plan.json").read_text())
    pins = json.loads((HERE / "pins.json").read_text())
    for name in ("manifest.json", "decisions.jsonl", "public_states.jsonl", "result.json", "error.json"):
        if (run / name).exists():
            (run / name).unlink()
    if (run / "responses").exists():
        shutil.rmtree(run / "responses")
    condition = hk["condition"]
    task = hk["task_definition"]
    group = (hk.get("group") or {}).get("id") or f"{hk['task']}-s{hk['seed']}"
    number = int(run.name.rsplit("-a", 1)[1]) if "-a" in run.name and run.name.rsplit("-a", 1)[1].isdigit() else 1
    caps = hk["caps"]
    checkpoint = {
        k: v for k, v in ((hk.get("model_health_before") or {}).get("metadata") or {}).items() if k not in LOCAL_FIELDS
    }
    game_fp = hashlib.sha256(
        canonical(pins["game"]["files"]).encode() + canonical(pins["game"]["bridge_source"]).encode()
    ).hexdigest()
    intervention = None
    if condition != "no_injection":
        d = hk["fault"]["definition"]
        refs = REFS.get(condition) or (
            f"checkpoint:{plan['clean_weights_sha256']}",
            f"checkpoint:{plan['alternate_weights_sha256']}",
        )
        intervention = {
            "component": condition,
            **{k: d[k] for k in ("family", "instance", "trigger_rule", "changed_artifact")},
            "original_ref": refs[0],
            "altered_ref": refs[1],
            "builder": {
                "form": hk["fault"]["form"],
                "code": "integrations/hollow-knight-forestdango/exp0/v0/faults.py",
            },
        }
    manifest = {
        "format": "honing-exp0-episode-v0",
        "contract_version": "0.2.0",
        "game_id": GAME_ID,
        "shape": "option",
        "run": run.name,
        "condition": condition,
        "reset_group": group,
        "attempt": {"number": number, "slot": f"{group}/{condition}"},
        "scenario": {
            "task": task["name"],
            "start": {
                "definition": task["start"],
                "bridge_start": task["scene"],
                "win": task.get("win"),
                "loss": task["loss"],
                **(
                    {
                        "start_id": hk["start"]["id"],
                        "prefix_script": hk["start"].get("prefix"),
                        "prefix_steps": hk["start"]["steps"],
                        "prefix_frames_per_step": hk["start"].get("prefix_frames"),
                        "prefix_sha256": hk["start"]["prefix_sha256_computed"],
                        "prefix": hk["start"]["script"],
                    }
                    if hk.get("start")
                    else {}
                ),
                **(
                    {"waypoint": (hk.get("group") or {}).get("waypoint")}
                    if (hk.get("group") or {}).get("waypoint")
                    else {}
                ),
                "handover": hk.get("handover"),
            },
            "seed": hk["seed"],
            "caps": {
                "decisions": caps["decisions"],
                "native_ticks": caps["frames"] + 120 + 3600,
                "wall_seconds": caps["wall_seconds"],
            },
        },
        "selection": {"rule": "argmax", "seed": None},
        "pins": {
            "game_build": {
                "id": f"ForestDango Hollow-Knight-Demo, bridge {pins['game']['bridge_version']}"
                + (
                    f", player {Path(hk['game']['player']).name}"
                    if hk["game"].get("player")
                    else ", default build linux64_bridge"
                ),
                "fingerprint": game_fp,
            },
            "code": {
                "upstream_sha": UPSTREAM_SHA,
                "fingerprint": hashlib.sha256(canonical(hk["pins"]).encode()).hexdigest(),
            },
            "harness": {
                "id": hk["harness"]["name"],
                "fingerprint": fingerprint_tree(raw / "harness_snapshot" / hk["harness"]["name"]),
            },
            "controller": {
                "id": "hk_v9 core/env_keys.py + core/env_forestdango.py + faults.py (Exp 0 copy)",
                "fingerprint": hashlib.sha256(
                    canonical({k: hk["pins"][k] for k in ("controller", "faults_sha256")}).encode()
                ).hexdigest(),
            },
            "observation": {
                "condition": observation_id(hk["harness"]["name"]),
                "knobs": {**hk["game"]["bridge_options"], "harness_knobs": hk["harness"].get("knobs_effective")},
            },
            "checkpoint": checkpoint,
        },
        "clean_weights_sha256": plan["clean_weights_sha256"],
        "intervention": intervention,
        "started_at": hk.get("started"),
        "runtime": {
            "python": hk.get("python"),
            "harness_policy": hk.get("policy"),
            "endpoint": hk.get("endpoint"),
            "questions": list(KEYS),
        },
        "extra": {"raw_record": "extra/", "task": hk["task"], "label": hk.get("label"), "capture": hk.get("capture")},
    }
    write_json(run / "manifest.json", manifest)

    decisions = rows(session / "decisions.jsonl")
    states = {r["decision"]: r["state"] for r in rows(session / "states.jsonl")}
    windows = rows(session / "windows.jsonl")
    bridge = rows(session / "bridge_windows.jsonl")
    visible = rows(session / "visible_observations.jsonl")
    debug = rows(session / "observations.jsonl")
    by_id = {o["frame"]["id"]: o for o in visible}
    deaths = enemy_deaths(debug)
    # bridge windows per decision: the leading adapter windows (before the first harness window) go to an auto row
    lead = len(bridge) - sum(1 + w.get("adapter_windows", 0) for w in windows)
    if lead < 0:
        raise ValueError("window accounting failed")
    per: dict = {}
    reset_windows = lead  # boot adapter windows + start prefix windows: the reset procedure (no rows)
    bi = lead
    for w in windows:
        group_ = bridge[bi : bi + 1 + w.get("adapter_windows", 0)]
        bi += len(group_)
        if w.get("prefix") is not None:
            reset_windows += len(group_)
            continue
        key = ("absent", w["absent_before"]) if w.get("absent_before") is not None else (w.get("decision") or 0)
        for b in group_:
            per.setdefault(key, []).append((b, None if str(b.get("why", "")).startswith("adapter:") else w))
    if bi != len(bridge):
        raise ValueError(f"window accounting failed: {bi} of {len(bridge)} bridge windows assigned")
    if 0 in per:
        raise ValueError("harness windows without a decision, prefix or absent tag")

    def contract_windows(items) -> list[dict]:
        out = []
        for b, w in items:
            cmd = {k: v for k, v in b.items() if k not in ("bridge_id", "bridge_tick", "why")}
            if w is None:
                out.append(
                    {
                        "native_action": adapter_action(cmd),
                        "ticks": int(cmd["frames"]),
                        "start_tick": int(b["bridge_tick"]),
                        "frames": int(cmd["frames"]),
                        "reason": "adapter: no player can act in this segment",
                        "extra": {"by": "adapter", "why": b.get("why")},
                    }
                )
                continue
            sent = {"hold": cmd.get("hold", []), "press": cmd.get("press", []), "first": cmd.get("first", [])}
            if sent != {k: w[k] for k in ("hold", "press", "first")} or int(cmd["frames"]) != int(w["frames"]):
                raise ValueError(
                    f"the game received other keys than the env recorded at bridge window {b['bridge_id']}"
                )
            win = {
                "native_action": faults.window_text(sent),
                "ticks": int(cmd["frames"]),
                "start_tick": int(b["bridge_tick"]),
                "frames": int(cmd["frames"]),
                "reason": str(w.get("label", "")),
            }
            asked = w.get("requested")
            if asked is not None:
                if faults.window_text(asked) != win["native_action"]:
                    win["intended_action"] = faults.window_text(asked)
                if int(asked["frames"]) != win["ticks"]:
                    win["intended_ticks"] = int(asked["frames"])
            out.append(win)
        return out

    def events_for(items) -> list[dict]:
        out = []
        for b, _ in items:
            prev, cur = by_id.get(b["bridge_id"]), by_id.get(b["bridge_id"] + 1)
            if prev is not None and cur is not None:
                out.extend(obs_events(prev, cur))
            start, end = int(b["bridge_tick"]), int(b["bridge_tick"]) + int(b["frames"])
            for frame in range(start + 1, end + 1):
                for seen in deaths.get(frame, []):
                    out.append(  # noqa: PERF401
                        {
                            "tick": frame,
                            "kind": "kill",
                            "text": "An enemy on screen died." if seen else "An enemy off screen died.",
                            "visible": seen,
                        }
                    )
        return out

    out_rows, state_lines, changed_rows = [], [], []
    if lead > 0 and not (hk.get("start") or {}).get("steps"):  # with a start prefix the boot is part of the reset
        boot = [(b, None) for b in bridge[:lead]]
        bw = contract_windows(boot)
        out_rows.append(
            {
                "index": 0,
                "kind": "auto",
                "native_before": {
                    "tick": int(visible[0]["frame"]["tick"]),
                    "frame": int(visible[0]["frame"]["tick"]),
                    "sim_seconds": visible[0]["frame"].get("simulation_time"),
                },
                "public_observation": None,
                "public_state": None,
                "request": None,
                "clean_request": None,
                "response": None,
                "probabilities": None,
                "server_choice": None,
                "model_choice": None,
                "executed": {"option": None, "windows": bw},
                "fault": None,
                "events": events_for(boot),
                "native_after": {
                    "tick": bw[-1]["start_tick"] + bw[-1]["ticks"],
                    "frame": bw[-1]["start_tick"] + bw[-1]["ticks"],
                    "sim_seconds": None,
                },
                "timing": {"model_seconds": None, "execution_seconds": None, "wall_seconds": None},
                "extra": {"what": "adapter windows before the first decision (no player could act)"},
            }
        )
    (run / "responses").mkdir()
    for d in decisions:
        k = d["decision"]
        absent = per.get(("absent", k))
        if absent:
            aw = contract_windows(absent)
            out_rows.append(
                {
                    "index": len(out_rows),
                    "kind": "auto",
                    "native_before": {"tick": aw[0]["start_tick"], "frame": aw[0]["start_tick"], "sim_seconds": None},
                    "public_observation": None,
                    "public_state": None,
                    "request": None,
                    "clean_request": None,
                    "response": None,
                    "probabilities": None,
                    "server_choice": None,
                    "model_choice": None,
                    "executed": {"option": None, "windows": aw},
                    "fault": None,
                    "events": events_for(absent),
                    "native_after": {
                        "tick": aw[-1]["start_tick"] + aw[-1]["ticks"],
                        "frame": aw[-1]["start_tick"] + aw[-1]["ticks"],
                        "sim_seconds": None,
                    },
                    "timing": {"model_seconds": None, "execution_seconds": None, "wall_seconds": None},
                    "extra": {"what": "the harness's no-key windows while the knight was not on screen (no option)"},
                }
            )
        items = per.get(k, [])
        wins = contract_windows(items)
        dst = f"responses/{k:04d}.raw"
        src = session / d["responseRawFile"]
        if src.exists():
            shutil.copyfile(src, run / dst)
        rejected = "rejected" in d
        before_tick = int(d["bridge_tick"])
        state = states.get(k)
        obs = (state or {}).get("agent", {}).get("view", {}).get("__view__")
        sim = by_id.get(d["bridge_obs_id"], {}).get("frame", {}).get("simulation_time")
        index = len(out_rows)
        row = {
            "index": index,
            "kind": "model",
            "native_before": {"tick": before_tick, "frame": before_tick, "sim_seconds": sim},
            "public_observation": public_observation(obs) if obs else {},
            "public_state": len(state_lines) if state else None,
            "request": d["request"],
            "clean_request": d.get("clean_request"),
            "response": {
                "file": dst,
                "tries": [
                    {
                        "file": dst,
                        "status": "rejected" if rejected else "accepted",
                        "reason": d.get("rejected"),
                        "seconds": d.get("seconds"),
                    }
                ],
            },
            "probabilities": None,
            "server_choice": None,
            "model_choice": None,
            "executed": None,
            "fault": None,
            "events": events_for(items),
            "native_after": None,
            "timing": {
                "model_seconds": d.get("seconds"),
                "execution_seconds": round(d["t_done"] - d["t_response"], 3) if d.get("t_done") else None,
                "wall_seconds": round(d["t_done"] - d["t_ask"], 3) if d.get("t_done") else None,
            },
            "extra": {
                "decision": k,
                "facts": d.get("facts"),
                "harness_result": d.get("result"),
                "forecast_check": d.get("check"),
                "bridge_obs_id": d.get("bridge_obs_id"),
                "request_file": f"extra/session/{d['requestFile']}",
            },
        }
        if not rejected:
            row["probabilities"] = {q: d["answers"][q]["probabilities"] for q in KEYS}
            row["server_choice"] = {q: d["server_choice"][q] for q in KEYS}
            row["model_choice"] = {q: d["choice"][q] for q in KEYS}
            row["executed"] = {"option": {q: d["executed_option"][q] for q in KEYS}, "windows": wins}
            end = before_tick
            for w_ in wins:
                end = max(end, w_["start_tick"] + w_["ticks"])
            row["native_after"] = {"tick": end, "frame": end, "sim_seconds": None}
        if condition == "harness" and d.get("harness_fault") is not None:
            hf = d["harness_fault"]
            row["fault"] = {
                "component": "harness",
                "changed": canonical(d["clean_request"]) != canonical(d["request"]),
                "detail": hf["detail"],
                "questions": hf.get("questions", []),
            }
        elif condition == "weights":
            row["fault"] = {"component": "weights", "changed": None, "detail": "served by the alternate checkpoint"}
        elif condition == "controller" and row["executed"] is not None and d.get("controller") is not None:
            c = d["controller"]
            mismatch = any("intended_action" in w or "intended_ticks" in w for w in wins)
            switched = row["executed"]["option"] != row["model_choice"]
            row["fault"] = {
                "component": "controller",
                "changed": bool(mismatch or switched),
                "detail": c["detail"] if (mismatch or switched) else "armed at this selection; nothing sent differed",
                "questions": c.get("questions", []),
            }
        if row["fault"] and row["fault"].get("changed"):
            changed_rows.append(index)
        if state:
            state_lines.append({"index": index, "state": state})
        out_rows.append(row)
    terminal = (res or {}).get("terminal") or {}
    if (
        terminal.get("outcome") in ("win", "loss", "left_area")
        and out_rows
        and out_rows[-1]["kind"] == "model"
        and out_rows[-1]["executed"]
    ):
        tick = (
            terminal.get("frame") if isinstance(terminal.get("frame"), int) else out_rows[-1]["native_before"]["tick"]
        )
        tick = min(max(tick, out_rows[-1]["native_before"]["tick"]), out_rows[-1]["native_after"]["tick"])
        text = {
            "reached_x": "The knight reached the goal place of this task.",
            "boss_death_sequence": "The boss's death sequence began: the fight is won.",
            "knight_death": "The episode ended with the knight's death.",
            "scene_change": "The knight left the room of this task.",
        }.get(terminal.get("event"), "The episode ended.")
        out_rows[-1]["events"].append(
            {
                "tick": tick,
                "kind": "goal" if terminal["outcome"] == "win" else "terminal",
                "text": text,
                "visible": True,
            }
        )
    with (run / "decisions.jsonl").open("x") as f:
        for r in out_rows:
            f.write(canonical(r) + "\n")
    with (run / "public_states.jsonl").open("x") as f:
        for s in state_lines:
            f.write(canonical(s) + "\n")
    events: dict[str, int] = {}
    for r in out_rows:
        for e in r["events"]:
            events[e["kind"]] = events.get(e["kind"], 0) + 1
    end_health = ((res or {}).get("model_health_after") or {}).get("metadata") or {}
    failed = err is not None or res is None or terminal.get("event") in FAILED
    if failed:
        status = {
            "guard_rejection": "rejected_response",
            "error": "collector_error",
            "wall_cap": "interrupted",
            "model_identity_changed": "identity_changed",
        }.get(terminal.get("event"), "other")
        write_json(
            run / "error.json",
            {
                "status": status,
                "type": terminal.get("event") or (err or {}).get("type") or "unknown",
                "message": str(terminal.get("detail") or (err or {}).get("message") or ""),
                "at_index": len(out_rows) - 1,
            },
        )
        return {"run": run.name, "written": "error.json", "status": status}
    out_status = {"win": "win", "loss": "loss", "timeout": "timeout", "left_area": "other_terminal"}[
        terminal["outcome"]
    ]
    last_tick = out_rows[-1]["native_after"]["tick"]
    result = {
        "status": out_status,
        "reason": REASON[terminal["event"]],
        "terminal_tick": last_tick,
        "rows": len(out_rows),
        "model_decisions": len(decisions),
        "forced_decisions": 0,
        "auto_rows": sum(r["kind"] == "auto" for r in out_rows),
        "windows": sum(len(r["executed"]["windows"]) for r in out_rows if r["executed"]),
        "native_ticks": last_tick - out_rows[0]["native_before"]["tick"],
        "native_frames": last_tick,
        "sim_seconds": round(last_tick / 30.0, 2),
        "wall_seconds": res.get("wall_seconds"),
        "event_counts": dict(sorted(events.items())),
        "trigger_index": changed_rows[0] if changed_rows else None,
        "fault_changed_rows": changed_rows,
        "checkpoint_end": {k: end_health.get(k) for k in ("weightsSha256", "role", "repo", "revision")},
        "extra": {
            "terminal_event": terminal,
            "raw_status": res.get("status"),
            "fault_summary": res.get("fault"),
            "best_x_debug": res.get("best_x"),
            "harness_log": res.get("harness_log"),
        },
    }
    write_json(run / "result.json", result)
    return {
        "run": run.name,
        "written": "result.json",
        "status": out_status,
        "reason": result["reason"],
        "trigger_index": result["trigger_index"],
    }


def localize(run: Path) -> dict:
    """Move the local-only raw files of one attempt out of the attempt directory (to <root>/_local/<run>/, same layout)
    and remove local-only fields (the service's checkpoint `path`) from the raw manifest / result, keeping the removed
    values in _local/<run>/removed_fields.json. Idempotent. Nothing is deleted."""
    run = Path(run).resolve()
    local = run.parent / "_local" / run.name
    moved = []
    for rel in LOCAL_ONLY:
        src = run / rel
        if src.exists():
            dst = local / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            moved.append(rel)
    removed = {}
    for name, key in (("extra/manifest.json", "model_health_before"), ("extra/result.json", "model_health_after")):
        path = run / name
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        meta = (data.get(key) or {}).get("metadata")
        if isinstance(meta, dict) and any(f in meta for f in LOCAL_FIELDS):
            removed[name] = {f: meta.pop(f) for f in LOCAL_FIELDS if f in meta}
            meta["local_fields_removed"] = list(removed[name])
            path.write_text(json.dumps(data, indent=1, allow_nan=False) + "\n")
    if moved or removed:
        local.mkdir(parents=True, exist_ok=True)
        log = local / "removed_fields.json"
        old = json.loads(log.read_text()) if log.exists() else {}
        old.update(removed)
        log.write_text(json.dumps(old, indent=1) + "\n")
        with (local / "moved.txt").open("a") as f:
            for rel in moved:
                f.write(rel + "\n")
    return {"run": run.name, "moved": moved, "fields_removed": sorted(removed)}


def write_plan(
    root: Path,
    task: str,
    groups_spec: str,
    dry_run: bool,
    harness_instances: list[str],
    controller_instances: list[str],
    form: str = "persistent",
    per_condition: int = 10,
    reserve: str = "",
) -> Path:
    """collection.json of one collection root (common contract section 8)."""
    plan = json.loads((HERE / "plan.json").read_text())
    groups = []
    for i, seed in enumerate(_seeds(groups_spec) + _seeds(reserve)):
        groups.append(
            {
                "id": f"{task}-s{seed}",
                "split": "train" if i % 2 == 0 else "val",
                "scenario": {"task": task, "seed": seed, "start": plan["tasks"][task]["scene"]},
            }
        )

    def defs(instances):
        out = [
            {k: faults.definition(i, form)[k] for k in ("family", "instance", "trigger_rule", "changed_artifact")}
            for i in instances
        ]
        return out[0] if len(out) == 1 else out

    weights = {
        k: faults.definition("laya-ra2-v33-v1", "persistent")[k]
        for k in ("family", "instance", "trigger_rule", "changed_artifact")
    }
    coll = {
        "schema": "honing-exp0-collection-v0",
        "game_id": GAME_ID,
        "dataset_dir": f"{GAME_ID}{'-dryrun' if dry_run else ''}/v0/",
        "shape": "option",
        "question_keys": list(KEYS),
        "observation": OBSERVATION_ID,
        "per_condition": per_condition,
        "initial_groups": len(_seeds(groups_spec)),
        "selection": {"rule": "argmax", "seed": None},
        "clean_weights_sha256": plan["clean_weights_sha256"],
        "alternate_weights_sha256": plan["alternate_weights_sha256"],
        "weight_difference_tolerance": plan["weight_difference_tolerance"],
        "probe": {"first_decisions": 4, "identity_tolerance": 0.01, "tie": 0.01},
        "diagnosis": {
            "max_tokens": 9000,
            "time": {"unit": "native_tick", "before": 100000, "after": 0},
            "preamble": (
                "Full Hollow Knight (ForestDango build) at 30 native frames per second; the renderer's "
                "time coordinate counts these frames. Every decision answers two questions, the movement "
                "keys and the attack key, and both answers' keys are held for the next 9 frames (0.3 s), "
                "sent as two or three native windows."
            ),
            "windows": {"max_runs_listed": 12, "max_mismatches_listed": 6},
            "public_observation_max_chars": 1400,
        },
        "outcomes": OUTCOMES,
        "faults": {"harness": defs(harness_instances), "weights": weights, "controller": defs(controller_instances)},
        "groups": groups,
        "attempt_cap_per_slot": 3,
        "neutral_id_prefix": "h",
        "shuffle_seed": 200,
        "response_tries": 1,
    }
    if dry_run:
        coll["dry_run"] = True
        coll["notes"] = (
            "DRY RUN of the release chain on smoke data (2026-11-25): not a dataset, never uploaded; the "
            "harness is the stand-in v9_keys and the task the provisional smoke task."
        )
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "collection.json", coll)
    return root / "collection.json"


def write_frozen_plan(root: Path, plan_path: Path | None = None) -> Path:
    """collection.json of the release collection root from the FROZEN hk plan (plan.json `collection` section):
    the reset groups (initial ones first, then reserves, in listed order), one validated fault instance per
    component, the task, caps, attempt caps and the diagnosis settings (contract section 8)."""
    plan = json.loads(Path(plan_path or HERE / "plan.json").read_text())
    spec = plan["collection"]
    if not str(plan.get("status", "")).startswith("FROZEN"):
        raise ValueError("the hk plan is not frozen")
    harness_name = Path(plan["release"]["harness"]["dir"]).name
    frames = int((plan["release"]["harness"].get("knobs") or {}).get("frames", spec.get("frames_per_decision", 13)))
    secs = f"{frames / 30:.2f}".rstrip("0").rstrip(".")
    groups = [
        {
            "id": g["id"],
            "split": g["split"],
            "scenario": {
                "task": spec["task"],
                "seed": g["seed"],
                "start": g["start"],
                "prefix_sha256": plan["starts"][g["start"]].get("prefix_sha256"),
                "prefix_steps": plan["starts"][g["start"]].get("steps", 0),
                "waypoint": g.get("waypoint"),
                "reserve": bool(g.get("reserve")),
            },
        }
        for g in plan["groups"]
    ]

    def one(instance):
        d = faults.definition(instance, "persistent")
        return {k: d[k] for k in ("family", "instance", "trigger_rule", "changed_artifact")}

    coll = {
        "schema": "honing-exp0-collection-v0",
        "game_id": GAME_ID,
        "dataset_dir": f"{GAME_ID}/v0/",
        "shape": "option",
        "question_keys": list(KEYS),
        "observation": observation_id(harness_name),
        "per_condition": spec["per_condition"],
        "initial_groups": sum(not g.get("reserve") for g in plan["groups"]),
        "selection": {"rule": "argmax", "seed": None},
        "clean_weights_sha256": plan["clean_weights_sha256"],
        "alternate_weights_sha256": plan["alternate_weights_sha256"],
        "weight_difference_tolerance": plan["weight_difference_tolerance"],
        "probe": {"first_decisions": 4, "identity_tolerance": 0.01, "tie": 0.01},
        "diagnosis": {
            "max_tokens": 9000,
            "time": {"unit": "native_tick", "before": 100000, "after": 0},
            "preamble": (
                "Full Hollow Knight (ForestDango build) at 30 native frames per second; the renderer's "
                "time coordinate counts these frames. Every decision answers two questions, the movement "
                f"keys and the attack key, and both answers' keys are held for the next {frames} frames "
                f"({secs} s), sent as two or three native windows."
            ),
            "windows": {"max_runs_listed": 12, "max_mismatches_listed": 6},
            "public_observation_max_chars": 1400,
        },
        "outcomes": OUTCOMES,
        "faults": {
            "harness": one(spec["faults"]["harness"]),
            "weights": one(spec["faults"]["weights"]),
            "controller": one(spec["faults"]["controller"]),
        },
        "groups": groups,
        "attempt_cap_per_slot": spec["attempt_cap_per_slot"],
        "neutral_id_prefix": "h",
        "shuffle_seed": 200,
        "response_tries": 1,
        "notes": spec.get("notes", ""),
    }
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "collection.json", coll)
    return root / "collection.json"


def _seeds(spec: str) -> list[int]:
    out = []
    for part in [p for p in spec.split(",") if p]:
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--plan", type=Path, default=None, help="write <root>/collection.json")
    ap.add_argument("--task", default="kp-smoke-d60")
    ap.add_argument("--groups", default="1-2")
    ap.add_argument("--reserve", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--per-condition", type=int, default=10)
    ap.add_argument("--harness-instances", default=faults.DEFAULT_INSTANCE["harness"])
    ap.add_argument("--controller-instances", default=faults.DEFAULT_INSTANCE["controller"])
    ap.add_argument("--form", default="persistent")
    ap.add_argument("--localize", action="store_true")
    ap.add_argument(
        "--frozen", action="store_true", help="with --plan: write collection.json from the frozen plan.json"
    )
    a = ap.parse_args()
    if a.plan and a.frozen:
        print(write_frozen_plan(a.plan))
    elif a.plan:
        print(
            write_plan(
                a.plan,
                a.task,
                a.groups,
                a.dry_run,
                a.harness_instances.split(","),
                a.controller_instances.split(","),
                a.form,
                a.per_condition,
                a.reserve,
            )
        )
    for r in a.runs:
        try:
            if a.localize:
                print(json.dumps(localize(r)))
            print(json.dumps(convert(r)))
        except Exception as exc:  # noqa: PERF203
            print(json.dumps({"run": str(r), "error": f"{type(exc).__name__}: {exc}"}))


if __name__ == "__main__":
    main()
