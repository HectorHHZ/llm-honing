"""Exp 0 v0 collector for VVVVVV (native shape): one attempt = one episode in a fresh game process.

Clean release: the original VVVVVV source @40e1ec3b built with the bridge of upstream main 9d5327f
(integrations/vvvvvv; build/VVVVVV), the integration's request builder `run.request_for` (15 native ticks per
decision, the integration default) and its six actions, a Laya service, and the plan's selection rule.
Writes the common contract (integrations/_shared/exp0/CONTRACT.md 0.1.0). The bridge's own session files (observation,
command, frames) go to <run>/extra/session/.

Episode ("一局", see EPISODES_PROPOSAL.md): the native Space Station 1 time-trial start (after its countdown) in
"Welcome Aboard"; win = the player enters a room other than "Welcome Aboard" [113,105] and "Conundrum"
[114,105] (i.e. has crossed Conundrum); loss = the first death (native death counter / death sequence), unless
the plan sets `deaths_end_episode: false`; timeout = the plan's decision cap.

Changes for the v0 release collection (2026-11-26, experiments/exp0/20261126_0045_vvvvvv_v0_collection):
- persistent fault forms `omit-local-terrain-persistent-v1` (harness: every model decision) and
  `swap-left-right-persistent-v1` (controller: every decision whose selected input has a direction); the phase-1
  one-shot forms stay available unchanged;
- every reply must pass both the common guard and the merged integration's own receiver (`checked_action`, which
  requires the exact six-option set) before any native input;
- own Xvfb displays only in :420..:439 (`DISPLAYS`), optionally one fixed display per stream (`--display`).

Post-freeze change (2026-11-26, after the 40 release attempts had been collected; recorded in the collection's
PROGRESS.md): the absolute local paths of the common tools, the desktop assets and the SDL3 library were replaced by
run-time resolution (`runtime.py`; `$VVVVVV_ASSETS`, `$VVVVVV_SDL_LIB`), so that no published file names a local path
or account. It resolves to the same locations on the collection server, so behavior and collected bytes are unchanged;
the release manifests' `pins.code.fingerprint` (19bb14f9...) refers to the collection-time file.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "_shared" / "exp0"))
sys.path.insert(0, str(HERE))
from contract import (
    CONTRACT_VERSION,
    FORMAT,
    ResponseRejected,
    Sampler,
    ServiceClient,
    fingerprint_files,
    guard_response,
    same_request,
    sha256_file,
    write_json,
)
from decision import checked_action, strict_json_loads
from procs import OwnedXvfb, proc_start_ticks, receipt, stop_owned
from run import NativeClient, request_for
from runtime import ASSETS, GAME, SDL_LIB

START_ROOM, CONUNDRUM = [113, 105], [114, 105]
MIRROR = {"left": "right", "right": "left", "flip_left": "flip_right", "flip_right": "flip_left"}
UPSTREAM_SHA = "9d5327f1c5a53b8b134db847952ff6281267c0a6"
DISPLAYS = range(420, 440)  # the displays this collection owns (:350..:379 belong to the HK collection)
# fault instances: (form, kind); "once" = only the trigger decision (phase 1), "persistent" = every eligible decision
HARNESS_FORMS = {
    "omit-local-terrain-v1": "once",
    "omit-local-terrain-persistent-v1": "persistent",
    "withhold-flip-option-v1": "once",
}
CONTROLLER_FORMS = {"swap-first-horizontal-v1": "once", "swap-left-right-persistent-v1": "persistent"}


def line(value: Any) -> str:
    """One JSONL line in insertion order. NOT contract.canonical(): sorting keys would reorder the request's
    options, and Laya's probabilities depend on option order (checked: max |dp| 0.069 on a VVVVVV request), so a
    sorted log would not be the request the model saw. The bridge observation embedded in a request is also kept
    in its own key order, so that request_for re-renders it byte-identically."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def code_fingerprint() -> str:
    files = [
        HERE / "exp0_vvvvvv.py",
        HERE.parents[1] / "run.py",
        HERE.parents[1] / "decision.py",
        HERE.parents[1] / "HoningBridge.inc",
        HERE.parents[1] / "HoningBridge.h",
        HERE.parents[2] / "_shared" / "decision.py",
        HERE / "procs.py",
    ]
    return fingerprint_files(files, HERE.parents[2])


def public_observation(state: dict[str, Any]) -> dict[str, Any]:
    """What the screen shows that the model's state also carries, written before any harness rendering."""
    p = state["player"]
    return {
        "tick": state["tick"],
        "room": state["room"],
        "deaths": state["deaths"],
        "gravity": state["gravity"],
        "player": {k: p[k] for k in ("x", "y", "vx", "vy", "onGround", "onRoof")},
        "localOriginTiles": state["localOriginTiles"],
        "localSolidRows": state["localSolidRows"],
        "entities": state["entities"],
    }


def omit_terrain(state: dict[str, Any]) -> dict[str, Any]:
    """Harness fault omit-local-terrain-v1: the rendered state shows the 9x9 local rows without solid or spike
    cells ('#' and '!' rendered as '.'); every other field is unchanged."""
    changed = copy.deepcopy(state)
    changed["localSolidRows"] = [row.replace("#", ".").replace("!", ".") for row in state["localSolidRows"]]
    return changed


def died(before: dict[str, Any], after: dict[str, Any], start: dict[str, Any]) -> bool:
    return after["deaths"] > start["deaths"] or after.get("deathSequence", -1) != -1


def events_between(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    events = []
    # one event per death: when the native death sequence starts (or, if a window began and ended it, when the
    # counter rises); the counter may rise one window after the sequence started, which is the same death
    if before.get("deathSequence", -1) == -1 and (
        after.get("deathSequence", -1) != -1 or after["deaths"] > before["deaths"]
    ):
        events.append(
            {
                "tick": after["tick"],
                "kind": "death",
                "text": f"The player died (death counter {before['deaths']} -> {after['deaths']}).",
                "visible": True,
            }
        )
    if after["room"] != before["room"]:
        events.append(
            {"tick": after["tick"], "kind": "room", "text": f"Entered room {after['room']}.", "visible": True}
        )
    if after["checkpoint"] != before["checkpoint"]:
        events.append(
            {
                "tick": after["tick"],
                "kind": "checkpoint",
                "text": f"Checkpoint is now {after['checkpoint'][:4]}.",
                "visible": True,
            }
        )
    return events


def receiver_check(request: dict[str, Any], raw: bytes) -> None:
    """The merged integration's own acceptance boundary (API.md "Model acceptance boundary"): strict JSON decoding
    and `checked_action` (exact option set, ids, token audits, distribution). Raises ResponseRejected."""
    try:
        checked_action(strict_json_loads(raw.decode("utf-8")), request)
    except Exception as error:
        raise ResponseRejected(f"integration receiver: {type(error).__name__}: {error}") from None


def collect(
    root: Path,
    plan: dict[str, Any],
    group: dict[str, Any],
    condition: str,
    attempt: int,
    endpoint: str,
    *,
    capture: bool = False,
    display: int | None = None,
) -> Path:
    run = f"{group['id']}-{condition}-a{attempt}"
    out = root / run
    (out / "responses").mkdir(parents=True)
    (out / "extra").mkdir()
    session = out / "extra" / "session"
    session.mkdir()
    (session / "user").mkdir()
    if capture:
        (session / "frames").mkdir()
    client = ServiceClient(endpoint)
    checkpoint = client.checkpoint()
    expected = plan["alternate_weights_sha256" if condition == "weights" else "clean_weights_sha256"]
    if checkpoint.get("weightsSha256") != expected:
        raise ValueError(
            f"{endpoint} serves {checkpoint.get('weightsSha256')}, the {condition} condition needs {expected}"
        )
    scenario = group["scenario"]
    ticks, cap = scenario["ticks_per_decision"], scenario["caps"]["decisions"]
    deaths_end = scenario.get("deaths_end_episode", True)
    selection = {"rule": plan["selection"]["rule"], "seed": scenario.get("policy_seed")}
    sampler = Sampler(selection["seed"]) if selection["rule"] == "sample" else None
    fault_def = plan["faults"].get(condition)
    refs = {
        "harness": ("decisions.jsonl:clean_request", "decisions.jsonl:request"),
        "weights": (f"checkpoint:{plan['clean_weights_sha256']}", f"checkpoint:{plan['alternate_weights_sha256']}"),
        "controller": ("decisions.jsonl:model_choice", "decisions.jsonl:executed"),
    }
    manifest = {
        "format": FORMAT,
        "contract_version": CONTRACT_VERSION,
        "game_id": plan["game_id"],
        "shape": "native",
        "run": run,
        "condition": condition,
        "reset_group": group["id"],
        "attempt": {"number": attempt, "slot": f"{group['id']}/{condition}"},
        "scenario": {
            "task": scenario["task"],
            "start": scenario["start"],
            "seed": scenario.get("policy_seed"),
            "caps": scenario["caps"],
            "ticks_per_decision": ticks,
            "deaths_end_episode": deaths_end,
            "win_rule": scenario["win_rule"],
        },
        "selection": selection,
        "pins": {
            "game_build": {
                "id": "VVVVVV@40e1ec3b+integrations/vvvvvv@9d5327f",
                "sha256": sha256_file(GAME),
                "assets_sha256": "c767809594f6472da9f56136e76657e38640d584164a46112250ac6293ecc0ea",
            },
            "code": {"upstream_sha": UPSTREAM_SHA, "fingerprint": code_fingerprint()},
            "harness": {
                "id": "integrations/vvvvvv run.request_for (6 actions, 15 ticks)",
                "fingerprint": fingerprint_files(
                    [HERE.parents[1] / "run.py", HERE.parents[1] / "decision.py"], HERE.parents[1]
                ),
            },
            "controller": {
                "id": "integrations/vvvvvv NativeClient + bridge command.txt",
                "fingerprint": fingerprint_files(
                    [HERE.parents[1] / "run.py", HERE.parents[1] / "HoningBridge.inc"], HERE.parents[1]
                ),
            },
            "observation": {
                "condition": plan["observation"],
                "knobs": {"local_tiles": "9x9", "entities": "current room", "pixels": False},
            },
            "checkpoint": checkpoint,
        },
        "clean_weights_sha256": plan["clean_weights_sha256"],
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
    xvfb = OwnedXvfb(
        out / "extra" / "xvfb.log",
        size="800x600x24",
        prefer=display,
        displays=[display] if display is not None else DISPLAYS,
    )
    if display is not None and display not in DISPLAYS:
        raise ValueError(f"display :{display} is outside this collection's displays :420..:439")
    process = None
    started = time.perf_counter()
    rows_file = out / "decisions.jsonl"
    states_file = out / "public_states.jsonl"
    index = 0
    harness_done = controller_done = False
    changed_rows: list[int] = []
    event_counts: dict[str, int] = {}
    status = reason = None
    state = start = None
    auto_rows = 0
    model_rows_n = 0
    try:
        display = xvfb.start()
        env = {
            **os.environ,
            "HONING_SESSION_DIR": str(session),
            "DISPLAY": f":{display}",
            "LD_LIBRARY_PATH": SDL_LIB,
            "SDL_AUDIODRIVER": "dummy",
            "SDL_RENDER_DRIVER": "software",
        }
        env.pop("HONING_CAPTURE", None)
        if capture:
            env["HONING_CAPTURE"] = "1"
        command = [str(GAME), "-assets", str(ASSETS), "-basedir", str(session / "user")]
        log = (out / "extra" / "player.log").open("wb")
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(out / "extra" / "launch.json", {"game": receipt(process), "xvfb": xvfb.info})
        native = NativeClient(session, process)
        state = start = native.observe()
        write_json(out / "extra" / "start_observation.json", start)
        decisions = 0
        while status is None:
            if decisions >= cap:
                status, reason = "timeout", "decision_cap"
                break
            row_started = time.perf_counter()
            before = state
            if not deaths_end and before.get("deathSequence", -1) != -1:
                # deaths allowed: while the native death sequence runs, inputs have no effect; the collector plays
                # `wait` windows (auto rows, no model call) until the game has respawned the player at its checkpoint
                after = native.step(before, "wait", ticks)
                events = events_between(before, after)
                if after.get("deathSequence", -1) == -1:
                    events.append(
                        {
                            "tick": after["tick"],
                            "kind": "other",
                            "text": f"Respawned at the checkpoint in room {after['room']}.",
                            "visible": True,
                        }
                    )
                for event in events:
                    event_counts[event["kind"]] = event_counts.get(event["kind"], 0) + 1
                with rows_file.open("a") as f:
                    f.write(
                        line(
                            {
                                "index": index,
                                "kind": "auto",
                                "native_before": {
                                    "tick": before["tick"],
                                    "frame": before["engineFrame"],
                                    "sim_seconds": before["simulationMilliseconds"] / 1000.0,
                                },
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
                                        {
                                            "native_action": "wait",
                                            "ticks": ticks,
                                            "start_tick": before["tick"],
                                            "frames": ticks,
                                            "reason": "death sequence: no input until respawn",
                                        }
                                    ],
                                },
                                "fault": None,
                                "events": events,
                                "native_after": {
                                    "tick": after["tick"],
                                    "frame": after["engineFrame"],
                                    "sim_seconds": after["simulationMilliseconds"] / 1000.0,
                                },
                                "timing": {
                                    "model_seconds": None,
                                    "execution_seconds": time.perf_counter() - row_started,
                                    "wall_seconds": time.perf_counter() - row_started,
                                },
                                "extra": {"after_observation": after},
                            }
                        )
                        + "\n"
                    )
                state = after
                index += 1
                auto_rows += 1
                continue
            with states_file.open("a") as f:
                f.write(line({"index": index, "state": {"observation": before, "ticks": ticks}}) + "\n")
            state_line = model_rows_n
            request = request_for(before, ticks)
            clean = fault = None
            harness_instance = (fault_def or {}).get("instance") if condition == "harness" else None
            if harness_instance == "omit-local-terrain-persistent-v1":
                # persistent form: applied at every model decision; `changed` says whether the request differs
                clean, request = request, request_for(omit_terrain(before), ticks)
                fault = {
                    "component": "harness",
                    "changed": not same_request(request, clean),
                    "detail": "local terrain rows rendered without solid or spike cells",
                }
            elif (
                harness_instance == "omit-local-terrain-v1"
                and not harness_done
                and any(("#" in r or "!" in r) for r in before["localSolidRows"])
            ):
                clean, request = request, request_for(omit_terrain(before), ticks)
                fault = {
                    "component": "harness",
                    "changed": request != clean,
                    "detail": "local terrain rows rendered without solid or spike cells",
                }
                harness_done = True
            elif harness_instance == "withhold-flip-option-v1" and not harness_done:
                # alternative instance (not the default): the `flip` input is not offered at the first decision
                clean = request
                request = copy.deepcopy(clean)
                del request["questions"]["action"]["criteria"]["flip"]
                fault = {
                    "component": "harness",
                    "changed": request != clean,
                    "detail": "one input was not offered at this decision",
                }
                harness_done = True
            elif condition == "harness" and harness_instance not in HARNESS_FORMS:
                raise ValueError(f"unknown harness fault instance {harness_instance!r}")
            tries, parsed = [], None
            for number in range(1, plan["response_tries"] + 1):
                name = f"responses/{index:04d}.raw" if number == 1 else f"responses/{index:04d}-t{number}.raw"
                try:
                    raw, seconds = client.decide_raw(request)
                except OSError as error:
                    tries.append({"file": None, "status": "transport_error", "reason": str(error), "seconds": None})
                    continue
                (out / name).write_bytes(raw)
                try:
                    parsed = guard_response(request, raw, plan["question_key"])
                    receiver_check(request, raw)
                except ResponseRejected as rejection:
                    parsed = None
                    tries.append({"file": name, "status": "rejected", "reason": str(rejection), "seconds": seconds})
                    continue
                tries.append({"file": name, "status": "accepted", "reason": None, "seconds": seconds})
                break
            native_before = {
                "tick": before["tick"],
                "frame": before["engineFrame"],
                "sim_seconds": before["simulationMilliseconds"] / 1000.0,
            }
            base = {
                "index": index,
                "kind": "model",
                "native_before": native_before,
                "public_observation": public_observation(before),
                "public_state": state_line,
                "request": request,
                "clean_request": clean,
                "response": {"file": tries[-1]["file"] if parsed else None, "tries": tries},
            }
            model_seconds = sum(t["seconds"] or 0.0 for t in tries)
            if parsed is None:
                with rows_file.open("a") as f:
                    f.write(
                        line(
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
                                    "model_seconds": model_seconds,
                                    "execution_seconds": None,
                                    "wall_seconds": time.perf_counter() - row_started,
                                },
                            }
                        )
                        + "\n"
                    )
                raise ResponseRejected("no accepted response: " + "; ".join(str(t["reason"]) for t in tries))
            server_choice = parsed["server_choice"]
            model_choice = (
                sampler.choose(request, parsed["probabilities"], plan["question_key"]) if sampler else server_choice
            )
            executed = model_choice
            window: dict[str, Any] = {
                "native_action": executed,
                "ticks": ticks,
                "start_tick": before["tick"],
                "frames": ticks,
            }
            controller_instance = (fault_def or {}).get("instance") if condition == "controller" else None
            if condition == "controller" and controller_instance not in CONTROLLER_FORMS:
                raise ValueError(f"unknown controller fault instance {controller_instance!r}")
            persistent_swap = controller_instance == "swap-left-right-persistent-v1"
            if condition == "controller" and (persistent_swap or not controller_done) and model_choice in MIRROR:
                # once (phase 1): only the first directional selection; persistent: every directional selection
                executed = MIRROR[model_choice]
                window = {
                    "native_action": executed,
                    "ticks": ticks,
                    "start_tick": before["tick"],
                    "frames": ticks,
                    "intended_action": model_choice,
                }
                fault = {
                    "component": "controller",
                    "changed": True,
                    "detail": f"selected {model_choice}; sent {executed}",
                }
                controller_done = True
            exec_started = time.perf_counter()
            after = native.step(before, executed, ticks)  # checks id, tick count, action echo, engine frame
            events = events_between(before, after)
            if died(before, after, start):
                status, reason = ("loss", "player_died") if deaths_end else (None, None)
            if status is None and after["room"] not in (START_ROOM, CONUNDRUM):
                status, reason = "win", "crossed_conundrum"
                events.append(
                    {
                        "tick": after["tick"],
                        "kind": "goal",
                        "text": f"Crossed Conundrum into room {after['room']}.",
                        "visible": True,
                    }
                )
            for event in events:
                event_counts[event["kind"]] = event_counts.get(event["kind"], 0) + 1
            if fault and fault.get("changed"):
                changed_rows.append(index)
            row = {
                **base,
                "probabilities": parsed["probabilities"],
                "server_choice": server_choice,
                "model_choice": model_choice,
                "executed": {"option": executed, "windows": [window]},
                "fault": fault,
                "events": events,
                "native_after": {
                    "tick": after["tick"],
                    "frame": after["engineFrame"],
                    "sim_seconds": after["simulationMilliseconds"] / 1000.0,
                },
                "timing": {
                    "model_seconds": model_seconds,
                    "execution_seconds": time.perf_counter() - exec_started,
                    "wall_seconds": time.perf_counter() - row_started,
                },
                "extra": {"after_observation": after},
            }
            with rows_file.open("a") as f:
                f.write(line(row) + "\n")
            state = after
            index += 1
            decisions += 1
            model_rows_n += 1
        native.command(state, "stop", 0)
        process.wait(timeout=15)
        end = client.checkpoint()
        write_json(
            out / "result.json",
            {
                "status": status,
                "reason": reason,
                "terminal_tick": state["tick"],
                "rows": index,
                "model_decisions": decisions,
                "forced_decisions": 0,
                "auto_rows": auto_rows,
                "windows": index,
                "native_ticks": state["tick"] - start["tick"],
                "native_frames": state["engineFrame"] - start["engineFrame"],
                "sim_seconds": state["simulationMilliseconds"] / 1000.0,
                "wall_seconds": time.perf_counter() - started,
                "event_counts": event_counts,
                "trigger_index": changed_rows[0] if changed_rows else None,
                "fault_changed_rows": changed_rows,
                "checkpoint_end": {"weightsSha256": end.get("weightsSha256"), "role": end.get("role")},
                "deaths": state["deaths"] - start["deaths"],
                "final_room": state["room"],
            },
        )
    except BaseException as error:
        kind = (
            "rejected_response"
            if isinstance(error, ResponseRejected)
            else (
                "interrupted"
                if isinstance(error, KeyboardInterrupt)
                else (
                    "game_crash"
                    if isinstance(error, RuntimeError)
                    else "bridge_timeout" if isinstance(error, TimeoutError) else "collector_error"
                )
            )
        )
        write_json(
            out / "error.json",
            {
                "status": kind,
                "type": type(error).__name__,
                "message": str(error),
                "at_index": index,
                "traceback": traceback.format_exc(),
            },
        )
        if isinstance(error, KeyboardInterrupt):
            raise
    finally:
        cleanup = {}
        if process is not None:
            cleanup["game"] = {"pid": process.pid, "returncode": stop_owned(process, proc_start_ticks(process.pid))}
        cleanup["xvfb"] = xvfb.stop()
        write_json(out / "extra" / "cleanup.json", cleanup)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path, help="collection root holding collection.json")
    ap.add_argument("--group", required=True)
    ap.add_argument("--condition", required=True, choices=["no_injection", "harness", "weights", "controller"])
    ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--display", type=int, help="the one Xvfb display to use (in :420..:439)")
    args = ap.parse_args()
    if not args.endpoint.rstrip("/").endswith((":8813", ":8814")):
        ap.error("this collection uses only Laya 8813 (clean) and 8814 (alternate)")
    plan = json.loads((args.root / "collection.json").read_text())
    group = next(g for g in plan["groups"] if g["id"] == args.group)
    out = collect(
        args.root.resolve(),
        plan,
        group,
        args.condition,
        args.attempt,
        args.endpoint,
        capture=args.capture,
        display=args.display,
    )
    result = out / "result.json"
    print(
        json.dumps(
            {
                "run": out.name,
                **(
                    json.loads(result.read_text()) if result.exists() else json.loads((out / "error.json").read_text())
                ),
            }
        )[:600],
        flush=True,
    )


if __name__ == "__main__":
    main()
