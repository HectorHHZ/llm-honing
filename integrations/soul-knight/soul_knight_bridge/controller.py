#!/usr/bin/env python3
"""Run an actual /v1/decide policy against the native Unity smoke bridge.

The observation, choices, action duration and argmax readout here are temporary
interface-smoke defaults. They do not define a research benchmark or reward.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .native import NativeClient
from .runtime import external_output

MOVES = {
    "stay": [0, 0],
    "east": [1, 0],
    "west": [-1, 0],
    "north": [0, 1],
    "south": [0, -1],
    "north_east": [1, 1],
    "north_west": [-1, 1],
    "south_east": [1, -1],
    "south_west": [-1, -1],
}
BUTTONS = ("none", "attack", "interact", "skill", "switch_weapon")


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def encoded(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prompt_json(value: Any) -> str:
    """Keep exact native states in artifacts, but round prompt floats to 3 decimals."""

    def compact(item: Any) -> Any:
        if isinstance(item, float):
            return round(item, 3)
        if isinstance(item, dict):
            return {key: compact(v) for key, v in item.items()}
        if isinstance(item, list):
            return [compact(v) for v in item]
        return item

    return json.dumps(compact(value), ensure_ascii=False, separators=(",", ":"))


def actions_match(expected: dict, actual: Any) -> bool:
    """Only aim tolerates the native float32 JSON round trip; controls stay exact."""
    if not isinstance(actual, dict) or set(actual) != set(expected):
        return False
    if any(expected[key] != actual[key] for key in expected if key != "aim"):
        return False
    return len(actual["aim"]) == 2 and all(
        math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-5) for a, b in zip(expected["aim"], actual["aim"], strict=True)
    )


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def call(endpoint: str, route: str, body: Any = None) -> tuple[Any, bytes]:
    request = Request(
        endpoint.rstrip("/") + route,
        data=None if body is None else encoded(body),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=90) as response:
        raw = response.read()
    return json.loads(raw), raw


def unwrap_state(value: dict) -> dict:
    """Support the native client's plain snapshot and response envelope."""
    return value.get("state", value)


def make_request(state: dict, decision_id: int, previous: list[dict]) -> tuple[dict, dict]:
    player = state.get("player") or {}
    x, y = float(player.get("x", 0)), float(player.get("y", 0))
    aims = {"east": [x + 5, y], "west": [x - 5, y], "north": [x, y + 5], "south": [x, y - 5]}
    enemies = [e for e in state.get("enemies", []) if not e.get("dead") and e.get("active", True)]
    # The model chooses among every offered target; no nearest-enemy controller is used.
    for index, enemy in enumerate(enemies[:6]):
        aims[f"enemy_{index}"] = [float(enemy["x"]), float(enemy["y"])]
    essential = [
        "You control the knight in this actual running top-down SoulKnight Unity game.",
        "Goal: survive, explore the current level, defeat enemies, and progress toward the boss.",
        "World coordinates: +x east/right; +y north/up. Move and aim are independent.",
        "The game waits during decisions; each action advances a bounded batch of rendered frames.",
        "Buttons are single presses, so attack again on later decisions when appropriate.",
        "Interact opens chests, picks weapons, or uses a nearby exit. Move closer first if necessary.",
        "Current frame/scene/player/progress: "
        + prompt_json({k: state.get(k) for k in ("bridge_frame", "sim_time", "scene", "player", "progress")}),
    ]
    details = ["Enemy: " + prompt_json(e) for e in enemies[:24]]
    for key in ("landmarks", "interactables", "doors", "objects", "room", "projectiles"):
        if key in state:
            value = state[key]
            if isinstance(value, list):
                details.extend(f"{key}: " + prompt_json(item) for item in value[: 12 if key == "projectiles" else 24])
            else:
                details.append(f"{key}: " + prompt_json(value))
    request = {
        "id": decision_id,
        "parts": {
            "essential": essential,
            "details": details,
            "memory": [],
            "logs": [prompt_json(p) for p in previous[-4:]],
        },
        "questions": {
            "move": {
                "type": "choice",
                "instructions": "Choose the next movement to survive and progress.",
                "criteria": {key: f"Move {key.replace('_', ' ')} ({v[0]},{v[1]})" for key, v in MOVES.items()},
            },
            "button": {
                "type": "choice",
                "instructions": "Choose one button for the next movement interval.",
                "criteria": {
                    "none": "Move without pressing an action button",
                    "attack": "Fire the equipped weapon once",
                    "interact": "Interact with a nearby chest, weapon, or exit",
                    "skill": "Activate the knight skill",
                    "switch_weapon": "Switch equipped weapon",
                },
            },
            "aim": {
                "type": "choice",
                "instructions": "Choose where to aim in world coordinates; aim at an enemy to attack it.",
                "criteria": {key: f"Aim at {key}: x={v[0]:.2f}, y={v[1]:.2f}" for key, v in aims.items()},
            },
        },
    }
    return request, aims


def choose(response: dict, request: dict, aims: dict) -> tuple[dict, dict]:
    selected = {}
    for key, question in request["questions"].items():
        answer = response["answers"][key]
        probabilities = answer["probabilities"]
        if set(probabilities) != set(question["criteria"]):
            raise ValueError(f"Model probability support differs from offered {key} choices")
        if not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in probabilities.values()):
            raise ValueError("Invalid model probability")
        choice = answer["choice"]
        if choice not in probabilities or probabilities[choice] != max(probabilities.values()):
            raise ValueError("Model choice does not equal returned probability argmax")
        selected[key] = choice
    action = {
        "move": MOVES[selected["move"]],
        "aim": aims[selected["aim"]],
        **{button: selected["button"] == button for button in BUTTONS if button != "none"},
    }
    return selected, action


def run_episode(
    ipc_dir: Path,
    endpoint: str,
    output: Path,
    decisions: int = 12,
    frames: int = 50,
    reset: bool = False,
    seed: int = 7,
    run_id: str | None = None,
    timeout_seconds: float = 90,
) -> dict[str, Any]:
    """Record actual model choices and the matching native execution envelopes."""
    if decisions < 1 or not 1 <= frames <= 250 or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("Use decisions >=1 and frames in1..250")
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("decision server must be an HTTP(S) base URL without credentials or query")
    output = external_output(output)
    ipc_dir = ipc_dir.resolve()
    if output.exists():
        raise ValueError("Trace output must be a fresh path; existing artifacts are never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    run_id = run_id or output.name
    if (output / "decisions.jsonl").exists():
        raise ValueError("Refusing to overwrite an existing trace")
    health, raw_health = call(endpoint, "/v1/health")
    if health.get("ready") is not True:
        raise RuntimeError("Actual model endpoint is not ready")
    write_json(output / "model_health.json", health)
    (output / "model_health.raw.json").write_bytes(raw_health)
    write_json(
        output / "run_manifest.json",
        {
            "run_id": run_id,
            "started_utc": utc(),
            "model_endpoint": endpoint,
            "model_health_sha256": digest(raw_health),
            "model_health": health,
            "bridge_dir": str(ipc_dir.resolve()),
            "decisions_requested": decisions,
            "frames_per_action": frames,
            "reset_requested": reset,
            "seed_if_reset": seed if reset else None,
            "observation_defaults": {
                "prompt_float_decimals": 3,
                "enemy_detail_limit": 24,
                "aim_enemy_limit": 6,
                "projectile_limit": 12,
                "other_detail_limit": 24,
                "history_length": 4,
                "history_player_fields": ["x", "y", "hp", "defence", "energy"],
                "raw_native_states_preserved_exactly": True,
            },
            "controller_sha256": digest(Path(__file__).read_bytes()),
            "smoke_defaults_only": True,
        },
    )
    model_name = health.get("metadata", {}).get("repo", "unknown served model")
    client = NativeClient(ipc_dir, timeout_seconds=timeout_seconds)
    if reset:
        reset_state = client.reset(seed=seed)
        write_json(output / "reset.json", reset_state)
    previous = []
    started = utc()
    status = {
        "run_id": run_id,
        "model_name": model_name,
        "stage": "starting",
        "started_utc": started,
        "endpoint": endpoint,
        "smoke_defaults_only": True,
        "decisions_completed": 0,
    }
    write_json(output / "status.json", status)
    try:
        with (output / "decisions.jsonl").open("w") as trace:
            for decision_id in range(decisions):
                before_raw = client.state()
                before = unwrap_state(before_raw)
                player = before.get("player") or {}
                if player.get("dead") or before.get("terminal"):
                    status["stop_reason"] = "native_terminal_or_dead"
                    break
                request, aims = make_request(before, decision_id, previous)
                request["id"] = f"{run_id}:{decision_id}"
                status.update(
                    stage="waiting_model",
                    decision_id=decision_id,
                    native_frame=before.get("bridge_frame"),
                    action=None,
                )
                write_json(output / "status.json", status)
                request_start = utc()
                start = time.perf_counter()
                response, raw = call(endpoint, "/v1/decide", request)
                request_finish = utc()
                latency_ms = (time.perf_counter() - start) * 1000
                if response.get("id") != request["id"]:
                    raise ValueError("Decision endpoint did not echo request id")
                selected, action = choose(response, request, aims)
                execution_start = utc()
                pending = {
                    "run_id": run_id,
                    "decision_id": decision_id,
                    "model_name": model_name,
                    "request_started_utc": request_start,
                    "request_finished_utc": request_finish,
                    "request": request,
                    "response": response,
                    "response_raw": raw.decode(),
                    "request_sha256": digest(encoded(request)),
                    "response_sha256": digest(raw),
                    "selected_choices": selected,
                    "selected_action": action,
                    "native_before": before,
                    "execution_started_utc": execution_start,
                    "execution_pending": True,
                }
                write_json(output / "current_decision.pending.json", pending)
                status.update(
                    stage="executing", selected_choices=selected, action=action, execution_started_utc=execution_start
                )
                write_json(output / "status.json", status)
                receipt = client.step(action, frames=frames)
                execution_finish = utc()
                native_request = client.last_request
                native_response = client.last_response
                after_raw = client.state()
                after = unwrap_state(after_raw)
                checks = {
                    "native_action_matches_model": actions_match(action, receipt.get("executed_action")),
                    "native_frames_match_request": receipt.get("advanced_frames") == frames,
                    "native_frame_advance_matches": receipt.get("bridge_frame")
                    == before.get("bridge_frame", 0) + frames,
                    "ipc_request_action_matches": native_request.get("action") == action,
                    "ipc_reply_matches_request": native_request.get("id") == native_response.get("id")
                    and native_response.get("ok") is True,
                    "post_state_frame_matches_receipt": after.get("bridge_frame") == receipt.get("bridge_frame"),
                }
                record = {
                    "run_id": run_id,
                    "decision_id": decision_id,
                    "model_name": model_name,
                    "request_started_utc": request_start,
                    "request_finished_utc": request_finish,
                    "execution_started_utc": execution_start,
                    "execution_finished_utc": execution_finish,
                    "request_latency_ms": latency_ms,
                    "request": request,
                    "response": response,
                    "response_raw": raw.decode(),
                    "request_sha256": digest(encoded(request)),
                    "response_sha256": digest(raw),
                    "selected_choices": selected,
                    "selected_action": action,
                    "frames_requested": frames,
                    "native_before": before,
                    "execution_receipt": receipt,
                    "native_execution_request": native_request,
                    "native_execution_response": native_response,
                    "execution_checks": checks,
                    "native_after": after,
                    "native_before_sha256": digest(encoded(before)),
                    "native_after_sha256": digest(encoded(after)),
                    "smoke_defaults_only": True,
                }
                trace.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                trace.flush()
                write_json(output / "current_decision.json", record)
                if not all(checks.values()):
                    raise RuntimeError("Native execution correspondence failed; mismatched receipt preserved in trace")
                (output / "current_decision.pending.json").unlink()
                previous.append(
                    {
                        "decision": decision_id,
                        "choice": selected,
                        "before": {
                            k: (before.get("player") or {}).get(k) for k in ("x", "y", "hp", "defence", "energy")
                        },
                        "after": {
                            k: (after.get("player") or {}).get(k) for k in ("x", "y", "hp", "defence", "energy")
                        },
                    }
                )
                status.update(
                    stage="idle",
                    decisions_completed=decision_id + 1,
                    native_frame=after.get("bridge_frame"),
                    execution_finished_utc=execution_finish,
                )
                write_json(output / "status.json", status)
                print(
                    json.dumps(
                        {
                            "decision": decision_id,
                            "selected": selected,
                            "native_frame": after.get("bridge_frame"),
                            "latency_ms": round(latency_ms, 1),
                        }
                    ),
                    flush=True,
                )
    except BaseException as exc:
        status.update(stage="error", error=f"{type(exc).__name__}: {exc}", finished_utc=utc())
        write_json(output / "status.json", status)
        raise
    status.update(stage="complete", finished_utc=utc())
    write_json(output / "status.json", status)
    return status
