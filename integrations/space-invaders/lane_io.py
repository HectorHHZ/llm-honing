"""Visible OCAtari state and Honing decision requests for the final lane harness."""

from __future__ import annotations

import json
import time
import urllib.request
from typing import Any

ACTION_NAMES = ("NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE")
PLAYER_MIN_X = 34
PLAYER_MAX_X = 116


def visible_objects(env: Any, kind: str) -> list[Any]:
    return [obj for obj in env.objects if type(obj).__name__ == kind]


def state_parts(env: Any, info: dict[str, Any], step: int, variant: str = "baseline") -> dict[str, list[str]]:
    """Use only the player's current observation, including observed motion."""
    players = visible_objects(env, "Player")
    if len(players) != 1:
        raise RuntimeError(f"Expected one visible player, got {len(players)}")
    player = players[0]
    px, py = player.xy
    aliens = visible_objects(env, "Alien")
    bullets = []
    for bullet in visible_objects(env, "Bullet"):
        previous = getattr(bullet, "_prev_xy", None)
        if previous is None:
            continue
        dx, dy = bullet.xy[0] - previous[0], bullet.xy[1] - previous[1]
        if 0 < dy <= 20 and bullet.xy[1] < py + 12:
            bullets.append((bullet.xy[0], bullet.xy[1], dx, dy))
    bullets.sort(key=lambda item: (abs(item[0] - px), py - item[1]))

    lines = [f"step={step} frame={info['episode_frame_number']} lives={info['lives']}", f"player x={px} y={py}"]
    if aliens:
        lines.append(
            f"aliens count={len(aliens)} x_range={min(a.xy[0] for a in aliens)}.."
            f"{max(a.xy[0] for a in aliens)} lowest_y={max(a.xy[1] for a in aliens)}"
        )
        nearest = sorted(aliens, key=lambda alien: (abs(alien.xy[0] - px), -alien.xy[1]))[:3]
        lines.append("nearest aliens=" + ", ".join(f"({a.xy[0]},{a.xy[1]})" for a in nearest))
        if variant in ("geometry", "relative"):
            target_x, target_y = nearest[0].xy
            dx = target_x - px
            direction = "LEFT" if dx < -6 else "RIGHT" if dx > 6 else "ALIGNED"
            lines.append(
                f"nearest target x={target_x} y={target_y}; target_x - player_x={dx}; target direction={direction}"
            )
    else:
        lines.append("aliens count=0")
    lines.append(
        "descending bullets=" + (", ".join(f"({x},{y},dx={dx},dy={dy})" for x, y, dx, dy in bullets[:5]) or "none")
    )
    return {"essential": lines, "details": [], "memory": [], "logs": []}


def jev_action(
    url: str, parts: dict[str, list[str]], request_id: str, question: dict[str, Any]
) -> tuple[str, dict[str, Any], float]:
    body = json.dumps({"id": request_id, "parts": parts, "questions": question}).encode()
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/decide", body, {"Content-Type": "application/json"}, method="POST"
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)
    latency_ms = (time.perf_counter() - start) * 1000
    choice = result["answers"]["control"]["choice"]
    if choice not in question["control"]["criteria"]:
        raise RuntimeError(f"Decision server returned an invalid control: {choice!r}")
    return choice, result, latency_ms
