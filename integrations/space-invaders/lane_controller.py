"""Visible-screen firing-lane controller for Space Invaders.

The model may choose a lane. Code handles exact movement, line of sight, firing,
and short bullet dodges between those strategic choices.
"""

from __future__ import annotations

from statistics import median
from typing import Any

from lane_io import PLAYER_MAX_X, PLAYER_MIN_X, visible_objects

SHIELD_RGB = (181, 83, 40)
MUZZLE_OFFSET = 5
ACTUAL_MUZZLE_OFFSET = 4
ALIEN_RGB = (134, 134, 29)
MAX_BONUS_TRAVEL_STEPS = 16


def formation_shift(previous: list[tuple[int, int]], aliens: list[Any]) -> float:
    """Measure common displacement without confusing a killed edge alien with motion."""
    shifts = []
    for alien in aliens:
        x, y = alien.xy
        matches = [x - old_x for old_x, old_y in previous if abs(y - old_y) <= 2 and abs(x - old_x) <= 8]
        if matches:
            shifts.append(min(matches, key=abs))
    return float(median(shifts)) if len(shifts) >= 2 else 0.0


def clear_shot(frame: Any, bullet_x: int) -> bool:
    """Check the visible pixels between muzzle and aliens, including damaged shields."""
    return not any(tuple(frame[y, bullet_x]) == SHIELD_RGB for y in range(151, 177))


def ambush_line(env: Any) -> int | None:
    """Choose one clear, central firing line and let a lone alien cross it."""
    player_x = visible_objects(env, "Player")[0].xy[0]
    current_line = player_x + MUZZLE_OFFSET
    desired_line = max(65, min(95, current_line))
    frame = env.render()
    options = [x for x in range(65, 96) if clear_shot(frame, x)]
    if not options:
        options = [
            x for x in range(PLAYER_MIN_X + MUZZLE_OFFSET, PLAYER_MAX_X + MUZZLE_OFFSET + 1) if clear_shot(frame, x)
        ]
    return min(options, key=lambda x: (abs(x - desired_line), abs(x - current_line))) if options else None


def aim_at_alien(env: Any, alien_x: int, lowest_y: int, velocity_x: float) -> int | None:
    """Update the firing x on every game step while keeping the same alien column."""
    player_x = visible_objects(env, "Player")[0].xy[0]
    flight_steps = max(2, round((185 - lowest_y) / 8))
    predicted_center = alien_x + 4 + round(velocity_x * flight_steps)
    predicted_center = max(PLAYER_MIN_X + MUZZLE_OFFSET, min(PLAYER_MAX_X + MUZZLE_OFFSET, predicted_center))
    frame = env.render()
    options = [
        x
        for x in range(predicted_center - 3, predicted_center + 4)
        if PLAYER_MIN_X + MUZZLE_OFFSET <= x <= PLAYER_MAX_X + MUZZLE_OFFSET and clear_shot(frame, x)
    ]
    if not options:
        return None
    return min(options, key=lambda x: (abs(x - predicted_center), abs(x - player_x - MUZZLE_OFFSET)))


def aim_at_satellite(env: Any, satellite_x: int, velocity_x: float) -> int | None:
    """Lead the fast bonus ship and require an unobstructed full-height shot."""
    player_x = visible_objects(env, "Player")[0].xy[0]
    flight_steps = round((185 - 12) / 8)
    center = satellite_x + 3 + velocity_x * flight_steps
    for _ in range(4):
        travel_steps = abs(center - MUZZLE_OFFSET - player_x) / 2
        center = satellite_x + 3 + velocity_x * (flight_steps + travel_steps)
    if abs(center - MUZZLE_OFFSET - player_x) / 2 > MAX_BONUS_TRAVEL_STEPS:
        return None
    center = round(center)
    frame = env.render()
    options = [
        x
        for x in range(center - 3, center + 4)
        if PLAYER_MIN_X + MUZZLE_OFFSET <= x <= PLAYER_MAX_X + MUZZLE_OFFSET
        and clear_shot(frame, x)
        and not any(tuple(frame[y, x]) == ALIEN_RGB for y in range(22, 151))
    ]
    return min(options, key=lambda x: (abs(x - center), abs(x - player_x - MUZZLE_OFFSET))) if options else None


def follow_target(
    env: Any, target: dict[str, Any], alien_velocity_x: float, satellite_velocity_x: float
) -> int | None:
    """Follow the chosen object between strategic decisions; report lost targets."""
    if target["target_type"] == "satellite":
        satellites = visible_objects(env, "Satellite")
        if not satellites:
            return None
        target["target_x"] = satellites[0].xy[0]
        return aim_at_satellite(env, target["target_x"], satellite_velocity_x)
    columns: dict[int, int] = {}
    for alien in visible_objects(env, "Alien"):
        columns[alien.xy[0]] = max(columns.get(alien.xy[0], 0), alien.xy[1])
    if not columns:
        return None
    nearest = min(columns, key=lambda x: abs(x - target["target_x"]))
    if abs(nearest - target["target_x"]) > 8:
        return None
    target["target_x"] = nearest
    return aim_at_alien(env, nearest, columns[nearest], alien_velocity_x)


def candidate_lanes(
    env: Any, velocity_x: float, locked_x: int | None = None, satellite_velocity_x: float = 0.0
) -> list[dict[str, Any]]:
    """Rank unobstructed firing lines, strongly favoring the lowest alien row."""
    aliens = visible_objects(env, "Alien")
    satellites = visible_objects(env, "Satellite")
    if not aliens and not satellites:
        return []
    player_x = visible_objects(env, "Player")[0].xy[0]
    columns: dict[int, list[Any]] = {}
    for alien in aliens:
        columns.setdefault(alien.xy[0], []).append(alien)
    lanes = []
    for alien_x, members in columns.items():
        lowest_y = max(alien.xy[1] for alien in members)
        bullet_x = aim_at_alien(env, alien_x, lowest_y, velocity_x)
        if bullet_x is None:
            continue
        aim_x = bullet_x - MUZZLE_OFFSET
        predicted_center = alien_x + 4 + round(velocity_x * max(2, round((185 - lowest_y) / 8)))
        score = (
            0.16 * lowest_y
            + 0.25 * len(members)
            - 0.12 * abs(aim_x - player_x)
            + (3.0 if locked_x is not None and abs(locked_x - bullet_x) <= 4 else 0.0)
        )
        lanes.append(
            {
                "target_type": "alien",
                "target_x": alien_x,
                "bullet_x": bullet_x,
                "aim_x": aim_x,
                "lowest_alien_y": lowest_y,
                "aliens_in_column": len(members),
                "predicted_center": predicted_center,
                "score": round(score, 3),
                "description": (
                    f"Lock clear firing lane x={bullet_x}; lowest alien y={lowest_y}; "
                    f"{len(members)} aliens in this column; move cannon to x={aim_x}; "
                    f"planner score={score:.2f}"
                ),
            }
        )
    if satellites:
        satellite_x = satellites[0].xy[0]
        bullet_x = aim_at_satellite(env, satellite_x, satellite_velocity_x)
        if bullet_x is not None:
            aim_x = bullet_x - MUZZLE_OFFSET
            score = 42.0 - 0.12 * abs(aim_x - player_x)
            lanes.append(
                {
                    "target_type": "satellite",
                    "target_x": satellite_x,
                    "bullet_x": bullet_x,
                    "aim_x": aim_x,
                    "lowest_alien_y": 12,
                    "aliens_in_column": 0,
                    "predicted_center": bullet_x,
                    "score": round(score, 3),
                    "description": f"Intercept bonus Command Alien Ship at x={bullet_x}; move cannon to x={aim_x}; "
                    f"planner score={score:.2f}",
                }
            )
    lanes.sort(key=lambda item: (-item["score"], abs(item["aim_x"] - player_x), item["bullet_x"]))
    distinct = []
    for lane in lanes:
        if all(abs(lane["bullet_x"] - other["bullet_x"]) >= 10 for other in distinct):
            distinct.append(lane)
    return distinct


def forecast_escape_direction(
    player_x: int,
    player_y: int,
    target_x: int,
    bullets: list[Any],
    *,
    max_steps: int = 16,
    corridor_px: int = 10,
    allow_hold: bool = False,
) -> str | None:
    """Leave a descending bullet's future collision corridor before it arrives."""
    impacts: list[tuple[float, float]] = []
    for bullet in bullets:
        previous = getattr(bullet, "_prev_xy", None)
        if previous is None:
            continue
        dx = bullet.xy[0] - previous[0]
        dy = bullet.xy[1] - previous[1]
        if not 0 < dy <= 20:
            continue
        steps = (player_y - bullet.xy[1]) / dy
        if 1 <= steps <= max_steps:
            impacts.append((steps, bullet.xy[0] + dx * steps))
    if not impacts:
        return None

    def projected_x(direction: int, steps: float) -> float:
        return max(PLAYER_MIN_X, min(PLAYER_MAX_X, player_x + direction * 2 * steps))

    def toward_target(steps: float) -> float:
        return max(player_x - 2 * steps, min(player_x + 2 * steps, target_x))

    if all(abs(toward_target(steps) + 3 - bullet_x) > corridor_px for steps, bullet_x in impacts):
        return None

    if allow_hold and all(abs(player_x + 3 - bullet_x) > corridor_px for _, bullet_x in impacts):
        return "HOLD"

    def clearance(direction: int) -> float:
        return min(abs(projected_x(direction, steps) + 3 - bullet_x) - corridor_px for steps, bullet_x in impacts)

    left = clearance(-1)
    right = clearance(1)
    if left == right:
        left_room = player_x - PLAYER_MIN_X
        right_room = PLAYER_MAX_X - player_x
        return "LEFT" if left_room > right_room else "RIGHT"
    return "LEFT" if left > right else "RIGHT"


def physical_action(
    env: Any,
    locked_bullet_x: int | None,
    fire_while_dodging: bool = False,
    *,
    allow_fire: bool = True,
    muzzle_offset: int = MUZZLE_OFFSET,
    verify_actual_shot: bool = False,
    predictive_dodge: bool = False,
    guarded_predictive_dodge: bool = False,
    lowest_alien_y: int | None = None,
) -> tuple[str, str]:
    """Navigate to the locked line, then fire in place; dodge only immediate threats."""
    player = visible_objects(env, "Player")[0]
    player_x, player_y = player.xy
    threats = []
    for bullet in visible_objects(env, "Bullet"):
        previous = getattr(bullet, "_prev_xy", None)
        if previous is None:
            continue
        dy = bullet.xy[1] - previous[1]
        if 0 < dy <= 20 and 135 <= bullet.xy[1] < player_y + 10:
            threats.append(bullet)

    def exposed(x: int, margin: int) -> bool:
        return any(abs(bullet.xy[0] - (x + 3)) <= margin for bullet in threats)

    if predictive_dodge:
        target_x = locked_bullet_x - muzzle_offset if locked_bullet_x is not None else player_x
        direction = forecast_escape_direction(player_x, player_y, target_x, visible_objects(env, "Bullet"))
        if direction is not None:
            projected_x = min(player_x + 2, PLAYER_MAX_X) if direction == "RIGHT" else max(player_x - 2, PLAYER_MIN_X)
            if projected_x != player_x:
                if allow_fire and fire_while_dodging:
                    frame = env.render()
                    if clear_shot(frame, player_x + muzzle_offset) and clear_shot(frame, projected_x + muzzle_offset):
                        direction += "FIRE"
                return direction, "forecast_dodge"

    if exposed(player_x, 8):
        options = [x for x in (player_x - 2, player_x + 2) if PLAYER_MIN_X <= x <= PLAYER_MAX_X]
        if options:
            safer = max(options, key=lambda x: (not exposed(x, 8), min(abs(x + 3 - b.xy[0]) for b in threats)))
            direction = "RIGHT" if safer > player_x else "LEFT"
            if allow_fire and fire_while_dodging:
                frame = env.render()
                if clear_shot(frame, player_x + muzzle_offset) and clear_shot(frame, safer + muzzle_offset):
                    direction += "FIRE"
            return direction, "dodge"
    if guarded_predictive_dodge and (lowest_alien_y is None or lowest_alien_y < 151):
        target_x = locked_bullet_x - muzzle_offset if locked_bullet_x is not None else player_x
        direction = forecast_escape_direction(
            player_x,
            player_y,
            target_x,
            visible_objects(env, "Bullet"),
            max_steps=6,
            corridor_px=8,
            allow_hold=True,
        )
        if direction == "HOLD":
            can_fire = allow_fire and clear_shot(env.render(), player_x + muzzle_offset)
            return ("FIRE" if can_fire else "NOOP"), "forecast_hold"
        if direction in ("LEFT", "RIGHT"):
            projected_x = min(player_x + 2, PLAYER_MAX_X) if direction == "RIGHT" else max(player_x - 2, PLAYER_MIN_X)
            if projected_x != player_x:
                if allow_fire and fire_while_dodging:
                    frame = env.render()
                    if clear_shot(frame, player_x + muzzle_offset) and clear_shot(frame, projected_x + muzzle_offset):
                        direction += "FIRE"
                return direction, "guarded_dodge"
    if locked_bullet_x is None:
        return ("FIRE" if allow_fire else "NOOP"), "fallback"
    target_x = locked_bullet_x - muzzle_offset
    if player_x < target_x - 1:
        projected_x = min(player_x + 2, PLAYER_MAX_X)
        if exposed(projected_x, 12):
            return (
                "FIRE" if allow_fire and clear_shot(env.render(), player_x + muzzle_offset) else "NOOP"
            ), "wait_threat"
        frame = env.render()
        safe = allow_fire and all(clear_shot(frame, x + muzzle_offset) for x in (player_x, projected_x))
        return ("RIGHTFIRE" if safe else "RIGHT"), "align"
    if player_x > target_x + 1:
        projected_x = max(player_x - 2, PLAYER_MIN_X)
        if exposed(projected_x, 12):
            return (
                "FIRE" if allow_fire and clear_shot(env.render(), player_x + muzzle_offset) else "NOOP"
            ), "wait_threat"
        frame = env.render()
        safe = allow_fire and all(clear_shot(frame, x + muzzle_offset) for x in (player_x, projected_x))
        return ("LEFTFIRE" if safe else "LEFT"), "align"
    if not allow_fire or (verify_actual_shot and not clear_shot(env.render(), player_x + muzzle_offset)):
        return "NOOP", "wait_shot"
    return "FIRE", "locked_fire"
