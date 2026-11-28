"""Visible-state prediction and timed firing for the last Space Invaders alien."""

from __future__ import annotations

from collections import deque
from statistics import median
from typing import Any

from lane_controller import ACTUAL_MUZZLE_OFFSET, clear_shot
from lane_io import PLAYER_MAX_X, PLAYER_MIN_X, visible_objects

ALIEN_MIN_X = 22
ALIEN_MAX_X = 129
ALIEN_CENTER_OFFSET = 4
MUZZLE_Y = 177
BULLET_STEP_Y = 8
MAX_FORECAST_STEPS = 48
SOLO_LANDING_LIMIT_Y = 169


class SoloTracker:
    """Estimate the lone alien's current direction without averaging across turns."""

    def __init__(self) -> None:
        self.last_xy: tuple[int, int] | None = None
        self.same_row_shifts: deque[int] = deque(maxlen=8)
        self.direction = 0
        self.age = 0

    def reset(self) -> None:
        self.last_xy = None
        self.same_row_shifts.clear()
        self.direction = 0
        self.age = 0

    def observe(self, xy: tuple[int, int]) -> None:
        if self.last_xy is not None:
            old_x, old_y = self.last_xy
            dx = xy[0] - old_x
            if xy[1] > old_y and self.direction:
                self.direction = -self.direction
            elif dx and xy[1] == old_y:
                self.direction = 1 if dx > 0 else -1
                if 1 <= abs(dx) <= 8:
                    self.same_row_shifts.append(abs(dx))
        self.last_xy = xy
        self.age += 1

    @property
    def speed(self) -> int:
        return round(median(self.same_row_shifts)) if self.same_row_shifts else 0

    def forecast(self, steps: int) -> tuple[int, int]:
        if self.last_xy is None:
            raise ValueError("observe an alien before forecasting")
        x, y = self.last_xy
        direction = self.direction
        speed = self.speed
        for _ in range(steps):
            if not direction or not speed:
                break
            next_x = x + direction * speed
            if next_x < ALIEN_MIN_X:
                x, y, direction = ALIEN_MIN_X, y + 10, 1
            elif next_x > ALIEN_MAX_X:
                x, y, direction = ALIEN_MAX_X, y + 10, -1
            else:
                x = next_x
        return x, y


def player_bullet(env: Any) -> Any | None:
    """The player's missile rises eight pixels per four-frame game decision."""
    for bullet in visible_objects(env, "Bullet"):
        previous = getattr(bullet, "_prev_xy", None)
        if previous is not None and -12 <= bullet.xy[1] - previous[1] < 0:
            return bullet
    return None


def bullet_wait_steps(env: Any) -> int:
    active = player_bullet(env)
    if active is None:
        return 0
    # A missile is still present at y=13; FIRE can launch another only on the
    # decision after it disappears. Include that one-decision cooldown.
    return max(1, (active.xy[1] - 16 + BULLET_STEP_Y - 1) // BULLET_STEP_Y + 1)


def shot_arrival_step(
    forecasts: list[tuple[int, int]],
    bullet_x: int,
    launch_delay: int,
    hit_margin: int = 3,
    landing_limit_y: int = 181,
) -> int | None:
    """Return the first future step when a new missile overlaps the alien."""
    for flight_steps in range(1, 21):
        arrival = launch_delay + flight_steps
        if arrival >= len(forecasts):
            break
        alien_x, alien_y = forecasts[arrival]
        if alien_y >= landing_limit_y:
            break
        bullet_y = MUZZLE_Y - BULLET_STEP_Y * flight_steps
        if (
            bullet_y <= alien_y + 9
            and bullet_y + 9 >= alien_y
            and abs(bullet_x - (alien_x + ALIEN_CENTER_OFFSET)) <= hit_margin
        ):
            return arrival
        if bullet_y + 9 < alien_y:
            break
    return None


def candidate_solo_plans(
    env: Any,
    tracker: SoloTracker,
    hit_margin: int = 3,
    landing_limit_y: int = 181,
) -> list[dict[str, Any]]:
    """Offer distinct, shield-clear intercept plans based on future crossing time."""
    if tracker.last_xy is None or not tracker.direction or not tracker.speed:
        return []
    player_x = visible_objects(env, "Player")[0].xy[0]
    frame = env.render()
    wait_steps = bullet_wait_steps(env)
    forecasts = [tracker.forecast(t) for t in range(MAX_FORECAST_STEPS + 1)]
    plans = []
    for bullet_x in range(PLAYER_MIN_X + ACTUAL_MUZZLE_OFFSET, PLAYER_MAX_X + ACTUAL_MUZZLE_OFFSET + 1):
        # The cannon advances two pixels per decision; a one-pixel aim error can leave
        # the real muzzle under a shield even when the proposed line is clear.
        if (bullet_x - ACTUAL_MUZZLE_OFFSET - player_x) % 2:
            continue
        if not clear_shot(frame, bullet_x):
            continue
        move_steps = (abs(bullet_x - ACTUAL_MUZZLE_OFFSET - player_x) + 1) // 2
        first_launch = max(move_steps, wait_steps)
        for launch_delay in range(first_launch, MAX_FORECAST_STEPS - 1):
            arrival = shot_arrival_step(forecasts, bullet_x, launch_delay, hit_margin, landing_limit_y)
            if arrival is None:
                continue
            score = 100.0 - arrival - 0.08 * move_steps - 0.015 * abs(bullet_x - 80)
            plans.append(
                {
                    "target_type": "solo_intercept",
                    "bullet_x": bullet_x,
                    "aim_x": bullet_x - ACTUAL_MUZZLE_OFFSET,
                    "arrival_steps": arrival,
                    "launch_steps": launch_delay,
                    "score": round(score, 3),
                    "description": (
                        f"Intercept last alien on clear firing lane x={bullet_x}; "
                        f"predicted hit in {arrival} decisions; move {move_steps} decisions"
                    ),
                }
            )
            break
    plans.sort(key=lambda plan: (-plan["score"], plan["bullet_x"]))
    distinct = []
    for plan in plans:
        if all(abs(plan["bullet_x"] - chosen["bullet_x"]) >= 4 for chosen in distinct):
            distinct.append(plan)
        if len(distinct) == 8:
            break
    return distinct


def solo_fire_ready(
    env: Any,
    tracker: SoloTracker,
    bullet_x: int,
    hit_margin: int = 3,
    landing_limit_y: int = 181,
) -> bool:
    """Fire before a predicted crossing, only when the real muzzle has a clear path."""
    if tracker.last_xy is None or not tracker.direction:
        return False
    wait_steps = bullet_wait_steps(env)
    if wait_steps > 4:
        return False
    player_x = visible_objects(env, "Player")[0].xy[0]
    actual_x = player_x + ACTUAL_MUZZLE_OFFSET
    if abs(actual_x - bullet_x) > 1 or not clear_shot(env.render(), actual_x):
        return False
    forecasts = [tracker.forecast(t) for t in range(25)]
    return shot_arrival_step(forecasts, actual_x, wait_steps, hit_margin, landing_limit_y) is not None


def emergency_solo_line(env: Any, tracker: SoloTracker, landing_limit_y: int = SOLO_LANDING_LIMIT_Y) -> int | None:
    """Pick a reachable clear line near the alien's remaining visible path."""
    if tracker.last_xy is None:
        return None
    player_x = visible_objects(env, "Player")[0].xy[0]
    current_line = player_x + ACTUAL_MUZZLE_OFFSET
    wait_steps = bullet_wait_steps(env)
    forecasts = [tracker.forecast(t) for t in range(17)]
    remaining = [(t, x + ALIEN_CENTER_OFFSET) for t, (x, y) in enumerate(forecasts) if y < landing_limit_y]
    if not remaining:
        remaining = [(0, tracker.last_xy[0] + ALIEN_CENTER_OFFSET)]
    frame = env.render()
    lines = [
        line
        for line in range(PLAYER_MIN_X + ACTUAL_MUZZLE_OFFSET, PLAYER_MAX_X + ACTUAL_MUZZLE_OFFSET + 1)
        if (line - current_line) % 2 == 0 and clear_shot(frame, line)
    ]
    if not lines:
        return None
    max_move_steps = min(6, max(0, len(remaining) - 3))
    nearby = [line for line in lines if abs(line - current_line) <= 2 * max_move_steps]
    if nearby:
        lines = nearby

    def lane_cost(line: int) -> tuple[float, int, int]:
        move_steps = abs(line - current_line) // 2
        crossings = [
            abs(line - alien_center) + 0.12 * t
            for t, alien_center in remaining
            if t >= max(move_steps, wait_steps) + 2
        ]
        if not crossings:
            crossings = [abs(line - alien_center) + 0.12 * t + 8 for t, alien_center in remaining]
        return min(crossings) + 0.08 * move_steps, move_steps, abs(line - 80)

    return min(lines, key=lane_cost)
