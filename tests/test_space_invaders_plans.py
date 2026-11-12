"""Behavior checks for the Space Invaders plan middleware."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def load_shared() -> ModuleType:
    folder = Path(__file__).resolve().parents[1] / "integrations" / "space-invaders"
    spec = importlib.util.spec_from_file_location("space_invaders_shared", folder / "lane_io.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_lane() -> ModuleType:
    folder = Path(__file__).resolve().parents[1] / "integrations" / "space-invaders"
    spec = importlib.util.spec_from_file_location("space_invaders_lane", folder / "lane_controller.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(folder))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(folder))
    return module


def load_lane_pilot() -> ModuleType:
    folder = Path(__file__).resolve().parents[1] / "integrations" / "space-invaders"
    spec = importlib.util.spec_from_file_location("space_invaders_lane_pilot", folder / "lane_pilot.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(folder))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(folder))
    return module


def load_solo_intercept() -> ModuleType:
    folder = Path(__file__).resolve().parents[1] / "integrations" / "space-invaders"
    spec = importlib.util.spec_from_file_location("space_invaders_solo_intercept", folder / "solo_intercept.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(folder))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(folder))
    return module


class Player:
    def __init__(self, xy: tuple[int, int]) -> None:
        self.xy = xy


class Alien:
    def __init__(self, xy: tuple[int, int]) -> None:
        self.xy = xy


class Satellite:
    def __init__(self, xy: tuple[int, int]) -> None:
        self.xy = xy


class Bullet:
    def __init__(self, xy: tuple[int, int], previous: tuple[int, int]) -> None:
        self.xy = xy
        self._prev_xy = previous


class Environment:
    def __init__(self, *objects: Any) -> None:
        self.objects = objects


class Screen:
    def __init__(self, shield_columns: set[int] | None = None) -> None:
        self.shield_columns = shield_columns or set()

    def __getitem__(self, position: tuple[int, int]) -> tuple[int, int, int]:
        y, x = position
        return (181, 83, 40) if y == 160 and x in self.shield_columns else (0, 0, 0)


class ScreenEnvironment(Environment):
    def __init__(self, screen: Screen, *objects: Any) -> None:
        super().__init__(*objects)
        self.screen = screen

    def render(self) -> Screen:
        return self.screen


def test_final_manifest_identifies_one_model_backed_lane_controller() -> None:
    folder = Path(__file__).resolve().parents[1] / "integrations" / "space-invaders"
    manifest = json.loads((folder / "final-harness.json").read_text())
    assert manifest["runner"] == "lane_pilot.py"
    assert manifest["model"]["policy"] == "jev"
    assert len(manifest["model"]["weights_sha256"]) == 64
    assert "--landing-priority" in manifest["runner_args"]
    assert all((folder / name).is_file() for name in manifest["runtime_files"])


def test_final_visible_state_encodes_only_observed_objects() -> None:
    shared = load_shared()
    env = Environment(Player((80, 185)), Alien((66, 151)), Bullet((81, 140), (81, 132)))
    parts = shared.state_parts(env, {"episode_frame_number": 400, "lives": 2}, 100, "plan_pairs")
    assert parts["essential"] == [
        "step=100 frame=400 lives=2",
        "player x=80 y=185",
        "aliens count=1 x_range=66..66 lowest_y=151",
        "nearest aliens=(66,151)",
        "descending bullets=(81,140,dx=0,dy=8)",
    ]


def test_lane_controller_avoids_blocked_shots_and_prefers_lower_aliens() -> None:
    lane = load_lane()
    env = ScreenEnvironment(Screen(set(range(56, 63))), Player((80, 185)), Alien((55, 130)), Alien((86, 95)))
    candidates = lane.candidate_lanes(env, velocity_x=0)
    assert len(candidates) == 1
    assert candidates[0]["bullet_x"] in range(87, 94)
    assert candidates[0]["lowest_alien_y"] == 95

    clear_env = ScreenEnvironment(Screen(), Player((80, 185)), Alien((55, 130)), Alien((86, 95)))
    candidates = lane.candidate_lanes(clear_env, velocity_x=0)
    assert candidates[0]["lowest_alien_y"] == 130


def test_lane_controller_ambushes_a_lone_alien_from_a_clear_central_line() -> None:
    lane = load_lane()
    assert lane.ambush_line(ScreenEnvironment(Screen(), Player((83, 185)))) == 88
    assert lane.ambush_line(ScreenEnvironment(Screen(set(range(65, 70))), Player((37, 185)))) == 70


def test_lane_teacher_question_omits_the_codes_answer_score() -> None:
    pilot = load_lane_pilot()
    lanes = [
        {"description": "Lock lane x=70; planner score=12.5"},
        {"description": "Lock lane x=80; planner score=8.0"},
    ]
    question = pilot.lane_question(lanes, include_score=False)
    assert question["control"]["criteria"] == {"A": "Lock lane x=70", "B": "Lock lane x=80"}
    orders = [pilot.reverse_options(595, step, balanced=True) for step in range(0, 1000, 16)]
    assert 20 <= sum(orders) <= 43
    assert pilot.reverse_options(595, 0, balanced=False)


def test_urgent_lane_menu_excludes_bonus_and_shallow_columns() -> None:
    pilot = load_lane_pilot()
    lanes = [
        {"target_type": "satellite", "lowest_alien_y": 12, "bullet_x": 60},
        {"target_type": "alien", "lowest_alien_y": 143, "bullet_x": 80},
        {"target_type": "alien", "lowest_alien_y": 161, "bullet_x": 96},
        {"target_type": "alien", "lowest_alien_y": 161, "bullet_x": 112},
    ]
    assert pilot.urgent_lanes(lanes, 141) == lanes
    assert [lane["bullet_x"] for lane in pilot.urgent_lanes(lanes, 161)] == [96, 112]
    assert [lane["bullet_x"] for lane in pilot.depth_lanes(lanes, 151)] == [60, 96, 112]
    assert pilot.urgent_target_expired(
        {"target_type": "satellite", "target_x": 60}, [Alien((80, 143)), Alien((96, 161))]
    )
    assert pilot.urgent_target_expired({"target_type": "alien", "target_x": 80}, [Alien((80, 143)), Alien((96, 161))])
    assert not pilot.urgent_target_expired(
        {"target_type": "alien", "target_x": 96}, [Alien((96, 143)), Alien((96, 161))]
    )
    assert not pilot.urgent_target_expired(
        {"target_type": "satellite", "target_x": 60}, [Alien((80, 143)), Alien((96, 151))]
    )


def test_lane_controller_waits_until_dangerous_bullet_passes() -> None:
    lane = load_lane()
    threat = Bullet((91, 160), (91, 156))
    near = ScreenEnvironment(Screen(), Player((82, 185)), threat)
    assert lane.physical_action(near, locked_bullet_x=95) == ("LEFT", "dodge")
    assert lane.physical_action(near, locked_bullet_x=95, fire_while_dodging=True) == ("LEFTFIRE", "dodge")
    evaded = ScreenEnvironment(Screen(), Player((76, 185)), threat)
    assert lane.physical_action(evaded, locked_bullet_x=95) == ("FIRE", "wait_threat")


def test_lane_controller_forecasts_escape_before_a_boundary_bullet_arrives() -> None:
    lane = load_lane()
    approaching = ScreenEnvironment(Screen(), Player((41, 185)), Bullet((39, 125), (39, 121)))
    assert lane.physical_action(approaching, locked_bullet_x=41, muzzle_offset=4) == ("LEFTFIRE", "align")
    assert lane.physical_action(
        approaching, locked_bullet_x=41, muzzle_offset=4, fire_while_dodging=True, predictive_dodge=True
    ) == ("RIGHTFIRE", "forecast_dodge")
    at_boundary = ScreenEnvironment(Screen(), Player((36, 185)), Bullet((39, 137), (39, 133)))
    assert lane.physical_action(at_boundary, locked_bullet_x=41, muzzle_offset=4, predictive_dodge=True) == (
        "RIGHT",
        "forecast_dodge",
    )


def test_guarded_forecast_holds_safe_position_and_yields_to_landing_urgency() -> None:
    lane = load_lane()
    approaching = ScreenEnvironment(Screen(), Player((34, 185)), Bullet((55, 163), (55, 159)))
    assert lane.physical_action(approaching, locked_bullet_x=51, muzzle_offset=4) == ("RIGHTFIRE", "align")
    assert lane.physical_action(
        approaching,
        locked_bullet_x=51,
        muzzle_offset=4,
        guarded_predictive_dodge=True,
        lowest_alien_y=121,
    ) == ("FIRE", "forecast_hold")
    assert lane.physical_action(
        approaching,
        locked_bullet_x=51,
        muzzle_offset=4,
        guarded_predictive_dodge=True,
        lowest_alien_y=161,
    ) == ("RIGHTFIRE", "align")


def test_calibrated_cannon_checks_the_real_missile_column() -> None:
    lane = load_lane()
    env = ScreenEnvironment(Screen({80}), Player((76, 185)))
    assert lane.physical_action(env, locked_bullet_x=80) == ("FIRE", "locked_fire")
    assert lane.physical_action(env, locked_bullet_x=80, muzzle_offset=4, verify_actual_shot=True) == (
        "NOOP",
        "wait_shot",
    )


def test_solo_tracker_forecasts_a_visible_edge_turn() -> None:
    solo = load_solo_intercept()
    tracker = solo.SoloTracker()
    tracker.observe((122, 121))
    tracker.observe((127, 121))
    assert tracker.speed == 5
    assert tracker.forecast(1) == (129, 131)
    tracker.observe((129, 131))
    assert tracker.direction == -1
    assert tracker.forecast(1) == (124, 131)


def test_solo_shot_is_released_before_the_alien_crosses_the_clear_line() -> None:
    solo = load_solo_intercept()
    tracker = solo.SoloTracker()
    tracker.observe((66, 161))
    tracker.observe((71, 161))
    clear = ScreenEnvironment(Screen(), Player((76, 185)))
    assert solo.solo_fire_ready(clear, tracker, 80)
    assert not solo.solo_fire_ready(ScreenEnvironment(Screen({80}), Player((76, 185))), tracker, 80)
    busy = ScreenEnvironment(Screen(), Player((76, 185)), Bullet((80, 145), (80, 153)))
    assert not solo.solo_fire_ready(busy, tracker, 80)


def test_solo_plans_use_reachable_muzzle_columns() -> None:
    solo = load_solo_intercept()
    tracker = solo.SoloTracker()
    tracker.observe((66, 151))
    tracker.observe((71, 151))
    env = ScreenEnvironment(Screen(), Player((48, 185)), Alien((71, 151)))
    plans = solo.candidate_solo_plans(env, tracker)
    assert plans
    assert all((plan["aim_x"] - 48) % 2 == 0 for plan in plans)


def test_solo_rescue_excludes_post_landing_intercepts_and_finds_a_clear_line() -> None:
    solo = load_solo_intercept()
    forecasts = [(80, 159)] * 3 + [(80, 161)] * 20
    assert solo.shot_arrival_step(forecasts, 84, 1) == 3
    assert solo.shot_arrival_step(forecasts, 84, 1, landing_limit_y=161) is None

    tracker = solo.SoloTracker()
    tracker.observe((80, 159))
    env = ScreenEnvironment(Screen({84}), Player((76, 185)))
    line = solo.emergency_solo_line(env, tracker)
    assert line is not None and line != 84 and (line - 80) % 2 == 0

    far_tracker = solo.SoloTracker()
    far_tracker.observe((22, 159))
    nearby_line = solo.emergency_solo_line(ScreenEnvironment(Screen(), Player((116, 185))), far_tracker)
    assert nearby_line is not None and nearby_line >= 108


def test_solo_intercept_accounts_for_existing_missile_cooldown() -> None:
    solo = load_solo_intercept()
    tracker = solo.SoloTracker()
    tracker.observe((110, 151))
    tracker.observe((115, 151))
    env = ScreenEnvironment(Screen(), Player((48, 185)), Alien((115, 151)), Bullet((52, 165), (52, 173)))
    assert solo.bullet_wait_steps(env) == 20
    plans = solo.candidate_solo_plans(env, tracker)
    assert plans
    assert plans[0]["bullet_x"] != 50
    assert plans[0]["launch_steps"] >= 20


def test_solo_plan_is_kept_through_its_launch_window() -> None:
    pilot = load_lane_pilot()
    plan = {"launch_steps": 20}
    assert not pilot.solo_replan_due(942, 926, plan, 16)
    assert not pilot.solo_replan_due(946, 926, plan, 16)
    assert not pilot.solo_replan_due(947, 926, plan, 16)
    assert pilot.solo_replan_due(949, 926, plan, 16)


def test_solo_fire_can_be_held_as_previous_missile_exits() -> None:
    solo = load_solo_intercept()
    tracker = solo.SoloTracker()
    tracker.observe((69, 161))
    tracker.observe((64, 161))
    exiting = Bullet((52, 37), (52, 45))
    env = ScreenEnvironment(Screen(), Player((42, 185)), Alien((64, 161)), exiting)
    assert solo.bullet_wait_steps(env) == 4
    assert solo.solo_fire_ready(env, tracker, 46)


def test_lane_controller_tracks_a_faster_column_after_an_edge_kill() -> None:
    lane = load_lane()
    previous = [(50, 90), (66, 90), (82, 90)]
    current = [Alien((69, 90)), Alien((85, 90))]
    assert lane.formation_shift(previous, current) == 3
    target = {"target_type": "alien", "target_x": 66}
    env = ScreenEnvironment(Screen(), Player((70, 185)), *current)
    assert lane.follow_target(env, target, alien_velocity_x=1.5, satellite_velocity_x=0) == 91
    assert target["target_x"] == 69


def test_lane_controller_intercepts_bonus_ship_only_with_a_clear_path() -> None:
    lane = load_lane()
    env = ScreenEnvironment(Screen(), Player((110, 185)), Alien((50, 80)), Satellite((80, 12)))
    candidates = lane.candidate_lanes(env, velocity_x=0, satellite_velocity_x=0.5)
    assert candidates[0]["target_type"] == "satellite"
    assert candidates[0]["bullet_x"] == 98
    blocked = ScreenEnvironment(Screen(set(range(95, 102))), Player((110, 185)), Satellite((80, 12)))
    assert lane.candidate_lanes(blocked, velocity_x=0, satellite_velocity_x=0.5) == []
    far = ScreenEnvironment(Screen(), Player((110, 185)), Satellite((16, 12)))
    assert lane.aim_at_satellite(far, satellite_x=16, velocity_x=1) is None


def test_lane_controller_keeps_reachable_edge_lane_until_formation_turns() -> None:
    lane = load_lane()
    env = ScreenEnvironment(Screen(), Player((110, 185)), Alien((113, 95)))
    assert lane.aim_at_alien(env, alien_x=113, lowest_y=95, velocity_x=1) == 121
