"""Run a Space Invaders policy that chooses a firing lane infrequently."""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from lane_controller import (
    ACTUAL_MUZZLE_OFFSET,
    ambush_line,
    candidate_lanes,
    clear_shot,
    follow_target,
    formation_shift,
    physical_action,
)
from lane_io import ACTION_NAMES, jev_action, state_parts, visible_objects
from solo_intercept import (
    SOLO_LANDING_LIMIT_Y,
    SoloTracker,
    bullet_wait_steps,
    candidate_solo_plans,
    emergency_solo_line,
    player_bullet,
    solo_fire_ready,
)

EARLY_DEPTH_Y = 151
URGENT_ALIEN_Y = 161


def depth_lanes(lanes: list[dict[str, Any]], lowest_y: int | None) -> list[dict[str, Any]]:
    """Remove shallower alien columns while still letting Jev compare a bonus target."""
    if lowest_y is None or lowest_y < EARLY_DEPTH_Y:
        return lanes
    alien_lanes = [lane for lane in lanes if lane["target_type"] == "alien"]
    if not alien_lanes:
        return lanes
    deepest_clear_y = max(lane["lowest_alien_y"] for lane in alien_lanes)
    return [lane for lane in lanes if lane["target_type"] != "alien" or lane["lowest_alien_y"] == deepest_clear_y]


def urgent_lanes(lanes: list[dict[str, Any]], lowest_y: int | None) -> list[dict[str, Any]]:
    """Keep Jev's choice among the deepest clear alien columns when landing is near."""
    if lowest_y is None or lowest_y < URGENT_ALIEN_Y:
        return lanes
    alien_lanes = [lane for lane in lanes if lane["target_type"] == "alien"]
    if not alien_lanes:
        return lanes
    deepest_clear_y = max(lane["lowest_alien_y"] for lane in alien_lanes)
    return [lane for lane in alien_lanes if lane["lowest_alien_y"] == deepest_clear_y]


def urgent_target_expired(target: dict[str, Any] | None, aliens: list[Any]) -> bool:
    """Interrupt a bonus or shallower target as soon as a low alien becomes urgent."""
    if target is None or len(aliens) < 2:
        return False
    lowest_y = max(alien.xy[1] for alien in aliens)
    if lowest_y < URGENT_ALIEN_Y:
        return False
    if target["target_type"] != "alien":
        return True
    columns: dict[int, int] = {}
    for alien in aliens:
        x, y = alien.xy
        columns[x] = max(columns.get(x, 0), y)
    nearest_x = min(columns, key=lambda x: abs(x - target["target_x"]))
    return abs(nearest_x - target["target_x"]) > 8 or columns[nearest_x] < lowest_y


def lane_question(lanes: list[dict[str, Any]], include_score: bool = True) -> dict[str, Any]:
    def description(lane: dict[str, Any]) -> str:
        return lane["description"] if include_score else lane["description"].rsplit("; planner score=", 1)[0]

    return {
        "control": {
            "type": "choice",
            "instructions": (
                "Choose which visible target to track. Code updates the aim each game step as the alien column "
                "moves, leads shots, and briefly dodges incoming bullets. Lower aliens are more urgent; the "
                "Command Alien Ship is a 200-point bonus. Options have clear firing paths. The target is held "
                "until the next strategic choice, unless it disappears."
            ),
            "criteria": {"A": description(lanes[0]), "B": description(lanes[1])},
        }
    }


def solo_question(plans: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "control": {
            "type": "choice",
            "instructions": (
                "Choose one of two clear intercept plans for the last alien. Code predicts its next crossing, "
                "moves to the firing line, avoids descending bullets, and times the shot. "
                "Prefer an earlier reachable hit before the alien descends."
            ),
            "criteria": {"A": plans[0]["description"], "B": plans[1]["description"]},
        }
    }


def solo_plan_pair(state: dict[str, Any], plans: list[dict[str, Any]]) -> tuple[int, ...]:
    """Offer Jev two distinct reachable crossings when the forecast has them."""
    if len(plans) < 2:
        return (0,)
    second = next((i for i in range(1, len(plans)) if abs(plans[i]["bullet_x"] - plans[0]["bullet_x"]) >= 10), 1)
    return (0, second)


def solo_hit_margin(state: dict[str, Any]) -> int:
    """Allow a small horizontal collision window for the RAM alien sprite."""
    return 3


def solo_replan_due(step: int, last_choice_step: int, plan: dict[str, Any] | None, interval: int) -> bool:
    if plan is None:
        return True
    if step - last_choice_step < interval:
        return False
    remaining_to_launch = last_choice_step + plan["launch_steps"] - step
    return not -2 <= remaining_to_launch <= 4


def reverse_options(seed: int, step: int, balanced: bool) -> bool:
    """Keep legacy ordering by default; vary labels within each game for distillation."""
    if balanced:
        return bool(random.Random(seed * 1_000_003 + step * 9_176).getrandbits(1))
    return bool((seed + step) % 2)


def run_seed(args: argparse.Namespace, seed: int, record: Any, teacher_record: Any = None) -> dict[str, Any]:
    from ocatari.core import OCAtari

    env = OCAtari(
        "ALE/SpaceInvaders-v5",
        mode="ram",
        hud=True,
        obs_mode="obj",
        render_mode="rgb_array",
        create_buffer_stacks=[],
        repeat_action_probability=0.25,
        frameskip=4,
    )
    rng = random.Random(seed)
    total_reward = 0.0
    locked_x: int | None = None
    locked_target: dict[str, Any] | None = None
    last_choice_step = -args.replan_steps
    previous_aliens: list[tuple[int, int]] = []
    previous_satellite_x: int | None = None
    recent_shifts: deque[float] = deque(maxlen=8)
    recent_satellite_shifts: deque[int] = deque(maxlen=4)
    model_calls = 0
    strategic_choices = 0
    lane_changes = 0
    positive_reward_steps = 0
    bonus_hits = 0
    bonus_preempts = 0
    aim_updates = 0
    hysteresis_keeps = 0
    forced_opening_choices = 0
    opening_event_redirects = 0
    solo_model_calls = 0
    solo_fire_attempts = 0
    urgent_interrupts = 0
    solo_tracker = SoloTracker()
    solo_plan: dict[str, Any] | None = None
    last_solo_choice_step = -args.replan_steps
    wave = 0
    action_reasons: dict[str, int] = {}
    finished = False
    try:
        _, info = env.reset(seed=seed)
        if tuple(env._env.unwrapped.get_action_meanings()) != ACTION_NAMES:
            raise RuntimeError("Unexpected ALE action order")
        for step in range(args.max_steps):
            aliens = visible_objects(env, "Alien")
            satellites = visible_objects(env, "Satellite")
            player_x = visible_objects(env, "Player")[0].xy[0]
            wave_started = bool(aliens) and (not previous_aliens or len(aliens) >= len(previous_aliens) + 15)
            if wave_started:
                wave += 1
                recent_shifts.clear()
                locked_target = None
                locked_x = None
                solo_tracker.reset()
                solo_plan = None
            solo_predictive = len(aliens) == 1 and args.solo_predictive
            if not solo_predictive and solo_tracker.age:
                solo_tracker.reset()
                solo_plan = None
            solo_ambush = len(aliens) == 1 and args.single_alien_strategy == "ambush" and not solo_predictive
            if solo_ambush and (locked_target is None or locked_target["target_type"] != "ambush"):
                locked_x = ambush_line(env)
                locked_target = {"target_type": "ambush", "target_x": locked_x} if locked_x is not None else None
            elif not solo_ambush and locked_target is not None and locked_target["target_type"] == "ambush":
                locked_target = None
                locked_x = None
            if previous_aliens and aliens:
                recent_shifts.append(formation_shift(previous_aliens, aliens))
            previous_aliens = [alien.xy for alien in aliens]
            velocity_x = sum(recent_shifts) / len(recent_shifts) if recent_shifts else 0.0
            if satellites:
                satellite_x = satellites[0].xy[0]
                if previous_satellite_x is not None and abs(satellite_x - previous_satellite_x) <= 8:
                    recent_satellite_shifts.append(satellite_x - previous_satellite_x)
                previous_satellite_x = satellite_x
            else:
                previous_satellite_x = None
                recent_satellite_shifts.clear()
            satellite_velocity_x = (
                sum(recent_satellite_shifts) / len(recent_satellite_shifts) if recent_satellite_shifts else 0.0
            )
            lowest_alien_y = max((alien.xy[1] for alien in aliens), default=None)
            urgent = args.landing_priority and lowest_alien_y is not None and lowest_alien_y >= URGENT_ALIEN_Y
            early_depth = args.landing_priority and lowest_alien_y is not None and lowest_alien_y >= EARLY_DEPTH_Y
            if urgent and urgent_target_expired(locked_target, aliens):
                locked_target = None
                locked_x = None
                urgent_interrupts += 1
            # Replanning has a hard minimum interval; dodges never call the model.
            choice = None
            lanes = None
            offered_lanes = None
            selected_lane = None
            model_interval_elapsed = step - last_choice_step >= args.replan_steps
            if (
                not solo_ambush
                and not solo_predictive
                and (model_interval_elapsed or (locked_target is None and aliens))
            ):
                lanes = candidate_lanes(env, velocity_x, locked_x, satellite_velocity_x)
                if urgent:
                    lanes = urgent_lanes(lanes, lowest_alien_y)
                elif early_depth:
                    lanes = depth_lanes(lanes, lowest_alien_y)
                if lanes:
                    offered = lanes[:2]
                    strategic_choices += len(offered) == 2
                    if len(offered) == 2 and reverse_options(seed, step, args.balanced_option_order):
                        offered.reverse()
                    offered_lanes = [dict(lane) for lane in offered]
                    if teacher_record is not None and len(offered) == 2 and model_interval_elapsed:
                        teacher_record.write(
                            json.dumps(
                                {
                                    "seed": seed,
                                    "step": step,
                                    "parts": state_parts(env, info, step, "plan_pairs"),
                                    "question": lane_question(offered, include_score=False),
                                    "label": "A" if offered[0]["score"] > offered[1]["score"] else "B",
                                    "delta": round(offered[0]["score"] - offered[1]["score"], 3),
                                    "teacher": "code_lane_score",
                                }
                            )
                            + "\n"
                        )
                    close_choice = (
                        len(offered) == 2 and abs(offered[0]["score"] - offered[1]["score"]) <= args.model_margin
                    )
                    use_model = len(offered) == 2 and (args.model_choice_mode == "all" or close_choice)
                    if args.policy == "jev" and use_model and model_interval_elapsed:
                        parts = state_parts(env, info, step, "plan_pairs")
                        choice, _, _ = jev_action(
                            args.decision_server,
                            parts,
                            f"lane:{seed}:{step}",
                            lane_question(offered, include_score=not args.scoreless_question),
                        )
                        selected = offered[0 if choice == "A" else 1]
                        model_calls += 1
                    elif args.policy == "random" and len(offered) == 2:
                        selected = rng.choice(offered)
                    else:
                        selected = max(offered, key=lambda lane: lane["score"])
                    forced_opening_choices += bool(selected.get("meta_forced"))
                    if (
                        locked_target is not None
                        and locked_target["target_type"] == "alien"
                        and selected["target_type"] == "alien"
                        and not selected.get("meta_forced")
                    ):
                        current = next(
                            (
                                lane
                                for lane in lanes
                                if lane["target_type"] == "alien"
                                and abs(lane["target_x"] - locked_target["target_x"]) <= 8
                            ),
                            None,
                        )
                        if (
                            current is not None
                            and abs(selected["aim_x"] - player_x) > 40
                            and abs(current["aim_x"] - player_x) < 20
                            and selected["score"] - current["score"] < 8
                        ):
                            selected = current
                            hysteresis_keeps += 1
                    if selected["bullet_x"] != locked_x:
                        lane_changes += 1
                    selected_lane = dict(selected)
                    locked_x = selected["bullet_x"]
                    locked_target = {"target_type": selected["target_type"], "target_x": selected["target_x"]}
                else:
                    locked_x = None
                    locked_target = None
                if model_interval_elapsed:
                    last_choice_step = step
            elif (
                not solo_ambush
                and not solo_predictive
                and not urgent
                and satellites
                and (locked_target is None or locked_target["target_type"] != "satellite")
            ):
                # A transient 200-point bonus target should not wait for the next model call.
                bonus_lanes = candidate_lanes(env, velocity_x, locked_x, satellite_velocity_x)
                bonus = next((lane for lane in bonus_lanes if lane["target_type"] == "satellite"), None)
                if bonus is not None:
                    locked_target = {"target_type": "satellite", "target_x": bonus["target_x"]}
                    locked_x = bonus["bullet_x"]
                    bonus_preempts += 1
            if locked_target is not None and not solo_ambush and not solo_predictive:
                updated_x = follow_target(env, locked_target, velocity_x, satellite_velocity_x)
                opening_event_redirects += bool(locked_target.get("meta_opening_event"))
                if updated_x is None:
                    fallback = candidate_lanes(env, velocity_x, locked_x, satellite_velocity_x)
                    if urgent:
                        fallback = urgent_lanes(fallback, lowest_alien_y)
                    elif early_depth:
                        fallback = depth_lanes(fallback, lowest_alien_y)
                    if fallback:
                        selected = fallback[0]
                        locked_target = {"target_type": selected["target_type"], "target_x": selected["target_x"]}
                        updated_x = selected["bullet_x"]
                    else:
                        locked_target = None
                if updated_x is not None:
                    aim_updates += updated_x != locked_x
                locked_x = updated_x
            fire_ready = True
            hit_margin = None
            if solo_predictive:
                solo_tracker.observe(aliens[0].xy)
                solo_state = {
                    "wave": wave,
                    "alien_x": aliens[0].xy[0],
                    "alien_y": aliens[0].xy[1],
                    "direction": solo_tracker.direction,
                    "speed": solo_tracker.speed,
                    "solo_age": solo_tracker.age,
                    "player_x": player_x,
                    "bullet_wait_steps": bullet_wait_steps(env),
                    "locked_bullet_x": locked_x,
                }
                hit_margin = solo_hit_margin(dict(solo_state))
                if type(hit_margin) is not int or not 0 <= hit_margin <= 5:
                    raise ValueError("solo_hit_margin must return an integer from 0 to 5")
                if solo_plan is not None and (
                    not clear_shot(env.render(), solo_plan["bullet_x"]) or (solo_plan["aim_x"] - player_x) % 2
                ):
                    solo_plan = None
                emergency_window = args.solo_deadline_rescue and aliens[0].xy[1] >= 151
                if emergency_window and solo_plan is not None and not solo_plan.get("emergency"):
                    deadline = next((t for t in range(49) if solo_tracker.forecast(t)[1] >= SOLO_LANDING_LIMIT_Y), 49)
                    remaining_arrival = solo_plan["arrival_steps"] - (step - last_solo_choice_step)
                    if remaining_arrival >= deadline:
                        solo_plan = None
                if solo_replan_due(step, last_solo_choice_step, solo_plan, args.replan_steps):
                    landing_limit = SOLO_LANDING_LIMIT_Y if emergency_window else 181
                    options = candidate_solo_plans(env, solo_tracker, hit_margin, landing_limit)
                    if options:
                        pair = solo_plan_pair(dict(solo_state), [dict(plan) for plan in options])
                        if not isinstance(pair, (tuple, list)) or not 1 <= len(pair) <= 2:
                            raise ValueError("solo_plan_pair must return one or two plan indices")
                        if len(options) >= 2 and len(pair) != 2:
                            raise ValueError("solo_plan_pair must offer Jev two plans when two are available")
                        if any(type(index) is not int or not 0 <= index < len(options) for index in pair) or len(
                            set(pair)
                        ) != len(pair):
                            raise ValueError("solo_plan_pair returned invalid plan indices")
                        offered = [options[index] for index in pair]
                    else:
                        offered = []
                    if len(offered) == 2 and reverse_options(seed, step, args.balanced_option_order):
                        offered.reverse()
                    offered_lanes = [dict(plan) for plan in offered]
                    if offered:
                        if len(offered) == 2:
                            strategic_choices += 1
                        if args.policy == "jev" and len(offered) == 2:
                            choice, _, _ = jev_action(
                                args.decision_server,
                                state_parts(env, info, step, "plan_pairs"),
                                f"solo:{seed}:{step}",
                                solo_question(offered),
                            )
                            solo_model_calls += 1
                            model_calls += 1
                            solo_plan = offered[0 if choice == "A" else 1]
                        elif args.policy == "random" and len(offered) == 2:
                            solo_plan = rng.choice(offered)
                        else:
                            solo_plan = offered[0]
                        last_solo_choice_step = step
                    else:
                        solo_plan = None
                if emergency_window and solo_plan is None:
                    emergency_x = emergency_solo_line(env, solo_tracker)
                    if emergency_x is not None:
                        solo_plan = {
                            "target_type": "solo_emergency",
                            "bullet_x": emergency_x,
                            "aim_x": emergency_x - ACTUAL_MUZZLE_OFFSET,
                            "arrival_steps": 0,
                            "launch_steps": 0,
                            "emergency": True,
                            "description": f"Emergency clear firing line x={emergency_x}",
                        }
                        last_solo_choice_step = step
                locked_x = solo_plan["bullet_x"] if solo_plan else None
                selected_lane = dict(solo_plan) if solo_plan else None
                locked_target = {"target_type": "solo_intercept", "target_x": aliens[0].xy[0]}
                fire_ready = locked_x is not None and (
                    (solo_plan is not None and bool(solo_plan.get("emergency")))
                    or solo_fire_ready(
                        env,
                        solo_tracker,
                        locked_x,
                        hit_margin,
                        SOLO_LANDING_LIMIT_Y if emergency_window else 181,
                    )
                )
            action, reason = physical_action(
                env,
                locked_x,
                fire_while_dodging=args.fire_while_dodging,
                allow_fire=fire_ready,
                muzzle_offset=ACTUAL_MUZZLE_OFFSET if solo_predictive or args.calibrated_geometry else 5,
                verify_actual_shot=solo_predictive or args.calibrated_geometry,
                predictive_dodge=args.predictive_dodge,
                guarded_predictive_dodge=args.guarded_predictive_dodge,
                lowest_alien_y=lowest_alien_y,
            )
            if solo_predictive and "FIRE" in action:
                solo_fire_attempts += 1
            action_reasons[reason] = action_reasons.get(reason, 0) + 1
            bullets = (
                [
                    {"xy": bullet.xy, "previous_xy": getattr(bullet, "_prev_xy", None)}
                    for bullet in visible_objects(env, "Bullet")
                ]
                if record is not None
                else None
            )
            lives_before = info["lives"]
            solo_alien_xy = aliens[0].xy if solo_predictive else None
            player_bullet_before = player_bullet(env) if record is not None else None
            player_bullet_before_xy = player_bullet_before.xy if player_bullet_before else None
            _, reward, terminated, truncated, next_info = env.step(ACTION_NAMES.index(action))
            total_reward += reward
            positive_reward_steps += reward > 0
            bonus_hits += reward >= 100
            if record is not None:
                player_bullet_after = player_bullet(env)
                record.write(
                    json.dumps(
                        {
                            "seed": seed,
                            "step": step,
                            "player_x": player_x,
                            "locked_bullet_x": locked_x,
                            "action": action,
                            "reason": reason,
                            "reward": reward,
                            "lives_before": lives_before,
                            "lives_after": next_info["lives"],
                            "bullets": bullets,
                            "model_choice": choice,
                            "target_type": locked_target["target_type"] if locked_target else None,
                            "alien_count": len(aliens),
                            "satellite_visible": bool(satellites),
                            "formation_velocity_x": round(velocity_x, 3),
                            "satellite_velocity_x": round(satellite_velocity_x, 3),
                            "lane_options": offered_lanes,
                            "selected_lane": selected_lane,
                            "lowest_alien_y": lowest_alien_y,
                            "urgent_low_aliens": urgent,
                            "early_depth_priority": early_depth,
                            "wave": wave,
                            "actual_muzzle_x": player_x + ACTUAL_MUZZLE_OFFSET,
                            "player_bullet_before_xy": player_bullet_before_xy,
                            "player_bullet_after_xy": player_bullet_after.xy if player_bullet_after else None,
                            "solo_plan": solo_plan if solo_predictive else None,
                            "solo_alien_xy": solo_alien_xy,
                            "solo_speed": solo_tracker.speed if solo_predictive else None,
                            "solo_direction": solo_tracker.direction if solo_predictive else None,
                            "solo_fire_ready": fire_ready if solo_predictive else None,
                            "solo_hit_margin": hit_margin,
                            "meta_solo": (
                                locked_target.get("meta_observation") if len(aliens) == 1 and locked_target else None
                            ),
                            "meta_opening_event": bool(locked_target and locked_target.get("meta_opening_event")),
                        }
                    )
                    + "\n"
                )
            info = next_info
            if terminated or truncated:
                finished = True
                break
        return {
            "seed": seed,
            "policy": args.policy,
            "replan_steps": args.replan_steps,
            "steps": step + 1,
            "reward": total_reward,
            "lives": info["lives"],
            "finished": finished,
            "model_calls": model_calls,
            "lane_changes": lane_changes,
            "strategic_choices": strategic_choices,
            "positive_reward_steps": positive_reward_steps,
            "action_reasons": action_reasons,
            "bonus_hits": bonus_hits,
            "bonus_preempts": bonus_preempts,
            "aim_updates": aim_updates,
            "hysteresis_keeps": hysteresis_keeps,
            "forced_opening_choices": forced_opening_choices,
            "opening_event_redirects": opening_event_redirects,
            "solo_model_calls": solo_model_calls,
            "solo_fire_attempts": solo_fire_attempts,
            "urgent_interrupts": urgent_interrupts,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("best", "random", "jev"), default="best")
    parser.add_argument("--decision-server")
    parser.add_argument("--replan-steps", type=int, default=16)
    parser.add_argument("--model-margin", type=float, default=2.0)
    parser.add_argument("--model-choice-mode", choices=("close", "all"), default="close")
    parser.add_argument("--scoreless-question", action="store_true")
    parser.add_argument("--balanced-option-order", action="store_true")
    parser.add_argument("--single-alien-strategy", choices=("track", "ambush"), default="track")
    parser.add_argument("--solo-predictive", action="store_true")
    parser.add_argument("--solo-deadline-rescue", action="store_true")
    parser.add_argument("--calibrated-geometry", action="store_true")
    parser.add_argument("--fire-while-dodging", action="store_true")
    dodge_mode = parser.add_mutually_exclusive_group()
    dodge_mode.add_argument("--predictive-dodge", action="store_true")
    dodge_mode.add_argument("--guarded-predictive-dodge", action="store_true")
    parser.add_argument(
        "--landing-priority", action="store_true", help="Prioritize deepest alien columns near the landing line"
    )
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--teacher-record", type=Path)
    args = parser.parse_args()
    if args.replan_steps <= 0 or args.max_steps <= 0 or args.model_margin < 0:
        parser.error("replan-steps and max-steps must be positive; model-margin must be nonnegative")
    if args.policy == "jev" and not args.decision_server:
        parser.error("Jev policy requires --decision-server")
    if args.teacher_record and not args.balanced_option_order:
        parser.error("Code-teacher records require --balanced-option-order")
    with ExitStack() as stack:
        record = None
        teacher_record = None
        if args.record:
            args.record.parent.mkdir(parents=True, exist_ok=True)
            record = stack.enter_context(args.record.open("w"))
        if args.teacher_record:
            args.teacher_record.parent.mkdir(parents=True, exist_ok=True)
            teacher_record = stack.enter_context(args.teacher_record.open("w"))
        results = [run_seed(args, int(seed), record, teacher_record) for seed in args.seeds.split(",")]
    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()
