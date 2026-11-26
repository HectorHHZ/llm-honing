"""Write a VVVVVV Exp 0 v0 collection plan (pilot or main). The main plan is frozen only after the user's review."""

import argparse
import json
from pathlib import Path

FAULTS = {
    "harness": {
        "family": "observation_omission",
        "instance": "omit-local-terrain-v1",
        "trigger_rule": "first model decision whose 9x9 local rows contain a solid or spike cell (in practice decision 0)",
        "changed_artifact": "localSolidRows in the model request (solid and spike cells rendered as clear)",
    },
    "weights": {
        "family": "alternate_checkpoint",
        "instance": "laya-ra2-v33-checkpoint-v1",
        "trigger_rule": "every decision of the episode",
        "changed_artifact": "decision checkpoint weights (jayzou3773/laya-ra2-v33@51d2f5a0, service 8814)",
    },
    "controller": {
        "family": "post_selection_direction_swap",
        "instance": "swap-first-horizontal-v1",
        "trigger_rule": "first model decision whose selected action has a horizontal direction (left, right, flip_left, flip_right)",
        "changed_artifact": "native action sent to the game (left and right swapped once)",
    },
}
# Persistent forms for the v0 release (2026-11-26, experiments/exp0/20261126_0045_vvvvvv_v0_collection). The brief's
# default harness fault (remove `flip`) is invalid for the integration's receiver (exact six-option set), so the
# harness fault is the persistent omission of the local floor/ground information (#185), as the HK collection's
# omit-ground-v1; family names follow the HK collection (code/exp0/hk_v9/faults.py).
FAULTS_PERSISTENT = {
    "harness": {
        "family": "observation-omission",
        "instance": "omit-local-terrain-persistent-v1",
        "trigger_rule": (
            "every model decision: the request is rendered from the state with the 9x9 local rows "
            "showing solid and spike cells as clear; the trigger is the first decision at which "
            "this changes the request (decision 0 from the start)"
        ),
        "changed_artifact": "localSolidRows in the model request (solid '#' and spike '!' cells rendered as clear '.')",
    },
    "weights": {
        "family": "alternate-checkpoint",
        "instance": "laya-ra2-v33-v1",
        "trigger_rule": "every decision of the episode",
        "changed_artifact": "decision checkpoint weights (jayzou3773/laya-ra2-v33@51d2f5a0, service 8814)",
    },
    "controller": {
        "family": "post-selection-input-remap",
        "instance": "swap-left-right-persistent-v1",
        "trigger_rule": (
            "every model decision whose selected input has a horizontal direction (left, right, "
            "flip_left, flip_right): left and right are exchanged after the selection; the trigger "
            "is the first such decision"
        ),
        "changed_artifact": "native input sent to the game (left and right exchanged; flip edges kept)",
    },
}
OUTCOMES = {"win": ["crossed_conundrum"], "loss": ["player_died"], "timeout": ["decision_cap"], "other_terminal": []}


def scenario(seed: int, deaths_end: bool, cap: int) -> dict:
    return {
        "task": "ss1-welcome-aboard-to-past-conundrum",
        "start": {
            "mode": "Space Station 1 time trial",
            "room": [113, 105],
            "room_name": "Welcome Aboard",
            "player": [200, 161],
            "gravity": "down",
            "deaths": 0,
            "setup": "native countdown, 165 ticks",
        },
        "policy_seed": seed,
        "ticks_per_decision": 15,
        "deaths_end_episode": deaths_end,
        "win_rule": "the player enters a room other than [113,105] Welcome Aboard and [114,105] Conundrum",
        "caps": {"decisions": cap},
    }


RELEASE_RULES = {
    "frozen": "2026-11-26 01:12 PST (San Diego), before the first release attempt; brief experiments/exp0/20261126_0045_vvvvvv_v0_collection/BRIEF.md",
    "episode": (
        "variant A: start at the native Space Station 1 time-trial start (room [113,105] 'Welcome Aboard', after the "
        "native 165-tick countdown); win = the player enters a room other than [113,105] and [114,105] 'Conundrum'; a "
        "death respawns the player at the game's checkpoint (the death animation runs as `auto` rows of `wait` "
        "windows, no model call) and the episode goes on; timeout = 150 model decisions"
    ),
    "request": "the merged integration's own run.request_for: rule text + the full bridge observation; 6 raw inputs, each held 15 native ticks (0.51 s); no autopilot, no hidden or ranked options",
    "game": "the pinned original build code/exp0/games/vvvvvv/build/VVVVVV (no speed-up bridge, no developer hooks, no lookahead)",
    "selection": "seeded sampling from the model's probabilities (contract.Sampler), one fixed policy seed per reset group, shared by the group's four conditions",
    "groups": "s01-s10 initial (seeds 1-10), s11-s16 reserves (seeds 11-16) in listed order; one layout only, so the groups are correlated (#185: extra seeds do not establish new layouts)",
    "scored_selection": (
        "Griddly v0 PROTOCOL.md line 9 as implemented by code/exp0/common/audit_set.py choose(): per slot the first "
        "eligible attempt; the first ten listed groups per condition; interventions non-wins first in plan order; "
        "controls the first ten, swapping in one of the other outcome when all ten share an outcome"
    ),
    "reserves": (
        "collected only (a) for a slot of an initial group that has no eligible attempt after its attempt cap (failed "
        "validation, incomplete, untriggered fault, missing evidence), in listed reserve order and for that condition "
        "only; or (b) for controls, when all ten initial controls share one outcome: reserve controls in listed order "
        "until one of the other outcome exists. An intervention win does not by itself trigger reserves (as Griddly)"
    ),
    "attempt_cap_per_slot": 3,
    "response_tries": "2 per decision (the identical request again after a rejected reply)",
    "concurrency": "at most 4 game instances: three clean-model streams (one per condition) on 8813, the weights stream on 8814",
    "budgets": {
        "collection_wall_hours_estimate": "0.8-1.0 (40 attempts; pilot: 1.9 s per call on 8813 with three clients, "
        "0.10 s per window, ~315 s per 150-decision attempt)",
        "reserve_wall_hours_estimate": "up to 0.5 more",
        "codex_calls": "80 primary (read + probe, run first) + 40 pre-registered probe variant (diagnosis_prompt_probe_variant.txt)",
        "diagnosis_timeout_seconds": 240,
        "diagnosis_max_attempts": 2,
        "laya_calls_outside_episodes": "weights re-ask <= 4 per weights attempt; probes <= 2 x 2 x 4 per episode",
    },
    "diagnosis_prompts": {
        "primary": "code/exp0/common/diagnosis_prompt.txt (Griddly's prompt, game nouns only; P59 option A)",
        "variant": "code/exp0/common/diagnosis_prompt_probe_variant.txt (primary + one sentence; probe mode only; reported separately)",
    },
    "history": (
        "phase 1 (2026-11-23) inspected pilot groups p01-p04 (seeds 9001-9004) and dev probes (seeds 1000+, 3000+, 4000+); "
        "the 2026-11-26 persistent-fault pilot re-used p01. None of the release seeds 1-16 was run before the freeze"
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--kind", choices=["pilot", "main", "release"], required=True)
    ap.add_argument("--faults", choices=["once", "persistent"], default="once")
    ap.add_argument("--pilot-groups", default="", help="pilot only: comma-separated pilot ids, e.g. p01")
    ap.add_argument("--deaths-end", choices=["yes", "no"], default="yes")
    ap.add_argument("--cap", type=int, default=120)
    args = ap.parse_args()
    deaths_end = args.deaths_end == "yes"
    if args.kind == "pilot":
        groups = [
            {"id": f"p{n:02d}", "split": "train", "scenario": scenario(9000 + n, deaths_end, args.cap)}
            for n in range(1, 9)
        ]
        initial = 5
        if args.pilot_groups:
            groups = [g for g in groups if g["id"] in args.pilot_groups.split(",")]
            initial = len(groups)
    else:
        splits = ["train", "train", "val", "val"]
        groups = [
            {"id": f"s{n:02d}", "split": splits[n % 4], "scenario": scenario(n, deaths_end, args.cap)}
            for n in range(1, 17)
        ]
        initial = 10
    plan = {
        "schema": "honing-exp0-collection-v0",
        "game_id": "vvvvvv",
        "dataset_dir": "vvvvvv/v0/",
        "shape": "native",
        "question_key": "action",
        "observation": "vvvvvv-integration-default-v1 (9x9 local tiles + current-room entities)",
        "per_condition": 10,
        "initial_groups": initial,
        "selection": {"rule": "sample", "seed": None, "per_group_seed": "scenario.policy_seed"},
        "clean_weights_sha256": "4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e",
        "alternate_weights_sha256": "2ed46ac94c366db2dc68751d0eabfe945ceaeaf430e9fea1ec9e6c60c0207925",
        "weight_difference_tolerance": 0.01,
        "probe": {"first_decisions": 4, "identity_tolerance": 0.01, "tie": 0.01},
        "diagnosis": {
            "max_tokens": 9000,
            # 2026-11-26: the #168 renderer prints times as seconds, so the release uses game seconds
            # (native_before.sim_seconds = tick x 34 ms); phase-1 plans used native ticks
            "time": (
                {"unit": "sim_seconds", "before": 3600, "after": 0}
                if args.faults == "persistent"
                else {"unit": "native_tick", "before": 100000, "after": 0}
            ),
            "preamble": (
                "Times are seconds of game time: one native tick is 34 ms, and every model decision holds one "
                "of six inputs for 15 ticks (0.51 s); the game pauses while the model decides. Gravity can be "
                "flipped only while standing on a floor or ceiling."
                if args.faults == "persistent"
                else "VVVVVV counts native ticks of 34 ms; every model decision holds one of six inputs for 15 ticks. "
                "Gravity can be flipped only while standing on a floor or ceiling."
            ),
            "windows": {"max_runs_listed": 12, "max_mismatches_listed": 6},
            "public_observation_max_chars": 1500,
        },
        "outcomes": OUTCOMES,
        "faults": FAULTS_PERSISTENT if args.faults == "persistent" else FAULTS,
        "groups": groups,
        "attempt_cap_per_slot": 3,
        "neutral_id_prefix": "v",
        "shuffle_seed": 200,
        "response_tries": 2,
        "status": (
            "PILOT (development, not scored)"
            if args.kind == "pilot"
            else (
                "DRAFT main plan - not frozen; needs the user's review of the episode definition"
                if args.kind == "main"
                else "FROZEN v0 release plan (see release_rules)"
            )
        ),
    }
    if args.kind == "pilot" and args.pilot_groups:
        plan["per_condition"], plan["dry_run"] = len(groups), True
    if args.kind == "release":
        plan["release_rules"] = RELEASE_RULES
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=1) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
