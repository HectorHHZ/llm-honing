"""Explicit full-game runs with video, typed decisions, and verifiable trajectories."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from honing.backends.craftax.catalog import ACTIONS, DEFAULT_GOALS, PROTOCOL
from honing.backends.craftax.engine import CraftaxEngine, runtime_versions
from honing.backends.craftax.harness import load_harness, parse_harness, seed_harness
from honing.backends.craftax.model import DecisionClient, make_request
from honing.backends.craftax.scenarios import CraftaxScenario
from honing.core import CampaignRuleError, HarnessSpec

FORMAT = "craftax-match-v1"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def summary(frame: dict[str, Any], reward: float) -> dict[str, Any]:
    return {
        "status": frame["status"],
        "success": frame["status"] == "goal_complete",
        "steps": frame["step"],
        "reward": reward,
        "health": frame["observation"]["player"]["health"],
        "achievements": frame["achievements"],
        "final_frame_hash": frame["frame_hash"],
    }


class Video:
    """Use the original renderer and preserve a playable partial video on interruption."""

    def __init__(self, out: Path, fps: int) -> None:
        import imageio.v2 as imageio

        self._imageio, self._out = imageio, out
        self._writer = imageio.get_writer(out / "replay.mp4", fps=fps, codec="libx264", quality=8, macro_block_size=1)
        self._last: Any = None

    def append(self, pixels: Any) -> None:
        if self._last is None:
            self._imageio.imwrite(self._out / "first-frame.png", pixels)
        self._writer.append_data(pixels)
        self._last = pixels

    def close(self) -> None:
        self._writer.close()
        if self._last is not None:
            self._imageio.imwrite(self._out / "last-frame.png", self._last)


def run_match(
    scenario: CraftaxScenario,
    out: Path,
    spec: HarnessSpec,
    *,
    policy: str = "random",
    policy_seed: int = 1,
    decision_server: str | None = None,
    actions: tuple[str, ...] = (),
    record_video: bool = True,
    video_fps: int = 10,
    policy_checksum: str | None = None,
) -> dict[str, Any]:
    """No model failure substitutes a scripted action. Every run gets a new directory."""
    if policy not in ("random", "actions", "model"):
        raise ValueError("unknown controller")
    if (policy == "model") != (decision_server is not None):
        raise ValueError("only policy=model requires decision-server")
    if type(policy_seed) is not int or not 0 <= policy_seed < 2**32:
        raise ValueError("policy_seed must be uint32")
    if type(video_fps) is not int or not 1 <= video_fps <= 60:
        raise ValueError("video_fps must be in 1..60")
    if policy == "actions" and (len(actions) < scenario.max_steps or any(a not in ACTIONS for a in actions)):
        raise ValueError("the action sequence must contain valid names and cover max_steps")
    if policy != "actions" and actions:
        raise ValueError("an action sequence requires policy=actions")
    out.mkdir(parents=True, exist_ok=False)
    started, video = time.perf_counter(), None
    try:
        harness = parse_harness(spec)
        client = DecisionClient(decision_server) if decision_server is not None else None
        health = client.health() if client else None
        manifest = {
            "format": FORMAT,
            "protocol": PROTOCOL,
            "scenario": scenario.record(),
            "kind": "evaluation" if client else "scripted",
            "controller": policy,
            "policy_seed": policy_seed,
            "runtime": runtime_versions(),
            "actions": list(ACTIONS),
            "harness": list(harness.files),
            "spec_id": spec.spec_id,
            "policy_checksum": policy_checksum,
            "model": health,
            "record_video": record_video,
            "video_fps": video_fps,
            "clock": "turn-based; exactly one engine step per action; no advancement during inference",
        }
        write_json(out / "manifest.json", manifest)
        with (
            CraftaxEngine(scenario) as engine,
            (out / "steps.jsonl").open("x") as steps,
            (out / "decisions.jsonl").open("x") as decisions,
        ):
            frame = engine.reset()

            def record(stream: Any, row: dict[str, Any]) -> None:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()

            record(steps, {"event": "reset", "frame": frame})
            if record_video:
                video = Video(out, video_fps)
                video.append(engine.pixels())
            rng, total_reward = random.Random(policy_seed), 0.0
            history: list[dict[str, Any]] = []
            while frame["status"] == "running":
                before, request, response = frame, None, None
                tick = time.perf_counter()
                if client:
                    request = make_request(frame, scenario, harness, history)
                    action, response = client.decide(request)
                elif policy == "actions":
                    action = actions[frame["step"]]
                else:
                    action = rng.choices(("LEFT", "RIGHT", "UP", "DOWN", "DO"), weights=(1, 1, 1, 1, 3))[0]
                latency_ms = (time.perf_counter() - tick) * 1000
                frame = engine.step(action)
                total_reward += frame["reward"]
                record(
                    decisions,
                    {
                        "step": before["step"],
                        "before_frame_hash": before["frame_hash"],
                        "request": request,
                        "response": response,
                        "chosen": {"action": action},
                        "decision_ms": latency_ms,
                    },
                )
                record(steps, {"event": "step", "action": action, "frame": frame})
                history.append(
                    {
                        "action": action,
                        "reward": frame["reward"],
                        "health": frame["observation"]["player"]["health"],
                        "achievements": frame["achievements"],
                    }
                )
                if video is not None:
                    video.append(engine.pixels())
                if frame["step"] == 1 or frame["step"] % 50 == 0 or frame["status"] != "running":
                    print(
                        json.dumps(
                            {
                                "event": "progress",
                                "step": frame["step"],
                                "status": frame["status"],
                                "reward": total_reward,
                                "elapsed_seconds": time.perf_counter() - started,
                            }
                        ),
                        flush=True,
                    )
            if video is not None:
                video.close()
                video = None
            result = {**summary(frame, total_reward), "elapsed_seconds": time.perf_counter() - started}
            write_json(out / "result.json", result)
            return result
    except (Exception, KeyboardInterrupt) as error:
        write_json(
            out / "error.json",
            {
                "kind": "harness" if isinstance(error, CampaignRuleError) else "match",
                "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "error",
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise
    finally:
        if video is not None:
            video.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--policy-seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--goals", default=",".join(DEFAULT_GOALS))
    parser.add_argument("--policy", choices=("random", "actions", "model"), default="random")
    parser.add_argument("--decision-server")
    parser.add_argument("--policy-checksum", help="Policy identity supplied by Honing's episode runner")
    parser.add_argument("--actions", type=Path, help="JSON list of action names; requires policy=actions")
    parser.add_argument("--harness-dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=10)
    args = parser.parse_args(argv)
    integration = Path(__file__).resolve().parents[3] / "integrations/craftax"
    actions = json.loads(args.actions.read_text()) if args.actions else []
    if not isinstance(actions, list) or any(not isinstance(a, str) for a in actions):
        parser.error("actions must be a JSON list of action names")
    if args.policy == "actions" and not args.actions:
        parser.error("policy=actions requires --actions")
    scenario = CraftaxScenario(
        args.seed,
        args.max_steps if args.max_steps is not None else len(actions) if actions else 300,
        tuple(args.goals.split(",")),
    )
    spec = load_harness(args.harness_dir) if args.harness_dir else seed_harness(integration)
    out = args.out or integration / "runs" / f"full-s{args.seed}-{args.policy}-{uuid.uuid4().hex[:12]}"
    result = run_match(
        scenario,
        out,
        spec,
        policy=args.policy,
        policy_seed=args.policy_seed,
        decision_server=args.decision_server,
        actions=tuple(actions),
        record_video=not args.no_video,
        video_fps=args.video_fps,
        policy_checksum=args.policy_checksum,
    )
    print(json.dumps({"event": "saved", "out": str(out), **result}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
