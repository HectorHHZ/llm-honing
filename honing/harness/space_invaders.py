"""Optional ALE Space Invaders environment and deterministic action replay.

Policy selection belongs to callers. This module accepts native action indices;
it never imports a model or controller.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

GAME = "ALE/SpaceInvaders-v5"
ACTION_REPEAT = 4
STICKY_ACTION_PROBABILITY = 0.25


def make_space_invaders(*, pooled: bool = True) -> Any:
    """Create the six-action RGB game with four physical frames per decision.

    ``pooled=False`` returns ALE's native four-frame observation. ``pooled=True``
    repeats in this wrapper and max-pools the last two physical frames.
    """
    try:
        import ale_py
        import gymnasium as gym
    except ImportError as exc:
        raise RuntimeError("Install llm-honing[space-invaders] and provide the ALE Space Invaders ROM") from exc

    gym.register_envs(ale_py)
    if not pooled:
        return gym.make(GAME, obs_type="rgb", frameskip=4, repeat_action_probability=STICKY_ACTION_PROBABILITY)
    env = gym.make(GAME, obs_type="rgb", frameskip=1, repeat_action_probability=STICKY_ACTION_PROBABILITY)
    return MaxPoolRepeat(env)


class MaxPoolRepeat:
    """Repeat one action for four ALE frames, stopping at either terminal flag."""

    def __init__(self, env: Any) -> None:
        if getattr(env.unwrapped, "_frameskip", None) != 1:
            raise ValueError("ALE internal frameskip must be 1 before applying action repeat")
        self.env = env
        self.last_raw_frame: Any = None
        self.underlying_frames = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(self, *, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        import numpy as np

        frame, info = self.env.reset(seed=seed)
        self.last_raw_frame = np.asarray(frame).copy()
        self.underlying_frames = 0
        return frame, info

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        import numpy as np

        recent: list[Any] = []
        total = 0.0
        for _ in range(ACTION_REPEAT):
            frame, reward, terminated, truncated, info = self.env.step(action)
            recent.append(np.asarray(frame).copy())
            recent = recent[-2:]
            total += float(reward)
            self.underlying_frames += 1
            if terminated or truncated:
                break
        self.last_raw_frame = recent[-1]
        pooled = np.maximum(*recent) if len(recent) == 2 else recent[0]
        return pooled, total, bool(terminated), bool(truncated), info

    def close(self) -> None:
        self.env.close()


def _frame_hash(frame: Any) -> str:
    return hashlib.sha256(frame.tobytes()).hexdigest()


def _write(path: Path, record: dict[str, Any]) -> None:
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def run_episode(
    policy: Callable[[Any, dict[str, Any]], int],
    *,
    seed: int,
    output: Path,
    pooled: bool = True,
    max_decisions: int = 100_000,
) -> dict[str, Any]:
    """Play one game and record actions and physical outcomes for replay.

    The policy may call a Jev server, use a controller, or be a fixed action.
    A capped game is recorded as incomplete and cannot pass replay verification.
    """
    if max_decisions < 1:
        raise ValueError("max_decisions must be positive")
    env = make_space_invaders(pooled=pooled)
    try:
        frame, info = env.reset(seed=seed)
        actions = list(env.unwrapped.get_action_meanings())
        if actions != ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]:
            raise ValueError(f"Unexpected Space Invaders action set: {actions}")
        rows = []
        total = 0.0
        terminated = truncated = False
        for _ in range(max_decisions):
            action = policy(frame, info)
            if not isinstance(action, int) or not 0 <= action < len(actions):
                raise ValueError(f"Policy must return a native action index in [0, {len(actions) - 1}]")
            frame, reward, terminated, truncated, info = env.step(action)
            total += reward
            rows.append(
                {
                    "action": action,
                    "reward": reward,
                    "lives": int(info["lives"]),
                    "frame_sha256": _frame_hash(frame),
                    "terminated": terminated,
                    "truncated": truncated,
                }
            )
            if terminated or truncated:
                break
        record = {
            "schema": 1,
            "game": GAME,
            "ale_py_version": importlib.metadata.version("ale-py"),
            "gymnasium_version": importlib.metadata.version("gymnasium"),
            "seed": seed,
            "pooled": pooled,
            "action_repeat": ACTION_REPEAT,
            "sticky_action_probability": STICKY_ACTION_PROBABILITY,
            "steps": rows,
            "reward": total,
            "terminated": terminated,
            "truncated": truncated,
            "lives": int(info["lives"]),
            "natural_end": bool(terminated and not truncated and info["lives"] == 0),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        _write(output, record)
        return record
    finally:
        env.close()


def verify_replay(path: Path) -> dict[str, Any]:
    """Recreate each recorded action and reject divergence or incomplete games."""
    record = json.loads(path.read_text())
    if (
        record.get("schema") != 1
        or record.get("game") != GAME
        or record.get("ale_py_version") != importlib.metadata.version("ale-py")
        or record.get("gymnasium_version") != importlib.metadata.version("gymnasium")
        or record.get("action_repeat") != ACTION_REPEAT
        or record.get("sticky_action_probability") != STICKY_ACTION_PROBABILITY
        or not isinstance(record.get("pooled"), bool)
        or not record.get("natural_end")
        or not record.get("steps")
    ):
        raise ValueError("Unsupported or incomplete Space Invaders replay")
    env = make_space_invaders(pooled=record["pooled"])
    try:
        frame, info = env.reset(seed=record["seed"])
        total = 0.0
        for index, expected in enumerate(record["steps"]):
            frame, reward, terminated, truncated, info = env.step(expected["action"])
            actual = {
                "action": expected["action"],
                "reward": reward,
                "lives": int(info["lives"]),
                "frame_sha256": _frame_hash(frame),
                "terminated": terminated,
                "truncated": truncated,
            }
            if actual != expected:
                raise ValueError(f"Replay diverged at decision {index}")
            total += reward
            if (terminated or truncated) and index != len(record["steps"]) - 1:
                raise ValueError("Replay ended before trace")
        if not (
            terminated
            and not truncated
            and info["lives"] == 0
            and total == record["reward"]
            and record["terminated"]
            and not record["truncated"]
            and record["lives"] == 0
        ):
            raise ValueError("Replay outcome does not match a natural game end")
        return {
            "verified": True,
            "seed": record["seed"],
            "decisions": len(record["steps"]),
            "reward": total,
            "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Record or verify a Space Invaders action trajectory")
    commands = parser.add_subparsers(dest="command", required=True)
    play = commands.add_parser("play", help="run one complete fixed-action smoke game")
    play.add_argument("--seed", type=int, required=True)
    play.add_argument("--action", choices=("NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"), default="NOOP")
    play.add_argument("--observation", choices=("native", "pooled"), default="pooled")
    play.add_argument("--max-decisions", type=int, default=100_000)
    play.add_argument("--output", type=Path, required=True)
    replay = commands.add_parser("replay", help="verify a complete recorded game")
    replay.add_argument("trace", type=Path)
    args = parser.parse_args()
    if args.command == "play":
        action = ("NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE").index(args.action)
        result = run_episode(
            lambda _frame, _info: action,
            seed=args.seed,
            output=args.output,
            pooled=args.observation == "pooled",
            max_decisions=args.max_decisions,
        )
        print(
            json.dumps(
                {
                    "natural_end": result["natural_end"],
                    "reward": result["reward"],
                    "decisions": len(result["steps"]),
                    "trace": str(args.output),
                }
            )
        )
    else:
        print(json.dumps(verify_replay(args.trace)))


if __name__ == "__main__":
    main()
