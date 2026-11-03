"""Render and verify a recorded ALE Space Invaders episode."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from lane_io import ACTION_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("fps must be positive")
    with args.trace.open() as source:
        rows = [row for line in source if (row := json.loads(line))["seed"] == args.seed]
    if not rows:
        parser.error(f"no trace rows for seed {args.seed}")

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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "160x210",
        "-r",
        str(args.fps),
        "-i",
        "-",
        "-vf",
        "scale=640:840:flags=neighbor",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(args.output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        _, info = env.reset(seed=args.seed)
        assert tuple(env._env.unwrapped.get_action_meanings()) == ACTION_NAMES
        for row in rows:
            assert row["lives_before"] == info["lives"]
            _, reward, _, _, info = env.step(ACTION_NAMES.index(row["action"]))
            assert reward == row["reward"] and info["lives"] == row["lives_after"]
            assert process.stdin is not None
            process.stdin.write(env.render().tobytes())
    finally:
        env.close()
        if process.stdin is not None:
            process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("ffmpeg failed")
    print(
        json.dumps(
            {
                "seed": args.seed,
                "frames": len(rows),
                "reward": sum(row["reward"] for row in rows),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
