"""Encode the contiguous game frames of a completed capture run."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def validate_capture(session: Path) -> int:
    """Reject incomplete or missing-frame sessions before invoking ffmpeg."""
    result = json.loads((session / "result.json").read_text())
    if not result.get("capture") or result.get("heuristicFallback"):
        raise ValueError("Session is not a completed model-controlled capture")
    observations = (session / "observations.jsonl").read_text().splitlines()
    if not observations:
        raise ValueError("No game observations were recorded")
    count = json.loads(observations[-1])["frame"]
    actual = sorted(path.name for path in (session / "frames").glob("*.png"))
    if type(count) is not int or count <= 0 or actual != [f"{index:06d}.png" for index in range(count)]:
        raise ValueError("Captured frame sequence is missing frames or is not contiguous from zero")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True, help="Output directory of run.py --capture")
    parser.add_argument("--output", type=Path, help="New MP4 path, default <session>/gameplay.mp4")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    count = validate_capture(args.session)
    output = args.output or args.session / "gameplay.mp4"
    subprocess.run(
        [
            args.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-n",
            "-framerate",
            "30",
            "-i",
            str(args.session / "frames/%06d.png"),
            # The game window is 523 pixels high and this codec needs an even height.
            "-vf",
            "crop=trunc(iw/2)*2:trunc(ih/2)*2:0:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )
    print(f"{output}: {count} frames, {count / 30:g}s at 30 fps")


if __name__ == "__main__":
    main()
