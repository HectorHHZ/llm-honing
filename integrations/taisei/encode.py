"""Encode Taisei's native PNG frame dump at its 60 Hz simulation rate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    frames = sorted((args.session / "frames").glob("frame-*.png"))
    if not frames:
        parser.error("No native frames found")
    expected = [f"frame-{index:08d}.png" for index in range(len(frames))]
    if [path.name for path in frames] != expected:
        parser.error("Native frame sequence has gaps or does not begin at zero")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        args.ffmpeg,
        "-hide_banner",
        "-y",
        "-framerate",
        "60",
        "-i",
        str(args.session / "frames/frame-%08d.png"),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(args.output),
    ]
    subprocess.run(command, check=True)
    subprocess.run([args.ffmpeg, "-v", "error", "-i", str(args.output), "-f", "null", "-"], check=True)
    receipt = {
        "nativePngFrames": len(frames),
        "fps": 60,
        "videoSeconds": len(frames) / 60,
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "decodeValidated": True,
        "inferenceWaitOmitted": True,
        "audio": False,
    }
    (args.session / "video.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
