"""Encode every accepted native tick from a completed Celeste Classic capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def encode(session: Path, output: Path, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> dict:
    result = json.loads((session / "result.json").read_text())
    count = result["frames"]
    if not result["capture"] or type(count) is not int or count <= 0 or output.exists():
        raise ValueError("completed capture and fresh output required")
    files = sorted((session / "frames").glob("*.bmp"))
    if [path.name for path in files] != [f"{number:06d}.bmp" for number in range(1, count + 1)]:
        raise ValueError("capture frames are missing, duplicated or unexpected")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-n",
            "-framerate",
            "30",
            "-start_number",
            "1",
            "-i",
            str(session / "frames/%06d.bmp"),
            "-frames:v",
            str(count),
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
    subprocess.run([ffmpeg, "-v", "error", "-i", str(output), "-f", "null", "-"], check=True)
    probe = json.loads(
        subprocess.check_output(
            [
                ffprobe,
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames,width,height,r_frame_rate,duration",
                "-of",
                "json",
                str(output),
            ]
        )
    )
    stream = probe["streams"][0]
    if int(stream["nb_read_frames"]) != count or abs(float(stream["duration"]) - count / 30) > 0.05:
        raise ValueError("encoded duration/frame count differs from capture")
    receipt = {
        "frames": count,
        "fps": 30,
        "simulationSeconds": count / 30,
        "waitingOmitted": True,
        "audioIncluded": False,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "probe": stream,
    }
    (session / "video.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    print(json.dumps(encode(args.session, args.output, args.ffmpeg, args.ffprobe), indent=2))


if __name__ == "__main__":
    main()
