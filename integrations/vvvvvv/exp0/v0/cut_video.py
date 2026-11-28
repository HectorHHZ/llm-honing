"""Two local videos of one captured contract attempt (presentation only; nothing here enters a dataset):

original:  every captured frame at game speed, no captions (inference waits are not in it: the game is paused
           while the model decides).
edited:    sped up (--speed), every decision kept, one caption bar per frame: decision number, the option the
           policy executed, the model's top option and the executed option's probability, and the native events
           of that decision (death, room, goal, kill, damage). The end card states the outcome.

python3 cut_video.py <run_dir> --frames <dir with %06d.png|bmp> --fps <game fps> --out-prefix <path> [--speed 2]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# 2026-11-26: no absolute local path in the code: $FFMPEG, else ffmpeg on PATH, else the cse153 conda environment's
FFMPEG = (
    os.environ.get("FFMPEG")
    or shutil.which("ffmpeg")
    or str(Path(sys.base_prefix).resolve() / "envs" / "cse153" / "bin" / "ffmpeg")
)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]  # noqa: E741


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--fps", required=True, help="game frames per second, e.g. 30 or 500/17")
    ap.add_argument("--out-prefix", type=Path, required=True)
    ap.add_argument("--speed", type=int, default=2)
    ap.add_argument("--scale", type=int, default=1, help="integer upscale of small frames (nearest neighbour)")
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    run = args.run
    manifest = json.loads((run / "manifest.json").read_text())
    result = json.loads((run / "result.json").read_text())
    rs = rows(run / "decisions.jsonl")
    frames = sorted(p for p in args.frames.iterdir() if p.suffix in (".png", ".bmp"))
    if not frames:
        raise SystemExit("no frames")
    # map a game tick to its row: frame i is the picture after tick i + 1
    spans = []
    model_n = 0
    for r in rs:
        if r["kind"] == "model":
            model_n += 1
        for w in (r.get("executed") or {}).get("windows", []):
            spans.append((w["start_tick"], w["start_tick"] + w["ticks"], r, model_n))  # noqa: PERF401
    spans.sort(key=lambda s: s[0])
    total = result["model_decisions"]
    ext = frames[0].suffix
    # original: all frames, game speed
    original = Path(str(args.out_prefix) + "_original.mp4")
    scale = f"scale=iw*{args.scale}:ih*{args.scale}:flags=neighbor," if args.scale > 1 else ""
    subprocess.run(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            args.fps,
            "-i",
            str(args.frames / f"%06d{ext}"),
            "-vf",
            scale + "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            "-movflags",
            "+faststart",
            str(original),
        ],
        check=True,
    )
    # edited: every --speed-th frame with a caption bar
    first = Image.open(frames[0])
    w, h = first.size[0] * args.scale, first.size[1] * args.scale
    bar = 64 if w >= 600 else 52
    font = ImageFont.truetype(FONT, 15 if w >= 600 else 12)
    bold = ImageFont.truetype(FONT_B, 15 if w >= 600 else 12)
    title = args.title or f"{manifest['game_id']} · group {manifest['reset_group']} · {manifest['condition']}"
    outcome = f"outcome: {result['status']} ({result['reason']}) after {total} decisions"
    with tempfile.TemporaryDirectory(dir=str(args.out_prefix.parent)) as tmp:
        tmpdir = Path(tmp)
        count = 0
        j = 0
        for i in range(0, len(frames), args.speed):
            tick = i + 1
            while j + 1 < len(spans) and spans[j + 1][0] < tick:
                j += 1
            span = spans[j] if spans else None
            img = Image.open(frames[i]).convert("RGB")
            if args.scale > 1:
                img = img.resize((w, h), Image.NEAREST)
            canvas = Image.new("RGB", (w, h + bar), (12, 12, 16))
            canvas.paste(img, (0, 0))
            d = ImageDraw.Draw(canvas)
            if span is not None:
                r, n = span[2], span[3]
                if r["kind"] == "model":
                    p = r["probabilities"].get(r["model_choice"])
                    top = r["server_choice"]
                    executed = (r.get("executed") or {}).get("option")
                    if executed != r["model_choice"]:  # 2026-11-26: show a selected-vs-sent change
                        text1 = f"decision {n}/{total}: model chose {r['model_choice']} (p {p:.2f}) -> sent {executed}"
                    else:
                        text1 = (
                            f"decision {n}/{total}: {executed}  (model top: {top}; p({r['model_choice']}) = {p:.2f})"
                        )
                else:
                    text1 = f"no decision: {r['executed']['windows'][0].get('reason', 'auto')}"
                ev = "; ".join(e["text"] for e in r.get("events", []))[:110]
            else:
                text1, ev = "", ""
            d.text((8, h + 4), title + f"  ·  x{args.speed}", font=bold, fill=(230, 230, 230))
            d.text((8, h + 4 + bar // 3), text1, font=font, fill=(255, 220, 120))
            d.text((8, h + 4 + 2 * bar // 3), ev or " ", font=font, fill=(255, 140, 140))
            canvas.save(tmpdir / f"{count:06d}.png")
            count += 1
        # end card: 2 seconds
        card = Image.new("RGB", (w, h + bar), (12, 12, 16))
        d = ImageDraw.Draw(card)
        d.text((20, (h + bar) // 2 - 30), title, font=bold, fill=(230, 230, 230))
        d.text((20, (h + bar) // 2), outcome, font=bold, fill=(255, 220, 120))
        d.text(
            (20, (h + bar) // 2 + 26),
            "inference waits are not shown (the game is paused while the model decides)",
            font=font,
            fill=(200, 200, 200),
        )
        fps_num = eval(args.fps) if "/" in args.fps else float(args.fps)
        for _ in range(int(2 * fps_num)):
            card.save(tmpdir / f"{count:06d}.png")
            count += 1
        edited = Path(str(args.out_prefix) + "_edited.mp4")
        subprocess.run(
            [
                FFMPEG,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                args.fps,
                "-i",
                str(tmpdir / "%06d.png"),
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "22",
                "-movflags",
                "+faststart",
                str(edited),
            ],
            check=True,
        )
    note = Path(str(args.out_prefix) + "_edited.txt")
    note.write_text(
        f"{edited.name}: every {args.speed}th captured frame of {run.name} at the game frame rate ({args.fps}), "
        f"i.e. {args.speed}x speed, all decisions kept, one caption bar per frame, 2 s end card.\n"
        f"{original.name}: all {len(frames)} captured frames at game speed, no captions.\n"
        f"Neither contains inference waits (the game is paused while the model decides).\n{outcome}\n"
    )
    print(
        json.dumps({"original": str(original), "edited": str(edited), "frames": len(frames), "edited_frames": count})
    )


if __name__ == "__main__":
    main()
