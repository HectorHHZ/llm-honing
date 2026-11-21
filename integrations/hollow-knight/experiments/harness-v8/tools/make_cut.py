# fmt: off
# isort: skip_file
# ruff: noqa
"""Edited demo cut of a finished, captured run: the stretches in which the knight only stands and waits are
played faster, everything else at normal speed. Nothing is left out entirely; what was shortened is listed.

  python3 make_cut.py <run_dir> [--wait-speed 6] [--speed 1.0] [--out NAME.mp4] [--first-tick A --last-tick B]
                                [--target-kbps N] [--output-fps F]

A "waiting stretch" is a run of native windows whose input is `wait` while the knight stands and the harness
says why it waits: watching a saw / a moving platform / an enemy, waiting for the moment to jump, the sword
cooldown gap between strikes is NOT counted (it is part of a fight), the fade guard, the respawn after a
death, the `wait` behaviour itself. Such a stretch is shown at --wait-speed (every Nth frame is kept); all
other frames are shown at --speed (1.0 = every frame, 30 fps). Stretches shorter than 12 frames are left alone.
Writes <run_dir>/<out> and <run_dir>/<out>.txt (what was shortened, speed factors, lengths). --out may be an
absolute path. --target-kbps encodes in two passes at that average bitrate and --output-fps keeps fewer frames of
the same timeline; together they make the copy committed with this experiment small enough for the repository.
The uncut video is <run_dir>/gameplay.mp4 (made by core.env.encode_video / make_demo.py).
Evaluation / presentation only: reads windows.jsonl and the captured frames.
"""
import argparse
import json
import os
import pathlib
import subprocess
import tempfile

FFMPEG = os.environ.get("HONING_FFMPEG", "ffmpeg")
WAIT_WORDS = (":watch", ":watching", ":timing", ":look", "fade-guard", "respawn", ":await", ":out-of-sight",
              "wait:", ":stand", "hurt-lock")

ap = argparse.ArgumentParser()
ap.add_argument("run", type=pathlib.Path)
ap.add_argument("--wait-speed", type=int, default=6)
ap.add_argument("--speed", type=float, default=1.0)
ap.add_argument("--out", default=None)
ap.add_argument("--first-tick", type=int, default=0)
ap.add_argument("--last-tick", type=int, default=None)
ap.add_argument("--target-kbps", type=int, default=None)
ap.add_argument("--output-fps", type=float, default=None)
a = ap.parse_args()

session = a.run / "session"
frames_dir = session / "frames"
windows = [json.loads(l) for l in (session / "windows.jsonl").read_text().splitlines() if l.strip()]
n_frames = len(list(frames_dir.glob("*.png")))
last = n_frames if a.last_tick is None else min(a.last_tick, n_frames)


def waiting(w: dict) -> bool:
    why = (w.get("why") or "")
    label = (w.get("label") or "")
    if w["action"] != "wait":
        return False
    if label.endswith(" wait") and why.startswith("wait"):
        return True
    return any(k in why for k in WAIT_WORDS)


# runs of waiting windows -> stretches (first tick, last tick exclusive, kinds)
stretches, cur = [], None
for w in windows:
    t0, t1 = w["tick"], w["tick"] + w["frames"]
    if waiting(w):
        kind = (w.get("why") or "").split("|")[0]
        if cur is not None and cur[1] == t0:
            cur[1] = t1
            cur[2].add(kind)
        else:
            cur = [t0, t1, {kind}]
            stretches.append(cur)
    else:
        cur = None
stretches = [s for s in stretches if s[1] - s[0] >= 12 and s[1] > a.first_tick and s[0] < last]

fast = [False] * (last + 1)
for s0, s1, _ in stretches:
    for t in range(max(s0, a.first_tick), min(s1, last)):
        fast[t] = True
# inside a waiting stretch keep every Nth frame (a step never jumps past the end of its stretch)
out_frames = []
t = a.first_tick
while t < last:
    out_frames.append(t)
    if fast[t]:
        nxt = t + a.wait_speed
        end = t
        while end < last and fast[end]:
            end += 1
        t = min(nxt, end)
    else:
        t += 1

fps = 30.0 * a.speed
name = a.out or f"cut_wait_x{a.wait_speed}.mp4"
out = a.run / name
with tempfile.TemporaryDirectory() as tmp:
    lst = pathlib.Path(tmp) / "frames.txt"
    with lst.open("w") as f:
        for t in out_frames:
            f.write(f"file '{(frames_dir / f'{t:06d}.png').resolve()}'\nduration {1.0 / fps:.6f}\n")
    source = [FFMPEG, "-hide_banner", "-loglevel", "warning", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
              "-vf", f"fps={a.output_fps or fps}", "-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if a.target_kbps is None:
        subprocess.run([*source, "-movflags", "+faststart", str(out)], check=True)
    else:
        rate = ["-preset", "placebo", "-tune", "animation", "-b:v", f"{a.target_kbps}k",
                "-passlogfile", str(pathlib.Path(tmp) / "pass")]
        subprocess.run([*source, *rate, "-pass", "1", "-an", "-f", "null", os.devnull], check=True)
        subprocess.run([*source, *rate, "-pass", "2", "-movflags", "+faststart", str(out)], check=True)

total = last - a.first_tick
cut_frames = sum(min(s1, last) - max(s0, a.first_tick) for s0, s1, _ in stretches)
lines = [
    f"edited cut of {a.run.name}: {out.name}",
    f"source: {total} captured frames = {total / 30:.1f} s at 30 fps",
    f"this cut: {len(out_frames)} frames = {len(out_frames) / fps:.1f} s at {fps:.0f} fps "
    f"(overall x{total / 30 / (len(out_frames) / fps):.2f})",
    f"normal play is shown at x{a.speed:g}; {len(stretches)} waiting stretches ({cut_frames} frames = {cut_frames / 30:.1f} s) "
    f"are shown at x{a.wait_speed * a.speed:g} (every {a.wait_speed}th frame kept). Nothing else is removed.",
]
if a.target_kbps is not None or a.output_fps is not None:
    lines.append(f"encoded for the repository: {a.output_fps or fps:g} fps on the same timeline, "
                 f"{a.target_kbps or 'default'} kbit/s H.264")
lines.append("waiting stretches (first frame - last frame, seconds, why the knight waits):")
for s0, s1, kinds in stretches:
    lines.append(f"  {s0:6d} - {s1:6d}  {(s1 - s0) / 30:5.1f} s  {', '.join(sorted(kinds))[:110]}")
(a.run / (name + ".txt")).write_text("\n".join(lines) + "\n")
print("\n".join(lines[:4]))
