# fmt: off
# isort: skip_file
# ruff: noqa
"""Evaluation-only: did a run hit the game freeze P37, and which enemy deaths could have caused one?

The game fades a dead enemy with a coroutine that never yields once the sprite's alpha is 0; if the next
frame is a paused one (every native window starts with one) the game hangs for good. A harness can guard
only the deaths it sees on screen. This tool reads the game's debug trace of a finished run and lists every
enemy death: where the knight was (on screen or not), what killed it (the sword, or a hazard it walked
into), and how many frames after it the run ended. A run counts as frozen when its status is the
observation timeout.

  python3 freeze_check.py <run_dir> [<run_dir> ...] [--json]

Nothing read here is fed back into a harness decision.
"""
import json
import pathlib
import sys

VIEW_X, VIEW_UP, VIEW_DOWN = 8.89, 7.0, 3.0      # the camera shows this much around the knight's centre


def load(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def check(run):
    run = pathlib.Path(run)
    s = run / "session" if (run / "session").is_dir() else run
    rows = load(s / "observations.jsonl")
    windows = load(s / "windows.jsonl")
    summary = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else {}
    status = summary.get("status", "")
    attack_ids = {w["obs_id"] for w in windows if w["action"].startswith("attack_")}
    deaths = []
    for a, b in zip(rows, rows[1:]):
        if b["scene"] != a["scene"] or (a["health"] == 0 and b["health"] > 0):
            continue                                   # scene (re)loaded: everything is new
        before = {e["name"]: e for e in a.get("enemies", [])}
        for e in b.get("enemies", []):
            p = before.get(e["name"])
            if p is None or not (p["health"] > 0 and e["health"] == 0):
                continue
            dx, dy = e["x"] - b["x"], e["y"] - b["y"]
            on_screen = abs(dx) <= VIEW_X and -VIEW_DOWN <= dy - 0 <= VIEW_UP
            by_sword = a["id"] in attack_ids and abs(dx) < 3.5 and abs(dy) < 2.5
            deaths.append({"frame": b["frame"], "window": b["id"], "enemy": e["name"], "x": round(e["x"], 2),
                           "knight_x": round(b["x"], 2), "scene": b["scene"], "on_screen": on_screen,
                           "by": "sword" if by_sword else "hazard", "frames_before_end": rows[-1]["frame"] - b["frame"],
                           "earliest_frames_before_end": rows[-1]["frame"] - a["frame"]})
    frozen = "TimeoutError" in status and "no new visible observation" in status
    cause = None
    if frozen:
        # The fade ends about 57 simulation frames after the death (a few more render frames, depending on
        # the windows in between), and the death happened somewhere inside the window in which the health
        # went to 0. Candidates: deaths whose window lies 50..80 render frames before the end. A sword kill
        # is guarded by the harness (v7: one long wait; v8: no window ends on the bad frame), so a hazard
        # kill among the candidates is taken as the cause.
        near = [d for d in deaths if d["earliest_frames_before_end"] >= 50 and d["frames_before_end"] <= 80]
        near.sort(key=lambda d: (d["by"] == "sword", d["on_screen"]))
        cause = near[0] if near else None
    return {"run": str(run), "status": status, "frozen": frozen, "freeze_cause": cause,
            "enemy_deaths": len(deaths), "by_sword": sum(d["by"] == "sword" for d in deaths),
            "by_hazard_on_screen": sum(d["by"] == "hazard" and d["on_screen"] for d in deaths),
            "by_hazard_off_screen": sum(d["by"] == "hazard" and not d["on_screen"] for d in deaths), "deaths": deaths}


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = [check(a) for a in args]
    if "--json" in sys.argv:
        print(json.dumps(out, indent=1))
    else:
        for r in out:
            c = r["freeze_cause"]
            print(f"{pathlib.Path(r['run']).name}: frozen={r['frozen']} | enemy deaths {r['enemy_deaths']} "
                  f"(sword {r['by_sword']}, hazard on screen {r['by_hazard_on_screen']}, hazard off screen {r['by_hazard_off_screen']})"
                  + (f" | freeze cause: {c['enemy']} died by {c['by']} at x={c['x']}, knight at x={c['knight_x']}, "
                     f"{'on' if c['on_screen'] else 'off'} screen, {c['frames_before_end']} frames before the end" if c else "")
                  + (f" | status: {r['status']}" if r["status"] else ""))
