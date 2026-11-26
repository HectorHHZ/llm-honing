"""Native replay of one full-HK Exp 0 attempt on the deterministic build (2026-11-26, Exp 0 v0 HK collection).

  python3 -B native_replay.py <attempt dir> --out <new dir> --display N [--drawn] [--capture] [--size 960x540]
                              [--report <file.json>]

The game is started again exactly as the attempt started it (same executable, bridge options, HONING_* variables, seed,
scene "new"), and every command the game received during the attempt (extra/session/bridge_windows.jsonl: the boot's
adapter windows, the start prefix, every model decision's windows, the adapter's own windows) is sent again in order.
No harness, no model, no adapter logic runs: an open-loop replay of the recorded native input.

Then the game's own logs of the replay are compared with the attempt's, record by record:
  debug state per window     observations.jsonl          (every field; exact)
  visible observation        visible_observations.jsonl  (every field; exact; --drawn: view.window.ui_rects excepted)
  state per frame            frames.jsonl                (HONING_FRAME_LOG=1: RNG state, rigid bodies, state machines,
                                                          enemies, animations, the knight, camera, clock; left out:
                                                          `wall`, the wall-clock time, and `nGo`, the count of all
                                                          game objects, which differs by 1-2 in a few frames when
                                                          particle effects are recycled (they use the engine's own
                                                          random seed; the devmode study saw the same between two
                                                          drawn runs, and it reaches no observation); the frames in
                                                          which nGo differs are counted in the report)
and the replay passes when all three are identical over the whole attempt (boot, prefix and episode). The report
also gives the native outcome markers of the replay (last debug row) next to the attempt's.

--drawn uses the drawn deterministic player (release.game.player_drawn, hk_det.sh) under Xvfb on the given display;
with --capture it saves the frames (for the videos; the frames stay local and are never part of a release).
Processes: one Xvfb (:350-:379) and one game, started and stopped through their handles (core/env_forestdango.py).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from core.env_forestdango import ForestDangoEnv

DISPLAYS = range(350, 380)


class ReplayGame(ForestDangoEnv):
    """The env's launcher and bridge I/O, without its automatic flow handling (the recording already has it)."""

    def _advance_flow(self, why: str) -> int:
        return 0


def rows(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def strip_visible(o: dict, drawn: bool) -> dict:
    o = json.loads(json.dumps(o))
    o.pop("adapter", None)
    if drawn:
        ((o.get("view") or {}).get("window") or {}).pop("ui_rects", None)
    return o


def strip_frame(o: dict) -> dict:
    o = dict(o)
    o.pop("wall", None)
    o.pop("nGo", None)
    return o


def ngo_differences(a: list | None, b: list | None) -> int | None:
    if a is None or b is None:
        return None
    return sum(1 for x, y in zip(a, b) if x.get("nGo") != y.get("nGo"))  # noqa: B905


def compare(name: str, a: list | None, b: list | None, strip) -> dict:
    if a is None or b is None:
        return {"what": name, "available": False, "identical_all": False}
    same, first, example = 0, None, None
    for i, (x, y) in enumerate(zip(a, b)):  # noqa: B905
        if strip(x) == strip(y):
            same += 1
        elif first is None:
            first = i
            sx, sy = strip(x), strip(y)
            keys = sorted(k for k in set(sx) | set(sy) if sx.get(k) != sy.get(k)) if isinstance(sx, dict) else []
            example = {"index": i, "fields": keys[:8]}
    return {
        "what": name,
        "available": True,
        "rows_attempt": len(a),
        "rows_replay": len(b),
        "identical": same,
        "first_difference": first,
        "example": example,
        "identical_all": len(a) == len(b) == same,
    }


def replay(
    run: Path, out: Path, display: int, drawn: bool = False, capture: bool = False, size: str | None = None
) -> dict:
    run = Path(run).resolve()
    raw = run / "extra"
    session = raw / "session"
    hk = json.loads((raw / "manifest.json").read_text())
    plan = json.loads((raw / "plan.json").read_text())
    if display not in DISPLAYS:
        raise ValueError("display must be within :350-:379")
    options = dict(hk["game"]["bridge_options"])
    options.pop("display", None)
    player = hk["game"].get("player") or ""
    if drawn:
        player = plan["release"]["game"]["player_drawn"]
    if player:
        options["player"] = player
    if size:
        options["size"] = size
    if capture:
        options["frames"] = "jpg"
    options["display"] = str(display)
    commands = rows(session / "bridge_windows.jsonl")
    env = ReplayGame(
        Path(out),
        scene=hk["task_definition"]["scene"],
        capture=capture,
        options=options,
        bridge_env=hk["game"].get("bridge_env") or {},
    )
    t0 = time.monotonic()
    error = None
    try:
        env.start()
        boot = time.monotonic() - t0
        for c in commands:
            cmd = {"action": c["action"], "frames": int(c["frames"])}
            for k in ("hold", "press", "first"):
                if c.get(k):
                    cmd[k] = c[k]
            env._send(cmd, "replay")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        boot = None
    finally:
        close = env.close()
    seconds = round(time.monotonic() - t0, 1)
    return dict(
        compare_dirs(run, Path(out), drawn),
        player=player,
        capture=capture,
        commands=len(commands),
        error=error,
        boot_seconds=None if boot is None else round(boot, 1),
        seconds=seconds,
        close=close,
    )


def compare_dirs(run: Path, out: Path, drawn: bool) -> dict:
    run = Path(run).resolve()
    session = run / "extra" / "session"
    rs = Path(out).resolve() / "session"
    results = [
        compare(
            "debug_state_per_window",
            rows(session / "observations.jsonl"),
            rows(rs / "observations.jsonl"),
            lambda o: o,
        ),
        compare(
            "visible_observation_per_window",
            rows(session / "visible_observations.jsonl"),
            rows(rs / "visible_observations.jsonl"),
            lambda o: strip_visible(o, drawn),
        ),
        compare("state_per_frame", rows(session / "frames.jsonl"), rows(rs / "frames.jsonl"), strip_frame),
    ]
    results[2]["object_count_nGo_differs_in_frames"] = ngo_differences(
        rows(session / "frames.jsonl"), rows(rs / "frames.jsonl")
    )
    last_a = (rows(session / "observations.jsonl") or [{}])[-1]
    last_b = (rows(rs / "observations.jsonl") or [{}])[-1]
    keys = ("frame", "scene", "x", "y", "health", "deaths", "grounded")
    return {
        "run": run.name,
        "replay_dir": str(Path(out).resolve()),
        "drawn": drawn,
        "results": results,
        "last_debug": {"attempt": {k: last_a.get(k) for k in keys}, "replay": {k: last_b.get(k) for k in keys}},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--display", type=int, required=True)
    ap.add_argument("--drawn", action="store_true")
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--size", default=None)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--compare-only", action="store_true", help="compare an existing replay directory (no game)")
    a = ap.parse_args()
    if a.compare_only:
        report = dict(compare_dirs(a.run, a.out, a.drawn), compare_only=True, error=None, commands=None, seconds=None)
    else:
        report = replay(a.run, a.out, a.display, a.drawn, a.capture, a.size)
    report["ok"] = report["error"] is None and all(r["identical_all"] for r in report["results"])
    text = json.dumps(report, indent=1)
    (a.report or (a.out / "native_replay.json")).write_text(text + "\n")
    print(
        json.dumps(
            {
                "run": report["run"],
                "ok": report["ok"],
                "seconds": report["seconds"],
                "error": report["error"],
                "results": [
                    (r["what"], r.get("identical"), r.get("rows_attempt"), r.get("first_difference"))
                    for r in report["results"]
                ],
            }
        )
    )
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
