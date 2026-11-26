"""Collect the planned attempts of one small-game Exp 0 v0 collection (Griddly collect_set.py, generalized).

  python3 collect_set.py <game: vvvvvv|hallowdale> <collection_root> --stream clean|alternate [--reserve]
         [--groups s01,s02] [--capture-first]

`collection_root/collection.json` is the frozen plan. Stream `clean` runs no_injection, harness and controller on
the clean service (8813); stream `alternate` runs weights on the alternate service (8814); the two streams can
run at the same time. For every (group, condition) slot the next attempt number is used when the previous
attempts are incomplete (error.json), up to the plan's attempt cap; nothing is overwritten or deleted. Initial
groups first; --reserve runs the reserve groups (after the initial ones were audited). Every finished attempt
prints one JSON line; the log of the whole call is the attempt log."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

GAMES = {
    "vvvvvv": Path(__file__).resolve().parent / "exp0_vvvvvv.py",
    "hallowdale": Path(__file__).resolve().parents[1] / "hallowdale" / "exp0_hallowdale.py",
}
ENDPOINTS = {"clean": "http://127.0.0.1:8813", "alternate": "http://127.0.0.1:8814"}
STREAMS = {"clean": ("no_injection", "harness", "controller"), "alternate": ("weights",)}


def load(game: str):
    spec = importlib.util.spec_from_file_location(f"exp0_{game}", GAMES[game])
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(GAMES[game].parent))
    spec.loader.exec_module(module)
    return module


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("game", choices=sorted(GAMES))
    ap.add_argument("root", type=Path)
    ap.add_argument("--stream", choices=sorted(STREAMS), required=True)
    ap.add_argument("--reserve", action="store_true")
    ap.add_argument("--groups", default="")
    ap.add_argument("--capture-first", action="store_true", help="capture frames for the first clean attempt (video)")
    ap.add_argument("--conditions", default="", help="only these conditions of the stream, e.g. harness (2026-11-26)")
    ap.add_argument("--display", type=int, help="one fixed Xvfb display for this stream (VVVVVV only, 2026-11-26)")
    args = ap.parse_args()
    root = args.root.resolve()
    plan = json.loads((root / "collection.json").read_text())
    groups = plan["groups"][plan["initial_groups"] :] if args.reserve else plan["groups"][: plan["initial_groups"]]
    if args.groups:
        wanted = set(args.groups.split(","))
        groups = [g for g in plan["groups"] if g["id"] in wanted]
    collector = load(args.game)
    conditions = STREAMS[args.stream]
    if args.conditions:
        wanted_conditions = args.conditions.split(",")
        if not set(wanted_conditions) <= set(conditions):
            ap.error(f"stream {args.stream} runs only {conditions}")
        conditions = tuple(c for c in conditions if c in wanted_conditions)
    options = {"display": args.display} if args.display is not None else {}
    first = True
    for group in groups:
        for condition in conditions:
            for number in range(1, plan["attempt_cap_per_slot"] + 1):
                run = root / f"{group['id']}-{condition}-a{number}"
                if run.exists():
                    if (run / "result.json").exists():
                        break  # a complete attempt exists for this slot
                    continue  # incomplete attempt kept; try the next number
                started = time.time()
                capture = args.capture_first and first and condition == "no_injection"
                out = collector.collect(
                    root, plan, group, condition, number, ENDPOINTS[args.stream], capture=capture, **options
                )
                first = False
                done = out / "result.json"
                info = json.loads(done.read_text()) if done.exists() else json.loads((out / "error.json").read_text())
                print(
                    json.dumps(
                        {
                            "run": out.name,
                            "status": info.get("status"),
                            "reason": info.get("reason"),
                            "model_decisions": info.get("model_decisions"),
                            "trigger_index": info.get("trigger_index"),
                            "wall_seconds": round(time.time() - started, 1),
                        }
                    ),
                    flush=True,
                )
                if done.exists():
                    break


if __name__ == "__main__":
    main()
