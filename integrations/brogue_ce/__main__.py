"""Run a seeded Brogue demo and watch or verify the native recording."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path

from integrations.brogue_ce.engine import VERSION, BrogueSession, check_engine


def match(engine: Path, seed: int, out: Path, actions: list[str]) -> None:
    out.mkdir(parents=True)
    with BrogueSession(engine, seed) as game, (out / "trajectory.jsonl").open("x", encoding="utf-8") as trace:
        frame = game.observe()
        print(json.dumps({"key": None, "observation": asdict(frame)}), file=trace, flush=True)

        # Explicitly quit at the end of a bounded demo so Brogue closes the recording.
        for key in [*actions, "Q", "y"]:
            if frame.outcome is not None:
                break
            if any("You die..." in row for row in frame.rows[:3]):
                key = " "
            frame = game.step(key)
            print(json.dumps({"key": key, "observation": asdict(frame)}), file=trace, flush=True)

        if frame.outcome is None:
            raise RuntimeError(
                "The game is still at a prompt. End custom action lists with a native quit or game over."
            )
        recording = out / "match.broguerec"
        game.save_recording(recording)
    result = {
        "engine": f"Brogue CE {VERSION}",
        "seed": seed,
        "outcome": frame.outcome,
        "deepest_level": frame.deepest_level,
        "gold": frame.gold,
        "recording": recording.name,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**result, "out": str(out)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run the seeded native auto-explore demo")
    demo.add_argument("--engine", type=Path, required=True)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--out", type=Path, required=True)
    demo.add_argument("--actions", type=Path, help="JSON array of native keyboard inputs")

    verify = sub.add_parser("verify", help="check the recording with Brogue's native replay verifier")
    verify.add_argument("--engine", type=Path, required=True)
    verify.add_argument("recording", type=Path)

    replay = sub.add_parser("replay", help="open the recording in the original graphical renderer")
    replay.add_argument("--tiles", type=Path, required=True)
    replay.add_argument("recording", type=Path)

    play = sub.add_parser("play", help="open a new seeded game in the original graphical renderer")
    play.add_argument("--tiles", type=Path, required=True)
    play.add_argument("--seed", type=int, default=7)

    args = parser.parse_args()
    if args.command == "demo":
        actions = json.loads(args.actions.read_text(encoding="utf-8")) if args.actions else ["x"] * 20
        if not isinstance(actions, list) or not all(isinstance(key, str) for key in actions):
            parser.error("--actions must contain a JSON array of key strings")
        match(args.engine, args.seed, args.out.resolve(), actions)
        return
    if args.command == "verify":
        engine = args.engine.resolve()
        check_engine(engine)
        with tempfile.TemporaryDirectory(prefix="honing-br-verify-") as directory:
            subprocess.run([engine, "-vn", args.recording.resolve()], cwd=directory, check=True, timeout=120)
        return
    if args.command == "replay":
        native_args = ["-v", str(args.recording.resolve(strict=True))]
    else:
        native_args = ["-s", str(args.seed)]
    subprocess.run([args.tiles.resolve(), "-G", *native_args], check=True)


if __name__ == "__main__":
    main()
