"""Independent replay-only verification. Never imports or invokes a game policy."""

import argparse
import hashlib
import json
from pathlib import Path

from ra2_game_env import Engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match", required=True, type=Path)
    parser.add_argument("--game-dir", required=True, type=Path)
    args = parser.parse_args()
    expected = json.loads((args.match / "result.json").read_text())
    replay = args.match / "match.rplx"
    with Engine(args.game_dir) as engine:
        step = engine.load_replay(replay, observe=False)
        while not step.done and step.tick < expected["tick"]:
            before = step.tick
            step = engine.step(ticks=min(450, expected["tick"] - step.tick), observe=False)
            if step.replay_mismatch or step.tick <= before:
                break
        tick, value = engine.hash()
        actual = {"tick": tick, "hash": value}
        verified = bool(
            step.done
            and not step.replay_mismatch
            and actual == expected["hash"]
            and step.ended == expected["ended"]
            and step.truncated == expected["truncated"]
        )
        report = {
            "verified": verified,
            "expectedHash": expected["hash"],
            "actualHash": actual,
            "ended": step.ended,
            "truncated": step.truncated,
            "replayMismatch": step.replay_mismatch,
            "replaySha256": hashlib.sha256(replay.read_bytes()).hexdigest(),
            "playbackEngine": engine.hello,
            "policyInvoked": False,
        }
    (args.match / "replay-verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    raise SystemExit(0 if verified else 1)


if __name__ == "__main__":
    main()
