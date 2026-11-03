"""Deterministic, three-step fake for the integration pipeline state-machine tests."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

CODE_NAMES = ("lane_pilot.py", "lane_controller.py", "solo_intercept.py", "lane_io.py", "random_ab_server.py")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--code-dir", type=Path, required=True)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.run / "config.json").read_text())
    environment = config["environment"]
    sha = hashlib.sha256(json.dumps(environment, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    code = {name: hashlib.sha256((args.code_dir / name).read_bytes()).hexdigest() for name in CODE_NAMES}
    scores = {"incumbent": 100, "candidate": 120, "random": 80, "base": 90, "code": 105}
    if args.arm == "candidate":
        scores["candidate"] = config.get("fake_candidate_score", scores["candidate"])
    games = []
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.trace, "wt") as output:
        for seed in map(int, args.seeds.split(",")):
            games.append(
                {
                    "seed": seed,
                    "reward": scores[args.arm],
                    "steps": 3,
                    "finished": True,
                    "lives": 0,
                    "model_calls": 0 if args.arm == "code" else 1,
                }
            )
            for step in range(3):
                row = {
                    "seed": seed,
                    "step": step,
                    "reward": scores[args.arm] if step == 2 else 0,
                    "lives_before": 1,
                    "lives_after": 0 if step == 2 else 1,
                    "wave": 1,
                    "reason": "dodge",
                    "action": "FIRE",
                    "player_x": 50,
                    "bullets": [],
                    "lowest_alien_y": 130,
                    "alien_count": 2,
                    "target_type": "alien",
                    "urgent_low_aliens": False,
                    "locked_bullet_x": 50,
                    "model_choice": "A" if step == 0 and args.arm != "code" else None,
                    "lane_options": (
                        [{"bullet_x": 50}, {"bullet_x": 70}] if step == 0 and args.arm != "code" else None
                    ),
                    "selected_lane": {"bullet_x": 50} if step == 0 and args.arm != "code" else None,
                }
                output.write(json.dumps(row) + "\n")
    metadata = {"code_sha256": code, "weights_sha256": args.weights_sha256 or None, "environment_sha256": sha}
    if args.weights_sha256:
        health_path = args.summary.parent / "server-health.json"
        health_path.write_text(json.dumps({"ready": True, "metadata": {"weightsSha256": args.weights_sha256}}))
        metadata["model_health_sha256"] = hashlib.sha256(health_path.read_bytes()).hexdigest()
    data = {"results": games, "pipeline_metadata": metadata}
    args.summary.write_text(json.dumps(data))
    if args.weights_sha256:
        snapshot = {
            "weights_sha256": args.weights_sha256,
            "checkpoint": "fake/checkpoint",
            "parameters": 1000,
            "architecture": {"num_layers": 2, "hidden_size": 16, "attention_heads": 2, "ffn_size": 32},
        }
        (args.summary.parent / "model-snapshot.json").write_text(json.dumps(snapshot))


if __name__ == "__main__":
    main()
