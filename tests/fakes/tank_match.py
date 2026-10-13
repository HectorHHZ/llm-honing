"""Stands in for integrations/tank-duel/scripts/match.mjs: same flags, same output files, no game.

The decision server URL decides the result: a URL containing "strong" wins, "weak" loses, anything
else runs out the clock. ``--teacher`` records a teacher demonstration. FAKE_TANK_ERROR=harness
refuses the harness and FAKE_TANK_ERROR=budget fails as an over-long state does; FAKE_TANK_SEED
records another seed than the one asked for.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

args = dict(a.removeprefix("--").partition("=")[::2] for a in sys.argv[1:])
out = Path(args["out"])
out.mkdir(parents=True)
error = os.environ.get("FAKE_TANK_ERROR")
if error:
    message = {
        "harness": ("tank/controller.json has unknown setting decisionInterval", "harness"),
        "budget": ("Model did not preserve complete decision context", "match"),
        "crash": ("Wall penetration", "match"),
    }[error]
    (out / "error.json").write_text(json.dumps({"error": message[0], "kind": message[1]}))
    sys.exit(1)
harness_dir = Path(args["harness-dir"])
harness = []
for entry in json.loads((harness_dir / "index.json").read_text()):
    content = (harness_dir / entry["path"]).read_text()
    role = {"tank/controller.json": "controller", "tank/questions.json": "questions"}.get(entry["path"], "note")
    harness.append({**entry, "role": role, "sha256": hashlib.sha256(content.encode()).hexdigest()})
teacher = "teacher" in args
manifest = {
    "kind": "teacher-demonstration" if teacher else "rollout" if "sample-seed" in args else "evaluation",
    "seed": int(os.environ.get("FAKE_TANK_SEED", args["seed"])),
    "seat": args["seat"],
    "opponent": args["opponent"],
    "seconds": float(args["seconds"]),
    "interval_ms": int(args["interval-ms"]),
    "opponent_interval_ms": int(args["opponent-interval-ms"]),
    "sample_seed": int(args["sample-seed"]) if "sample-seed" in args else None,
    "harness": harness,
    "arguments": args,
}
(out / "manifest.json").write_text(json.dumps(manifest))
seat, other = args["seat"], "red" if args["seat"] == "blue" else "blue"
url = "" if teacher else args["decision-server"]
mine = 1 if "strong" in url else -1 if "weak" in url else 0
outcome = {seat: mine, other: -mine}
result = {
    "winner": seat if mine == 1 else other if mine == -1 else None,
    "reason": "timeout" if mine == 0 else "hit",
    "outcome": outcome,
    "sim_seconds": 30.0 if mine == 0 else 12.5,
    "shots": {seat: 3, other: 4},
    "decisions": {seat: 60, other: 40},
    "self_hits": [],
    "interventions": {seat: 2, other: 1},
}
(out / "result.json").write_text(json.dumps(result))
motion = {
    "type": "choice",
    "instructions": "Choose a motion.",
    "criteria": {"stop_none": "stop_none: use the forecast.", "forward_none": "forward_none: use the forecast."},
}
weapon = {"type": "choice", "instructions": "Fire?", "criteria": {"fire": "Fire now.", "hold": "Hold."}}
rows = [
    {
        "contextParts": {"essential": ["Seed state."], "details": [], "memory": [], "logs": []},
        "questions": {"motion": motion},
        "answers": {"motion": {"choice": "forward_none", "probabilities": {"stop_none": 0.25, "forward_none": 0.75}}},
        "chosen": {"motion": "stop_none"},
    },
    {
        "contextParts": {"essential": ["Weapon state."], "details": [], "memory": [], "logs": []},
        "questions": {"weapon": weapon},
        "answers": {"weapon": {"choice": "hold", "probabilities": {"fire": 0.4, "hold": 0.6}}},
        "chosen": {"weapon": "fire"},
    },
]
(out / "decisions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
