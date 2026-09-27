"""Stands in for integrations/ra2/integration/laya/run.mjs: same flags, same output files, no engine.

The decision server URL decides the result: a URL containing "strong" wins, "weak" loses,
anything else runs to the tick limit. FAKE_MATCH_EXIT sets the exit code.
"""

import json
import os
import sys
import time
from pathlib import Path

args = dict(a.removeprefix("--").split("=", 1) for a in sys.argv[1:])
out = Path(args["out"])
out.mkdir(parents=True)
started = time.time()
time.sleep(float(os.environ.get("FAKE_MATCH_SECONDS", "0")))
seat, other = args["seat"], "blue" if args["seat"] == "red" else "red"
url = args["decision-server"]
outcome = {seat: 1, other: -1} if "strong" in url else {seat: -1, other: 1} if "weak" in url else {seat: 0, other: 0}
truncated = outcome[seat] == 0
result = {
    "tick": int(args["ticks"]) if truncated else 9000,
    "ended": not truncated,
    "truncated": truncated,
    "outcome": outcome,
    "hash": {"tick": 9000, "hash": "abc"},
    "stats": {seat: {"submitted": 40, "accepted": 35, "rejected": 5, "filtered": 0}},
    "finalPlayers": [
        {
            "name": seat,
            "credits": 400,
            "credits_gained": 5200,
            "units_built": {"building": 6, "infantry": 4, "vehicle": 5, "aircraft": 0},
            "units_killed": {"building": 1, "infantry": 12, "vehicle": 2, "aircraft": 0},
            "units_lost": {"building": 0, "infantry": 3, "vehicle": 5, "aircraft": 0},
        }
    ],
    "arguments": args,
    "window": [started, time.time()],
}
(out / "result.json").write_text(json.dumps(result))
question = {"type": "choice", "instructions": "Next?", "criteria": {"A": "Wait", "B": "Power plant"}}
stage = {
    "stageIndex": 1,
    "tick": 3,
    "contextParts": {"essential": ["Tick 3."]},
    "questions": {"g1": question},
    "answers": {"g1": {"choice": "B", "probabilities": {"A": 0.3, "B": 0.7}}},
}
rows = [
    {"tick": 3, "decisionMode": "staged", "stages": [stage]},
    {"tick": 6, "decisionMode": "staged", "stages": [{**stage, "tick": 6, "questions": {}, "answers": {}}]},
]
(out / "decisions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
sys.exit(int(os.environ.get("FAKE_MATCH_EXIT", "0")))
