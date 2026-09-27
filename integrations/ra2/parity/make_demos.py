"""Labelled RA2-style training matches in #49's directory format, generated without the game.

Six matches (seeds 100 to 105 on protocol training maps); seed 101 is the held-out validation seed.
Questions mimic the staged controller: a root intent question (option A waits) and parameter stages.
"""

import hashlib
import json
import random
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
MAPS = ["mp03t4.map", "mp06t2.map", "tn01t2.map"]
INTENTS = [
    "Establish or expand the base and its economy",
    "Recruit additional combat or specialist units",
    "Command troops to fight the visible enemy",
    "Send combat troops to scout for the enemy base",
    "Maintain resource collection, resume paused production or repair damage",
]
BUILDS = ["Power plant", "Ore refinery", "Barracks", "War factory", "Radar tower", "Service depot"]
UNITS = ["Rhino tank", "Conscript", "Flak trooper", "Terror drone", "War miner"]


def row(rng, seed, seat, map_name, index):
    tick = 3 * (index + 1)
    credits = rng.randint(0, 12000)
    army = rng.randint(0, 24)
    parts = {
        "essential": [
            f"Decision stage 1, tick {tick}. Simulation paused; this decision's actions have not executed yet.",
            f"Own economy: {credits} credits, power {rng.randint(0, 400)}/{rng.randint(0, 400)}, army {army}.",
        ],
        "details": [f"Visible enemy: {rng.randint(0, 9)} units near ({rng.randint(0, 120)}, {rng.randint(0, 120)})."],
        "memory": [f"Enemy last seen at tick {max(0, tick - rng.randint(0, 300))}."],
        "logs": [
            f"t{tick - k}: {rng.choice(BUILDS)} order {rng.choice(['queued', 'placed', 'cancelled'])}"
            for k in range(rng.randint(0, 12))
        ],
    }
    stage = rng.choice([1, 1, 2, 3])
    if stage == 1:
        options = {"A": "Take no action for this group this turn."}
        options.update({chr(66 + i): label for i, label in enumerate(rng.sample(INTENTS, rng.randint(2, 5)))})
        instructions = "Choose the next intent. Select the action first; unresolved parameters will be asked next."
    else:
        pool = BUILDS if stage == 2 else UNITS
        options = {"A": "Cancel this pending action; submit no command for it."}
        options.update({chr(66 + i): label for i, label in enumerate(rng.sample(pool, rng.randint(2, 5)))})
        instructions = f"For the previously selected action, choose the {'structure' if stage == 2 else 'actor'}."
    questions = {"g1": {"type": "choice", "instructions": instructions, "criteria": options}}
    answers = {"g1": {"choice": rng.choice(list(options))}}
    if rng.random() < 0.3:
        extra = {
            "A": "Take no action for this group this turn.",
            "B": "Reposition or defend ore harvesters; these vehicles provide mining income",
        }
        questions["g2"] = {"type": "choice", "instructions": "Worker orders for this turn.", "criteria": extra}
        answers["g2"] = {"choice": rng.choice(list(extra))}
    return {
        "split": "train",
        "seed": seed,
        "seat": seat,
        "map": map_name,
        "stage": stage,
        "parts": parts,
        "questions": questions,
        "answers": answers,
    }


for seed in range(100, 106):
    rng = random.Random(seed)
    seat = "red" if seed % 2 == 0 else "blue"
    map_name = MAPS[seed % len(MAPS)]
    d = OUT / f"demo-{map_name.removesuffix('.map')}-s{seed}-{seat}"
    (d / "source").mkdir(parents=True, exist_ok=True)
    teacher = f"// synthetic teacher for seed {seed}\n".encode()
    (d / "source" / "teacher.mjs").write_bytes(teacher)
    manifest = {
        "split": "train",
        "seed": seed,
        "seat": seat,
        "map": map_name,
        "kind": "teacher-demonstration",
        "sourceSha256": {"teacher.mjs": hashlib.sha256(teacher).hexdigest()},
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (d / "result.json").write_text(json.dumps({"notModelEvaluation": True, "kind": "teacher-demonstration"}))
    with (d / "samples.jsonl").open("w") as f:
        for i in range(90):
            f.write(json.dumps(row(rng, seed, seat, map_name, i)) + "\n")
    print(d)
