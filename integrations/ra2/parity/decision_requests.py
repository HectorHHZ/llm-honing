"""The decision requests the serving parity checks send: RA2-shaped states and 1 to 3 choice questions.

Some requests carry 400 log entries, which Laya's 2,048-token input has to trim.
"""

import random

BUILDINGS = ["Power plant", "Ore refinery", "Barracks", "War factory", "Radar", "Battle lab", "Wall", "Service depot"]


def requests(n=24, seed=0):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        essential = [f"Tick {30 * (i + 1)}. Credits {rng.randint(0, 10000)}. Army {rng.randint(0, 20)}."]
        logs = [
            f"t{j}: order {rng.choice(BUILDINGS)} queued and {rng.choice(['placed', 'cancelled', 'delayed'])}"
            for j in range(rng.choice([0, 5, 400]))
        ]
        parts = {
            "essential": essential,
            "details": [f"Visible enemy units: {rng.randint(0, 9)}"],
            "memory": [],
            "logs": logs,
        }
        questions = {}
        for q in range(rng.randint(1, 3)):
            opts = rng.sample(BUILDINGS, rng.randint(2, 8))
            questions[f"q{q}"] = {
                "type": "choice",
                "instructions": f"Which structure should be built next ({q})?",
                "criteria": {f"o{k}": label for k, label in enumerate(opts)},
            }
        out.append({"id": i + 1, "parts": parts, "questions": questions})
    return out
