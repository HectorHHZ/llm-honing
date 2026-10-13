#!/usr/bin/env bash
# Serving end to end: Honing's decision server as its own process, and the RA2 match's Node client stack against it.
# Usage: scripts/e2e/serving.sh [out-dir]    (Python with Honing installed, Node 22)
# DECISION_SERVER=<url> checks a decision server that is already running, such as a real model on a GPU, instead of
# starting the CPU fixture.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${1:-$(mktemp -d)}; mkdir -p "$OUT"
# 24 controller-shaped requests from a fixed seed: 1 to 3 choice questions, some with 400 log entries.
python - "$OUT/requests.json" <<'PY'
import json, random, sys
rng = random.Random(0)
BUILDINGS = ["Power plant", "Ore refinery", "Barracks", "War factory", "Radar", "Battle lab", "Wall", "Service depot"]
requests = []
for i in range(24):
    parts = {
        "essential": [f"Tick {30 * (i + 1)}. Credits {rng.randint(0, 10000)}. Army {rng.randint(0, 20)}."],
        "details": [f"Visible enemy units: {rng.randint(0, 9)}"],
        "memory": [],
        "logs": [f"t{j}: order {rng.choice(BUILDINGS)} queued" for j in range(rng.choice([0, 5, 400]))],
    }
    questions = {
        f"q{q}": {
            "type": "choice",
            "instructions": f"Which structure should be built next ({q})?",
            "criteria": {f"o{k}": label for k, label in enumerate(rng.sample(BUILDINGS, rng.randint(2, 8)))},
        }
        for q in range(rng.randint(1, 3))
    }
    requests.append({"parts": parts, "questions": questions})
json.dump(requests, open(sys.argv[1], "w"))
PY
if [ -n "${DECISION_SERVER:-}" ]; then
  node "$ROOT/scripts/e2e/serving_client.mjs" "$DECISION_SERVER" "$OUT/requests.json" --real
  exit
fi
python "$ROOT/scripts/e2e/fake_decision_server.py" > "$OUT/server.out" 2> "$OUT/server.err" &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 1 50); do grep -q '"ready"' "$OUT/server.out" 2>/dev/null && break; sleep 0.2; done
URL=$(python -c "import json,sys; print(json.loads(open(sys.argv[1]).readline())['url'])" "$OUT/server.out")
node "$ROOT/scripts/e2e/serving_client.mjs" "$URL" "$OUT/requests.json"
