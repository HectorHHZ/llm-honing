#!/usr/bin/env bash
# Kev serving parity: three kev.serve references at the pinned commit and Honing's KevDecisionModel, same requests.
#   :8091 bf16, no CUDA graphs, prefix cache on -> the same computation as Honing (a state pass, then the question rows)
#   :8092 bf16, kev.serve's CUDA defaults (CUDA graphs, prefix cache) -> what kev.serve serves
#   :8093 fp32, no CUDA graphs, prefix cache on -> Kev's exact path, against Honing in fp32
# Usage: KEV_PYTHON=... KEV_SOURCE_DIR=... [KEV_MODEL=kev-4b] kev_serving.sh <out-dir>
#   (GPUs 1-4; HF_HUB_OFFLINE with the pinned snapshots; KEV_MODEL defaults to kev-0.8b)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); OUT=$1; mkdir -p "$OUT"
PY=${KEV_PYTHON:?}; export HF_HUB_OFFLINE=1 KEV_MODEL=${KEV_MODEL:-kev-0.8b}
SNAP=$("$PY" -c "import json,sys; sys.path.insert(0, '$KEV_SOURCE_DIR'); from kev.checkpoint import resolve_run; p=json.load(open('$HERE/../pins.json'))['kevModels']['$KEV_MODEL']; print(resolve_run(p['repo'] + '@' + p['revision']))")
cd "$KEV_SOURCE_DIR"
CUDA_VISIBLE_DEVICES=2 KEV_DTYPE=bf16 KEV_CUDA_GRAPHS=0 "$PY" -m kev.serve --run "$SNAP" --port 8091 > "$OUT/serve-8091.log" 2>&1 & A=$!
CUDA_VISIBLE_DEVICES=3 "$PY" -m kev.serve --run "$SNAP" --port 8092 > "$OUT/serve-8092.log" 2>&1 & B=$!
CUDA_VISIBLE_DEVICES=1 KEV_DTYPE=fp32 KEV_CUDA_GRAPHS=0 "$PY" -m kev.serve --run "$SNAP" --port 8093 > "$OUT/serve-8093.log" 2>&1 & C=$!
for port in 8091 8092 8093; do for _ in $(seq 1 120); do curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null && break; sleep 5; done; done
cd /tmp
"$PY" "$HERE/kev_serving.py" 4 "$OUT" bf16 8091:same-computation 8092:as-served
"$PY" "$HERE/kev_serving.py" 4 "$OUT" fp32 8093:exact-fp32
kill $A $B $C
