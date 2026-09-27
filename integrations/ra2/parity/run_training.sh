#!/usr/bin/env bash
# Training parity for #56: #49's train_laya.py twice and Honing's LocalJevTrainer once, same data and seed.
# Usage: PR49_DIR=... LAYA_PYTHON=... LAYA_SOURCE_DIR=... run_training.sh <work-dir> [--deterministic] [epochs]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd)
W=$1; MODE=${2:-}; EPOCHS=${3:-1}; PY=${LAYA_PYTHON:-python3}; mkdir -p "$W"
"$PY" "$HERE/make_demos.py" "$W/demos" > "$W/demos.txt"; M=$(tr '\n' ' ' < "$W/demos.txt")
WRAP=(); OURS_FLAG=()
if [ "$MODE" = "--deterministic" ]; then WRAP=("$HERE/deterministic_49.py"); OURS_FLAG=(--deterministic); fi
cd "$PR49_DIR"
"$PY" "${WRAP[@]}" league/train_laya.py --matches $M --out "$W/pr49-a" --gpu "${GPU_A:-5}" --epochs "$EPOCHS" --seed 7 > "$W/pr49-a.log" 2>&1 &
"$PY" "${WRAP[@]}" league/train_laya.py --matches $M --out "$W/pr49-b" --gpu "${GPU_B:-6}" --epochs "$EPOCHS" --seed 7 > "$W/pr49-b.log" 2>&1 &
(cd "$REPO" && PYTHONPATH="$REPO" "$PY" "$HERE/train_honing.py" "$W/honing" "${GPU_C:-7}" "$EPOCHS" $M "${OURS_FLAG[@]}" > "$W/honing.log" 2>&1) &
wait
PARENT=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['path'])" "$PR49_DIR/.cache/models/typed-decisions.json")
"$PY" "$HERE/compare_training.py" "$PARENT" "$W/pr49-a" "$W/pr49-b" "$(ls -d "$W"/honing/train-*)"
