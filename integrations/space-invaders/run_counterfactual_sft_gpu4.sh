#!/usr/bin/env bash
set -euo pipefail

pilot=${SFT_PILOT_DIR:?Set SFT_PILOT_DIR}
code_root=${SFT_CODE_ROOT:?Set SFT_CODE_ROOT}
runtime=${SFT_RUNTIME:?Set SFT_RUNTIME}
source_dir=${SFT_SOURCE_DIR:?Set SFT_SOURCE_DIR}
cache=${SFT_CACHE:?Set SFT_CACHE}
hold_script=${SFT_HOLD_SCRIPT:?Set SFT_HOLD_SCRIPT}
hold_unit=${SFT_HOLD_UNIT:?Set SFT_HOLD_UNIT}
gpu_uuid=${SFT_GPU_UUID:?Set SFT_GPU_UUID}
pins=${SFT_PINS:?Set SFT_PINS}
base=${SFT_BASE_CHECKPOINT:?Set SFT_BASE_CHECKPOINT}
base_sha256=${SFT_BASE_SHA256:?Set SFT_BASE_SHA256}
work_root=${SFT_WORK_ROOT:?Set SFT_WORK_ROOT}
tag=${SFT_TAG:?Set SFT_TAG}
input="$pilot/runs/counterfactual-sft-$tag"
work_dir="$work_root/counterfactual-sft-$tag"
train_per_label=${SFT_TRAIN_PER_LABEL:-100}
validation_per_label=${SFT_VALIDATION_PER_LABEL:-30}
replay_per_label=${SFT_REPLAY_PER_LABEL:-0}
reserved_game_seeds=${SFT_RESERVED_GAME_SEEDS:?Set SFT_RESERVED_GAME_SEEDS}
hold_stopped=0

cleanup() {
  rc=$?
  trap - EXIT INT TERM
  if [[ "$hold_stopped" == 1 ]]; then
    for attempt in 1 2 3 4 5; do
      systemctl --user reset-failed "$hold_unit" 2>/dev/null || true
      if systemd-run --user --unit=gpu4-new-task-hold \
        --description='Temporary GPU4 memory hold for user new task' \
        --setenv="CUDA_VISIBLE_DEVICES=$gpu_uuid" "$runtime" "$hold_script"; then break; fi
      sleep 2
    done
    sleep 2
    systemctl --user is-active "$hold_unit" || rc=1
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv,noheader | grep "$gpu_uuid" || rc=1
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

test -f "$input/train.jsonl"
test -f "$input/validation.jsonl"
if [[ "$replay_per_label" -gt 0 ]]; then test -f "$input/replay.jsonl.gz"; fi
test -f "$input/train_rollout_pair_sft.py"
test -f "$base/best.safetensors"
[[ $(sha256sum "$base/best.safetensors" | cut -d ' ' -f 1) == "$base_sha256" ]] || {
  echo 'Base checkpoint weights differ from the frozen incumbent' >&2
  exit 1
}
python3 - "$input/train.jsonl" "$input/validation.jsonl" <<'PY'
import json
import sys
train = [json.loads(line) for line in open(sys.argv[1])]
validation = [json.loads(line) for line in open(sys.argv[2])]
if len(train) < 200 or len(validation) < 60:
    raise ValueError(f'Insufficient matched labels: {len(train)} train, {len(validation)} validation')
if {row['seed'] for row in train} & {row['seed'] for row in validation}:
    raise ValueError('Training and validation seeds overlap')
PY
mkdir -p "$work_dir"
hold_pid=$(systemctl --user show "$hold_unit" -p MainPID --value)
[[ "$hold_pid" =~ ^[0-9]+$ && "$hold_pid" -gt 0 ]]
[[ $(ps -p "$hold_pid" -o args=) == "$runtime $hold_script" ]]
[[ $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$gpu_uuid, $hold_pid") == 1 ]]
systemctl --user stop "$hold_unit"
hold_stopped=1
for attempt in $(seq 1 20); do
  if ! nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -q "$gpu_uuid"; then break; fi
  sleep 1
done
if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -q "$gpu_uuid"; then
  echo 'GPU4 did not become free' >&2
  exit 1
fi

replay_args=()
if [[ "$replay_per_label" -gt 0 ]]; then
  replay_args=(--replay-records "$input/replay.jsonl.gz" --replay-per-label "$replay_per_label"
               --replay-min-delta 0)
fi
timeout 7200 env PYTHONPATH="$code_root" HF_HOME="$cache" HF_HUB_OFFLINE=1 \
  USE_TF=0 TOKENIZERS_PARALLELISM=false "$runtime" "$input/train_rollout_pair_sft.py" \
  --train-records "$input/train.jsonl" \
  --validation-records "$input/validation.jsonl" \
  --work-dir "$work_dir" --pins "$pins" --source-dir "$source_dir" \
  --base-checkpoint "$base" --gpu 4 --epochs 1 \
  --train-per-label "$train_per_label" --validation-per-label "$validation_per_label" \
  --min-delta 20 "${replay_args[@]}" \
  --validation-seeds auto --reserved-game-seeds "$reserved_game_seeds" \
  --batch-name spaceinvaders-current-sft-counterfactual-v1 \
  --teacher-description 'same-harness matched A/B ALE continuations with current SFT after the first choice' \
  --teacher-value-description 'paired natural-end game score; both arms finish with zero lives' \
  > "$input/train.log" 2>&1
cat "$work_dir/dataset-audit.json"
cat "$work_dir/train-result.json"
