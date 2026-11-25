#!/usr/bin/env bash
# The GPU check a model or engine change needs (scripts/gpu/paths.txt), from the repository root:
#   check.sh <out-dir> <gpu> <gpu> <gpu>
# with three free GPUs of one machine, by nvidia-smi index, after sourcing the env.sh scripts/gpu/setup.sh wrote.
#
# 1. gpu-tests: every test marked gpu, on the first GPU. The step fails when none ran or one was skipped, since a
#    skip here means the machine is missing something.
# 2. tank-direction-edit: configs/e2e/tank_direction_edit.json on a real NanoJev (two serving GPUs, one for the
#    edit), and its train move must publish.
# Writes <out-dir>/summary.md (a table, then the gate's line) and each step's log, and exits 1 if a step failed.
set -uo pipefail
out=$(mkdir -p "$1" && cd "$1" && pwd)
gpus=("$2" "$3" "$4")
failed=0
printf '| Step | Result | Time |\n| --- | --- | --- |\n' > "$out/summary.md"

step() {
  local name=$1 start=$SECONDS result=pass
  shift
  "$@" > "$out/$name.log" 2>&1 || { result=fail; failed=1; }
  printf '| %s | %s | %d min |\n' "$name" "$result" $(((SECONDS - start + 30) / 60)) >> "$out/summary.md"
}

gpu_tests() {
  # Only the files with a gpu test: other files skip themselves at import when an optional extra is missing.
  mapfile -t files < <(grep -rl --include="*.py" "mark.gpu" tests)
  NANOJEV_SOURCE_DIR=$HONING_NANOJEV_SOURCE NANOJEV_GPU=${gpus[0]} python -m pytest -m gpu -p no:cacheprovider \
    -rs --junitxml="$out/gpu-tests.xml" "${files[@]}" || return 1
  python - "$out/gpu-tests.xml" <<'PY'
import sys
import xml.etree.ElementTree as ET

suite = ET.parse(sys.argv[1]).getroot().find("testsuite")
ran, skipped = int(suite.get("tests")), int(suite.get("skipped"))
print(f"{ran} gpu tests, {skipped} skipped")
sys.exit(ran == 0 or skipped > 0)
PY
}

tank_direction_edit() {
  HONING_NANOJEV_GPUS=${gpus[0]},${gpus[1]} HONING_NANOJEV_EDIT_GPU=${gpus[2]} \
    python scripts/e2e/tank_campaign.py --config configs/e2e/tank_direction_edit.json --work-dir "$out/tank" &&
    python scripts/e2e/check_campaign.py "$out/tank" --expect train=published
}

step gpu-tests gpu_tests
step tank-direction-edit tank_direction_edit
grep -h "^train:" "$out/tank-direction-edit.log" | head -1 | sed 's/^/\n/' >> "$out/summary.md"
exit $failed
