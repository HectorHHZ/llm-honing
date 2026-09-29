#!/usr/bin/env bash
# The Kev model environment: Python 3.12 with Kev's declared stack (torch 2.8, transformers 5.x, peft), the Kev source
# at the commit in pins.json, and the pinned kev-0.8b and Qwen base snapshots in the Hugging Face cache.
#
# Usage: scripts/setup-kev.sh [env-dir]    (default .cache/kev-env; the source goes to deps/kev)
# Then: export KEV_PYTHON=<env-dir>/bin/python KEV_SOURCE_DIR=$PWD/deps/kev
set -euo pipefail
cd "$(dirname "$0")/.."
ENV=${1:-.cache/kev-env}
UV=${UV:-uv}
PIN() { python3 -c "import json,sys; p=json.load(open('pins.json')); print(eval(sys.argv[1]))" "$1"; }
SOURCE_URL=$(PIN "p['kevSource']['url']"); SOURCE_REV=$(PIN "p['kevSource']['revision']")
MODEL=$(PIN "p['kevModels']['kev-0.8b']['repo']"); MODEL_REV=$(PIN "p['kevModels']['kev-0.8b']['revision']")
BASE=$(PIN "p['kevModels']['kev-0.8b']['base']"); BASE_REV=$(PIN "p['kevModels']['kev-0.8b']['baseRevision']")
[ -x "$ENV/bin/python" ] || $UV venv -q --python 3.12 "$ENV"
$UV pip install -q --python "$ENV/bin/python" "torch==2.8.*" --index-url https://download.pytorch.org/whl/cu128
# Kev's declared dependencies at the pinned commit, plus fastapi and uvicorn for its reference server (parity/).
$UV pip install -q --python "$ENV/bin/python" "transformers>=5.17,<6" "peft>=0.21" "accelerate>=1.15" "datasets>=3.0" \
  "scikit-learn>=1.9.1" safetensors "numpy>=2" "pydantic>=2.9" fastapi uvicorn huggingface_hub
# flash-linear-attention is not a Kev dependency, and with the Triton torch 2.8 ships (3.4) it refuses the gated delta
# rule backward on Hopper GPUs (fla #640), so training fails. Qwen3.5 runs its torch kernels without it.
$UV pip uninstall -q --python "$ENV/bin/python" flash-linear-attention fla-core 2>/dev/null || true
$UV pip install -q --python "$ENV/bin/python" --no-deps -e ../..
mkdir -p deps
[ -d deps/kev/.git ] || git clone -q "$SOURCE_URL" deps/kev
git -C deps/kev fetch -q origin "$SOURCE_REV"
git -C deps/kev checkout -q "$SOURCE_REV"
"$ENV/bin/python" - "$MODEL" "$MODEL_REV" "$BASE" "$BASE_REV" <<'PY'
import sys
from huggingface_hub import snapshot_download
model, model_rev, base, base_rev = sys.argv[1:]
print(snapshot_download(model, revision=model_rev))
print(snapshot_download(base, revision=base_rev))
PY
echo "export KEV_PYTHON=$PWD/$ENV/bin/python KEV_SOURCE_DIR=$PWD/deps/kev"
