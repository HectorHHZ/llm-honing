#!/usr/bin/env bash
# The StartLux-Decision model environment: Python 3.12 with the stack StartLux was tested with (torch 2.11,
# transformers 5.8.1, flash-linear-attention 0.5.2, causal-conv1d 1.7.0), the StartLux source at the commit in
# pins.json, and the pinned models in the Hugging Face cache. causal-conv1d builds against the installed torch, so
# nvcc must be on PATH. Without these two kernels StartLux refuses to start on a GPU.
#
# Usage: [STARTLUX_MODELS="startlux-0.8b startlux-4b"] scripts/setup-startlux.sh [env-dir]
#   (default: every model in pins.json's startluxModels; env default .cache/startlux-env; the source goes to
#   deps/startlux). The 27B model is about 54 GB; set HF_HOME to a disk with room for it.
# Then: export STARTLUX_PYTHON=<env-dir>/bin/python STARTLUX_SOURCE_DIR=$PWD/deps/startlux
set -euo pipefail
cd "$(dirname "$0")/.."
ENV=${1:-.cache/startlux-env}
UV=${UV:-uv}
PIN() { python3 -c "import json,sys; p=json.load(open('pins.json')); print(eval(sys.argv[1]))" "$1"; }
SOURCE_URL=$(PIN "p['startluxSource']['url']"); SOURCE_REV=$(PIN "p['startluxSource']['revision']")
MODELS=${STARTLUX_MODELS:-$(PIN "' '.join(p['startluxModels'])")}
mkdir -p deps
[ -d deps/startlux/.git ] || git clone -q "$SOURCE_URL" deps/startlux
git -C deps/startlux fetch -q origin "$SOURCE_REV"
git -C deps/startlux checkout -q "$SOURCE_REV"
[ -x "$ENV/bin/python" ] || $UV venv -q --python 3.12 "$ENV"
$UV pip install -q --python "$ENV/bin/python" "torch==2.11.*" --index-url https://download.pytorch.org/whl/cu130
$UV pip install -q --python "$ENV/bin/python" "transformers==5.8.1" "flash-linear-attention==0.5.2" safetensors numpy \
  pillow huggingface_hub accelerate ninja packaging setuptools wheel
$UV pip install -q --python "$ENV/bin/python" --no-build-isolation "causal-conv1d==1.7.0"
$UV pip install -q --python "$ENV/bin/python" --no-deps -e ../..
"$ENV/bin/python" - $MODELS <<'PY'
import json, sys
from huggingface_hub import snapshot_download
from honing.backends.jev.startlux import DOWNLOAD_PATTERNS
pins = json.load(open("pins.json"))["startluxModels"]
for name in sys.argv[1:]:
    pin = pins[name]
    print(name, snapshot_download(pin["repo"], revision=pin["revision"], allow_patterns=DOWNLOAD_PATTERNS))
PY
echo "export STARTLUX_PYTHON=$(cd "$ENV" && pwd)/bin/python STARTLUX_SOURCE_DIR=$PWD/deps/startlux"
