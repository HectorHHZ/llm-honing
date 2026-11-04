#!/usr/bin/env bash
# The Nimble-v4-max model environment: Python 3.12 with the stack Bespoke's Decision Index run used (torch 2.8 on
# cu128, transformers 5.17.0, accelerate, flash-linear-attention 0.5.2, causal-conv1d 1.7.0), the Decision Index kit
# at the commit in pins.json, and in the Hugging Face cache: the pinned model, its base model's tokenizer, and the
# pinned snapshot of Bespoke's inference code. causal-conv1d builds against the installed torch, so nvcc must be on
# PATH.
#
# Usage: scripts/setup-nimble.sh [env-dir]      (env default .cache/nimble-env; the kit goes to deps/decision-index)
#   The model is about 55 GB; set HF_HOME to a disk with room for it.
# Then: export the three variables the script prints.
set -euo pipefail
cd "$(dirname "$0")/.."
ENV=${1:-.cache/nimble-env}
UV=${UV:-uv}
PIN() { python3 -c "import json,sys; p=json.load(open('pins.json')); print(eval(sys.argv[1]))" "$1"; }
KIT_URL=$(PIN "p['decisionIndexKit']['url']"); KIT_REV=$(PIN "p['decisionIndexKit']['revision']")
mkdir -p deps .cache
[ -d deps/decision-index/.git ] || git clone -q "$KIT_URL" deps/decision-index
git -C deps/decision-index fetch -q origin "$KIT_REV"
git -C deps/decision-index checkout -q "$KIT_REV"
[ -x "$ENV/bin/python" ] || $UV venv -q --python 3.12 "$ENV"
$UV pip install -q --python "$ENV/bin/python" "torch==2.8.*" --index-url https://download.pytorch.org/whl/cu128
$UV pip install -q --python "$ENV/bin/python" "transformers==5.17.0" "flash-linear-attention==0.5.2" accelerate \
  safetensors numpy huggingface_hub httpx ninja packaging setuptools wheel
$UV pip install -q --python "$ENV/bin/python" --no-build-isolation "causal-conv1d==1.7.0"
$UV pip install -q --python "$ENV/bin/python" --no-deps -e ../..
"$ENV/bin/python" - <<'PY' | tee .cache/nimble-paths.txt
import json
from huggingface_hub import snapshot_download
from honing.backends.jev.nimble import BASE_TOKENIZER_PATTERNS, DOWNLOAD_PATTERNS, SOURCE_PATTERNS
pins = json.load(open("pins.json"))
for name, pin in pins["nimbleModels"].items():
    print(name, snapshot_download(pin["repo"], revision=pin["revision"], allow_patterns=DOWNLOAD_PATTERNS))
    print(name, "base tokenizer", snapshot_download(pin["base"], revision=pin["baseRevision"],
                                                    allow_patterns=BASE_TOKENIZER_PATTERNS))
source = pins["nimbleSource"]
print("source", snapshot_download(source["repo"], repo_type=source["repoType"], revision=source["revision"],
                                  allow_patterns=SOURCE_PATTERNS))
PY
SOURCE=$(awk '$1 == "source" {print $2}' .cache/nimble-paths.txt)
echo "export NIMBLE_PYTHON=$(cd "$ENV" && pwd)/bin/python NIMBLE_SOURCE_DIR=$SOURCE NIMBLE_KIT_DIR=$PWD/deps/decision-index"
