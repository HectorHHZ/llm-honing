#!/usr/bin/env bash
# The djev environment for DiffusionGemma, without Docker: djev's own runtime recipe (runtime/Dockerfile) on a venv.
#
# 1. vLLM at djev's base commit and torch 2.13+cu130, the versions djev's runtime/sources.json pins;
# 2. djev's installer (runtime/install.py): it refuses any other torch, vLLM, transformers, or Pillow version,
#    checks every base and downloaded source file against its hashes, then writes the structured-read patches;
# 3. djev's API dependencies, after which the pinned versions are checked again;
# 4. the pinned DiffusionGemma snapshot (about 52 GB) in the Hugging Face cache.
#
# Needs an NVIDIA driver for CUDA 13 and a GPU that holds the BF16 weights (an 80 GB H100 does; djev targets B200).
# Usage: scripts/setup-djev.sh [env-dir]    (default .cache/djev-env; the source goes to deps/djev)
# Then: export DJEV_PYTHON=<env-dir>/bin/python DJEV_SOURCE_DIR=$PWD/deps/djev, and keep HF_HOME on local disk.
set -euo pipefail
cd "$(dirname "$0")/.."
ENV=${1:-.cache/djev-env}
UV=${UV:-uv}
PIN() { python3 -c "import json,sys; p=json.load(open('pins.json')); print(eval(sys.argv[1]))" "$1"; }
SOURCE_URL=$(PIN "p['djevSource']['url']"); SOURCE_REV=$(PIN "p['djevSource']['revision']")
if ! nvidia-smi | grep -Eq "CUDA Version: 1[3-9]"; then
  echo "djev's runtime needs an NVIDIA driver for CUDA 13 (torch 2.13+cu130)" >&2; exit 1
fi
mkdir -p deps .cache
[ -d deps/djev/.git ] || git clone -q "$SOURCE_URL" deps/djev
git -C deps/djev fetch -q origin "$SOURCE_REV"
git -C deps/djev checkout -q "$SOURCE_REV"
VERSION() { python3 -c "import json,sys; print(json.load(open('deps/djev/runtime/sources.json'))[sys.argv[1]] if sys.argv[1] != 'pkg' else json.load(open('deps/djev/runtime/sources.json'))['packages'][sys.argv[2]])" "$@"; }
BASE=$(VERSION base_commit); VLLM=$(VERSION pkg vllm); TORCH=$(VERSION pkg torch)
TRANSFORMERS=$(VERSION pkg transformers); PILLOW=$(VERSION pkg Pillow)
[ -x "$ENV/bin/python" ] || $UV venv -q --python 3.12 "$ENV"
$UV pip install -q --python "$ENV/bin/python" "vllm==$VLLM" "torch==$TORCH" \
  --extra-index-url "https://wheels.vllm.ai/$BASE" --extra-index-url https://download.pytorch.org/whl/cu130 \
  --index-strategy unsafe-best-match
$UV pip install -q --python "$ENV/bin/python" "transformers==$TRANSFORMERS" "Pillow==$PILLOW"
(cd deps/djev && "../../$ENV/bin/python" -m runtime.install > ../../.cache/djev-install-proof.json)
(cd deps/djev && $UV pip install -q --python "../../$ENV/bin/python" '.[tokenizer]')
$UV pip install -q --python "$ENV/bin/python" --no-deps -e ../..
"$ENV/bin/python" - <<'PY'
import importlib.metadata, json
pins = json.load(open("deps/djev/runtime/sources.json"))["packages"]
found = {name: importlib.metadata.version(name) for name in pins}
if found != pins:
    raise SystemExit(f"djev's pinned packages changed while installing its API: {found} != {pins}")
PY
$UV pip freeze --python "$ENV/bin/python" > .cache/djev-freeze.txt
"$ENV/bin/python" - <<'PY'
import json
from huggingface_hub import snapshot_download
for name, pin in json.load(open("pins.json"))["djevModels"].items():
    print(name, snapshot_download(pin["repo"], revision=pin["revision"]))
PY
echo "export DJEV_PYTHON=$PWD/$ENV/bin/python DJEV_SOURCE_DIR=$PWD/deps/djev"
