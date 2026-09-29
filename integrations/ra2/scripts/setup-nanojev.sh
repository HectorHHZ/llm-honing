#!/usr/bin/env bash
# The NanoJev model environment: Python 3.12 with NanoJev's own pinned requirements (requirements-toy.txt at the
# source commit in pins.json: torch 2.14, transformers 5.17, safetensors, numpy), the NanoJev source at that commit,
# and the pinned model's inference files in the Hugging Face cache.
#
# Usage: scripts/setup-nanojev.sh [env-dir]    (default .cache/nanojev-env; the source goes to deps/nanojev)
# Then: export NANOJEV_PYTHON=<env-dir>/bin/python NANOJEV_SOURCE_DIR=$PWD/deps/nanojev
set -euo pipefail
cd "$(dirname "$0")/.."
ENV=${1:-.cache/nanojev-env}
UV=${UV:-uv}
PIN() { python3 -c "import json,sys; p=json.load(open('pins.json')); print(eval(sys.argv[1]))" "$1"; }
SOURCE_URL=$(PIN "p['nanojevSource']['url']"); SOURCE_REV=$(PIN "p['nanojevSource']['revision']")
mkdir -p deps
[ -d deps/nanojev/.git ] || git clone -q "$SOURCE_URL" deps/nanojev
git -C deps/nanojev fetch -q origin "$SOURCE_REV"
git -C deps/nanojev checkout -q "$SOURCE_REV"
[ -x "$ENV/bin/python" ] || $UV venv -q --python 3.12 "$ENV"
TORCH=$(grep '^torch==' deps/nanojev/requirements-toy.txt)
$UV pip install -q --python "$ENV/bin/python" "$TORCH" --index-url https://download.pytorch.org/whl/cu130
$UV pip install -q --python "$ENV/bin/python" -r <(grep -v '^torch==' deps/nanojev/requirements-toy.txt) huggingface_hub
$UV pip install -q --python "$ENV/bin/python" --no-deps -e ../..
"$ENV/bin/python" - <<'PY'
import json
from huggingface_hub import snapshot_download
from honing.backends.jev.nanojev import DOWNLOAD_PATTERNS
for name, pin in json.load(open("pins.json"))["nanojevModels"].items():
    print(name, snapshot_download(pin["repo"], revision=pin["revision"], allow_patterns=DOWNLOAD_PATTERNS))
PY
echo "export NANOJEV_PYTHON=$PWD/$ENV/bin/python NANOJEV_SOURCE_DIR=$PWD/deps/nanojev"
