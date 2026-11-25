#!/usr/bin/env bash
# Prepare a GPU machine for scripts/gpu/check.sh, from the repository root: setup.sh <env-dir>.
#
# Into <env-dir>: Node 22, NanoJev at the revision integrations/ra2/pins.json names, the pinned NanoJev and base
# weights in a Hugging Face cache, and a Python 3.12 venv with NanoJev's requirements-toy.txt and this checkout of
# Honing (editable, with its dev extra). Each piece already there is kept, so a second run only reinstalls Honing.
# Writes <env-dir>/env.sh, which check.sh's caller sources. Needs uv, git and curl.
set -euo pipefail
env_dir=$(mkdir -p "$1" && cd "$1" && pwd)
root=$(pwd)
source=$(python3 -c "import json; s = json.load(open('integrations/ra2/pins.json'))['nanojevSource']; print(s['url'], s['revision'])")
read -r nanojev_url nanojev_revision <<< "$source"

if [ ! -x "$env_dir/node/bin/node" ]; then
  curl -fsSL https://nodejs.org/dist/v22.23.5/node-v22.23.5-linux-x64.tar.xz -o "$env_dir/node.tar.xz"
  mkdir -p "$env_dir/node" && tar -xJf "$env_dir/node.tar.xz" -C "$env_dir/node" --strip-components=1
  rm "$env_dir/node.tar.xz"
fi

[ -d "$env_dir/nanojev" ] || git clone -q "$nanojev_url" "$env_dir/nanojev"
git -C "$env_dir/nanojev" fetch -q origin 2>/dev/null || true
git -C "$env_dir/nanojev" checkout -q "$nanojev_revision"

[ -d "$env_dir/venv" ] || uv venv -q --python 3.12 "$env_dir/venv"
uv pip install -q -p "$env_dir/venv/bin/python" -r "$env_dir/nanojev/requirements-toy.txt" huggingface_hub
uv pip install -q -p "$env_dir/venv/bin/python" -e "$root[dev]"

HF_HOME="$env_dir/hf" "$env_dir/venv/bin/python" - <<'PY'
import json
from huggingface_hub import snapshot_download

for model in json.load(open("integrations/ra2/pins.json"))["nanojevModels"].values():
    snapshot_download(model["repo"], revision=model["revision"])
    snapshot_download(model["base"], revision=model["baseRevision"])
PY

cat > "$env_dir/env.sh" <<ENV
export PATH="$env_dir/node/bin:$env_dir/venv/bin:\$PATH"
export HF_HOME="$env_dir/hf" HF_HUB_OFFLINE=1
export HONING_NANOJEV_SOURCE="$env_dir/nanojev"
ENV
echo "ready: source $env_dir/env.sh"
