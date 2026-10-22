#!/usr/bin/env bash
# Installs inference dependencies only; never starts a game or trains a model.
set -euo pipefail
if [[ $# == 0 || ( $# == 1 && ( "$1" == --help || "$1" == -h ) ) ]]; then
  cat <<'HELP'
Usage: bash setup-linux.sh --install

Explicitly download and verify the pinned Laya v33 release, check out its pinned
Laya source, and install a CPU-only Python inference environment. No model or
game is loaded. Existing release/source files must match their pins.

Requires x86_64 Linux, Git, Python 3.12 with venv/pip support, and network access.
Node.js 22+, the prepared RA2 assets, SDK 0.1.1 and Supalosa dependencies must be
prepared separately. RA2_LAB_DIR defaults to $HOME/ra2-lab.

RA2_ENV_FILE=/absolute/path/to/trusted.env opts into shell configuration.
LAYA_BOOTSTRAP_PYTHON overrides the Python used to create the environment.
LAYA_VENV_DIR selects the environment; LAYA_PYTHON must point to its bin/python.
HELP
  exit 0
fi
if [[ $# != 1 || "$1" != --install ]]; then
  echo 'Setup requires exactly --install; use --help for preparation details.' >&2
  exit 2
fi
RA2_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$RA2_SCRIPT_DIR/env.sh"
ra2_load_environment
ra2_require_linux
LAYA_BOOTSTRAP_PYTHON="${LAYA_BOOTSTRAP_PYTHON:-python3}"
if [[ "$LAYA_PYTHON" != "$LAYA_VENV_DIR/bin/python" ]]; then
  echo 'Set LAYA_VENV_DIR and use its bin/python for LAYA_PYTHON when running setup.' >&2
  exit 1
fi
"$LAYA_BOOTSTRAP_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Setup requires Python 3.12"'
command -v git >/dev/null
LAYA_SOURCE_REV=970dc8c5f63d7b886a68409493f37d569424f933
mkdir -p "$RA2_LAB_DIR"
if [[ ! -e "$LAYA_SOURCE_DIR" ]]; then
  git clone --no-checkout https://github.com/NandhaKishorM/laya.git "$LAYA_SOURCE_DIR"
  git -C "$LAYA_SOURCE_DIR" checkout --detach "$LAYA_SOURCE_REV"
fi
if [[ "$(git -C "$LAYA_SOURCE_DIR" rev-parse HEAD)" != "$LAYA_SOURCE_REV" ]] || \
   [[ -n "$(git -C "$LAYA_SOURCE_DIR" status --porcelain --untracked-files=no)" ]]; then
  echo 'The Laya source directory differs from the selected pin; refusing to overwrite.' >&2
  exit 1
fi
"$LAYA_BOOTSTRAP_PYTHON" "$RA2_SCRIPT_DIR/download-release.py" --destination "$LAYA_RELEASE_DIR"
if [[ ! -d "$LAYA_VENV_DIR" ]]; then
  "$LAYA_BOOTSTRAP_PYTHON" -m venv "$LAYA_VENV_DIR"
fi
LAYA_INSTALL_PYTHON="$LAYA_VENV_DIR/bin/python"
"$LAYA_INSTALL_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), "The existing environment must use Python 3.12"'
"$LAYA_INSTALL_PYTHON" -m pip install --disable-pip-version-check 'torch==2.7.0' --index-url https://download.pytorch.org/whl/cpu
"$LAYA_INSTALL_PYTHON" -m pip install --disable-pip-version-check \
  'transformers==5.17.0' 'tokenizers==0.23.2' 'safetensors==0.8.0' \
  'huggingface_hub==1.32.0' 'numpy==1.26.4'
"$LAYA_INSTALL_PYTHON" -m pip install --disable-pip-version-check --no-deps "$LAYA_SOURCE_DIR"
"$LAYA_INSTALL_PYTHON" -m pip check
"$LAYA_INSTALL_PYTHON" -m pip freeze > "$RA2_LAB_DIR/laya-v33-cpu-pip-freeze.txt"
USE_TF=0 "$LAYA_INSTALL_PYTHON" - <<'PY'
import json,torch,transformers,laya
if torch.version.cuda is not None:
    raise RuntimeError('Expected a CPU-only PyTorch build')
print(json.dumps({'event':'cpu_environment_ready','torch':torch.__version__,
                  'transformers':transformers.__version__,'laya':laya.__version__,
                  'cuda':torch.version.cuda,'modelLoaded':False,'matchStarted':False}))
PY
