#!/bin/bash
# Build the environment the slime agent worker runs in (honing/backends/slime/agent_worker.py), off Slime's image.
#
# Slime's adapter and harness import torch, and Slime does not pip-install on macOS, so this makes a venv with
# CPU torch and the few packages slime.agent imports, and puts a full checkout of Slime at the pinned commit
# next to it (the agent package needs slime/utils too). The interpreter is picked for this machine's
# architecture: on an Apple Silicon Mac, a Python 3.12 built for x86_64 runs under Rosetta and gets torch 2.2,
# the last release for that platform, which then fails against NumPy 2. The commit is read from pyproject's
# runtime group, so it cannot drift from the pin.
#
#   scripts/setup-slime-agent-env.sh DEST
#   export HONING_SLIME_AGENT_ENV=DEST   # DEST/env/bin/python, with DEST/slime on PYTHONPATH
set -euo pipefail
DEST="${1:?usage: scripts/setup-slime-agent-env.sh DEST}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIN="$(sed -n 's|.*slime\.git@\([0-9a-f]\{40\}\).*|\1|p' "${REPO_ROOT}/pyproject.toml")"
[ "${#PIN}" -eq 40 ] || { echo "no 40-character slime commit in pyproject.toml" >&2; exit 1; }

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) PYTHON=cpython-3.12-macos-aarch64-none; MACHINE=arm64 ;;
  Linux-x86_64) PYTHON=cpython-3.12-linux-x86_64-gnu; MACHINE=x86_64 ;;
  Linux-aarch64) PYTHON=cpython-3.12-linux-aarch64-gnu; MACHINE=aarch64 ;;
  *) echo "no slime agent environment for $(uname -s)-$(uname -m)" >&2; exit 1 ;;
esac
# PyPI's Linux torch wheels bundle CUDA; the CPU build is enough to import slime's types.
TORCH_INDEX=()
if [ "$(uname -s)" = Linux ]; then TORCH_INDEX=(--index-url https://download.pytorch.org/whl/cpu); fi

mkdir -p "${DEST}"
uv venv --quiet --clear --python "${PYTHON}" "${DEST}/env"
GOT="$("${DEST}/env/bin/python" -c 'import platform; print(platform.machine())')"
[ "${GOT}" = "${MACHINE}" ] || { echo "${DEST}/env runs on ${GOT}, not ${MACHINE}" >&2; exit 1; }
VIRTUAL_ENV="${DEST}/env" uv pip install --quiet ${TORCH_INDEX[@]+"${TORCH_INDEX[@]}"} torch==2.14.1
# transformers and numpy as Slime's image has them at the pin: SGLang v0.5.15.post1 (the image's base,
# docker/Dockerfile:1-2) pins transformers==5.12.1, and docker/Dockerfile:109 pins numpy==1.26.4 for Megatron.
VIRTUAL_ENV="${DEST}/env" uv pip install --quiet \
  transformers==5.12.1 tokenizers==0.22.2 aiohttp==3.14.5 httpx==0.28.1 numpy==1.26.4

if [ ! -d "${DEST}/slime/.git" ]; then git -C "${DEST}" init -q slime; fi
git -C "${DEST}/slime" fetch -q --depth 1 https://github.com/THUDM/slime.git "${PIN}"
git -C "${DEST}/slime" checkout -q --force FETCH_HEAD
FETCHED="$(git -C "${DEST}/slime" rev-parse HEAD)"
[ "${FETCHED}" = "${PIN}" ] || { echo "fetched ${FETCHED}, expected ${PIN}" >&2; exit 1; }

PYTHONPATH="${DEST}/slime" "${DEST}/env/bin/python" -c \
  'import slime.agent.adapters, slime.agent.aiohttp_threaded, slime.agent.harness, slime.agent.sandbox'
echo "slime agent environment at ${DEST}: slime ${PIN}, $("${DEST}/env/bin/python" -c 'import torch; print("torch", torch.__version__)')"
