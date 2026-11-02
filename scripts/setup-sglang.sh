#!/usr/bin/env bash
# The SGLang serving environment: Python 3.12 with exactly the packages in scripts/sglang.lock (SGLang 0.5.21, its
# torch 2.13 built for CUDA 13.0, its kernels). Honing never imports SGLang; it starts the server from this
# environment's interpreter (SGLANG_PYTHON), which refuses to start when the installed versions differ from the pin.
#
# Usage: scripts/setup-sglang.sh [env-dir]   (default .cache/sglang-env)
# Then: export SGLANG_PYTHON=<env-dir>/bin/python
set -euo pipefail
cd "$(dirname "$0")/.."
ENV=${1:-.cache/sglang-env}
UV=${UV:-uv}
[ -x "$ENV/bin/python" ] || $UV venv -q --python 3.12 "$ENV"
$UV pip sync -q --python "$ENV/bin/python" scripts/sglang.lock
"$ENV/bin/python" -c "import sglang, torch; print('sglang', sglang.__version__, 'torch', torch.__version__)"
echo "export SGLANG_PYTHON=$(cd "$ENV" && pwd)/bin/python"
