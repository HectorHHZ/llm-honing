#!/usr/bin/env bash
set -euo pipefail
RA2_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$RA2_SCRIPT_DIR/env.sh"
ra2_load_environment
if [[ "${1:-}" == --check ]]; then
  ra2_require_linux
  shift
  ra2_select_node
  exec "$RA2_NODE_BIN" "$RA2_SCRIPT_DIR/check.mjs" "$@"
fi
if [[ "${1:-}" == --report ]]; then
  shift
  ra2_select_node
  exec "$RA2_NODE_BIN" "$RA2_SCRIPT_DIR/report.mjs" "$@"
fi
for RA2_ARGUMENT in "$@"; do
  if [[ "$RA2_ARGUMENT" == --run ]]; then
    ra2_require_linux
  fi
done
ra2_select_node
exec "$RA2_NODE_BIN" "$RA2_SCRIPT_DIR/run.mjs" "$@"
