#!/usr/bin/env bash
# Shared path configuration only. Sourcing this file does not run a model/game.

ra2_load_environment() {
  # Configuration files are executable shell code and are opt-in only.
  if [[ -n "${RA2_ENV_FILE:-}" ]]; then
    if [[ ! -r "$RA2_ENV_FILE" ]]; then
      echo "Cannot read RA2_ENV_FILE: $RA2_ENV_FILE" >&2
      return 1
    fi
    local ra2_previous_allexport=false
    [[ "$-" == *a* ]] && ra2_previous_allexport=true
    set -a
    # shellcheck disable=SC1090
    source "$RA2_ENV_FILE"
    if [[ "$ra2_previous_allexport" == false ]]; then set +a; fi
  fi
  export RA2_LAB_DIR="${RA2_LAB_DIR:-${HOME:?HOME or RA2_LAB_DIR is required}/ra2-lab}"
  export LAYA_RELEASE_DIR="${LAYA_RELEASE_DIR:-$RA2_LAB_DIR/laya-ra2-v33-51d2f5a}"
  export LAYA_MODEL_PATH="${LAYA_MODEL_PATH:-$LAYA_RELEASE_DIR}"
  export LAYA_SOURCE_DIR="${LAYA_SOURCE_DIR:-$RA2_LAB_DIR/laya-970dc8c}"
  export LAYA_VENV_DIR="${LAYA_VENV_DIR:-$RA2_LAB_DIR/laya-v33-cpu-venv}"
  export LAYA_PYTHON="${LAYA_PYTHON:-$LAYA_VENV_DIR/bin/python}"
  export SUPALOSA_DIR="${SUPALOSA_DIR:-$RA2_LAB_DIR/supalosa-165b77a}"
  export RA2_GAME_DIR="${RA2_GAME_DIR:-$RA2_LAB_DIR/game-assets-headless-v1}"
  export RA2_SDK_DIR="${RA2_SDK_DIR:-$RA2_SCRIPT_DIR/../../engine/ra2-game-env-0.1.1/js/package}"
  export LAYA_DEVICE=cpu USE_TF=0 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
}

ra2_require_linux() {
  if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo 'Model inference and engine checks require x86_64 Linux. Help, reports and offline tests also work on other hosts.' >&2
    return 1
  fi
}

ra2_select_node() {
  if [[ -n "${RA2_NODE_BIN:-}" ]]; then
    if ! command -v "$RA2_NODE_BIN" >/dev/null 2>&1; then
      echo "Node executable was not found: $RA2_NODE_BIN" >&2
      return 1
    fi
  elif [[ -x "$RA2_LAB_DIR/node-v22.23.4-linux-x64/bin/node" && "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]]; then
    RA2_NODE_BIN="$RA2_LAB_DIR/node-v22.23.4-linux-x64/bin/node"
  elif command -v node >/dev/null 2>&1; then
    RA2_NODE_BIN="$(command -v node)"
  else
    echo 'Node.js 22 or newer is required. Install it or set RA2_NODE_BIN to its executable.' >&2
    return 1
  fi
  export RA2_NODE_BIN
}
