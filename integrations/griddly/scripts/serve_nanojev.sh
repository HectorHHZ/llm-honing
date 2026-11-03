#!/usr/bin/env bash
set -euo pipefail
if [ -f /etc/profile.d/honing.sh ]; then
  source /etc/profile.d/honing.sh
fi
model_python=''
model_source=''
if [ -v NANOJEV_PYTHON ]; then model_python=$NANOJEV_PYTHON; fi
if [ -v NANOJEV_SOURCE_DIR ]; then model_source=$NANOJEV_SOURCE_DIR; fi
while [ "$#" -gt 0 ]; do
  case "$1" in
    --python|--source)
      if [ "$#" -lt 2 ]; then echo "Missing value for $1" >&2; exit 2; fi
      if [ "$1" = --python ]; then model_python=$2; else model_source=$2; fi
      shift 2
      ;;
    --help)
      echo 'Usage: serve_nanojev.sh [--python LINUX_PYTHON] [--source NANOJEV_CHECKOUT]'
      echo 'Defaults use NANOJEV_PYTHON and NANOJEV_SOURCE_DIR from the environment.'
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
cd "$(dirname "$0")/../../.."
if [ -z "$model_python" ] || [ ! -x "$model_python" ] || [ -z "$model_source" ] || [ ! -d "$model_source" ]; then
  echo 'Configure NANOJEV_PYTHON and NANOJEV_SOURCE_DIR, or supply --python and --source.' >&2
  echo 'See integrations/griddly/README.md for the separate Linux model environment.' >&2
  exit 1
fi
exec "$model_python" -m honing.backends.jev.server \
  --family nanojev --pins "$PWD/integrations/ra2/pins.json" \
  --nanojev-source "$model_source" --nanojev-inference reference \
  --gpu 0 --max-len 8192 --port 8803
