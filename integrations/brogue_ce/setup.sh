#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: bash integrations/brogue_ce/setup.sh NEW_ENGINE_DIRECTORY" >&2
    exit 2
fi

patch_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
engine_dir="$1"

if [ -e "$engine_dir" ]; then
    echo "Choose a new engine directory: $engine_dir" >&2
    exit 2
fi

git clone --depth 1 --branch v1.15.1 https://github.com/tmewett/BrogueCE.git "$engine_dir"
test "$(git -C "$engine_dir" rev-parse HEAD)" = 1ba4240b7a928ddf0ffb772717bf1d433cd63804
git -C "$engine_dir" apply "$patch_dir/webbrogue.patch"
make -C "$engine_dir" -j 4 GRAPHICS=NO TERMINAL=NO WEBBROGUE=YES RELEASE=YES
