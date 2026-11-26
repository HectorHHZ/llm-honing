#!/bin/bash
# The deterministic game without drawing (the fast mode): the devmode player with HONING_FAST=1, -batchmode -nographics.
# A drop-in "player" for code/hk-hitl (run_keys.py / run_agent.py: --game-opt player=<this file>); the Xvfb the
# adapter starts is simply not used. No pictures can be taken in this mode (use hk_det.sh for recordings).
export HONING_FAST=${HONING_FAST:-1}
export HONING_POLL=${HONING_POLL:-fast}
# without a window a "picture" is an empty 640x480 image and writing it costs 20 ms per frame (measured 2026-11-25,
# runs/a14_fast_capture): pictures are switched off here whatever the launcher asked for
export HONING_CAPTURE=0
exec "$(dirname "$(readlink -f "$0")")/HollowKnightDemo.x86_64" "$@" -batchmode -nographics
