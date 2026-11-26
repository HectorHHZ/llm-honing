#!/bin/bash
# The same deterministic game, drawn (for recordings and for looking at it): the devmode player with HONING_FAST=1.
export HONING_FAST=${HONING_FAST:-1}
export HONING_POLL=${HONING_POLL:-fast}
exec "$(dirname "$(readlink -f "$0")")/HollowKnightDemo.x86_64" "$@"
