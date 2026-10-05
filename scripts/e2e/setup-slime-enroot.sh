#!/bin/bash
# Import Slime's image with enroot and create a container named "slime" on this node, for clusters without Docker.
#
# The tag is the one docker/version.txt names at the pinned Slime commit (41014d1f, pyproject's runtime group). The
# image's own Slime checkout is a few commits older, so slime_grpo_step.py runs a checkout at the pin
# (--slime-source) ahead of it. The image is about 30 GB: keep it on node-local disk and rerun on each node.
#
#   scripts/e2e/setup-slime-enroot.sh [tag]
set -euo pipefail
TAG="${1:-nightly-dev-20260810a-cu129}"
ROOT="${ENROOT_ROOT:-/tmp/enroot}"
export ENROOT_CACHE_PATH="${ROOT}/cache" ENROOT_DATA_PATH="${ROOT}/data" ENROOT_TEMP_PATH="${ROOT}"
mkdir -p "${ENROOT_CACHE_PATH}" "${ENROOT_DATA_PATH}"
IMAGE="${ROOT}/slime-${TAG}.sqsh"
[ -f "${IMAGE}" ] || enroot import -o "${IMAGE}" "docker://slimerl/slime:${TAG}"
enroot create -f -n slime "${IMAGE}"
echo "container slime from slimerl/slime:${TAG}"
