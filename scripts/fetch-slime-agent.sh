#!/bin/bash
# Fetch slime's agent package at the pinned commit, for the sandbox tests (tests/test_slime_sandbox.py).
#
# Slime does not pip-install on macOS, but slime/agent/sandbox.py imports only the standard library, so the tests
# put this checkout on PYTHONPATH instead. The commit is read from pyproject's runtime group, so it cannot drift from
# the pin. Only slime/__init__.py and slime/agent/ are checked out, not slime/utils/: slime/agent/harness/ imports
# torch through slime.utils.misc for SingletonMeta alone, so the tests that import the harness put a stand-in module
# holding only that class in its place.
#
#   scripts/fetch-slime-agent.sh DEST
#   PYTHONPATH=DEST pytest tests/test_slime_sandbox.py
set -euo pipefail
DEST="${1:?usage: scripts/fetch-slime-agent.sh DEST}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIN="$(sed -n 's|.*slime\.git@\([0-9a-f]\{40\}\).*|\1|p' "${REPO_ROOT}/pyproject.toml")"
[ "${#PIN}" -eq 40 ] || { echo "no 40-character slime commit in pyproject.toml" >&2; exit 1; }
mkdir -p "${DEST}"
git -C "${DEST}" init -q
git -C "${DEST}" fetch -q --depth 1 https://github.com/THUDM/slime.git "${PIN}"
FETCHED="$(git -C "${DEST}" rev-parse FETCH_HEAD)"
[ "${FETCHED}" = "${PIN}" ] || { echo "fetched ${FETCHED}, expected ${PIN}" >&2; exit 1; }
git -C "${DEST}" checkout -q FETCH_HEAD -- slime/__init__.py slime/agent
echo "slime/agent at ${PIN} in ${DEST}"
