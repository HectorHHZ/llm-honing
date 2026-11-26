"""Run-time locations of the VVVVVV game runtime, resolved without absolute local paths in the code (2026-11-26).

The collector, the probe/replay adapter, the frame-capture replay and the checks import these names. Nothing here is
written into any record.

- GAME:    the pinned build at `work/vvvvvv/build/VVVVVV` of the repository (integrations/vvvvvv/README.md).
- ASSETS:  `$VVVVVV_ASSETS`, else `work/vvvvvv/data.zip` of the repository (the caller's official desktop data;
           never published).
- SDL_LIB: `$VVVVVV_SDL_LIB`, else the `lib` directory of the conda environment `cse153` of the running Python's conda
           installation (where this server's SDL3 3.4.0 lives), else empty (then the system loader paths apply).
- COMMON:  `integrations/_shared/exp0` (the shared Exp 0 tools).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[3]
COMMON = HERE.parents[2] / "_shared" / "exp0"
GAME = WORKSPACE / "work" / "vvvvvv" / "build" / "VVVVVV"
ASSETS = Path(os.environ.get("VVVVVV_ASSETS") or WORKSPACE / "work" / "vvvvvv" / "data.zip")


def _sdl_lib() -> str:
    explicit = os.environ.get("VVVVVV_SDL_LIB")
    if explicit:
        return explicit
    candidate = Path(sys.base_prefix).resolve() / "envs" / "cse153" / "lib"
    return str(candidate) if (candidate / "libSDL3.so.0").exists() else ""


SDL_LIB = _sdl_lib()
