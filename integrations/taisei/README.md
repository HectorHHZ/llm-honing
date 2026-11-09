# Taisei native decision demo

This integration runs the genuine [Taisei Project](https://github.com/taisei-project/taisei/tree/5bfc9b842fe6d9bb8d1610f76f442ba07c72c4cf)
source and original project resources. Taisei is an original Touhou-world
fangame with its own story, art, music and code, as its
[README](https://github.com/taisei-project/taisei/blob/5bfc9b842fe6d9bb8d1610f76f442ba07c72c4cf/README.rst)
explains. It is not an original commercial Touhou game.

The player flies through a vertical bullet-hell shooter, shoots enemies and
dodges dense bullet patterns. Focus reduces movement speed and exposes the
hitbox. This adapter offers nine movement choices while always holding shot
and focus; it excludes bombs. A real Honing decision service supplies every
movement choice. There is no heuristic fallback or training.

Scope is a short Stage 1 Easy segment with Youmu A. The full native game has six
story stages, an extra stage and generated spell-practice entries in
[`src/stageinfo.c`](https://github.com/taisei-project/taisei/blob/5bfc9b842fe6d9bb8d1610f76f442ba07c72c4cf/src/stageinfo.c).
The demo duration is measured in `validation/author-smoke.json`; full-game and
full-stage completion times have not been measured. The segment's end is a
controller tick budget, not stage completion. No completion, skill, independent
native rerun or deterministic replay result is claimed.

Compared with the existing Soul Knight arena and Hollow Knight imitation
platformer demonstrations, this exercises native 60 Hz flight and bullet
motion with a partial on-screen object sensor. It has no grounded jumping,
platform navigation or campaign grading. The API records which hazards are
omitted, and inference pauses the native loop.

## Install and run

Game assets, binaries, weights, recordings and full traces are external local
artifacts. This directory contains only source, build instructions, tests and
a sanitized receipt. Read upstream
[BUILD.rst](https://github.com/taisei-project/taisei/blob/5bfc9b842fe6d9bb8d1610f76f442ba07c72c4cf/doc/BUILD.rst)
and [COPYING.txt](https://github.com/taisei-project/taisei/blob/5bfc9b842fe6d9bb8d1610f76f442ba07c72c4cf/COPYING.txt)
for dependency and resource licensing details (code MIT, soundtrack and
commissioned portraits CC-BY 4.0). Keep the entire recursive checkout local.

Author platform: Ubuntu 22.04 x86-64, GCC 11.3, Meson 1.12.1, SDL 3.4.12 via the
pinned release's Meson wrap, Xvfb and Mesa llvmpipe. No root installation is
required if development packages are available in a user prefix. On a normal
Ubuntu installation, install the X11 development headers, freetype, zlib,
OpenGL, C/C++ compiler, CMake, Ninja, pkg-config, gettext and Python Meson
requirements; Meson fetches the remaining pinned wraps. The source build
requires `backports.zstd` on Python <3.14. `build.py` includes four narrow
zero-initialized VLA replacements for GCC 11, preserving upstream zeroing.

```bash
python -m pip install 'meson==1.12.1' backports.zstd
git clone --recursive --branch v1.4.6 https://github.com/taisei-project/taisei.git work/taisei/source
python integrations/taisei/build.py --source work/taisei/source --build work/taisei/build
python integrations/taisei/run.py \
  --game work/taisei/build/src/taisei --resources work/taisei/source/resources \
  --decision-url http://127.0.0.1:8809 --display :191 \
  --output work/taisei/episode --decisions 3 --frames 120 --capture
python integrations/taisei/encode.py --session work/taisei/episode --output work/taisei/demo.mp4
```

Use an existing service implementing Honing `/v1/health` and `/v1/decide`,
including probability, retained-context and token-budget audits. A generic chat
endpoint is insufficient. The author smoke used existing local
`convaiinnovations/laya-typed-decisions` CPU weights; the precise revision and
weight hash are in the receipt. Another machine must supply its own service and
local weights. No service credentials or weights are included.

To exercise synthetic faults against the native game and run portable tests:

```bash
python integrations/taisei/verify_native.py \
  --game work/taisei/build/src/taisei --resources work/taisei/source/resources \
  --display :191 --output work/taisei/native-faults
pytest tests/test_taisei.py
```

See [API.md](API.md) for state visibility, exact JSON/action semantics, timeout,
reset limits, rejection behavior, timing and trace contents. All runtime output
belongs outside tracked integration source. This is an experimental standalone
native interface, not a registered `EpisodeRunner` or `TaskGrader`.
