# Experimental Soul Knight native interface

This adapter gives the author release of
[huyanjueluowei/SoulKnight](https://github.com/huyanjueluowei/SoulKnight)
local JSON control and native Unity camera readback. An existing Honing
decision server can choose movement, a button and an aim target through
`POST /v1/decide`; the driver records the response and matching native action.

The source game has ordinary and elite encounters followed by a Boss room.
This integration is an interface experiment. Task suites, a harness
`EpisodeRunner`, grading, rewards and training remain separate work. The
author's student game is the baseline; commercial Soul Knight fidelity and a
completed model-controlled Boss route have not been established.

## External resources

Only adapter source, patch source, schemas and tests are included here. Obtain
the author's [v1.0 release](https://github.com/huyanjueluowei/SoulKnight/releases/tag/v1.0)
yourself and extract its `Game` directory to a private location outside this
checkout. [pins.json](pins.json) records the source/release commits, release
archive identity and three original binary hashes. The archive SHA-256 is
`ced16dffde890cdd51041c027ccbe1ddb15f71e07ff91dab6f31b5c30b115312`.
The author binary was not independently rebuilt from source.

The Linux runtime needs Python 3.10+, Wine, Xvfb and Mono's C# compiler with
Mono.Cecil. Native validation used Wine 11.0, Mono 6.8 and Cecil 0.11; Unity
reports 2020.3.0f1c1. Supply installed tool paths to the CLI. This integration
does not install system packages, download game/model resources or start a
model server. Serving instructions for existing models are in
[the RA2 integration](../ra2/README.md#using-a-decision-server).

## Run privately

Set these variables to caller-owned paths outside the checkout. The patched
game and each run/output directory must be new paths. `MONO_CECIL` names the
installed `Mono.Cecil.dll`; `wine`, `mono`, `mcs` and `Xvfb` below use `PATH`.

```bash
BRIDGE="$PWD/integrations/soul-knight/soul_knight.py"

python "$BRIDGE" prepare \
  --game-dir "$SOUL_KNIGHT_GAME" --output "$SOUL_KNIGHT_PATCHED" \
  --cecil "$MONO_CECIL"
python "$BRIDGE" launch \
  --game-dir "$SOUL_KNIGHT_PATCHED" --run-dir "$SOUL_KNIGHT_RUN"
python "$BRIDGE" run \
  --ipc-dir "$SOUL_KNIGHT_RUN/ipc" --decision-server "$HONING_DECISION_SERVER" \
  --output "$SOUL_KNIGHT_TRACE" --reset --seed 7 --decisions 12 --frames 50
python "$BRIDGE" stop --launch-receipt "$SOUL_KNIGHT_RUN/launch.json"
```

The absolute script path works from another working directory. Preparation
checks the original files and compiles instrumentation into a separate game
copy. Launch creates a private Wine prefix and an unused local X display;
the game starts paused in its menu. `run --reset` loads the source gameplay
scene. Stop verifies PID birth/command identities and shuts down only that
run's Wine prefix, including its sidecars. The original game and other game
instances are never overwritten or automatically attached.

The [interface document](API.md) covers Python/JSONL controls, each state field,
timing, termination, the shared decision envelope and artifact formats.
The client and runner use only Python's standard library. The optional
user-local Mono mode accepts `--mono`, `--mcs`, `--mono-root` and
`--mono-config`; launch accepts `--wine`, `--wineserver` and `--xvfb`.

## Evidence and limits

Each trace stores the exact HTTP request/response, selected argmax choices,
native request/response IDs, executed action, frame advancement and before/after
snapshots. Failed execution retains the pending HTTP exchange. Native PNGs
come from the original Unity `Camera.Render` and a RenderTexture; the adapter
switches the original HUD canvas to camera rendering for capture. The game
rules remain in the author player. Windowed mode stalled in the tested Wine
setup, so this adapter uses batch mode and native offscreen frames.

The bundled controller's observation limits, nine movement options, five
button options, target candidates, argmax choice and frame batches are smoke
defaults. Inference waits while gameplay is paused. A rendered frame is not a
physics tick; global Unity time is not reset and deterministic replay is not
promised. A video encoded at 50 PNGs/s omits model waits and does not measure
real-time inference throughput.

No recordings, screenshots, game binaries/assets/source, model files or
private run logs are part of this integration. Keep those resources private
and follow their respective terms. This adapter has no public hosting or
upload command.

## Checks

The standard repository suite discovers the interface regressions:

```bash
pytest tests/test_soul_knight_interface.py
pre-commit run --all-files
pytest
```

These tests use explicitly labelled transport/lifecycle fixtures. Real native
and model acceptance requires the external resources above; use fresh
directories and retain the emitted preparation, launch, trace and cleanup
receipts privately. Test fixtures are not gameplay or model inference evidence.
