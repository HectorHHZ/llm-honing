# Original VVVVVV: experimental native decision demo

This integration compiles the original [Terry Cavanagh VVVVVV source](https://github.com/TerryCavanagh/VVVVVV)
at `40e1ec3bbf2a990a5c600564fe312c2121178a50`, adds a paused input bridge,
and calls an existing Honing `/v1/decide` server. It retains the original physics,
collision, rendering, levels and checkpoint deaths. The resulting executable is
an experimental Honing modification; VVVVVV is copyright Terry Cavanagh.

VVVVVV replaces jumping with reversing gravity while touching the floor or
ceiling. Move left/right and time gravity flips to avoid spikes and explore.
The demo starts at the original Space Station 1 time trial's first checkpoint,
in “Welcome Aboard,” after its native countdown. This is a small interface demo,
with no EpisodeRunner, TaskGrader, training or completion claim.

## Resources and build

The original source uses a [custom noncommercial license](VVVVVV-LICENSE.md).
The pinned code requires **SDL3** despite its older desktop README mentioning
SDL2. Tested locally with GNU/Linux x86_64, GCC 11.3, CMake 3.22.1 and SDL 3.4.0.
The bridge is currently Linux-tested only.

Acquire desktop `data.zip` from an owned game copy or the official
[Make and Play source-assets link](https://thelettervsixtim.es/makeandplay/).
The upstream README expressly permits that download for personal source builds.
The free Make and Play executable itself omits original campaign levels; the
source checkout supplies those levels. No game assets, binaries or music are
distributed here. The local downloaded desktop data SHA-256 was
`c767809594f6472da9f56136e76657e38640d584164a46112250ac6293ecc0ea`.

From the Honing repository root, using a new caller-owned source checkout:

```bash
git clone --recursive https://github.com/TerryCavanagh/VVVVVV.git work/vvvvvv/source
git -C work/vvvvvv/source checkout 40e1ec3bbf2a990a5c600564fe312c2121178a50
git -C work/vvvvvv/source submodule update --init
python integrations/vvvvvv/build.py --source work/vvvvvv/source --build work/vvvvvv/build
```

`build.py` checks the source commit, copies the two bridge files, applies guarded
source edits and invokes CMake with four compile workers. Use
`--cmake-prefix /path/to/SDL3/prefix` when SDL3 is outside system directories.
Set the platform library search path if that SDL3 installation needs it.
Original source modifications stay in this isolated checkout. Keep the
source's original license with compiled copies.

## Native/model run

With a visible display (or a caller-owned Xvfb) and an existing ready decision
server, run:

```bash
python integrations/vvvvvv/run.py \
  --game work/vvvvvv/build/VVVVVV --assets /path/to/data.zip \
  --decision-url http://127.0.0.1:8809 \
  --output work/vvvvvv/runs/demo --decisions 20 --ticks 15 --capture
python integrations/vvvvvv/verify_native.py \
  --game work/vvvvvv/build/VVVVVV --assets /path/to/data.zip \
  --output work/vvvvvv/runs/native-check
```

Output directories must be new. Native saves/settings stay within the run's
`user/` directory. The launcher starts and cleans only its own game process;
it never starts, changes or stops the shared decision server or display.
Model waits have a 180-second HTTP timeout; an orphaned paused game exits after
10 minutes without a valid command. Full requests, raw responses, accepted
actions, before/after state and rejection diagnostics remain local.

`verify_native.py` exercises every offered input, native stale/invalid commands,
restart, stop and a deliberately invalid response against the real paused game.
Its response fixture is separate from the actual model run.

The local author run used the actual Laya typed-decisions weights, 20 decisions
and 15 native ticks per decision. All 20 selected `flip`; the player alternated
between floor and ceiling in one room, with no deaths or room exit. Capturing
300 original 320×240 frames produced **10.2 simulation seconds**. Inference
waiting and the startup countdown are omitted. To reproduce the 640×480 video
timeline from that run:

```bash
ffmpeg -framerate 500/17 -i work/vvvvvv/runs/demo/frames/%06d.bmp \
  -vf scale=640:480:flags=neighbor -c:v libx264 -crf 18 -pix_fmt yuv420p demo.mp4
```

The fixed frame rate above applies to that normal-speed run (34 ms/tick).
For slowdown modes, use actual per-tick durations; do not infer timing from
render count alone. The video and full private trace are intentionally local.
[Sanitized evidence](validation.json) records reproducible scope and hashes.

## Task length and comparison

The measured segment is 10.2 simulated seconds, not a room-completion time.
Original source has six time-trial start modes (Space Station 1, Laboratory,
Tower, Space Station 2, Warp Zone, Final Level), a 20×20 overworld map array,
separate final/tower content and 20 collectible trinkets. These are structural
facts from [`Game::starttrial`](https://github.com/TerryCavanagh/VVVVVV/blob/40e1ec3bbf2a990a5c600564fe312c2121178a50/desktop_version/src/Game.cpp),
[`scriptclass::startgamemode`](https://github.com/TerryCavanagh/VVVVVV/blob/40e1ec3bbf2a990a5c600564fe312c2121178a50/desktop_version/src/Script.cpp)
and [`Map.h`](https://github.com/TerryCavanagh/VVVVVV/blob/40e1ec3bbf2a990a5c600564fe312c2121178a50/desktop_version/src/Map.h), not a
measured full-game duration. For a human planning estimate, the
[GOG game page](https://www.gog.com/en/game/vvvvvv) displays HowLongToBeat's
roughly 2-hour main story and 3-hour main plus sides figures; this is a
third-party estimate and says nothing about the decision model's completion time.

Compared with Honing's existing Hollow Knight demonstration, this is original
C++/SDL VVVVVV with gravity reversal and six actions, rather than Unity sword,
jump and dash inputs. Compared with Soul Knight it is side-view platforming
with no independent aim, attack or weapon interface. It exposes one bounded
local tile neighborhood plus current-room entities and advances native fixed
ticks while inference pauses simulation. It does not yet expose a campaign
runner/grader or arbitrary deterministic savestates.

See [API.md](API.md) for every field, timing, reset and rejection semantics.
