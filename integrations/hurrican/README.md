# Hurrican native decision demo

This standalone integration connects an existing Honing decision server to
[HurricanGame/Hurrican](https://github.com/HurricanGame/Hurrican), a native
Turrican-inspired run-and-gun game. It uses the existing game, graphics and
physics. Hurrican is a separate freeware game; this is not original Cuphead.

Hurrican combines scrolling platform movement, ranged shooting, enemies,
powerups and bosses. The external game's `Hurrican/readme.txt` describes nine
levels plus a tutorial. This integration starts the ordinary first jungle
level at medium difficulty. A 12-decision demonstration with 40 updates per
input covers eight simulated seconds; the complete game's play time has not
been measured. It is an input demo, without a campaign task, EpisodeRunner,
grader, training or a completion result.

## External game and build

The source is pinned to `0d91d1f2d40f38d21e7f41afc01369ee55649fb6`.
The required `glm` and `sdl2-cmake-scripts` submodules are pinned by that commit.
Keep the game checkout, resources, binary and output outside this repository.

```bash
git clone --recurse-submodules https://github.com/HurricanGame/Hurrican /local/hurrican
git -C /local/hurrican checkout 0d91d1f2d40f38d21e7f41afc01369ee55649fb6
git -C /local/hurrican submodule update --init --recursive
python integrations/hurrican/build.py --source /local/hurrican --build-dir /local/hurrican-build
```

Use a C++17 compiler, CMake, SDL2, SDL2_image, SDL2_mixer and libepoxy development
packages. On Debian/Ubuntu these are `libsdl2-dev libsdl2-image-dev
libsdl2-mixer-dev libepoxy-dev`. The build helper refuses another game revision,
applies `native.patch` once and copies the two control hook source files.
`--jobs` defaults to four and accepts 1–8; repeat `--cmake-arg=-DNAME=value`
for a local dependency prefix. It changes only the external checkout and build
directory. The hooks are inactive unless `HONING_HURRICAN_SESSION` is set.

Validated on Linux x86_64, GCC 11.3, CMake 3.22.1, SDL 2.0.20, SDL2_image 2.0.5,
SDL2_mixer 2.0.4, libepoxy 1.5.10 and software Mesa OpenGL under Xvfb. Other
platforms have not been run; these hooks use POSIX stdin/poll/process groups.

## Run with an existing model

Start a normal Honing decision server separately. This client does not start,
reconfigure or terminate it. `--data` names the external `Hurrican/` directory
containing both `data/` and `lang/`.

```bash
python integrations/hurrican/run.py \
  --game /local/hurrican-build/hurrican \
  --data /local/hurrican/Hurrican \
  --decision-url http://127.0.0.1:8809 \
  --output /local/hurrican-demo --decisions 12 --frames 40 --capture
ffmpeg -framerate 30 -i /local/hurrican-demo/frame%06d.ppm \
  -c:v libx264 -pix_fmt yuv420p /local/hurrican-demo.mp4
```

Use an available display or a caller-owned Xvfb. The launcher runs one game
process, manages only its own process group and leaves display/server lifecycle
to the caller. `SDL_AUDIODRIVER=dummy` supports a machine without audio hardware.
The output directory must be new. It contains private raw model requests and
responses, action/state trace, native log, binary identity, cleanup receipts and
optional lossless PPM frames. Do not commit it.

The native simulation pauses between inputs. Captured video has one frame per
two 60 Hz updates and omits inference waits. It is not evidence of real-time
inference. The [API](API.md) defines timing, visibility, errors and lifecycle.

## Author-side validation

The local real-model run used the existing CPU Laya service with
`convaiinnovations/laya-typed-decisions` revision `1a793eb...`, 16 maximum
branches, a 2048 context and 512 head budget. All 12 accepted distributions chose
`shoot`; original native primary projectiles remained present at every boundary. The player stayed at the opening position with 100 energy: no level
progress or boss encounter was demonstrated. The eight-second 640×480 video
contains 240 decoded frames. The [sanitized receipt](validation.json) records
identities and measured results; assets, video and raw traces remain local.

A separate native probe exercised movement/jump and actual enemies. Final
isolated native probes confirmed malformed/stale commands and a rejected model
response did not advance state, fresh-process restart restored the opening
player fields, and invalid/partial command traffic could not evade the idle
watchdog. These are control checks, not model proficiency or deterministic
replay. Portable tests use synthetic responses and require no game resources.

Compared with the existing Soul Knight room-combat clone, this game uses a
scrolling platform level and ranged movement/shoot combinations. Compared with
the Hollow Knight imitation's melee inputs, it supplies a native C++ run-and-gun
world and bounded 60 Hz updates. RA2 and Tank have different vehicle/strategy
controls; Craftax, Griddly Spiders and the Isaac-inspired simulation remain
separate environments. This integration adds no shared protocol or dependency.

## Licensing and provenance

Hurrican's root `README.txt` grants noncommercial use and modification of its
source and assets. The same file describes SDL-port additions in existing files
under the original terms and original port files as MIT; the bundled
`Hurrican/readme.txt` also contains the original freeware distribution terms.
These mixed/custom terms are not an OSI open-source license. Obtain and retain
the upstream notices; evaluate their terms for your use. This PR distributes
only Honing adapter source, the narrow build patch, tests and documentation.
It includes no upstream game assets, complete game source, binary or recording.
