# Celeste Classic: native C-port input demo

An existing Honing decision server can control
[ccleste](https://github.com/lemon32767/ccleste), a community C source port of
**Celeste Classic (2015)**. This is the earlier PICO-8 platformer, a different game
from commercial Celeste (2018). The pinned engine supplies movement, collisions,
jumping, directional dash, deaths, rooms and graphics. The bridge adds bounded
inputs, paused observations and tick capture. It performs no policy fallback,
training, harness evolution or gameplay scripting.

Move left/right, jump and dash to climb each room; spikes kill the player and
restart that room. Jumping against walls enables wall jumps. There is no grab
button in Classic. Compared with Honing's Unity Hollow Knight imitation and Soul
Knight clone demos, this exercises compact precision platforming in an existing
port of the earlier game, without their combat/action surfaces. RA2/Craftax/
Griddly have campaign/task/grader surfaces; this standalone demo does not.

## Resources and build

Linux x86_64, gcc, SDL2 >=2.0.18, SDL2_mixer, Xvfb and ffmpeg/ffprobe are needed.
Only independently authored bridge/build/client/test source is supplied here.
Clone the game separately into local working state:

```sh
git clone https://github.com/lemon32767/ccleste work/ccleste
git -C work/ccleste checkout 261d96f15af430b8111abc7a5250229246654f52
python integrations/celeste-classic/build.py \
  --source work/ccleste --output work/celeste-classic
```

The upstream repository contains the game's map, graphics and audio. It has no
LICENSE file and GitHub's license metadata is null; this integration grants no
rights to redistribute those materials. Keep them local and review their terms
for your own use. No paid PICO-8 runtime, commercial Celeste installation, REA,
Everest or .NET is needed for this C-port route.

The builder refuses the wrong revision, changed tracked source/resources and an
existing output. It copies into a new directory, adds small source hooks and
builds with upstream floating-point arithmetic. The input checkout is unchanged.
`build.json` records source/resource/bridge/binary hashes and the compiler command.
A caller-managed SDL prefix can be supplied with `--sdl-prefix`; Linux multiarch
headers and libraries are supported. If its shared libraries need extra runtime
search paths, pass the same paths to the runner's `--library-path`.

## Model run and video

Start a server exposing Honing's existing `GET /v1/health` and `POST /v1/decide`
routes; see the [decision-server setup](../ra2/README.md#using-a-decision-server).
The runner owns a fresh Xvfb display and player. It leaves an existing model
service running.

```sh
python integrations/celeste-classic/run.py \
  --game work/celeste-classic/ccleste-honing \
  --decision-url http://127.0.0.1:8809 \
  --output work/celeste-classic-runs/model --decisions 20 --frames 12 --capture
python integrations/celeste-classic/encode.py \
  --session work/celeste-classic-runs/model --output work/celeste-classic-demo.mp4
```

The [API](API.md) specifies units, visibility, held/edge inputs, resets, errors and
response acceptance. The default run has 240 game ticks, or 8 simulation seconds
at 30 ticks/second. Inputs are simulated as fast as rendering permits, with both
update and draw frozen between decisions. The video omits inference waits and
contains no audio; it is simulation playback, not real-time model latency.

The source indexes rooms 0..30, with title screen 31. The first room is the demo
start. There is no measured full-game completion time here; 8 seconds describes
the requested demonstration, not how long a person or model needs to finish the
game. The room index and summit flag are native state, not a campaign grader.

## Author-side verification

The local Linux run used gcc 11.3.0, SDL2 2.0.20, SDL2_mixer 2.0.4, Xvfb, and the
existing Laya CPU decision server. Twenty real model responses each matched its
request and caused exactly 12 original update calls and 12 original draw calls;
the original modulo-30 engine clock also advanced correctly. The model selected
`dash_up` every time. The first dash occurred; later held dash inputs did not
create new press edges. It remained in the first room with zero deaths. This
establishes the connection and action semantics, not model competence or room
completion.

The local video has 240 fully decoded frames at 512x512/30 fps, duration 8.000s,
SHA256 `62a47fe0cda4e660743a61e190505e32adf48a96f53680ce6b3c3a694f02588d`.
Game imagery, video, raw requests/responses and private paths remain local.
[Sanitized receipts](validation.json) record the evidence boundary. Independent
native re-execution, cross-build determinism, replay grading and whole-game
completion remain unverified.

Portable checks need no game or model:

```sh
pytest tests/test_celeste_classic.py
```

For a separate scripted interface check, including every input as the first
accepted action after reset, paused-state stability, malformed/stale rejection,
reset and invalid-command watchdog flooding:

```sh
python integrations/celeste-classic/verify_native.py \
  --game work/celeste-classic/ccleste-honing \
  --output work/celeste-classic-runs/native
```

This check is explicitly scripted validation. It is separate from the model run.
No training, EpisodeRunner/TaskGrader, campaign configuration or durable replay
format is introduced. The temporary command files are a local implementation
inside this standalone integration, following the existing HK demo pattern;
Honing core protocols and serving contracts are unchanged.
