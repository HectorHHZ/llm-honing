# Hollow Knight imitation: Unity input demo

A caller-supplied Mono build of [Hollow-Knight-Imitation](https://github.com/DanielDFY/Hollow-Knight-Imitation)
can take inputs chosen by an existing Honing decision server. Unity runs the original movement,
combat, collisions and coroutines. The client sends observations and choices to `/v1/decide`,
then gives the selected input to a small file-based bridge inside an isolated copy of the game.

This is a standalone play-and-record demo. It has no campaign runner, reward function, training,
or deterministic reset/step contract. The game build, project assets, Unity, REA and model
weights are operator-supplied; only validation footage and traces are included here.

## Build the isolated demo

The tested target is macOS with Unity 2019.4.2f1, Mono, and the game revision in `pins.json`.
Apple Silicon also needs Rosetta for this editor's x86_64 Mono compiler and player.
Linux and IL2CPP have not been tested.

1. Clone the source and check out the pinned revision:

   ```sh
   git clone https://github.com/DanielDFY/Hollow-Knight-Imitation.git work/Hollow-Knight-Imitation
   git -C work/Hollow-Knight-Imitation checkout c443c6262ab85d007c8ed6f3abb42a8b077df007
   ```

2. Build its `Hollow Knight/` Unity project as a Mac Mono app, keeping the configured scenes.
   Supply that app and the editor path to the adapter:

   ```sh
   python integrations/hollow-knight/build.py \
     --source work/Hollow-Knight-Imitation \
     --game-app /path/to/HollowKnight.app \
     --unity /path/to/Unity.app \
     --output work/hollow-knight/Demo.app
   ```

The builder validates the source revision and input call sites, copies the app, patches copies
of two C# scripts, and compiles a new `Assembly-CSharp.dll` using the supplied Unity compiler.
It refuses an existing output or untracked C# inputs and never edits the input app or source
checkout. App symlinks are materialized inside the copy before compilation. The new app's
`honing-build.json` records the original and generated assembly hashes. REA is not needed to
compile or run the bridge, and the editor process does not need to run during compilation.

## Connect a decision model and record

Start any server exposing Honing's existing `GET /v1/health` and `POST /v1/decide` routes.
The [decision-server instructions](../ra2/README.md#using-a-decision-server) cover the supported
model families. Then, from the repository root:

```sh
python integrations/hollow-knight/run.py \
  --game work/hollow-knight/Demo.app \
  --decision-url http://127.0.0.1:8011 \
  --scene Level1 --decisions 50 --capture
```

The command prints its session directory under `work/hollow-knight/runs/`. Encode that session
with ffmpeg:

```sh
python integrations/hollow-knight/encode.py --session work/hollow-knight/runs/<session>
```

Use `--headless` instead of `--capture` for a state/control run without graphics. The client
refuses stale response IDs, incomplete or invalid probability distributions, and choices that
are not the model's argmax. A server error or stalled game stops the run and cleans up its
owned player process; there is no scripted policy fallback. `encode.py` requires a completed
capture and every numbered render frame before creating a video.

`--timeout` controls the model HTTP timeout (180 seconds by default); the bridge watchdog
receives the same value plus 30 seconds of headroom. Standard JSON responses are recorded
before validation; HTTP failures get error records in `decisions.jsonl`. Failed sessions
do not get a successful `result.json`.

## Observations and timing

The bridge reports player position, velocity, health and remaining jumps; nearby enemies;
left/right walls and local floor probes; and exit collider bounds. `frame`, `fixedSteps` and
`simulationTime` record render and physics progress. The 11 inputs include walking, jumping,
dashing and directional sword attacks. The game's own cooldowns and collision rules still
decide whether an input takes effect.

The default input window is 12 rendered frames at 30 fps. Unity game time pauses while the
model decides. Waiting frames are omitted from capture, so video duration is playback time,
not inference wall time. Resuming from a pause can produce a render frame without advancing
simulation time; this is why the demo is not a fixed-duration training step implementation.

Each session retains `model.json`, `observations.jsonl`, `decisions.jsonl`, `player.log`,
and `result.json` on success, plus `frames/` when capture is enabled. These files live in
ignored working state. The input adapter is inactive without `HONING_SESSION_DIR`.

## Local CPU reproduction

The Mac validation used the existing `LayaDecisionModel.predict` adapter with an upstream
Laya agent loaded on CPU. The normal CUDA loader and decision server are unchanged.
In a separate environment with Honing installed:

```sh
uv pip install torch==2.8.0 transformers==4.57.1 safetensors==0.6.2
git clone https://github.com/NandhaKishorM/laya.git work/laya
git -C work/laya checkout 970dc8c5f63d7b886a68409493f37d569424f933
uv pip install --no-deps -e work/laya
python -c 'from huggingface_hub import snapshot_download; snapshot_download("convaiinnovations/laya-typed-decisions", revision="1a793eb568e6718f15941d08f85432581df534e3", local_dir="work/laya-typed-decisions")'
```

Run the server in that environment, then run the game client in another terminal:

```python
import torch
import laya
from honing.backends.jev.laya import LayaDecisionModel
from honing.backends.jev.server import DecisionServer

torch.set_num_threads(4)
torch.manual_seed(1)
agent = laya.load("work/laya-typed-decisions", device="cpu")
model = LayaDecisionModel(agent, {"family": "laya", "device": "cpu"}, max_len=2048)
DecisionServer(("127.0.0.1", 8011), model).serve()
```

The pinned checkpoint has an out-of-range `choice:11+` temperature. Upstream Laya clamps it
to 0.5 and warns that these scores are uncalibrated. The displayed `p` is the returned
normalized action score, not a probability of completing the game.

## Recorded real-model run

The [20-second recording](validation/laya-cpu.mp4), [run metadata](validation/laya-cpu.json),
and [complete decision trace](validation/laya-cpu-trace.jsonl) come from the adapter in this
directory talking over HTTP to Honing's actual `DecisionServer` with Laya on CPU.
On an Apple M1 Pro, 50 decisions produced 600 frames at 960×540 and 30 fps. All 50
request/response IDs were matched to game observations, physics advanced at each input,
and the resulting video was decoded successfully. The original game assembly hash was unchanged.

The policy chose `jump_left` every time and remained near the left wall; it did not finish
the level. Health stayed at 5. Mean model latency in this run was about 851 ms. The video
shows the model-to-game connection, not a trained policy's proficiency or a win-rate estimate.
No weights or harness training was performed, and no training tracker run was needed.

## Static analysis and verification

REA 6.8.0 identified the original game assembly as Unity Mono / CIL, with 43 types, 158 fields,
218 methods and 546 call edges. It directly references Unity's physics, animation and input
modules: the existing build needs the Unity runtime, though running it needs no editor.
This static analysis does not provide a live environment or reconstruct the engine.
`pins.json` records the REA revision used. With a separately installed REA CLI:

```sh
rea inspect-managed-artifact /path/to/Assembly-CSharp.dll --json
rea inspect-managed-members /path/to/Assembly-CSharp.dll --json
```

The bridge was compiled from available game source, so this is not a blind reverse-engineering
reconstruction. Portable client, builder and capture checks need no game or model:

```sh
pytest tests/test_hollow_knight.py
```
