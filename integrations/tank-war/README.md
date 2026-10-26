# Tank War

Decision models need iteration.

https://github.com/user-attachments/assets/9dca9289-bb7d-4b51-bd0b-8b896fabd42d

Two tanks fire shells that bounce off the walls. One hit ends a round. A tank's
own shells can kill it. Model tanks drive forward or backward and rotate the
hull to aim. Human players can also select independent mouse aiming.

The video shows four controller versions with the same NanoJev weights. V1
spins, V2 fires but can hit itself, V3 fights, and V4 adds dodge assistance.
These are selected demonstration clips. They do not establish a win rate or
prove that each version passed a release gate. The changes are in the context,
action selection and execution checks; they are not new abilities learned by
updating the model's weights.

[Play against a model](https://play.honinglab.com/play/) in your browser. The
hosted game uses its configured opponent and queue. It does not accept arbitrary
model endpoints. We plan to open **Tank War Arena** for bring-your-own-model
matches. Public model submission is not open yet. You can connect your own
models to this local integration now.

## Quick start

Node.js 22 or newer is enough for local keyboard play. No build, Python, GPU or
external game assets are required. From the repository root:

```bash
cd integrations/tank-war
npm start
```

Open <http://127.0.0.1:4177/>. Before a human match, select Blue's controls:

| Control card    | Move                                              | Aim                                         | Fire       |
| --------------- | ------------------------------------------------- | ------------------------------------------- | ---------- |
| Mouse (default) | Follow the cursor; stop near it                   | Turn the hull toward the cursor             | Left click |
| Keyboard        | W/S forward/reverse                               | A/D turn the hull                           | Space      |
| Mouse + WASD    | W/S up/down; A/D left/right in screen coordinates | Point the independent barrel with the mouse | Left click |

In You vs model, arrow keys also work in Keyboard and Mouse + WASD modes.
In Local two-player, arrows and Enter remain reserved for Red. Red keeps hull
steering. P pauses. You can change Blue's control card while paused. Changing
controls clears held inputs. Backgrounding the tab pauses the game.

On a phone, use the fixed joystick and the Fire button with separate fingers.
The joystick controls hull steering. In landscape, starting a human match opens
a console layout with the arena between the large lower-left joystick and Fire.
The opponent's decision bars fit below Fire when space permits. The page does
not scroll during landscape play; Restart returns to the setup screen. Local
Red still needs a keyboard, so You vs model is the intended single-phone mode.

Human keyboard turns run at 5 rad/s; mouse and joystick turns run at up to
8.4 rad/s. Model turns remain at 2.5 rad/s. Mouse + WASD aims the barrel directly.
All modes use the same movement speed, wall/tank collision, ammunition, cooldown,
and ricochet rules. Diagonal movement does not give a speed bonus.

To enable **You vs model** and **Model vs model**, start a Honing decision
server, then give the game its base URL:

```bash
HONING_URL=http://127.0.0.1:8080 MODEL_LABEL=NanoJev npm start
```

The browser needs no GPU. The model server supplies inference. See
[RA2's model setup](../ra2/README.md#using-a-decision-server) for supported model
families and their server commands. RA2 itself needs a privately supplied engine
SDK and retail game assets; Tank War includes its game engine.

## Game and controllers

Model tanks retain hull steering with a fixed barrel. All matches use bouncing
shells, five active shells per tank, lethal self-hits, and first-to-five scoring.
Human control cards change movement and aiming only; model decision protocols
and model-vs-model evaluation remain unchanged. Exports record the selected
human control scheme. Compare human results only with the same control scheme.

| Decision version | Context                                     | Behavior                                                                                                                                                      |
| ---------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Planned Steering | Full tactical (`planned-aim-v1`)            | Choose one of nine drive/turn plans, then decide whether to fire after that motion. Geometry limits motion selection and checks the shot again before firing. |
| Free Controls    | Full tactical (`state-v7-avoidance-levels`) | Choose drive, turn and weapon independently, with aim hints and motion/shot forecasts.                                                                        |
| Free Controls    | Arena only (`arena-v1`)                     | Choose the same controls from positions, headings, walls, shells, timing and assistance state, without aim hints or trajectory forecasts.                     |

Each seat has separate model, context, cadence and assistance settings. Model,
context and cadence changes restart the match and invalidate pending replies.
Assistance changes apply immediately. Settings persist in the browser. The
model panels show probabilities, context and latency. Match exports record
configuration, requests, native responses, applied controls, assistance, shots
and kills. These audit records are not deterministic replay files.

### Context and action execution

Every context includes all active shells from both sides: position, velocity,
owner, bounce count and remaining lifetime. Ownership does not make a shell
safe. State includes both tanks, walls, ammunition, cooldowns and time.

Full tactical context adds verified firing directions, required turns, response
delay, navigation waypoints and forecasts for all nine drive/turn combinations.
Planned Steering retains the native probabilities for all nine plans. During
normal play it selects the most probable admissible plan. A second model request
selects fire or hold for that plan. The tank turns at its physical angular speed.
A fresh check cancels a shot if its plan changes or becomes unsafe.

Drive/turn bars for Planned Steering are marginals of the joint distribution.
They show model output. Geometry restrictions and execution changes are recorded
separately. Free Controls has no joint-plan filter and can cause self-hits.

### Dodge assistance

Dodge assistance is local geometry code. It does not make another inference
request. It predicts shell impacts and can briefly replace the model's motion.
The model's command resumes after the burst if it is still valid. Keyboard
players receive no assistance.

| Profile               | Predicted-hit trigger | Maximum burst | Cooldown |
| --------------------- | --------------------- | ------------- | -------- |
| Reactive (`light`)    | 480 ms                | 400 ms        | 280 ms   |
| Balanced (`standard`) | 620 ms                | 500 ms        | 200 ms   |
| Protective (`strong`) | 780 ms                | 650 ms        | 120 ms   |

Reactive checks every 60 ms. No profile changes shell speed, collision size or
damage. Turning dodge off does not remove observations or Planned Steering's
shot checks. Free Controls also has an optional shot guard.

The dodge fixture is separate from match evaluation:

```bash
node scripts/dodge-benchmark.mjs
```

It tests 32 incoming-shell situations: four gaps and eight hull headings, with
stationary model commands and no walls. Its survival counts are not match win rates.

### Maps, collisions and timing

Seeded, rotationally symmetric Loop Mazes use a 10 × 7 grid with 28–37 interior
wall segments. Each cell has at least two exits. All cells are reachable and no
single intermediate cell cuts the route between spawns. New arena changes the
seed; Restart retains it. Map revision: `loop-mazes-v3`.

Shells expire after 4.5 seconds and release an ammo slot. Fire cooldown is 300 ms.
The tank silhouette and barrel fit inside a 22 px collision disk. Tanks and
shells move in spatial steps of at most 2 px, so long frames cannot skip thin
walls or live tanks. Forecasts use the same dimensions as live physics.

The 150 ms owner grace applies only to an unbounced shell leaving its muzzle.
A bounced shell can immediately kill its owner.

Auto cadence uses each seat's last 40 successful requests. After four samples,
the scheduling floor follows the 20th percentile of round-trip time; forecasts
use the 75th percentile. Busy responses add a decaying backoff. Only one request
per seat is in flight. A 50 ms setting is a scheduling floor, not a guaranteed
inference time. Replies older than 1.6 seconds are discarded. Connection errors
pause the match. Current motion continues during inference until it expires.

## Connect a model

The game bridge calls `GET /v1/health` and `POST /v1/decide` on a configured
Honing server. A backend must support typed choice questions and return a native
probability for every option. Supported families include NanoJev, Laya, Kev,
StartLux-Decision, Nimble-v4-max and DiffusionGemma through
[djev](../djev/README.md). The game server does not load model weights.

A simplified independent-control request to `/v1/decide` is:

```json
{
    "id": "round-1:controls",
    "parts": {
        "essential": ["Full tank, wall and shell state..."],
        "details": [],
        "memory": [],
        "logs": []
    },
    "questions": {
        "drive": {
            "type": "choice",
            "instructions": "Choose drive.",
            "criteria": {
                "forward": "Drive forward",
                "reverse": "Reverse",
                "stop": "Stop"
            }
        },
        "turn": {
            "type": "choice",
            "instructions": "Choose hull rotation.",
            "criteria": {
                "left": "Turn left",
                "right": "Turn right",
                "none": "Do not turn"
            }
        },
        "weapon": {
            "type": "choice",
            "instructions": "Choose weapon.",
            "criteria": { "fire": "Fire", "hold": "Hold" }
        }
    }
}
```

The live request contains the complete observations and option forecasts.
Planned Steering sends a motion question, then a weapon question conditioned on
the selected motion. When the server's health reports `maxBranches` of at least
the number of admissible motions, as Honing's decision server does, both go in
one request: the weapon question of each admissible motion is a branch of the
motion request, the server answers every branch in the same model call, and the
bridge keeps the selected motion's. The recorded calls, probabilities and
controls are those of the two-request decision; `structured_requests` counts the
requests sent. The bridge rejects truncated context, missing candidates,
invalid probabilities and mismatched identities. It does not invent model
probabilities or substitute a scripted player after an error.

Set `HONING_TOKEN` for an upstream proxy that needs a Bearer header. Tokens and
upstream URLs stay on the server. Browser requests select registered IDs only.
Two seats have separate requests and context, but may share one model instance
and GPU. Separate seats do not guarantee concurrent inference.

To use an environment file:

```bash
cp .env.example .env
# Edit .env, then:
node --env-file=.env server.mjs
```

`npm start` reads the process environment, not `.env` automatically. `HOST`
defaults to `127.0.0.1` and `PORT` to `4177`. Set `PUBLIC_ORIGIN` to the exact
external origin when using an HTTPS reverse proxy. This local integration does
not include the hosted service's queue or admission controls. Shared deployments
need their own access controls and rate limits.

Tank-specific `/health` and `/decision` services can use `TANK_MODEL_URL` and
`TANK_MODEL_LABEL`. `NANOJEV_URL` adds another tank-specific endpoint. Supported
game protocols are `plans-v1` and `controls-v3`, both for hull steering. A service
can support only one; selecting the other blocks model play. Replies must echo
request, round, model and seat identity, preserve probabilities, and report
`state_truncated: false`.

### Multiple models and checkpoints

Each registration points to an already running model endpoint:

```bash
MODELS_JSON='[
  {"id":"gemma","label":"DiffusionGemma","version":"current","url":"http://127.0.0.1:8081"},
  {"id":"nano","label":"NanoJev","version":"checkpoint-1","url":"http://127.0.0.1:8082"}
]' npm start
```

`transport` defaults to `honing`; use `tank` for tank-specific services. An
optional `token` stays server-side. Registration labels do not load or verify
checkpoints. The UI reports service identity separately from context identity.
Each seat validates replies against its own selected endpoint.

### Serving pinned models

`scripts/serve-honing.py` is an optional launcher for a prepared Honing checkout
with pinned packages in `vendor/laya`, `vendor/kev` and model records in `models/`.
It checks revisions against `integrations/ra2/pins.json`. It does not download
packages. Use a separate environment with the required GPU libraries.

```bash
python scripts/serve-honing.py --root /path/to/prepared/honing --model english --gpu 3 --memory-gib 4 --port 28780 --allow-shared-gpu
```

Other launcher choices are `typed-decisions`, `kev-0.8b` and `kev-4b`. Its memory
cap is not an exclusive GPU reservation. It checks free memory and Honing's GPU
lock and does not evict other processes.

Laya permits 48 tokens per option pointer. The bridge keeps full forecasts in
essential state and uses short option references. Kev rounds probabilities to
four decimal places; validation permits that rounding error without changing
the returned distribution.

## Train and evolve with Honing

Tank War is a Honing task environment in `honing/backends/tank`. One task is one
seeded round against the planner opponent, which uses geometry without a model.
The campaign evaluates candidates on validation tasks. Training uses the
default selector: more paired wins than losses against the incumbent. GEPA
supplies its own selector: a strictly higher validation mean than the served
candidate's stored mean. Ties do not produce a release.

From `integrations/tank-war`, run a model round:

```bash
node scripts/match.mjs --seed=2001 --seat=blue --decision-server=http://127.0.0.1:8080 --out=runs/r1
```

The runner advances in lockstep: game time stops during inference. This isolates
action quality from server latency. It does not measure real-time browser
performance. The policy acts every 200 ms of game time and the opponent every
300 ms by default. These intervals belong to the task, not the evolved harness.

The output includes `manifest.json` (scenario, model and source/harness hashes),
`decisions.jsonl` (exact context, questions, probabilities and choices),
`events.jsonl` and `result.json`. Use a new output directory for each run.

| Harness file           | Kind         | Purpose                                                             |
| ---------------------- | ------------ | ------------------------------------------------------------------- |
| `tank/controller.json` | settings     | Controller, context, assistance, shot guard and planner parameters. |
| `tank/questions.json`  | instructions | Question instructions and weapon option descriptions.               |
| Other files            | either       | Operating notes supplied before state.                              |

Defaults are in `config/`. Unknown settings, unknown planner parameters and
missing question keys fail before play. GEPA rewrites instructions and notes;
it does not rewrite controller settings. The persisted agent ID remains
`tank_classic` for compatibility with existing Honing harness records.

`--teacher` uses `web/game/teacher.js` and records labels for SFT.
`--sample-seed=N` samples from the full native model probabilities for GRPO.
Safety can project a sampled motion to an admissible motion before execution.
The training label remains the sampled policy choice. The log keeps the applied
action and constraints separately. The weapon request uses the applied motion's
forecast. Each sampled decision receives the round's reward.

From the **repository root**, with Honing and its GEPA extra installed:

```bash
python scripts/e2e/tank_campaign.py --config configs/e2e/tank_dry_run.json
```

This checks a GEPA move, SFT and GRPO with a CPU stand-in model and trainers. It
does not demonstrate improved real model weights. See the
[campaign guide](../../scripts/e2e/README.md#the-tank-campaign). The task suite
contains 16 training, 8 validation and 8 held-out seeds, each played from both
seats. From this integration directory, regenerate it with:

```bash
python -m honing.backends.tank.scenarios tasks/suite.jsonl
```

## Verify

From `integrations/tank-war`:

```bash
npm ci
npm test
npm run format:check
node scripts/smoke-model.mjs --seconds 25
node scripts/smoke-model.mjs --seconds 25 --controller raw
```

Smoke checks require a configured model and a running game server. Pause model
matches in the browser before using the same seats. The checks measure
connectivity, decisions, shots, hits and latency; they are not quality benchmarks.

From the repository root, with Honing installed in the active Python environment:

```bash
node scripts/e2e/tank-war.mjs
```

This runs the real Python DecisionServer and the Node bridge with a CPU fixture
through both decision interfaces and the same game engine. Set `HONING_URL` to
use an existing real model server instead.

## Files and migration

`web/game/` contains the sole engine, context builders, planner and UI adapter.
`web/shared/` contains the UI, renderer, model client and collision primitives.
`model-bridge.mjs` converts game requests to Honing questions. `server.mjs` serves
the game at `/game/` (the root redirects there) and its API at `/api/tank-war/`.
`scripts/match.mjs` runs headless tasks. This directory is a standalone demo
package; Python training uses the main Honing package.

Existing checkouts must update integration paths to `integrations/tank-war`,
use the routes above and use the environment names in `.env.example`. Older
browser settings are not migrated. Existing custom question files must remove
the obsolete `movement` key. Regenerate sampled training rounds from older
code: their logged probabilities can differ from their sampling distribution. The separate screen-direction movement and
independent-turret implementation is no longer included.
