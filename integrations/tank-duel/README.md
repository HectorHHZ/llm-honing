# RICOCHET

Two tanks. One last shot. Two browser games in one English sketchbook UI,
with local keyboard play and an optional Honing model connection.

| Edition      | Movement                                    | Aiming and firing                                                                                                     |
| ------------ | ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **Classic**  | Forward/reverse; left/right rotate the hull | Barrel follows the hull. Planned steering chooses a joint motion, then asks the model whether to fire after turning.  |
| **Auto Aim** | Up/down/left/right in screen coordinates    | Independent turret turns toward a verified direct or ricochet shot. The model chooses movement and firing separately. |

Both have bouncing shells, five active shells per tank, one-hit rounds, a
first-to-five match, seeded arenas, model probabilities, full context inspection,
and downloadable match records. **Your own shells are lethal.**

Screenshots from connected model matches; model labels come from the configured service.

| Classic                      | Auto Aim                       |
| ---------------------------- | ------------------------------ |
| ![Classic](docs/classic.png) | ![Auto Aim](docs/auto-aim.png) |

## Play now

Requires Node.js 22 or newer. No build, Python, GPU, game assets or runtime npm
dependencies are needed for local play.

```bash
cd integrations/tank-duel
npm start
```

Open <http://127.0.0.1:4177/> and choose an edition. Direct links:

- <http://127.0.0.1:4177/classic/>
- <http://127.0.0.1:4177/auto-aim/>

Blue uses WASD + Space; Red uses arrow keys + Enter. In Classic, W/S drive and
A/D turn. In Auto Aim, WASD move in screen coordinates. P pauses; backgrounding
the tab also pauses the game. A keyboard is required; the layout scales to
small screens for viewing, but there are no touch controls.

Classic's **Match setup** has independent Blue and Red settings: model/checkpoint,
decision version, context, measured Auto or 50–1,000 ms scheduling, dodge toggle/strength and raw
shot guard. Collapse it for recordings. Settings persist in this browser/origin.
Model, decision, context or interval changes restart the match and invalidate
in-flight responses; assistance changes apply to that seat immediately. The
**Settings & model decisions** panel shows probabilities, actual context,
latency, trajectory previews, fullscreen and match export. Auto Aim keeps its
shared model and assist controls.

## Connect a model

Start a [Honing decision server](../../honing/backends/jev/server.py), then set
its **base URL** when launching the game server:

```bash
HONING_URL=http://127.0.0.1:8080 MODEL_LABEL=NanoJev npm start
```

This enables **Model vs model** and **You vs model** in both editions. The game
uses `GET /v1/health` and `POST /v1/decide`; the loaded backend must support
choice questions and return complete native probability distributions. The
bridge accepts the existing Laya, Kev and NanoJev decision contract. Additional
families, including a djev/DiffusionGemma backend, can use the same interface
when available in the configured Honing server. The game server does not load weights. The optional `scripts/serve-honing.py` launcher serves the pinned Laya / Laya Typed / Kev 0.8B / Kev 4B models using native Honing implementations; see [Arena lab](docs/arena-lab.md).

Use the model server's documented setup to prepare its checkpoint, pins and
GPU. Allow enough context for the full maze, all ten shells and every offered
choice; the bridge reports an error if that complete state cannot fit. `HONING_TOKEN`, when set, adds a server-side Bearer header for an upstream
proxy that requires it. It is never returned to the browser. The game server
does not accept upstream URLs or credentials from browser requests.

To use a configuration file, copy `.env.example` to `.env`, edit it, then run:

```bash
node --env-file=.env server.mjs
```

`npm start` reads the process environment; it does not load `.env` implicitly.
`PORT` defaults to 4177 and `HOST` to `127.0.0.1`. `DEFAULT_EDITION=classic` or
`auto-aim` redirects the home route. For an HTTPS reverse proxy, set
`PUBLIC_ORIGIN` to its exact external origin. This is a local demo server;
shared deployments should put their usual authentication and rate limits in
front of it. The static `web/` directory can also be served by any HTTP server;
without the game API it offers keyboard play only.

Existing tank-specific services can be retained using `CLASSIC_MODEL_URL`,
`CLASSIC_MODEL_LABEL`, `AUTOAIM_MODEL_URL` and `AUTOAIM_MODEL_LABEL`. These use
`/health` and `/decision` directly. `CLASSIC_NANOJEV_URL` adds a second Classic
backend. Each service must report its model, two seat IDs and support the selected
protocol: `plans-v1` for Classic planned steering, `controls-v3` for Classic
independent controls, or `controls-v2` for Auto Aim. A service can support just
one Classic decision version; selecting an unsupported version blocks play.
Responses must echo the request ID, round, seat, model and protocol, include
native probabilities and report `state_truncated: false`.

## Multiple models and checkpoints

Register additional Honing-compatible or tank-compatible endpoints using
`MODELS_JSON`. Each ID represents one running model/checkpoint endpoint. The
same family can appear multiple times with different `version` labels:

```bash
MODELS_JSON='[
  {"id":"gemma-current","label":"DiffusionGemma","version":"current","url":"http://127.0.0.1:8081","transport":"honing","editions":["classic"]},
  {"id":"nano-original","label":"NanoJev","version":"original checkpoint","url":"http://127.0.0.1:8082","transport":"honing","editions":["classic"]}
]' npm start
```

Use `transport: "tank"` for the legacy `/health` + `/decision` services. Optional
`token` is kept server-side. Registration labels do not load, pin or verify a
checkpoint: configure each actual model service first. **Model & context
identity** shows its reported model revision and adapter identity separately
from the context revision; an unreported checkpoint is explicitly labelled.
The browser only selects server-registered IDs and never submits upstream URLs.
Each seat validates responses against its own selected endpoint and seat ID.

The **Gemma planned vs NanoJev independent** preset selects the two named
families when available. Blue uses planned steering, Red uses independent
controls, both use Full tactical context, Auto scheduling and Reactive dodge.
Either dodge can then be turned off independently. This is a control-policy
comparison using the configured checkpoints, **not an assertion that an
archived first-generation prompt or older weights were restored**.

| Decision version | Context                                     | Behavior                                                                                                                                               |
| ---------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Classic Aim      | Full tactical (`planned-aim-v1`)            | Joint motion and conditional weapon, bounded turning, geometric selection and fresh shot validation.                                                   |
| Free Controls    | Full tactical (`state-v7-avoidance-levels`) | Independent drive/turn/weapon with existing aim hints and motion/shot forecasts; no joint-plan filtering.                                              |
| Free Controls    | Arena only · v1 (`arena-v1`)                | Newly added minimal profile: positions, headings, walls, all shells, timing and own assistance state; no aim hints, waypoints or trajectory forecasts. |

Planned steering requires its full tactical context. Every profile includes
shells from both sides. Dodge Off disables the local emergency controller for
that tank; it does not remove observations or planned steering's geometry and
shot checks. The optional raw shot guard is also independent per tank. Keyboard
players always use manual controls. Settings survive round changes and restarts.

## What the model sees and controls

- Both editions include **every active shell from both players**: position,
  velocity, owner, ricochet count and remaining lifetime. Ownership never makes
  a shell safe. State also includes tank positions, ammo, cooldowns and time.
- Classic Full tactical adds hull heading, wall geometry, verified hit directions, required
  turns, response-delay estimates, navigation waypoints and forecasts for all
  nine drive/turn combinations. Auto Aim supplies forecasts for each usable
  movement, including whole-path shell clearance and enemy-route progress.
- The model's essential state is sent intact. The bridge rejects missing
  candidates, invalid probabilities, mismatched responses and truncated state.
  It does not invent probabilities or substitute a scripted player on errors.
- Each seat has its own request/context and at most one request in flight.
  A shared endpoint may serialize both seats on the same model weights;
  independent seats do **not** imply separate model instances or GPUs.
- **50 ms is a scheduling floor, not an inference latency promise.** The UI
  reports request cadence and round-trip latency. Late decisions over 1.6 s
  are discarded; disconnects pause the match. Movement continues during
  inference until the current command expires.

Classic's default **Planned steering** is explicitly model + geometry assistance:
all nine native motion probabilities are retained, selection is restricted to
the geometrically admissible plans, then a second model request evaluates the
weapon conditioned on that exact motion. The hull turns at its physical speed.
If fresh geometry changes the motion or invalidates the shot, firing is
cancelled. Displayed drive/turn probabilities are marginals of the joint motion
distribution. **Raw controls** exposes independent drive/turn/fire choices for
experiments and can cause self-hits.

Classic's **Reactive** dodge (the persisted `light` ID) checks every 60 ms,
reacts to a predicted hit within 480 ms, takes control for at most 400 ms, then
cools down for 280 ms. **Balanced** uses a 620 ms trigger / 500 ms burst /
200 ms cooldown. **Protective** uses 780 / 650 / 120 ms. All three are bounded;
none changes bullet speed, collision size or damage. The model's original
command resumes after a burst if still valid. Auto Aim's separate optional
dodge checks 700 ms ahead every 50 ms and defaults off. Keyboard players are
never assisted. All interventions appear in match exports.

The export records each player's backend, reported model metadata, decision and
context versions, interval and assist settings. Each decision also snapshots
that configuration and the model identity it was validated against. Mid-match
assistance changes are events. Seed, physics revision, full context, native
responses, actual latency, applied plans, shots and kills are included.
It is an audit record, not a deterministic replay format. Treat probability
bars as model outputs; geometry restrictions and execution changes are shown
separately in the details and export.

## Classic arenas

Classic uses seeded, rotationally symmetric **Loop Mazes**, with 28–37 interior
wall segments on a 10 × 7 grid. Every cell has at least two exits, every cell is
reachable, and no single intermediate cell cuts the route between spawns.
Spawn exits have clear inward lanes. Corridors retain enough clearance for the
44 px tank footprint. Rounds change maps deterministically; New arena changes
the seed and Restart keeps it. Map revision: `loop-mazes-v3`.

Classic shells expire after **4.5 seconds** (previously 8), releasing their ammo
slot. Fire cooldown is **300 ms** (previously 480 ms). Five active shells per
tank and lethal self-hits remain. Forecasts use the same lifetime as physics.
Auto Aim retains its own maze and ammunition rules.

## Adaptive cadence and strategy experiments

Auto cadence learns each seat independently from its last 40 successful requests.
After four samples, the scheduling floor follows the 20th percentile of RTT;
forecasting uses the 75th percentile. Busy responses cause a decaying backoff.
One request per seat remains in flight; this measures attainable speed, not an
inference acceleration. Model changes clear calibration. Model identity details
show the floor, median, p95 and sample count. Manual scheduling remains available.

Strategy IDs preserve old baselines. Candidates are evaluated with real model
requests, paired seeds and swapped seats, then checked on unused seeds before
promotion. Losing trials remain in the experiment history and are not labelled
improvements. These are controller/context iterations, not model weight training.
See [Arena lab and measured results](docs/arena-lab.md).

## Collision rules

The complete tank silhouette, tracks and barrel fit inside a shared **22 px
radius** collision disk. Hull and turret rotations cannot extend into walls.
Wall ink fills the actual wall rectangle. Tanks and shells advance in spatial
steps of at most 2 px, so a delayed frame cannot jump over thin walls or another
live tank. Diagonal movement slides along walls while keeping full clearance.

Forecasts, live physics and displayed geometry use the same dimensions. The
22 px footprint is larger than the earlier experiment's 15 px center-only
collider; old match records are not byte-for-byte comparable. The existing
150 ms owner grace applies only to an unbounced shell leaving its muzzle;
**a bounced own shell is immediately lethal**.

## Verify

```bash
npm ci
npm test
npm run format:check
```

Tests cover both mechanics, self-ricochets, all-shell context, planned rotation
and shot revalidation, bounded assistance, wall/tank collision stress, server
routes, two-seat isolation, Honing request contracts and malformed responses.
CI also runs the real Python `DecisionServer`, the Node bridge and both game
engines against a clearly labelled CPU fixture:

```bash
# From the repository root, with Honing installed in the active Python env:
node scripts/e2e/tank-duel.mjs
```

To check a configured live model with the actual engine and collision assertions:

```bash
node scripts/smoke-model.mjs --edition classic --seconds 25
node scripts/smoke-model.mjs --edition auto-aim --seconds 25
# Optional: --url http://127.0.0.1:4177 --backend honing --controller raw
# Mixed Classic seats (use IDs registered on your server):
node scripts/smoke-model.mjs --edition classic --seconds 20 --players-json '[{"backend":"gemma-current","controllerMode":"plans","assistEnabled":true},{"backend":"nano-original","controllerMode":"raw","contextVersion":"full","assistEnabled":false}]'
```

Pause browser model matches before running a smoke check against the same
seats. The smoke check sends real requests and reports model identity, decision
counts, shots, hits and latency; it is a connectivity/gameplay check, not a model
quality benchmark.

## Files

`web/shared/` owns the UI, renderer, model client and collision primitives.
`web/classic/` and `web/auto-aim/` own their engines, context builders and thin UI
adapters. `model-bridge.mjs` maps the game contracts to Honing's typed choice
questions. `server.mjs` serves both editions and forwards configured model calls.
Everything in this directory can be copied as a standalone demo package.
