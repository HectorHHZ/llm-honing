# Arena lab

Six real model endpoints are connected in the live demo: DiffusionGemma + djev,
NanoJev, Laya English, Laya Typed Decisions, Kev 0.8B and Kev 4B. Model settings
are independent per seat. The four added models also pass Classic Free Controls
and Auto Aim requests containing all ten shells, with full native probabilities.

## Measured speed, 2026-10-01

Milliseconds, HTTP round trip from this Mac to shared ds-serv8 GPUs. Eight warm
serial samples and eight two-seat samples per model; these are measurements,
not hardware throughput ceilings. Planned control uses two conditional heads.

| Model                 | Serial median | Two-seat median | Two-seat p95 |
| --------------------- | ------------: | --------------: | -----------: |
| Laya                  |           300 |             315 |          342 |
| Laya Typed            |           283 |             320 |          355 |
| NanoJev               |           351 |             673 |          700 |
| DiffusionGemma + djev |           354 |             531 |         1280 |
| Kev 0.8B              |           526 |             652 |          745 |
| Kev 4B                |           688 |            1000 |         1086 |

Auto cadence recalibrates in the actual match as queueing and context change.
Latency includes networking, both decisions, and queueing. A 50 ms manual floor
cannot make a 700 ms response arrive faster. Responses older than 1.6 seconds
are dropped and reported rather than silently replayed.

## Experiment protocol

Run from `integrations/tank-duel` against a dedicated game API instance:

```bash
node scripts/arena-lab.mjs --mode calibrate --url http://127.0.0.1:4181 --backend laya --samples 12 --output arena-results/calibration
node scripts/arena-lab.mjs --candidate standoff-v1 --backend nanojev --seeds 701,702,703 --seconds 30 --output arena-results/screening
```

Each evaluation writes a unique timestamp/UUID directory, source hashes,
strategy definitions, model metadata, requests, native responses, interventions,
shots, kills, self-hits, range and actual latency. Simulated physics advances at
60 Hz while inference consumes real wall-clock time. It verifies wall clearance
and tank separation throughout. It has no fallback scripted player. The first
Angle Hunter trial predates source manifests; its full request/event logs were
retained, and it is not used as a promotion result.

A screen uses the same model on both sides, three seeds, with both seat orders.
Do not compare different-model wins as a strategy improvement. Candidates that
show useful screening results need at least twelve new paired-seed rounds,
no request/collision errors, no increase in self-hits and an out-of-sample score
above baseline before being offered as validated versions. Small samples remain
preliminary, and repeated hypothesis searches must be disclosed. Preserve the
baseline; never edit a published strategy's behavior under the same ID.

## Five directions tested — this round is complete

Thirty actual NanoJev duels, three paired seeds for each of five hypotheses:

| Direction                                        | Wins / losses | Candidate self-hits | Decision     |
| ------------------------------------------------ | ------------- | ------------------- | ------------ |
| Angle Hunter v1 — up to six firing headings      | 2 / 4         | 0                   | Not promoted |
| Keep Distance v1 — 220–360 px preferred range    | 2 / 4         | 0                   | Not promoted |
| Safe Lanes v1 — shell-clearance margin           | 2 / 4         | 0                   | Not promoted |
| Quick Shot v1 — shorter time-to-hit              | 2 / 4         | 0                   | Not promoted |
| Steady Aim v1 — preserve still-safe model intent | 3 / 3         | 1                   | Not promoted |

All 30 rounds completed without API, identity or collision assertion errors.
There were no held-out promotion trials because no candidate passed screening.
These small, sequentially selected experiments do not establish general win rates.
The UI keeps **Classic Aim** and **Free Controls**, and the Arena results page
preserves the five experiments and reflections. Candidate definitions remain in
`web/classic/strategies.js` for reproduction; they are not advertised as upgrades.
No automatic recurring runs were created after the five-direction stop request.

More firing angles alone did not help. Strong retreat pressure lost shot
opportunities. A clearance penalty further restricted offense. Favoring a short
time-to-hit did not translate into wins. Keeping a valid intent reached parity,
but one delayed self-ricochet exposed the limit of the three-second shot horizon
and the assumption that later movement is unknown. Shot safety is not guaranteed
for the entire 4.5-second shell lifetime. That experiment was rejected rather
than hiding the self-hit or weakening the damage rule.

The model sees per-plan outcomes and explicit geometric filtering. Current shot
forecasts hold the opponent stationary. A future moving-target forecast needs
matching context and execution checks, not merely a different prompt. This round
uses controlled engineering hypotheses rather than GEPA; no weight training or
GEPA optimization result is claimed.

## Dodge experiment

Run `node scripts/dodge-benchmark.mjs [path/to/previous/engine.js]`.
A reproducible fixture of four incoming-shell gaps × eight hull headings, with
stationary model commands and no walls: old Light survived 8/32, new Reactive
24/32, new Balanced 30/32 and new Protective 30/32. This measures those fixed
scenarios, not match win rates. All current levels cap the burst and enforce a
cooldown. Full compact results: [evaluation record](evaluations/2026-10-01.json).

## Native model serving

`scripts/serve-honing.py` is an opt-in launcher for an already prepared pinned
Honing checkout with `vendor/laya`, `vendor/kev` and model records in `models/`.
It verifies source revisions against `integrations/ra2/pins.json`. Laya model
records must match pins; Kev uses Honing's pinned checkpoint resolver. Install
the pinned packages into an isolated virtual environment before starting it.

```bash
python scripts/serve-honing.py --root /path/to/prepared/honing --model english --gpu 3 --memory-gib 4 --port 28780 --allow-shared-gpu
```

Use `typed-decisions`, `kev-0.8b` or `kev-4b` for the other services. The live
experiment uses allocator caps of 4, 4, 5 and 14 GiB respectively on a shared GPU.
The caps are not exclusive reservations or a guarantee against all GPU memory
pressure. The launcher checks free memory, respects Honing's GPU file lock and
does not evict other processes. It does not automatically download dependencies.

Laya supports 48 tokens per option pointer: the bridge places complete option
forecasts in essential state and sends short references. No forecast is removed.
Kev serializes probabilities to four decimal places; validation allows only
that accumulated rounding error and preserves native values and tied choices.
