# Hollow Knight (ForestDango): file bridge and Exp 0 v0

[ForestDango/Hollow-Knight-Demo](https://github.com/ForestDango/Hollow-Knight-Demo) is a Unity remake of the
opening of Hollow Knight: King's Pass, Dirtmouth and the False Knight. This directory holds the Honing side of
that game: a file bridge compiled into the game, and the Exp 0 v0 collector and frozen settings of the
`hollow-knight-forestdango` failure-attribution dataset ([#185](https://github.com/HectorHHZ/llm-honing/issues/185),
[#174](https://github.com/HectorHHZ/llm-honing/issues/174)). It is a different game from the
[Hollow Knight imitation](../hollow-knight/README.md) (DanielDFY's two levels) in `integrations/hollow-knight/`.

The game is not included. The project has no license and ships commercial Unity assets (PlayMaker, 2D Toolkit,
InControl), so the deterministic build the dataset was collected on is not redistributable. The collection ran on
its commit `974a49a385c38335e3a78c52d41b5eaaf0db96c1`, built with Unity 2020.3.26f1; `exp0/v0/pins.json` records
the SHA-256 of the player, `UnityPlayer.so` and both game assemblies. Offline scoring and the exact re-render of
every logged request work without the build; the native replay needs it.

## Research scope

Primary category: **Reactive combat**. Capability tags: platform movement, navigation, combat.

Move through King's Pass and fight its enemies with near-raw keys.

| Component | Responsibility in this configuration |
| --- | --- |
| Model | Choose one of 9 move options (left, right or stay, each without, with a short or with a long jump) and one of 5 attack options (none, strike forward, up or down, focus) at every decision. |
| Harness | The pinned `v12_standing` snapshot: describe what the screen and HUD show in text, offer every option in a fixed order, and turn the two choices into the keys of one window. Nothing is hidden, ranked or recommended, and no reflex acts without the model. |
| Engine and other policies | Run the game's own movement, combat, enemies, damage and room changes. |

The model never receives pixels. Observations are limited to what a player sees on the current screen and HUD
(`honing.visible.v1`); the bridge's global debug state is used only by the evaluation side.

## The bridge

`bridge/` holds the seven C# files compiled into the game's `Assembly-CSharp.dll` for the release's deterministic
build. The bridge is inert unless `HONING_SESSION_DIR` is set when the player starts. It then supplies the key
states through InControl's keyboard provider, fixes the time step (30 frames per second of game time) and the
random seed, stops the game between action windows, starts a new game the way a player does, and writes the
visible and the debug observation after every window (`HoningBridge.cs` describes the protocol).

`HONING_FAST=1` makes the game deterministic: game scripts read a clock that advances 1/30 s per frame, scene loads
are synchronous, the opening films are skipped, and "is this on screen" is computed from the camera rectangle.
With `-batchmode -nographics` the player then draws nothing; `player/hk_fast.sh` starts it that way and
`player/hk_det.sh` starts the same build with drawing, for recordings. On that build, the same keys give the same
frames whether drawn or not. The developer knobs in `HoningDev.cs` (revive in place, no damage, skip the death
wait, teleports) and the lookahead in `HoningLookahead.cs` are off unless their variables are set; none is set in
any dataset run.

The deterministic build also changes 16 of the author's scripts in 35 places. Each change sits behind the
determinism switches: off by default, on under `HONING_FAST=1`. Those modified copies are derived from the
unlicensed project and are not included here. `pins.json` records the resulting assembly's SHA-256, and the
bridge's own sources are the files in `bridge/` byte for byte.

## Exp 0 v0

`exp0/v0/` holds the collector and its tools, the pinned harness snapshot, the frozen plan, the pins, the release
driver and the collection's release documents. It uses the shared tooling in
[`integrations/_shared/exp0/`](../_shared/exp0/README.md).

The collection is complete and checked locally: 40 formal attempts, none excluded, ten per condition (no injection,
harness, weights, controller) from ten reset groups; every attempt valid, all 1,291 logged requests re-rendered byte
for byte, all 40 attempts identical under native replay, and the label-isolation check of the diagnosis inputs
passed. The diagnosis has not run: the Codex CLI subscription used for the diagnosis hit its usage limit on
2026-11-26 (the CLI's message: "try again at Nov 28th, 2026 11:57 PM"), and no scored call has been made. The
results, the Hugging Face release (`hollow-knight-forestdango/v0/` in `honinglab/exp0-data`) and the #174 receipt
follow after the Codex run. [`RUNBOOK.md`](exp0/v0/RUNBOOK.md) gives the one command that resumes the chain.

| Path | What it does |
| --- | --- |
| `collect.py`, `batch.py`, `release_jobs.py` | Collect one attempt in a fresh game process; run a job list, a few at a time; write the job lists of the formal collection from the frozen plan |
| `starts.py`, `starts/`, `verify_starts.py` | Scripted key prefixes from a new game to each start state, and their check on two boots |
| `faults.py` | The fault instances: what the harness renders, which checkpoint answers, which keys the controller sends |
| `state_codec.py`, `hk_adapter.py` | Exact save and restore of the harness state; the shared tools' re-render adapter |
| `replay_env.py`, `replay_check.py`, `native_replay.py` | Replay of an attempt from its recording, and natively on the deterministic build |
| `offline.py`, `offline_faults.py` | The collector chain and every fault on a recorded run, with no game and no model |
| `to_contract.py` | Writes the episode contract of an attempt from its raw record |
| `release_v0.py` | The release chain after the collection: `prepare`, `diagnose`, `cards`, `bundle`, and `resume` for after the usage limit |
| `release_dryrun.py`, `card_values_dryrun.json` | A dry run of the chain on a small smoke or pilot collection with the stub diagnoser |
| `runtime.py` | Run-time resolution of local paths, ffmpeg and Laya's tokenizer |
| `pins.py`, `pins.json`, `plan.json`, `collection.json` | The pinned build, bridge, harness, controller and services; the frozen plan and the contract plan rendered from it |
| `PROTOCOL.md`, `DATASET.md`, `LABELS.md`, `RESULTS.md`, `RUNBOOK.md`, `TODO.md` | The collection's release documents, filled from its files; `cards/` holds the templates `release_v0.py cards` fills |
| `harness/v12_standing/` | The harness snapshot, byte for byte; `.pre-commit-config.yaml` leaves it unformatted |
| `core/` | The game process and bridge client, the key-level controller, the Laya client and the evaluation watcher |

`SOURCES.json` lists the workspace file each file here was ported from, its SHA-256 and the rewrites applied.

The harness counts option tokens with Laya's tokenizer. Install `tokenizers` and set `HONING_LAYA_TOKENIZER` to
`tokenizer/tokenizer.json` of `convaiinnovations/laya-typed-decisions@1a793eb568e6718f15941d08f85432581df534e3`
(SHA-256 `6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30`), or let `huggingface_hub` fetch
it. Every collector, replay and probe tool refuses to run without it, and the tests that re-render requests are
skipped.

The frozen plan names the build and the bridge sources by paths of the collecting workspace
(`code/hk-devmode/...`). Plan paths resolve against the repository root, so an operator places the build there or
points `release.game.player`, `release.game.player_drawn` and `release.game.bridge_source` at their copy (the
bridge sources are `bridge/`). The defaults of `core/` and `pins.py` are under `work/hollow-knight-forestdango/`.
