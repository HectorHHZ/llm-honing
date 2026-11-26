# VVVVVV Exp 0 v0

Code, frozen settings and release documents of the VVVVVV failure-attribution dataset for
[Exp 0](https://github.com/HectorHHZ/llm-honing/issues/174), collected with the shared tools in
[`integrations/_shared/exp0/`](../../../_shared/exp0/README.md). Forty complete episodes: ten each with a changed
harness, changed weights, a changed controller, and no change. The dataset id is `vvvvvv`, its release path
`vvvvvv/v0/` in `honinglab/exp0-data`.

The collection is complete and checked locally: 40 formal attempts, none excluded; every attempt valid, all 5,671
logged requests re-rendered byte for byte, all 40 attempts reproduced under native replay, and the integration's
receiver accepted every executed reply. The diagnosis has not run: the Codex CLI subscription used for the
diagnosis hit its usage limit on 2026-11-26 (the CLI's message: "try again at Nov 28th, 2026 11:57 PM"), and no
scored call has been made. The results, the Hugging Face release and the #174 receipt follow after the Codex run.
[`RUNBOOK.md`](RUNBOOK.md) gives the one command that resumes the chain.

The decision request is the merged integration's own `run.request_for` (rule text, the full bridge observation, six
raw inputs held 15 native ticks each), and the controller is its `NativeClient` and bridge. The game is the
original source at `TerryCavanagh/VVVVVV@40e1ec3b`, built with this integration's bridge as
[`../../README.md`](../../README.md) describes; the game build and the official `data.zip` are supplied by the
operator. The release manifests record the harness and controller fingerprints over `run.py`, `decision.py` and
`HoningBridge.inc` of `integrations/vvvvvv` at `9d5327f`, named `integration/<file>` as in the collecting
workspace. Those are this integration's files on main byte for byte; `tests/test_exp0_vvvvvv.py` recomputes both
fingerprints from them. A collector run from this repository names the files relative to `integrations/vvvvvv`,
so its fingerprints differ from the release's for the same bytes.

| File | What it does |
| --- | --- |
| `exp0_vvvvvv.py` | Collects one attempt in a fresh game process and writes the episode contract; holds the fault instances |
| `collect_set.py` | Runs the planned attempts of a collection in streams, keeping every attempt |
| `adapter.py` | The shared tools' adapter: exact re-render of a decision's request, native replay of an attempt |
| `runtime.py` | Run-time locations of the game build, the data and SDL3 |
| `fault_checks.py` | Per-attempt checks of the three faults from the first decision, and the receiver on every executed reply |
| `capture_replay.py` | A native replay of a logged attempt that also captures its frames, for videos |
| `release_v0.py` | The release chain after the collection: `prepare`, `diagnose`, `cards`, `bundle`, and `resume` for after the usage limit |
| `procs.py` | Starts and stops only its own Xvfb and game processes, by pid and start time |
| `cut_video.py` | Edited and original videos of a recorded attempt, for people; never a model input |
| `plans/make_plan.py`, `plans/release_v0_frozen.json`, `collection.json` | The plan writer, the frozen release plan, and the collection root's copy of it (the same bytes) |
| `plans/pilot_persistent_p01.json` | The pilot plan of the persistent faults, outside the scored set |
| `pins.json` | Game build, assets, integration files and decision checkpoints the collection ran on |
| `PROTOCOL.md`, `DATASET.md`, `LABELS.md`, `RESULTS.md`, `RUNBOOK.md`, `TODO.md` | The collection's release documents, filled from its files; `cards/` holds the templates `release_v0.py cards` fills |
| `SOURCES.json` | The workspace file each file here was ported from, its SHA-256 and the rewrites applied |

The stricter label-isolation check of the diagnosis inputs is the shared
[`isolation_check.py`](../../../_shared/exp0/isolation_check.py).

Local paths resolve at run time, relative to the repository root: the game build is `work/vvvvvv/build/VVVVVV`
(where [`../../README.md`](../../README.md) builds it), `VVVVVV_ASSETS` names `data.zip` (default
`work/vvvvvv/data.zip`), and `VVVVVV_SDL_LIB` the directory of SDL3 when it is not on the loader path.
