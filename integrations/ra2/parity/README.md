# Parity with #49

These scripts compare the code in Honing with the original RA2 experiment in #49, run on the same inputs. They need a GPU node with #49's branch checked out (`PR49_DIR`, its `examples/ra2` directory, prepared as in its README), the model environment (`LAYA_PYTHON`), and the pinned Laya source (`LAYA_SOURCE_DIR`).

| Script | Compares | Result on an H100 |
|---|---|---|
| `decision_server.py <out>` | #49's stdio worker, Honing's stdio worker, and Honing's HTTP server (sequentially and from 4 concurrent clients), on 24 generated requests, 5 of which force trimming | identical answers, probabilities, state text, token audits, and trimming |
| `run_training.sh <work>` | #49's `train_laya.py` run twice and `LocalJevTrainer` run once on the same generated demonstrations and seed | #49 differs from itself (CUDA kernels are not deterministic), and Honing falls within that variation |
| `run_training.sh <work> --deterministic` | the same, with PyTorch's deterministic kernels on (`deterministic_49.py` enables them and then runs #49's file unmodified) | all three weight files byte-identical |

`make_demos.py` writes six labelled matches in #49's directory format, with seed 101 held out for validation, so training parity needs no game.

## NanoJev against #49

`nanojev_serving.py <out> <gpu-49> <gpu-stdio> <gpu-http>` sends the 24 serving requests to #49's native worker (twice), Honing's stdio worker, and Honing's HTTP server, sequentially and from 4 concurrent clients. It needs `PR49_DIR`, `NANOJEV_PYTHON`, and `NANOJEV_SOURCE_DIR`. `nanojev_training.py <gpus> <work> <matches>` checks `NanoJevTrainer` as `kev_training.py` checks Kev's. Results on an H100:

- On one GPU, all four paths agree with #49 on every field of all 24 requests (answers, probabilities, state text, token audits, 1,153 trimmed entries).
- NanoJev's bf16 forward does not repeat bit for bit across processes. In six runs of #49's own worker on one GPU, requests 4 and 21 took one of two values (at most 0.0015 apart) from run to run; the other 22 never changed. Honing's worker shows the same two values, and matches #49 whenever both land on the same one.
- The first HTTP server ran each call on the request's thread, which changed request 4 by 0.0015 against the loading thread. The server now runs every call on the thread that loaded the model, and matches.
- Training: sft validation loss 1.60 to 1.30 (584 train, 118 validation); the trained checkpoint serves from its manifest; grpo raises the favored option from 0.44 to 0.997, on 16 of 16 states; 1 minute 40 seconds for all three steps.

`nanojev_tree.py <out> <gpu> [requests]` compares the token-tree inference modes (`NanoJevSettings.inference`) with the reference on one GPU. It loads each mode in its own process, answers the serving requests twice and times the second pass, and measures every bf16 mode against the reference in fp32. For the tree modes it also answers the requests three to a call (`predict_many`, as the server answers branches) and compares them with the same requests answered alone, and checks that encoding a request in one tokenizer call gives the paths NanoJev's own `prepare_examples` gives. It needs `NANOJEV_SOURCE_DIR` and a Python with NanoJev's requirements that can import Honing. Results on an RTX A6000, 24 requests (50 questions):

| Mode | Largest probability difference from fp32 | Choices that differ from fp32 (fp32's top-two gap) | Median per request | Load | Peak reserved |
|---|---|---|---|---|---|
| reference, fp32 | (the yardstick) | | 88.4 ms | 18 s | 7.0 GiB |
| reference, bf16 | 0.0076 | 1 (0.0012) | 99.5 ms | 18 s | 4.3 GiB |
| `tree` | 0.0054 | 1 (0.00007) | 35.5 ms | 18 s | 3.5 GiB |
| `tree-compiled` | 0.0056 | 2 (0.00007, 0.0012) | 6.5 ms | 69 s with filled compile caches | 4.1 GiB |

The choices that differ are near ties, which bf16 rounding moves either way. Answered three to a call, the requests moved by at most 0.0057 (`tree`) and 0.0044 (`tree-compiled`) against answering them alone, the same rounding on a tree of another shape; some of those trees were larger than the captured graphs and ran without one. The one-call encoding gave every path `prepare_examples` gives, with no string encoded outside the batch. `tree-compiled`'s exact values depend on the kernels Inductor picked when it compiled: two runs on the same caches gave the same probabilities, while an earlier compile with other cache contents measured a largest difference of 0.0064 and no differing choice.

## Kev against Kev's own code

The Kev family has no counterpart in #49, so these compare Honing with Kev's own code at the commit in `pins.json`. They need the Kev environment from `scripts/setup-kev.sh` (`KEV_PYTHON`, `KEV_SOURCE_DIR`) and the pinned snapshots in the Hugging Face cache.

| Script | Compares | Result on an H100 |
|---|---|---|
| `kev_serving.sh <out>` | Honing's `KevDecisionModel` against `kev.serve` at the pinned commit, on the 24 serving requests plus 3 whose long logs make Kev trim the state: a reference running the same computation (bf16, no CUDA graphs), `kev.serve`'s CUDA defaults, and Kev's exact fp32 path | same computation: 56 of 56 questions choose the same option with identical probabilities; fp32: identical; `kev.serve` defaults (CUDA graphs): largest difference 0.008, one near-tie (0.4264 vs 0.4309) chosen differently |
| `kev_training.py <gpus> <work> <matches>` | `KevTrainer` sft on `make_demos.py`'s matches, the trained checkpoint served from its manifest, and grpo on synthetic rollouts favoring one option | sft validation loss 3.09 to 1.32 (584 train, 118 validation examples); the trained checkpoint serves at temperature 1.0; grpo raises the favored option from 0.10 to 0.91, on 16 of 16 states |

`KEV_MODEL=kev-4b` runs both for kev-4b:

| Script | Result for kev-4b on an H100 |
|---|---|
| `kev_serving.sh <out>` | same computation: 56 of 56 identical; fp32: identical; `kev.serve` defaults: largest difference 0.009, one near-tie (0.1918 vs 0.1932) chosen differently |
| `kev_training.py` | sft validation loss 2.01 to 1.31; the trained checkpoint serves at temperature 1.0; grpo raises the favored option from 0.39 to 0.9994, on 16 of 16 states; 3.5 minutes for all three steps |

## StartLux-Decision against StartLux's own server

`startlux_serving.py <our-gpu> <reference-gpu> <out>` starts `startlux_decision.server` at the pinned commit on one GPU and loads Honing's `StartLuxDecisionModel` on another. It sends both the 24 serving requests plus 3 with long logs that make Honing trim the state; the reference gets the state text Honing sent the model. It needs the environment from `scripts/setup-startlux.sh` (`STARTLUX_PYTHON`, `STARTLUX_SOURCE_DIR`), and `STARTLUX_MODEL` picks the size. Results on H100s, 56 questions per model:

| Model | Graph and eager paths (41 questions) | Requests StartLux runs on its shared-prefix path (15 questions) | Median latency, untrimmed requests |
|---|---|---|---|
| `startlux-0.8b` | identical probabilities | same choice 15 of 15, largest difference 0.008 | 8.3 ms |
| `startlux-4b` | identical probabilities | same choice 15 of 15, largest difference 0.028 | 14.4 ms |
| `startlux-27b` | identical probabilities | same choice 12 of 15, largest difference 0.012 | 49.0 ms |

The 15 differing questions come from requests with two or more questions over 4,096 tokens, where the reference reads the letters in bf16 and Honing, with that path off, reads them in fp32. The three 27B questions that choose differently are near-ties in the reference's bf16 values; one is an exact tie (0.3535 and 0.3535). All three servers passed StartLux's own graph self-test with a difference of 0.0 over 40 graphs. The long requests trimmed 2,873 entries in all, and the longest prompt was 8,179 tokens.

## Nimble-v4-max against Bespoke's engine

`nimble_serving.py <our-gpu> <reference-gpu> <out>` loads Honing's `NimbleDecisionModel` on one GPU and, in a second process on another, Bespoke's `NimbleEngine` from the pinned source snapshot with its Decision Index run's settings, called as the Decision Index kit calls it. Bespoke ships no server for the model, so the engine itself is the reference. It sends both the 24 serving requests, 3 with long logs that make Honing trim the state, and 3 that add a yes/no and a five-level score question; the reference gets the state text Honing sent the model. It needs the environment from `scripts/setup-nimble.sh` (`NIMBLE_PYTHON`, `NIMBLE_SOURCE_DIR`, `NIMBLE_KIT_DIR`). Results on H100s:

| Questions | Same choice | Largest probability difference |
|---|---|---|
| choice (62) | 62 of 62 | 0.000000 |
| yes/no (3) | 3 of 3 | 0.000000 |
| score (3) | 3 of 3 | 0.000000 |

The probabilities are identical, including after Honing converts a score answer from the engine's choice over level indices. The long requests trimmed 2,880 entries in all, the longest prompt was 8,165 tokens, and the median latency of the untrimmed requests was 102.9 ms.

## Real game controller parity on CPU

`game_parity.py` compares the original #49 controller with the extracted
controller using one shared, pinned Laya model on CPU. The original stdio
client talks through a transport-only HTTP bridge. Both games still use their
real policies, the engine and the original assets. This isolates controller
and transport parity; it does not replace GPU worker parity or training tests.

Prepare the game, SDK and Supalosa paths in `integrations/ra2/.env` as described
in the integration README, and export them before starting either subprocess:

```bash
set -a
source integrations/ra2/.env
set +a
"$LAYA_PYTHON" integrations/ra2/parity/game_parity.py \
  --reference "$PR49_DIR" --source "$LAYA_SOURCE_DIR" \
  --out "$WORK/game-parity" --ticks 1800 --seed 7 \
  --map mp03t4.map --seat red
# Use a Python environment with the updated SDK that supports load_replay.
for side in reference extracted; do
  node integrations/ra2/integration/laya/audit.mjs "$WORK/game-parity/$side"
  "$RA2_PYTHON" integrations/ra2/scripts/verify_replay.py \
    --match "$WORK/game-parity/$side" --game-dir "$RA2_GAME_DIR"
done
```

The output directory must not already exist. `PR49_DIR` is the original
`examples/ra2` directory and must contain its prepared `.cache/models` files.
`LAYA_PYTHON` needs the pinned Laya dependencies; `RA2_PYTHON` needs the updated
Python game SDK. No model or original game assets are included here.

The comparison removes `latencyMs` from decision/action records and otherwise
retains all fields. Replay comparison validates each chunk's CRC, removes only
`HEAD.game.savedAt`, and compares the remaining content including action bytes,
RNG state, simulation timestamps and hash checkpoints. Different raw replay
hashes are therefore expected when save times differ. Unit tests reject action
changes, simulation timestamp changes, invalid CRCs, truncation and trailing
bytes.

### Recorded result, 2026-09-28 UTC

The 1800-tick run completed in 1640.49 seconds on CPU with four PyTorch threads,
using typed-decisions revision `1a793eb568e6718f15941d08f85432581df534e3`
and Laya source `970dc8c5f63d7b886a68409493f37d569424f933`.
It ran at `fa14b66`, based on main `1663a23`; that additional SFT numerical-guard
commit does not modify the inference model, HTTP server or game controller.
Both games used mp03t4, seed 7, with Laya in the red seat.

- All 600 decisions and all 74 action records matched.
- Both final simulation hashes were `1678891593` at tick 1800.
- Canonical replay SHA-256 was
  `ba154fe0460d7423bd013743a47694e408552b0c915002117324733ae6caf223`.
- Both matches passed the current audit and independent replay-only verification;
  replay invoked no policy and reported no mismatch.

The tick limit truncated both games. This is one deterministic controller
regression, not a completed competitive match, win-rate estimate, concurrent
four-game serving test, or a training-stability result. Issue #50 still needs
its other in-game validation cases.
