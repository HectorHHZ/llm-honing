# Parity with #49

These scripts compare the code in Honing with the original RA2 experiment in #49, run on the same inputs. They need a GPU node with #49's branch checked out (`PR49_DIR`, its `examples/ra2` directory, prepared as in its README), the model environment (`LAYA_PYTHON`), and the pinned Laya source (`LAYA_SOURCE_DIR`).

| Script | Compares | Result on an H100 |
|---|---|---|
| `decision_server.py <out>` | #49's stdio worker, Honing's stdio worker, and Honing's HTTP server (sequentially and from 4 concurrent clients), on 24 generated requests, 5 of which force trimming | identical answers, probabilities, state text, token audits, and trimming |
| `run_training.sh <work>` | #49's `train_laya.py` run twice and `LocalJevTrainer` run once on the same generated demonstrations and seed | #49 differs from itself (CUDA kernels are not deterministic), and Honing falls within that variation |
| `run_training.sh <work> --deterministic` | the same, with PyTorch's deterministic kernels on (`deterministic_49.py` enables them and then runs #49's file unmodified) | all three weight files byte-identical |

`make_demos.py` writes six labelled matches in #49's directory format, with seed 101 held out for validation, so training parity needs no game.

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
