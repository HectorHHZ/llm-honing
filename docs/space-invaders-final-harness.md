# Final hand-tuned Space Invaders harness

`integrations/space-invaders/final-harness.json` is the machine-readable
identity of the final presented system. It selects the OCAtari lane runner,
the `--landing-priority` variant, and NanoJev weights with SHA-256
`287ff07466032fa0a8054783300ac4a3123f6c6a2bc691815a4f60287b36913b`.
The earlier game that prompted the landing-priority change used a different
controller configuration. The checkpoint bytes are supplied separately; this
PR does not contain a model or a game recording.

The OCAtari lane controller lives in `integrations/space-invaders/`. Its game
entry point is `lane_pilot.py`; `lane_controller.py` and `solo_intercept.py`
implement aiming, movement, firing, shield checks, dodging, and urgent
low-alien handling. `lane_io.py` encodes visible state and calls the decision
server. NanoJev chooses between valid target lanes or solo
intercept plans. The harness converts that choice to native ALE actions.

Install the pinned `ale-py`, `gymnasium`, and `ocatari` versions from
`integrations/space-invaders/requirements.txt` in a dedicated environment.
ALE also needs an externally supplied Space Invaders ROM. Run a code-selected
full game to check the environment and controller:

```sh
python integrations/space-invaders/lane_pilot.py \
  --policy best --seeds 61004 --max-steps 100000 \
  --replan-steps 16 --model-choice-mode all \
  --scoreless-question --balanced-option-order --solo-predictive \
  --calibrated-geometry --fire-while-dodging --landing-priority \
  --record /tmp/space-lane-best.jsonl
```

To run a Jev-selected game, start Honing's NanoJev decision server with a
checkpoint supplied outside this repository. A trained checkpoint's manifest
must point to the local directory containing `best.safetensors` after the
files have been copied from external storage. Use a pinned NanoJev source
checkout and a GPU available to you:

```sh
python -m honing.backends.jev.server --family nanojev \
  --pins integrations/ra2/pins.json \
  --nanojev-source /path/to/pinned/NanoJev \
  --checkpoint-manifest /path/to/checkpoint.json \
  --gpu 0 --port 8801
```

Then run the same controller with model selection:

```sh
python integrations/space-invaders/lane_pilot.py \
  --policy jev --decision-server http://127.0.0.1:8801 \
  --replan-steps 16 --model-choice-mode all \
  --scoreless-question --balanced-option-order --solo-predictive \
  --calibrated-geometry --fire-while-dodging --landing-priority \
  --seeds 61004 --max-steps 100000 --record /tmp/space-lane-jev.jsonl
```

Before playing, verify that `best.safetensors` in the external checkpoint has
the checksum in `final-harness.json`. The decision server's health metadata
also reports `weightsSha256`; it must match. Using a different checkpoint is
a different system even with the same controller flags.

`lane_pilot.py` records the offered options, model choice, selected lane,
native action, and rewards in JSONL. `--policy best` is a code selector and
`--policy random` randomly selects valid high-level options. Both exercise the
same harness, but neither is a Jev game. The controller can run without model
weights in these two modes. No checkpoint, generated training data, game
trace, result summary, or rendered demo is included in this PR.
