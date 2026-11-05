# Space Invaders environment and replay

The optional installed module supplies an ALE game environment and an action replay
validator. It contains no controller, Jev model, training code, checkpoint,
Atari ROM, or sealed incumbent artifact.

## Install and run

Use Python 3.10 or newer. Install `llm-honing[space-invaders]`. ALE must be able
to load the external Space Invaders ROM. The ROM and its redistribution rights
are outside this package; obtain it through a lawful source and configure ALE
according to its installation instructions. The CLI reports an ALE error if the
ROM is unavailable.

```sh
python -m honing.harness.space_invaders play --seed 61001 --action NOOP --output /tmp/space-invaders.json
python -m honing.harness.space_invaders replay /tmp/space-invaders.json
```

The fixed-action CLI is a smoke game. For a Jev or other decision model, call
`run_episode(policy, seed=..., output=...)` from Python. `policy(frame, info)`
receives the current RGB observation and ALE info and returns one of the six
native action indices: `NOOP=0`, `FIRE=1`, `RIGHT=2`, `LEFT=3`,
`RIGHTFIRE=4`, `LEFTFIRE=5`. A caller may turn the observation into model input
and map the model's choice to an index; no model code lives in this package.
The package validates the action set at runtime. Separate controller and
training sources are documented in `integrations/space-invaders/README.md`.

```python
from pathlib import Path
from honing.harness.space_invaders import run_episode, verify_replay

# A model adapter can replace this function without changing the environment.
def policy(frame, info):
    return 1  # FIRE

run_episode(policy, seed=61001, output=Path("/tmp/space-invaders.json"))
print(verify_replay(Path("/tmp/space-invaders.json")))
```

## Environment contract

Both modes use `ALE/SpaceInvaders-v5`, RGB, sticky-action probability `0.25`,
and the six-action minimal set. The default `pooled=True` mode creates ALE with
internal `frameskip=1`; the wrapper repeats a native action for up to four
physical frames, sums their rewards, and pixelwise max-pools the final two
frames. `pooled=False` asks ALE to use native `frameskip=4` and returns the
fourth frame. Both modes stop at the first `terminated` or `truncated` flag.
A life loss is not an episode end. A valid full-game replay requires natural
termination with zero lives, and rejects a truncated or decision-capped game.

The JSON replay contains the seed, ALE/Gymnasium versions, mode and environment constants, each native
action, observed reward/lives/terminal flags, and a SHA-256 of each returned RGB
frame. Verification recreates the game and checks every step and the final
score. Deterministic replay requires compatible ALE, ROM, and Gymnasium builds;
compare package versions when moving a trace between machines. This trace
format is an integration-level persisted format. Its merge needs an accepted
RFC under `CONTRIBUTING.md`.

The final hand-tuned OCAtari controller and its operating instructions are documented
in `docs/space-invaders-final-harness.md`. Model checkpoints and generated
experiment data are supplied separately.
