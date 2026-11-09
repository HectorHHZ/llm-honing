# Game runtimes and Honing adapters

The game runtime owns game state, rules, native engine calls, actions, rendering
and standalone controls. The Honing adapter owns task splits, harness files,
model requests, episode execution, grading and training data conversion.

| Game | Runtime and resources | Honing adapter |
| --- | --- | --- |
| Craftax | `craftax/engine.py`, `craftax/catalog.py`; pinned upstream Craftax package | `../honing/backends/craftax/` |
| Griddly Spiders | `griddly/engine.py`, `griddly/catalog.py`; pinned upstream Griddly package | `../honing/backends/griddly/` |
| REA-Issac | `rea_issac/engine.py`, `source_facts.json`, `policy.py`, `ui.py` and `web/` | `../honing/backends/rea_issac/` |
| Tank War | `tank-war/web/game/engine.js` and web resources | `../honing/backends/tank/` |
| RA2 | `ra2/` native bridge and runtime tools | `../honing/backends/ra2/` |
| Soul Knight, Hollow Knight | `soul-knight/`, `hollow-knight/` native control bridges | Integration-local demos |

Craftax and Griddly keep the upstream game as an optional pinned dependency.
Their local runtime modules expose game observations and explicit steps. They
load native packages when an engine is created. Importing these modules does
not load JAX, Craftax, Griddly or Honing.

Honing adapters import the runtimes. Runtime modules do not import Honing.
`EngineScenario` in each of the Craftax and Griddly runtime modules describes
read-only settings. The existing validated `CraftaxScenario` and
`GriddlyScenario` classes satisfy these types. Standalone callers can supply
objects with the same attributes. They must use values within the documented
scenario limits. Task validation and train/validation/heldout splits remain in
`honing/backends/`.

## Imports and installation

Use `integrations.craftax.engine`, `integrations.griddly.engine` and
`integrations.rea_issac.engine` for new engine imports. The old
`honing.backends.<game>.engine` imports re-export the same classes and functions.
Craftax and Griddly catalog imports, and REA-Issac policy and UI imports, also
have compatibility modules. Runner, grader and factory import paths are unchanged.

The `llm-honing` wheel includes only these three Python game packages, their
runtime data, and default harness/task resources. It does not install every
integration script as a Python package or bundle the upstream native engines.
Install each game's documented requirements to run Craftax or Griddly.
REA-Issac runs with the Python standard library.

The REA-Issac source directory is now `integrations/rea_issac/`. Use this path
for script commands and web assets. Existing source checkouts can retain old
`integrations/rea-issac/runs/` directories. Replay accepts an explicit run path;
it does not move or delete old recordings. Wire protocol names, task IDs and
harness file names still use `rea-issac`.

## Recordings and replay

Griddly's `source/` snapshot contains the backend Python modules and the game
modules under `source/integration/`. `source/integration_namespace.py` records
the Python integration package initializer. `adapter_sha256` covers every
snapshot name and file. Replay checks all these bytes before it accepts a run.
REA-Issac's runtime fingerprint reads the engine, facts and baseline from the
game directory. It does not hash the compatibility imports as game code.

Recording formats and action semantics are unchanged. Source hashes change
when imports or source locations change. Use the original commit and pinned
runtime to replay recordings made before this refactor. Do not edit recorded
hashes to bypass this check.

## Checks

```sh
python -m pytest tests/test_game_layout.py tests/test_craftax.py tests/test_griddly.py tests/test_rea_issac*.py
python -m build
python -m venv /tmp/honing-game-wheel
/tmp/honing-game-wheel/bin/python -m pip install --no-deps dist/*.whl
/tmp/honing-game-wheel/bin/python -I scripts/checks/game_install.py
```

The final command uses the installed wheel, with the source checkout excluded
from Python's import path. It checks packaged resources and executes a real
REA-Issac episode and replay. Native Craftax and Griddly tests need their
separate environments, as described in each game's README.
