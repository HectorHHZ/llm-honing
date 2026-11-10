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

The seven compatibility modules are scheduled for removal on **2027-02-10**:
`craftax/engine.py`, `craftax/catalog.py`, `griddly/engine.py`,
`griddly/catalog.py`, `rea_issac/engine.py`, `rea_issac/policy.py` and
`rea_issac/ui.py`, all under `honing/backends/`. Update external imports before
that date. Use the matching `integrations.<game>` module; the recording
identifier `FORMAT` is defined in `honing.backends.griddly.match`.
The old REA-Issac UI import shares the runtime module, including assignments
to `ROOT`. Removing these compatibility modules does not change recording
source identities because no match or audit uses them.

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

Both games identify the code used by headless matches and replay: adapter
modules that interpret the harness, build/validate decisions, advance play or
audit results, plus the game implementation and runtime data. Compatibility
modules, package initializers, launchers and training-only modules are excluded.
The game package initializers contain only docstrings; a test keeps them free
of runtime behavior. Each game's `honing/backends/<game>/source.py` names the
files explicitly. Add a new play/audit dependency there when adding code.

Griddly's `source/` snapshot contains the selected backend modules and runtime
modules under `source/integration/`. `adapter_sha256` covers every snapshot
name and file. REA-Issac's runtime fingerprint covers its selected backend
modules under `adapter/`, plus `engine.py`, `policy.py` and `source_facts.json`
from the game directory. Both audits still reject changed play/audit code.
Changing or removing compatibility imports does not invalidate recordings.

### Existing recordings and training demonstrations

Source identities change at this refactor. Old Griddly and REA-Issac runs need
their original commit and pinned runtime for audit/replay. This also affects
`GriddlyDemonstrationProvider`: it calls `audit_run` before accepting teacher
labels, so old demonstration directories cannot be passed to the new provider.
Keep the old directories and original checkout as archives. Do not overwrite
hashes or relabel old logs to make them pass a new audit. Already exported
`TrainingBatch`/`DecisionSample` records do not acquire a new wire format here.

To generate replacement Griddly demonstrations, run this from a checkout of
the new version with the pinned Griddly environment. It uses only the train
split and verifies every new run with the native engine. The output directory
is unique, so existing data is preserved.

```sh
integrations/griddly/.venv/bin/python - <<'PYCODE'
import json
import uuid
from pathlib import Path
from honing.backends.griddly.harness import seed_harness
from honing.backends.griddly.match import run_match
from honing.backends.griddly.replay import verify_replay
from honing.backends.griddly.scenarios import load_suite
from integrations.griddly import engine

integration = Path(engine.__file__).parent
suite = load_suite(integration / "tasks/suite.jsonl")
harness = seed_harness(integration)
destination = Path.cwd() / "runs" / ("griddly-teachers-" + uuid.uuid4().hex[:12])
runs = []
for record in suite.records:
    if record["split"] != "train":
        continue
    task_id = str(record["task_id"])
    out = destination / task_id
    run_match(suite.scenario(task_id), out, harness, policy="teacher")
    assert verify_replay(out)["verified"]
    runs.append(str(out))
print(json.dumps({"demonstration_runs": runs}))
PYCODE
```

Pass the printed run paths to `GriddlyDemonstrationProvider` (or the `runs`
argument of the Griddly e2e demonstrations factory). Use the same suite when
building the provider. It still refuses val/heldout tasks and performs native
replay before yielding training labels. These are newly generated rule-teacher
runs, not converted historical observations or a model performance result.

For a new REA-Issac baseline run, copy `seed`, `max_ticks` and
`decision_frames` from the old manifest's `scenario`, and reuse its saved
harness with `--harness-dir`. For example:

```sh
python -m honing.backends.rea_issac.match --policy baseline \
  --seed 7 --max-ticks 6000 --decision-frames 3 \
  --harness-dir /path/to/old-run/harness --out ./runs/rea-regenerated
python -m honing.backends.rea_issac.replay ./runs/rea-regenerated
```

Model runs additionally need the original checkpoint and decision server with
`--policy model --decision-server URL --policy-checksum CHECKSUM`. Such a run
is a new evaluation; it does not replace the historical result. Exact historical
audit still uses the original code and runtime. This PR does not delete or
modify archived model runs.

## Installed match commands

The three `python -m honing.backends.<game>.match` entry points find default
harness resources from the loaded game module. With no `--out`, they write
under `./runs/<game>/` in the current working directory. `--out DIR` selects
an explicit new directory. Neither mode writes match output into site-packages.
The standalone REA-Issac web UI still uses `ROOT` for its web assets and UI
recordings; this is separate from the match CLI.

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
