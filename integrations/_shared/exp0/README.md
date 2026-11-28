# Exp 0 v0 episode tooling

These standard-library tools turn the attempts a game collector records into a checked, scored and portable
Exp 0 v0 dataset release ([#174](https://github.com/HectorHHZ/llm-honing/issues/174)). They follow the
accepted Griddly Spiders v0 release, whose tools live in its own integration. The VVVVVV and full Hollow Knight
(ForestDango) v0 collections were validated, audited and given their diagnosis inputs with the files in this
directory; their diagnosis, bundles and Hugging Face releases follow once the Codex run can be made. Each of
those games adds its collector, an adapter and a release driver.

[CONTRACT.md](CONTRACT.md) defines one attempt on disk (`honing-exp0-episode-v0`, contract version 0.2.0): the
manifest, the decision rows, the raw replies, the saved public states, the fault records, the outcome, and what a
diagnoser may see. The JSON schemas are in `schemas/`.

| Step | Tool | Output |
| --- | --- | --- |
| Check every attempt | `validate_episode.py` | valid, incomplete or invalid, per attempt |
| Weights evidence by re-asking the clean checkpoint | `audit_set.py reask` | `weights-reask.jsonl` |
| Evidence, selection and neutral episode IDs | `audit_set.py audit` | `audit/episodes.json` (gold), `audit/attempts.jsonl` |
| Label-free diagnosis inputs | `diagnosis_inputs.py` | `read-inputs.jsonl` |
| Stricter label-isolation check of the read and probe inputs | `isolation_check.py` | `isolation-check.json` |
| Probe captures and the #170 identity check | `probes.py` | `probes/identity.jsonl`, `probes/probe-inputs.jsonl` |
| One fresh, tool-free Codex session per episode and mode | `diagnose.py` | `diagnosis-<mode>/` |
| Offline scores with random, majority and constant baselines | `score.py` | `scores.json` |
| Cards, bundle and release metadata | `fill_cards.py`, `bundle.py`, `release_metadata.py` | `<game-id>/v0/` |
| Fresh-environment check of a bundle | `verify_release.py`, copied into every bundle | a JSON receipt |
| Upload plan and guarded upload | `upload_release.py` | refuses dry-run bundles; reads the token from `HF_TOKEN_FILE` |

A game supplies an adapter (`adapter.py` defines the interface): its `rerender(state, history)` rebuilds a
decision's request with the clean harness from the saved public state. `probes.py selfcheck` requires it to
reproduce every logged request byte for byte before any probe call.

Two pieces of shared Honing code are pinned copies, so that a release can be reproduced exactly:

- `vendor/traces_9d5327f.py` is `honing/harness/traces.py` at `9d5327f1c5a53b8b134db847952ff6281267c0a6`, the
  #168 renderer both releases rendered their diagnosis inputs with. `upstream.py` loads only this copy and
  refuses it unless its git blob id is `4140ac647c85669cbe4bca518054b6f50aecf721`. The renderer on main has
  changed since (#216 and #217 list the fields that disappeared between decisions differently), so importing it
  would change the inputs the releases were diagnosed on.
- `vendor/probe_a151596.py` is `honing/harness/probe.py` of `feat/counterfactual-probe` at `a151596`, the #170
  identity check Griddly v0 used, also checked by blob id.

`honing.core`'s trace types come from this repository; they are unchanged since `9d5327f`.

`SOURCES.json` lists, for every file here and for `tests/test_exp0_common.py`, the workspace file it was ported
from, that file's SHA-256 at the time of the port, and the rewrites applied. A release bundle keeps the source
files themselves under `common-source/`.

## Tests

```bash
python -m pytest tests/test_exp0_common.py
```

The tests build synthetic episodes of both decision shapes, run every step above against fake decision services
on loopback ports and a stub Codex CLI, verify the bundle in a fresh virtual environment without packages, and
check that tampered bundles are refused. They use no model, game or network and take about a minute.
