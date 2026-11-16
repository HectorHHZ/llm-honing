# Exp 0 rapid test: split, budget, and commands

This records the experiment as run. It does not turn an after-the-fact description into a pre-test freeze. The source freeze is `blind-v7/sample-freeze.json` in the private artifact store; its SHA-256 is `a7dbeb68f361c73b9072e23266c95c197a3af67822f1844f22966f4ae21a98f1`.

## Which games were tested

[`blind-split.json`](blind-split.json) lists the 40 opaque episode IDs and SHA-256 hashes of the exact diagnosis inputs, sorted by ID. The private freeze was sealed before model answers at 2026-11-16 01:43:33 UTC. All 40 episodes are distinct-seed losses: ten each from harness, weights, controller, and no-change conditions. No few-shot examples were used in this rapid test. Pilot, development, fault-screening, and previously diagnosed games were excluded from this blind split. The private freeze retains seed, seat, source run, changed layer, technical reference decision, and source hashes; those gold fields are intentionally absent from the public split list.

The source collection ranges, eligibility checks, and deterministic SHA-256 selection rule are in [`protocol.json`](protocol.json). The exact private freeze is at `/data/xinkaiz/system0/exp0/tank-war/v1-critical-window-20261115/blind-v7/sample-freeze.json` on ds-serv2.

## Budget and actual usage

The pre-run execution limit was **three hours of wall time**. The plan allowed up to three available model-service GPUs (0, 5, and 6) for this experiment while leaving other users' work alone. Expressed as a GPU-time ceiling, that is **at most 9 GPU-hours of capacity** (3 GPUs × 3 hours), not a measurement of GPU utilization or an exclusive reservation. The retained records do not support a precise actual GPU-hour total. This conversion is documented here after the run; no separate numeric GPU-hour counter was sealed in `sample-freeze.json`.

The frozen diagnosis design limited this blind test to 40 cases, two modes, one call per case and mode: **80 planned Codex calls**, at most two concurrent. The runner rejected a diagnosis payload over 1,500,000 characters. There was **no pre-frozen token or dollar ceiling**, so neither is claimed here. The recorded actual usage was 4,570,610 input tokens and 98,080 output tokens across both modes. The $10.12 API-equivalent estimate in `SUMMARY.json` is not the user's ChatGPT-login bill and is not the experiment budget.

## Commands used for the final analysis

Run these on ds-serv2, where the private artifact tree and the original frozen scorer are present. The diagnosis runner refuses to overwrite an existing mode directory; these commands document the original run and are for a fresh copy, not an in-place rerun of sealed answers.

```sh
ROOT=/data/xinkaiz/system0/exp0/tank-war/v1-critical-window-20261115
python3 "$ROOT/repo/integrations/tank-war/scripts/run-critical-diagnosis.py" --out "$ROOT/blind-v7" --mode read
python3 "$ROOT/repo/integrations/tank-war/scripts/run-critical-diagnosis.py" --out "$ROOT/blind-v7" --mode probe
python3 "$ROOT/repo/integrations/tank-war/scripts/score-critical-diagnosis.py" --out "$ROOT/blind-v7"
python3 "$ROOT/repo/integrations/tank-war/scripts/summarize-critical-dataset.py" --root "$ROOT" --inventory "$ROOT/inventory-final-v3" --blind "$ROOT/blind-v7"
```

The scorer checks its own hash against the private freeze. The repository copy was formatted after the run; the original scorer is retained at `$ROOT/repo/` and its results were separately reproduced byte for byte with the formatted copy. Individual replay, probe, and Codex commands, exact inputs, output, timing, and usage remain in the private per-case attempt records. No new model call is needed to audit the published summary.

## Later per-game label and answer audit

After the user clarified that the System 0 output is one class per game, the
following game-local commands audited private technical labels and rescored
the **already sealed** answers. They did not call a model. To repeat the first
command, choose a new empty output directory because the auditor refuses to
overwrite existing labels.

```sh
ROOT=/data/xinkaiz/system0/exp0/tank-war/v1-critical-window-20261115
python3 "$ROOT/repo/integrations/tank-war/scripts/audit-critical-episode-labels.py" \
  --blind-root "$ROOT/blind-v7" \
  --output-dir "$ROOT/episode-label-audit-v1"
python3 "$ROOT/repo/integrations/tank-war/scripts/score-critical-episode-attribution.py" \
  --sealed-score "$ROOT/blind-v7/score.json" \
  --technical-labels "$ROOT/episode-label-audit-v1/labels.jsonl" \
  --out "$ROOT/episode-label-audit-v1/episode-attribution-summary.json"
```

The private and repository copies of `episode-attribution-summary.json` are
byte-identical (SHA-256
`99caa71776c62e498d72b43fb61380ba5c55e7e551c4e7df3cb5599e33b12372`).
The original prompt and scorer remain frozen. The new episode-only prompt and
schema were not inputs used in this historical rescore.

## Episode-only prompt sensitivity run

The clarified prompt and schema were then run on the **same 40 games** as a
post-test sensitivity check. The new prompt, schema, protocol, and exact
anonymous input/probe hashes were frozen before the new answers; this did not
create an independent test set. The private freeze is
`episode-only-v2/freeze.json` (SHA-256
`611c815d76da85916e792f057d0f5af5fd0dcdc1d7eefcc6a7cbcdcd1a12fe8f`).
No gold labels were passed to the isolated Codex processes.

```sh
ROOT=/data/xinkaiz/system0/exp0/tank-war/v1-critical-window-20261115
python3 "$ROOT/repo/integrations/tank-war/scripts/prepare-critical-episode-only.py" \
  --source-blind "$ROOT/blind-v7" --out "$ROOT/episode-only-v2"
python3 "$ROOT/repo/integrations/tank-war/scripts/run-critical-episode-only.py" \
  --source-blind "$ROOT/blind-v7" --out "$ROOT/episode-only-v2" --mode read
python3 "$ROOT/repo/integrations/tank-war/scripts/run-critical-episode-only.py" \
  --source-blind "$ROOT/blind-v7" --out "$ROOT/episode-only-v2" --mode probe
python3 "$ROOT/repo/integrations/tank-war/scripts/make-critical-episode-score-input.py" \
  --run "$ROOT/episode-only-v2" --out "$ROOT/episode-only-v2/score-input.json"
python3 "$ROOT/repo/integrations/tank-war/scripts/score-critical-episode-attribution.py" \
  --sealed-score "$ROOT/episode-only-v2/score-input.json" \
  --technical-labels "$ROOT/episode-label-audit-v1/labels.jsonl" \
  --out "$ROOT/episode-only-v2/score.json"
```

Both modes completed 40/40 calls without invalid output or tool use. Read
ran from approximately 04:47 to 04:55 UTC on 2026-11-16 (483 seconds of
observed wall time); probe ran from 04:55 to 05:01 UTC (367 seconds).
Read used 2,191,730 input and 33,126 output tokens; probe used 2,370,199
input and 22,658 output tokens. The private and public aggregate scores are
byte-identical, SHA-256
`c3866168a11ece0ed7ee3194cd5cf901f2539ff7975d49dc038f51fa1993726c`.
