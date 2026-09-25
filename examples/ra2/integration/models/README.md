# Experimental native choice adapters

The experimental NanoJev and Jev-Omni clients implement the same staged choice
interface as Laya and are selectable in the RA2 match runner. The default
controller remains Laya. A bounded head-only trainer and a two-model duel runner
are available below. Full native adaptation, promotion and repeated self-play
learning remain incomplete; interface checks do not establish game performance.

`NanoJevChoices` wraps the official NanoJev `DecisionPredictor` and
`prepare_examples`. It preserves the offered candidate IDs and order, measures
each native candidate path, and requires a complete native probability vector
and matching argmax answer. `JevOmniChoices` wraps the official merged-text
loader's `predict` function. It maps numbered option probabilities back to the
offered IDs and checks the actual native chat prompt before any model forward.
The adapters use the upstream revisions pinned in `league/model-sources.json`.

Both accept `predict(parts, questions)` with the same essential/details/memory/
logs sections and choice questions used by the staged controller. They return
answers, the exact retained state text, token accounting and removed records.
Packing keeps every essential record and every offered option. It removes
optional records whole (oldest logs first, then memory/details from the end).
If the remaining prompt is too large, or the upstream output is incomplete or
invalid, prediction fails. There is no local action substitute. The native
input limit is explicit; Laya's option/head budgets are not assumed to apply.

The worker verifies local sources/weights, locks the selected GPU, loads the
official predictor/model objects and passes them to the adapter. The adapters
themselves do not load weights or claim GPU ownership.

The CPU fixtures in `tests/test_ra2_native_choices.py` exercise packing,
choice-ID binding, malformed outputs and prompt drift without model weights.
A separate server preflight also checked actual pinned tokenizers and the
Jev-Omni upstream prompt on recorded normal-view RA2 requests, stopping before
model execution. Neither kind of check is a model inference or game evaluation.

## Experimental native worker

`worker.py` provides a separate JSONL process for the pinned upstream native APIs.
The common client factory connects it to the RA2 match runner. The worker has
CPU transport/provenance tests; those tests do not load models or establish
native game performance. Promotion still requires full, independently verified
games under the registered evaluation protocol.

Run from the RA2 example directory with a compatible, separately configured
Python environment and a prepared local model manifest:

```bash
python -m integration.models.worker --family nanojev \
  --prepared-manifest .cache/models/nanojev.json \
  --source-root /path/to/pinned/NanoJev --gpu 1 --max-length 2048

python -m integration.models.worker --family jev-omni \
  --prepared-manifest .cache/models/jev-omni.json --gpu 1 --max-length 2048
```

Before importing upstream code it verifies registry pins and every prepared
file hash, including all shards referenced by Jev-Omni's weight index. NanoJev
also requires an unchanged tracked checkout at its pinned source revision.
The worker uses the same device UUID lock as the Laya dispatcher, refuses an
occupied GPU, and sets CUDA visibility before importing Torch. The upstream
loaders retain FP32 parameters and BF16 forward autocast. Requests preserve the
native tokenizer/choice representations; no model or action fallback exists.

After successful model loading, stdout emits one `ready` record containing
verified provenance and runtime settings. Input lines contain exactly `id`,
`parts` and `questions`; IDs must be positive and strictly increasing. Responses
retain the request ID, complete native probabilities, packed context and token
audits. Duplicate JSON fields, nonfinite JSON, replayed IDs or inference errors
stop the worker. Loader diagnostics go to stderr. A weight-set digest identifies
the sorted mapping of native weight filenames to their hashes; it is explicitly
not a single-file weight hash or a trained checkpoint's promotion evidence.
Prepared manifests are left unchanged and retain their preparation-time status.

A server smoke check also loaded both pinned upstream models and answered three
requests per family using real inference on recorded normal-view training requests (root, target
and actor choices). Every offered choice retained a probability; no context
records were trimmed. Peak native input length was 1,454 tokens for NanoJev and
1,787 for Jev-Omni, within the explicit 2,048-token limit. This validates the
worker transport and native inference path only. No game, native training,
cross-model self-play or migration qualification is established by these six
requests. Full responses and verified file-set provenance are recorded in the
external `diagnostic-v49-native-inference-20260925` experiment.

## Common game client and independent verification

`factory.mjs` selects Laya by default, or a pinned native worker with
`--family=nanojev|jev-omni`. It uses the same context, candidates, staged
commitments, normal observation boundary and action limits for every family.
The same two-order mean applies to a target question with Cancel and exactly
two targets. Both actual native responses remain in the decision log.
`client.mjs` checks response IDs, the complete upstream probability vector,
argmax binding, retained context and encoding bounds. Worker errors or timeouts
stop the run without inventing an action.

After configuring the external engine, assets, baseline, compatible Python
runtime and prepared manifests:

```bash
export RA2_NATIVE_PYTHON=/path/to/compatible/python
node integration/laya/run.mjs --family=nanojev \
  --prepared-manifest=.cache/models/nanojev.json \
  --source-root=/path/to/pinned/NanoJev --gpu=1 \
  --map=mp03t4.map --seed=1000 --ticks=18000 --seat=red \
  --decision-log=gzip --out=runs/nanojev-dev

node integration/laya/run.mjs --family=jev-omni \
  --prepared-manifest=.cache/models/jev-omni.json --gpu=5 \
  --map=mp03t4.map --seed=1000 --ticks=18000 --seat=red \
  --decision-log=gzip --out=runs/jev-omni-dev

node integration/laya/audit.mjs runs/nanojev-dev
python league/verify_artifacts.py --match runs/nanojev-dev --split dev
```

These commands describe the integration path, not completed evaluations. Native
runs reject Laya's `--model` argument. `--native-max-length` defaults to 2048;
`NANOJEV_SOURCE_DIR` can supply NanoJev's source checkout. `RA2_NATIVE_PYTHON`
falls back to `LAYA_PYTHON`, then `python3` for direct model execution. The
artifact verifier requires one of the two Python environment variables.

Native manifests record `modelFamily`, the prepared manifest path/hash, source
revision, all verified file hashes and the native weight-set digest. Inference
counts mean native choice requests, not necessarily individual internal
transformer forwards. The common runner retains legacy `layaSeat` and
`layaMetrics` field names for compatibility.

The normal action audit reconstructs the original and permuted native output
bindings. The artifact verifier also runs `league/verify_native_encoding.py`
with the compatible model environment: it rehashes the actual pinned files and
source checkout, then independently rebuilds every request with the official
tokenizer/packing API. It compares exact retained state, removed records and
native token audits, including path or prompt hashes. This step constructs no
model objects and allocates no GPU. Its report includes verifier source hashes.
Original prepared files and sources must remain available for re-verification.
Evidence labels native weight-set hashes separately from Laya single-file
checkpoint hashes, and still requires normal action audits and replay hashes.

CPU fixtures cover worker failures, malformed native results, binary-target
mapping and the complete staged action-to-audit path. The latter is a synthetic
transcript, not an engine game. A separate real-weight client smoke check
(`diagnostic-v52-native-client-20260925`) reproduced all three earlier requests
per family with identical choices and probabilities through the JavaScript
transport. Full native game evaluation and qualification remain pending.


The production factory was then checked on four recorded training requests per
family, including a binary target requiring two native calls. All five native
requests per family passed independent rehashing and exact tokenizer
reconstruction (`diagnostic-v53-native-factory-20260925`). These are inference
transcripts, not engine games.

A separate 600-tick engine smoke run per family subsequently exercised the full
normal-view game path against Supalosa (`diagnostic-v54-native-engine-smoke-20260925`).
NanoJev made 546 native choice requests and submitted 9 accepted commands;
Jev-Omni made 210 requests and submitted 12 accepted commands. Every raw and
permuted request passed the action audit and independent native re-encoding;
both replays reproduced their recorded engine hashes. Both reached the short
cap without an engine winner. These two technical smoke runs are explicitly
excluded from training, promotion and full-match win-rate evidence. Original
upstream weights were used in those runs; the subsequent training and duel
checks below have separate identities and evidence limits.


## Two-model games

`duel.mjs` connects any pair of Laya, NanoJev and Jev-Omni to separate seats.
Both normal-view packets are frozen before either model predicts. Both choices
are resolved before advancing the engine. Each model gets one command per
3-tick decision; neither sees the other model's choices or hidden observation.
This differs from baseline matches, where Supalosa keeps its 16-command cap.

Reserve two GPUs from your allowed pool. The group wrapper acquires both UUID
leases together, releasing a partial acquisition immediately to avoid deadlock.
The example pool below is illustrative; supply the GPUs allocated to your job.

```bash
python league/gpu_group.py --pool 0 1 --record runs/duel-resources.json -- \
  node integration/models/duel.mjs \
  --red-family=nanojev --red-gpu={gpu0} \
  --red-prepared-manifest=.cache/models/nanojev.json \
  --red-source-root=/path/to/pinned/NanoJev \
  --blue-family=jev-omni --blue-gpu={gpu1} \
  --blue-prepared-manifest=.cache/models/jev-omni.json \
  --map=mp03t4.map --seed=104 --ticks=600 --out=runs/native-duel-smoke
python league/verify_duel.py --match runs/native-duel-smoke
```

For a Laya seat use `--red-family=laya --red-model=typed-decisions` and an
explicit `--red-checkpoint-manifest=upstream` or trained manifest path (use
`blue-` for the other seat). Each seat has its own decision log, provenance and
action audit; both share one engine replay. Verification also reconstructs the
actual native inputs. Required Python environments and game assets are the
same as baseline verification. A short capped game is a technical check, not
a full-match win-rate result. Playing a duel does not train either model.

## Native training data and head updates

`league/export_native_training.py` accepts a checked training-source registry
through `--data-check`. The registry must declare `passed: true`,
`developmentOrHeldoutUsed: false` and `trainingSources`, with each source's
`path`, `map`, `seed`, `seat`, `manifestSha256` and `samplesSha256`. The exporter
rechecks those files, episode identities and source snapshots, rejects episodes
outside `league/protocol.json`'s training split, and requires explicit teacher
or DAgger labels. Evaluation games and privileged observations are not labels.
This registry is an input contract, not an automatically generated curriculum.

The exporter packs every row with the official family-specific tokenizer. It
preserves essential state and all options, rejects conflicting labels, and
removes duplicate encoded inputs across splits. Its `dev.jsonl` is classifier
validation from training seed 105, not the game's development or sealed tests.
The separate verifier reconstructs every packed row and the removal decisions.
Neither command loads a model or allocates a GPU.

Run these commands with the configured model Python from `examples/ra2`:

```bash
"$RA2_NATIVE_PYTHON" league/export_native_training.py --family nanojev \
  --data-check /path/to/verified-training-sources.json \
  --prepared-manifest .cache/models/nanojev.json \
  --source-root /path/to/pinned/NanoJev --out runs/nanojev-supervision
"$RA2_NATIVE_PYTHON" league/verify_native_training_export.py \
  --directory runs/nanojev-supervision
"$RA2_NATIVE_PYTHON" league/train_native_head.py --family nanojev \
  --prepared-manifest .cache/models/nanojev.json \
  --source-root /path/to/pinned/NanoJev --data-dir runs/nanojev-supervision \
  --steps 8 --batch-questions 4 --head-lr 0.0001 --gpu 0 \
  --out runs/nanojev-head
"$RA2_NATIVE_PYTHON" league/verify_native_head.py \
  --directory runs/nanojev-head --gpu 0
```

Use `--family jev-omni` and its prepared manifest for Jev-Omni; omit
`--source-root`. Use fresh family-specific output directories and an available
reserved GPU. The default 32 distinct training rows and eight optimizer steps
are a gradient/reload diagnostic, not full training or validation-based model
selection. The trainer chooses a seeded sample with stage coverage, freezes
all backbone parameters, preserves runtime buffers, and updates native head
parameters with AdamW. It checks differentiable logits against the official
predictor before and after updates. Stored artifacts include head safetensors,
parent and data hashes, selected rows, optimizer-step records and probe logits.
The independent verifier loads the original parent and saved head in a fresh
process, checks frozen body/buffers, and reproduces every probe exactly.

To use a verified head in a baseline game, add
`--head-manifest=runs/nanojev-head/checkpoint.json` to the corresponding native
match command above. A duel accepts `--red-head-manifest` and
`--blue-head-manifest`. Prepared base weights remain required. The worker refuses
missing independent verification, altered head tensors or a different parent.
Learned weights receive a distinct composite identity; manifests retain the
upstream identity separately. There is no silent fallback to the original head.

Verified server diagnostics have completed actual eight-step head updates for
both families and independent fresh-process reloads. Three 600-tick duels also
exercised Laya/NanoJev, Laya/Jev-Omni and NanoJev/Jev-Omni. These demonstrate
learning and game plumbing separately, not learning from those duels, native
transfer success, stable wins, or a qualified champion. Repeated cross-model
learning and a portable end-to-end optimization command remain future work.
