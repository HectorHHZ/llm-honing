# Experimental native choice adapters

The experimental NanoJev and Jev-Omni clients implement the same staged choice
interface as Laya and are selectable in the RA2 match runner. The default
controller remains Laya. Native training and promotion are not implemented;
interface tests and inference checks do not establish native game performance,
migration qualification or self-play results.

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
native game performance. Training and promotion still require separate work.

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
transport. Full native game evaluation, learning and qualification remain pending.


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
upstream weights are still unadapted; no native learning or cross-model
self-play has been completed.
