# Experimental native choice adapters

These Python adapters prepare the choice contract for NanoJev and Jev-Omni.
They are **not connected to the RA2 match runner, training loop or promotion
gates**. They do not establish migration, model quality or self-play results.
The runnable game controller remains Laya.

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

Callers must verify local sources/weights, reserve the GPU, load the official
predictor/model objects, and pass them to the adapter. Loading, provenance,
worker lifecycle, training and game-runner integration remain separate required
work. The adapters themselves do not load weights or claim GPU ownership.

The CPU fixtures in `tests/test_ra2_native_choices.py` exercise packing,
choice-ID binding, malformed outputs and prompt drift without model weights.
A separate server preflight also checked actual pinned tokenizers and the
Jev-Omni upstream prompt on recorded normal-view RA2 requests, stopping before
model execution. Neither kind of check is a model inference or game evaluation.

## Experimental native worker

`worker.py` provides a separate JSONL process for the pinned upstream native APIs.
It is not yet wired into the RA2 match runner, training loop or promotion gates.
The worker has CPU transport/provenance tests; those tests do not load models or
establish native game performance.

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

A server smoke check also loaded both pinned upstream models and executed three
real forwards per family on recorded normal-view training requests (root, target
and actor choices). Every offered choice retained a probability; no context
records were trimmed. Peak native input length was 1,454 tokens for NanoJev and
1,787 for Jev-Omni, within the explicit 2,048-token limit. This validates the
worker transport and native inference path only. No game, native training,
cross-model self-play or migration qualification is established by these six
requests. Full responses and verified file-set provenance are recorded in the
external `diagnostic-v49-native-inference-20260925` experiment.
