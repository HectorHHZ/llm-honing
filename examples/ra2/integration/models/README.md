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
