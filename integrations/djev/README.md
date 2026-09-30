# DiffusionGemma through djev

Honing can serve **choice, noul and score** decisions from
[mmastrac/djev](https://github.com/mmastrac/djev), the structured-read server for
`google/diffusiongemma-26B-A4B-it`. This is inference integration, not a local
DiffusionGemma loader or trainer. The model probabilities come from djev's
diffusion reads; Honing does not turn generated text into invented confidence.

The adapter contract was checked against djev commit
[`e5841cf41e9211608e698492658685c36e24e77a`](https://github.com/mmastrac/djev/tree/e5841cf41e9211608e698492658685c36e24e77a).
That is a source reference, not verification of an operator's remote weights.
Health metadata explicitly reports `remoteWeightsVerified: false`.

## Connect an existing deployment

The upstream must be **djev**, not an ordinary vLLM chat port. Its
`GET /v1/models` must advertise the same served model name that its
`POST /v1/systemone` returns. Start the bridge from the Honing checkout:

```bash
python -m honing.backends.jev.server \
  --family djev --djev-url http://127.0.0.1:8011 --model dgemma \
  --djev-samples 1 --djev-steps 1 --djev-timeout 60 \
  --host 127.0.0.1 --port 8801
```

The base URL may end in `/v1`. There are no local GPU, tokenizer or model
dependencies. Do not pass `--pins` or `--checkpoint-manifest`: an HTTP bridge
cannot use those files to select or verify the upstream weights. The operator
must record the deployed model revision and engine build separately.

```bash
curl http://127.0.0.1:8801/v1/decide -H 'Content-Type: application/json' -d '{
  "id": "example",
  "parts": {"essential": ["An enemy shell is approaching from the right."]},
  "questions": {"action": {
    "type": "choice", "instructions": "Choose the next action.",
    "criteria": {"A": "Hold position", "B": "Move out of the shell trajectory"}
  }}
}'
```

`answers` keeps the upstream probabilities, confidence, score legend and score
expectation unchanged. `modelResult` keeps the complete upstream response,
including usage, sampling and timing diagnostics. Missing questions, missing
options, nonfinite/non-normalized probabilities, a non-maximal selected choice,
or a mismatched served model fail the request. Errors and timeouts propagate;
there is no retry with a substitute action or scripted policy.

State sections are joined in order: essential, details, memory, logs. Every
line is sent; no client trimming or token truncation occurs. Upstream context
overflow remains an error. djev reports total prompt tokens, so `tokenAudit`
labels them `upstream-request-total`, with `stateTokens` and `stateRoom` set to
null. These are not per-question token measurements or a verified context
budget. `stateTruncated: false` describes the adapter's lossless packing.

Current limits follow this djev contract: 1–64 questions, 2–26 alternatives per
choice/score, samples 1–32, steps 1–8. Fixed sampling defaults to one draw and
one denoising step. Upstream runtime settings can impose tighter limits; consult
the returned diagnostics for actual draws. Conditional questions, span/spans,
images and thinking options are not exposed by this adapter.

## RA2

Use the Honing bridge URL, not the djev URL:

```bash
cd integrations/ra2
npm run match -- --family=djev --decision-server=http://127.0.0.1:8801 \
  --out=runs/djev-example --ticks=1800 --seed=7 --map=mp03t4.map --seat=red
```

Duel seats similarly accept `--red-family=djev` and
`--red-decision-server=http://127.0.0.1:8801` (or blue). Existing engine setup
and retail game files are still required. A djev worker cannot be launched with
GPU flags; the family requires a running bridge. Family checking, target
permutation averaging and the audit understand the remote token accounting.
Logs count **structured decision requests**, not GPU forwards: a request can
contain multiple questions, samples and denoising steps. Raw djev diagnostics
remain in each response, including both halves of an averaged target decision.

## Upstream deployment reference

A GPU operator needs a vLLM build containing the structured diffusion features
from [vLLM #57250](https://github.com/vllm-project/vllm/pull/57250). Do not assume
that any installed release contains them. Follow the pinned djev README and
record the chosen vLLM commit, weights revision and hardware. A deployment has
three processes, for example:

```bash
# GPU model server (requires compatible vLLM and access to the model weights):
vllm serve google/diffusiongemma-26B-A4B-it --served-model-name dgemma \
  --diffusion-config '{"canvas_length":64}' --max-logprobs 32 \
  --enable-prefix-caching --host 127.0.0.1 --port 8000

# From the pinned djev checkout, in its inference environment:
python structured_server.py --upstream http://127.0.0.1:8000 \
  --model dgemma --tokenizer google/diffusiongemma-26B-A4B-it \
  --canvas 64 --host 127.0.0.1 --port 8011

# Then start the Honing bridge as above.
```

Optional djev `--constrained` and `--engine-samples` require additional engine
features and are not enabled here. Honing does not install vLLM, download the
26B checkpoint, claim a GPU, or change an existing deployment as part of this
adapter's startup.

## Verification boundary

CPU tests use a local HTTP stand-in with the pinned djev response shapes. They
cover state packing, unchanged probability transport, schema/result rejection,
model discovery, the CLI bridge and the real RA2 Node client. Node tests cover
the target permutation path and remote token audits. These tests do **not**
establish real DiffusionGemma accuracy, latency, GPU parity or a completed retail
RA2 match; no live djev GPU deployment was used for this PR.

```bash
python -m pytest tests/test_jev_djev.py
npm --prefix integrations/ra2 test
```
