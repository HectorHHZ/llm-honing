# Standalone game decision receiver

`decision.py` is the standard-library-only response validator, HTTP client and
strict JSON decoder used by the Hollow Knight, VVVVVV, Taisei, Hurrican,
Celeste Classic and hallowdale demos. The game scripts temporarily add their `integrations/` directory for this
import and remove that inserted entry afterward. They work from an arbitrary
working directory without an installed Honing package or model dependencies,
while preserving optional-runtime discovery and caller-owned import paths.

This consumes the existing serving response; it introduces no request or trace
format. A choice answer must include `type: "choice"`, as the serving backends
emit. Every submitted question and every offered option must be present exactly.
Probabilities must be finite non-Boolean numbers in `[0, 1]`, sum to one within
`option_count * 0.00005 + 1e-8` (Laya's four-decimal output), and select an argmax.
Ties are accepted. Values are never normalized or replaced by a policy fallback.

The response must echo the integer request ID and retained `stateText`, and
provide `trimmed` plus `tokenAudit` for exactly the submitted questions. Audit
keys `stateTokens`, `stateRoom`, `inputTokens`, `options` and `stateTruncated`
are required. Explicit null token counts support backends such as DJEV that do
not expose every count; missing keys fail. Available counts must be nonnegative
integers, consistent with each other and any `maxInputTokens` limit. A nonempty
state cannot have a zero known budget/count. `stateTruncated` must be false.

All context is required by default. A caller may explicitly allow whole entries
of `logs`, `memory` or `details` to be pruned. Removals must match submitted
entries in the existing server's order: oldest logs, newest memory, newest
details. Essential entries and token clipping are always rejected. Current
policies remain: Hollow Knight/Taisei/Hurrican/hallowdale allow no removal,
Celeste permits oldest logs only, and VVVVVV permits optional logs/memory/details.

The decoder rejects duplicate keys at every object depth, NaN/Infinity and
exponent overflow. When `raw_path` is supplied, the HTTP client writes the exact
response bytes before decoding, including HTTP error bodies before re-raising.
Each game still owns its native-action gate and local rejection artifacts.

Run the shared receiver tests once with no third-party test dependencies:

```sh
python3 -m unittest tests.test_game_decision_contract
```

The existing game tests cover their adapters and native boundaries. These
portable tests do not establish native game performance or real-model evidence.

The `exp0/` directory holds the Exp 0 v0 episode tooling that the VVVVVV and
full Hollow Knight (ForestDango) dataset releases share; see its
[README](exp0/README.md).
