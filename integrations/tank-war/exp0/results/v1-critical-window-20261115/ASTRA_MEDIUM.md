# Astra medium secondary check

On the same 40 previously diagnosed Tank War games, `gpt-6-astra` with
`medium` effort made one game-level attribution from the replay and one from
the replay plus controlled repeats. The sample, prompt, schema, and scoring
rules were frozen before these Astra answers. This is a secondary check on
known games, not an independent blind confirmation.

| Measure | Replay only | Replay plus controlled repeats |
| --- | ---: | ---: |
| Correct on 30 edited-fault losses | 11/30 | 30/30 |
| Harness | 1/10 | 10/10 |
| Weights | 2/10 | 10/10 |
| Controller | 8/10 | 10/10 |
| Unsupported harness/controller attribution on 10 unmodified losses | 10/10 | 3/10 |

The paired gain was 19/30, or 63.3 percentage points. The existing
layer-stratified bootstrap gives a 95% interval of 50.0 to 76.7 points.
There were no invalid calls or tool calls. Replay-only used 2,211,436 input
and 5,818 output tokens; controlled repeats used 2,388,186 input and 4,484
output tokens. Wall time was approximately 5 minutes 18 seconds and 4 minutes
28 seconds respectively, with at most two concurrent calls. Exact per-call
records and the freeze are in the private dataset root under `astra-medium-v1/`.
The freeze SHA-256 is
`fb9c2f45b1754ca1be03566679542b23e1514a32a8d24011f3a5e00f7a1f48d4`.

The controlled-repeat fields expose strong layer-specific evidence. Across
all ten harness cases, the repeated model input and question differed; across
all ten weights cases, the input stayed the same but the chosen action changed;
across all ten controller cases, the model choice stayed the same but the
command changed. These are generic comparisons, not explicit fault labels,
but they nearly determine the three-way answer for these fault families. The
30/30 result therefore measures whether Astra uses these signals on this
sample. It does not show that replay alone is reliable, that other fault
families generalize, or that a changed decision caused the final loss.

The Sol-high result on these same games was 17/30 replay-only and 19/30 with
controlled repeats. The original three final one-decision faults also failed
the required independent score-drop test. This secondary Astra result does
not change the scoped no-go recommendation.

The run used Codex CLI 0.158.0 with Bubblewrap isolation. The ChatGPT-login
CLI pinned model and effort, but did not expose the internal model revision
or permit setting temperature. The aggregate machine-readable score is
[`astra-medium-v1-summary.json`](astra-medium-v1-summary.json).
