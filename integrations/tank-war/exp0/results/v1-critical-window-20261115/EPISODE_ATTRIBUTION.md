# Episode-level attribution amendment

Zhuohang clarified the task after the 40-game blind run: System 0 gives **one
component attribution per entire game**, not a decision-window answer. The
three edited-fault labels are `harness`, `weights`, and `controller`. The ten
unmodified losses are controls and stay outside that three-class denominator.
No label claims that the edited component caused the final loss.

## Technical labels

For each of the 40 frozen games, the game-local
`audit-critical-episode-labels.py` checked the source manifest hash, complete
same-state probe set, recorded reference index, and absence of differences at
other decisions. It then assigned one private label per game from the measured
signature:

| Label | Same-state evidence |
| --- | --- |
| Harness | Model-visible input changed before selection, and selected action changed. |
| Weights | Input stayed identical; a real alternate checkpoint changed probabilities and selected action. |
| Controller | Input, probabilities, and selected action stayed the same; the command or execution changed afterward. |
| No injection | No technical difference at any checked decision and no fault manifest. |

All **40/40** passed: 10 per edited-fault class and 10 unmodified controls.
The per-game label file, manifests, and exact reference indices remain private
under `episode-label-audit-v1/` in the dataset root. Label file SHA-256:
`0108c5a690b822b450e00c31a08f6e9408ffc368ced01c528c17d27469b32557`.
[`episode-attribution-data-card.json`](episode-attribution-data-card.json)
records the public schema, provenance hashes, class counts, and limitations.
The audit uses these indices to establish the label, but System 0 is scored
once per game. A better clean-game outcome is a separate field; it occurred
in only six of the 30 blind fault games.

## Rescore of sealed answers

The existing Sol-high `read` and `probe` runs each returned one `cause` for
every game. [`episode-attribution-summary.json`](episode-attribution-summary.json)
scores that field alone. The original sample, prompt, schema, answers, and
window-oriented score have not been changed.

| Correct class on 30 edited-fault games | Read | Probe |
| --- | ---: | ---: |
| Harness | 3/10 | 4/10 |
| Weights | 5/10 | 5/10 |
| Controller | 0/10 | 1/10 |
| **All three classes** | **8/30** | **10/30** |

Probe improved by 2/30 games, or 6.7 percentage points. The paired,
class-stratified bootstrap 95% interval is **-16.7 to +30.0 points**. Its
lower bound is below zero; probe accuracy is the 10/30 balanced-set majority
baseline. Both modes made an unsupported harness or controller diagnosis on
1/10 unmodified losses. Eight of those ten losses were called `weights` by
each mode; that is not counted as an injected-weight fault or as a false alarm,
because ordinary poor model choices can occur without an edited checkpoint.

**No-go for reliable episode-level three-class attribution on this scoped
sample.** Controller recognition is the clearest failure. This is an
exploratory rescore after the user changed the target: the original prompt
also required a decision window and could have influenced the component
answer. A new episode-only prompt is provided in
[`episode-only-prompt-v2.txt`](episode-only-prompt-v2.txt) for a future,
unexposed blind set; it was not used to produce these numbers.

The 30-game human window-review page and the unadjudicated planner action
suggestions are retained as historical artifacts. They are unnecessary for
this technical three-class rescore. Codex's technical audit is not an
independent human failure-cause review, so the original #183 reviewer
criterion remains unmet unless the issue scope is amended.
