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

## Episode-only sensitivity run

To test the clarified output directly, Sol high answered once per mode on the
**same 40 anonymous games** using an output schema with no decision index or
loss-impact field. The episode-only prompt, schema, protocol, inputs, and
call limit were frozen before these 80 new answers. Freeze SHA-256:
`611c815d76da85916e792f057d0f5af5fd0dcdc1d7eefcc6a7cbcdcd1a12fe8f`.
All 80 calls were valid, with no tools. This is a **post-test sensitivity
run**, not an independent confirmation: the same games and their earlier
results were known when the prompt was written. Per-case answers remain
private under `episode-only-v2/`; the public aggregate is
[`episode-only-v2-summary.json`](episode-only-v2-summary.json).

| Correct class on 30 edited-fault games | Read | Probe |
| --- | ---: | ---: |
| Harness | 9/10 | 9/10 |
| Weights | 4/10 | 4/10 |
| Controller | 4/10 | 6/10 |
| **All three classes** | **17/30** | **19/30** |

Probe gained 2/30 games, or 6.7 percentage points. The paired,
class-stratified bootstrap 95% interval is **-16.7 to +30.0 points**, so a
probe benefit is unestablished. The balanced-set majority baseline is 10/30.
On the ten unmodified losses, **7/10** in each mode were called harness or
controller without supporting same-state evidence. The model used no
`uncertain` abstentions. This high false-alarm rate prevents treating its
fault-game accuracy alone as reliable attribution. Calling an unmodified loss
`weights` is reported separately because ordinary poor model choices can
occur without an edited checkpoint.

The read arm used 2,191,730 input and 33,126 output tokens in about 483
seconds of wall time; probe used 2,370,199 input and 22,658 output tokens in
about 367 seconds. Using the same frozen uncached standard-tier reference
as the historical test, their API equivalents are $4.71 and $4.97. These are
not ChatGPT-login bills. The CLI was `0.158.0`; the model's internal revision
and temperature were not pinned.

Representative errors from the sealed probe answers, without publishing
episode IDs or the private answer key:

- An unmodified loss had identical model inputs and choices under controlled
  repeats. System 0 still called it `harness`, treating ordinary “do not fire
  yet” guidance as a faulty instruction.
- In a real one-turn edited-checkpoint game, System 0 called `controller`.
  Its explanation focused on ordinary drive-command transformations elsewhere
  in the game, missing the same-input probability and choice change at the
  checkpoint call.
- In a one-turn controller drop-fire game, System 0 called `harness` because
  it focused on the normal firing guidance. The technical difference was
  after action selection, in the command sent for that one decision.

**Scoped no-go for reliable per-game three-class attribution.** The
episode-only prompt improves fault-game accuracy relative to the historical
window-oriented prompt, while control false alarms rise sharply. Because
this is a repeat on the same games and the internal model revision is
unavailable, the difference cannot be attributed solely to prompt wording.
Neither run shows a reliable paired gain from controlled probing.

## Historical rescore of sealed answers

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

In that historical prompt, controller recognition was the clearest failure.
The new episode-only run above used
[`episode-only-prompt-v2.txt`](episode-only-prompt-v2.txt); this historical
rescore did not. Both use the same 40 games and private technical labels.

The 30-game human window-review page and the unadjudicated planner action
suggestions are retained as historical artifacts. They are unnecessary for
this technical three-class rescore. Codex's technical audit is not an
independent human failure-cause review, so the original #183 reviewer
criterion remains unmet unless the issue scope is amended.
