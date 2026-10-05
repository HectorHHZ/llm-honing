<!--
Title this pull request as a Conventional Commit subject, e.g.
  feat(compression): add expert-drop scoring over router activation counts
Merges are squashed, so this title becomes the commit subject.
-->

## Motivation

<!-- What problem does this solve? Link an issue or RFC: Closes #N -->

## Changes

<!-- Describe the implementation in focused bullets. -->

## Compatibility and operational impact

<!--
Changes to plane protocols, configuration, persisted formats, HoneRelease
shape, dependencies, or resource usage. Write "None" if unaffected.
-->

## Verification

<!--
The commands that actually ran, and their real results. A claim that checks
pass, without the command and its output, is not review evidence.
-->

```
$ pytest
```

## End-to-end

<!--
CI runs the end-to-end checks on every pull request (the `e2e` job: the
all-planes pipeline, the harness runner contract, a dry-run RA2 campaign, and a
decision server with the match's Node client). Name its result. A new harness
runner also needs scripts/e2e/check_runner.py on its own config, and a new
trainer scripts/e2e/check_trainer.py.

A change to a model family, a trainer, the decision server, or the engine side
also needs an end-to-end run CI cannot do, on a GPU or in a real game: the
command, the hardware, and the output (for example integrations/ra2/parity/,
or a match with its audit and replay check). If it could not be run, say why.

A run that trains, whether a harness evolving or model weights being
post-trained, also links its Weights & Biases (wandb) log: every step with its
scores or losses, and the commit and configuration it ran. Paste the charts a
reviewer needs here too. If it could not log to wandb, say why and attach the
same numbers.
-->

## AI assistance

<!--
Write "None", or name the tools used and what they contributed. The author
remains responsible for every submitted line, claim, and test result.
-->

## Checklist

- [ ] The change is focused and contains no unrelated cleanup.
- [ ] The title is a valid Conventional Commits subject.
- [ ] Tests cover behavior changes, or this pull request does not change behavior.
- [ ] Public protocol changes include contract tests, or none are present.
- [ ] The design specification under `system-design/` is updated, or the change does not affect it.
- [ ] An accepted RFC is linked, or this change does not require one.
- [ ] `pre-commit run --all-files` and `pytest` pass locally.
- [ ] The CI `e2e` job passes, and a GPU or in-game run is reported above when the change needs one.
- [ ] An end-to-end run that trains (harness evolution or weight post-training) links its wandb log, or none was run.
- [ ] Non-trivial AI assistance is disclosed above, or none was used.
