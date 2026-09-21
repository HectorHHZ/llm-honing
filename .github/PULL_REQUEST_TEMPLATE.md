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
- [ ] Non-trivial AI assistance is disclosed above, or none was used.
