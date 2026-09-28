# RA2 issue progress snapshot

This branch preserves the locally saved SFT candidate while the development
server is unavailable. It is a review/archive snapshot, not a merge candidate.

## What is included

The code patch is unchanged from the saved candidate patch for server commit
`14d13d0`, based on trainer commit `2f71f010b7102275a5441bd5bb5141c237efc27c`.
It excludes cuDNN SDPA, rejects nonfinite gradients before optimizer updates,
and adds numerical tests. Learning rates, labels and the loss are unchanged.
The original base predates the merged GRPO trainer. Do not merge this branch
onto current main; use the newer server-side port once it is accessible.

For publication, the patch applied against the exact original blobs, Git's
whitespace check passed, and each changed Python source parsed successfully.
No training run was launched during this publication.

## Verified work already published

[PR #75](https://github.com/HectorHHZ/llm-honing/pull/75) contains the real-game
controller parity verifier at `2529a57`, based on main `1663a23`.
All six CI checks pass. It remains open with no review as of this snapshot.
The documented 1800-tick regression matched 600 decisions, 74 actions and
simulation hash 1678891593. Both independent replay checks and audits passed.
This is a truncated regression, not a completed-match win-rate result.

## Server-side progress not included in this branch

- SFT port `fa14b66` on `codex/sft-numerics-current-main`: 233 CPU tests passed
  with four GPU tests deselected. GPU reproduction and at least three epochs
  of CUDA training are still required; the epoch-2 NaN is not declared fixed.
- Issue #45 review at `7f2bd8a`: 38 decision-repository paths inspected,
  54 pinned GitHub sources, five calibration studies, and a census of 538
  deduplicated catalog entries. Framework screening, training comparisons
  and the ranked shortlist are incomplete. These review files are absent
  from the local backup and cannot be published from this snapshot.
- Four games sharing one CPU model/server were confirmed alive at elapsed
  49:20 (seed 7/8, red/blue, 1800 ticks each). Subsequent SSH connections
  timed out and then were refused. Completion and failure are both unknown;
  audits and independent replay verification remain pending.

## Remaining issue work

- [#50](https://github.com/HectorHHZ/llm-honing/issues/50): finish current-main
  in-game checks, including shared serving, training/gate and duel evidence.
- [#54](https://github.com/HectorHHZ/llm-honing/issues/54): the maintainer accepted
  seed parity and fake harness gate/rollback. GEPA integration awaits #24/#25.
- [#59](https://github.com/HectorHHZ/llm-honing/issues/59): document retirement
  and decide how once-only heldout use is recorded. No format change is made.
- [#45](https://github.com/HectorHHZ/llm-honing/issues/45): complete and publish
  the source-grounded review.

The full issue goal remains active. No issue is closed by this snapshot.
