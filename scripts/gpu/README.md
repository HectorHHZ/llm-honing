# The GPU check

CI on GitHub's runners has no GPU. A change to model or engine code (the paths in `paths.txt`) also needs
`check.sh` to pass on a GPU machine before it merges: the tests marked `gpu`, and the tank direction edit
(`configs/e2e/tank_direction_edit.json`) on a real NanoJev, whose train move must publish.

## Asking for it

A reviewer with write access comments `/gpu` on the pull request. The `gpu` workflow labels it `gpu-queued`,
and the commit status `gpu` reads "Queued". The runner then sets it to running and to the result, and comments
the summary. A push after that sets the status back to pending, so the check that merges ran on the commit that
merges. Only branches of this repository run, never forks, since the runner executes the pull request's code.

## The runner

Any machine with three free GPUs can be one. It polls for `gpu-queued` pull requests, and for each:

1. checks that the latest `/gpu` comment's author has write access and the head is a branch of this repository,
2. moves the label to `gpu-running` and sets the status to running,
3. downloads the head commit as a tarball (no credentials on the GPU machine), then runs
   `scripts/gpu/setup.sh <env-dir>`, sources `<env-dir>/env.sh` and runs `scripts/gpu/check.sh <out-dir> <gpu> <gpu> <gpu>`,
4. sets the status to success or failure, comments `summary.md`, and removes the label.

A pull request older than these scripts runs `main`'s copy of them. Today the runner is Xuan's, on Orchard H100s.

## By hand

```bash
scripts/gpu/setup.sh ~/honing-gpu
source ~/honing-gpu/env.sh
scripts/gpu/check.sh runs/gpu-check 0 1 2
```
