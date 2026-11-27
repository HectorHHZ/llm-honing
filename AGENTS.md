# Instructions for coding agents

These apply to every change an AI coding agent writes for llm-honing, whoever runs the agent.
CONTRIBUTING.md is the full rulebook. This file is the part an agent gets wrong most often.

## Before you start

- Check that nobody is already on it:
  ```bash
  gh issue view <number> --repo HectorHHZ/llm-honing --comments
  gh pr list --repo HectorHHZ/llm-honing --state open --search "<number> in:body"
  gh pr list --repo HectorHHZ/llm-honing --state open --search "<area keywords>"
  ```
  If an open pull request does the same thing, do not open another. If your approach differs,
  say how in the issue first.
- A change to a plane protocol, a persisted format or the HoneRelease shape needs an accepted RFC
  (CONTRIBUTING.md, Changes that need an RFC).
- No busywork: a lone typo, a style-only edit or an unrequested refactor is not a pull request.
- If you are unsure the change is wanted or right, stop and say what is missing instead of guessing.

## While you work

- Set up with uv: `uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"`, then
  `pre-commit install`.
- Every test declares exactly one kind: `unit`, `functional`, `integration` or `gpu`
  (CONTRIBUTING.md, Tests). A `unit` test that starts a process or opens a socket fails. A new
  feature needs unit tests and a functional test through its entry point, and CI fails a pull
  request whose changed Python lines are less than 80 percent covered.
- A public protocol needs a fake in `tests/fakes/` and a contract test in `tests/contracts/`.
- Run `pytest -m unit` before every commit and `pytest -m "not gpu"` before every push.
- Code that needs a GPU gets `gpu` tests. CI has no GPU: the GPU check runs them on a GPU machine
  after a maintainer comments `/gpu` on the pull request (scripts/gpu/README.md).

## Commits and the pull request

- Every commit subject is a Conventional Commit, checked in CI per commit:
  `python .github/scripts/check_commit_message.py --subject "<subject>"`.
- Update a branch by rebasing onto `main` and pushing with `--force-with-lease`. Never merge `main`
  into it: CI fails a pull request that contains a merge commit.
- Fill in every section of the pull request template. Verification lists the commands that ran,
  their exact summary lines (the pytest line with its time) and the commit they ran on.
- Fill in the AI assistance section. The human who opens the pull request has read
  every changed line and can defend it.
- Prose (docs, comments, pull request text) follows the Writing style section of CONTRIBUTING.md.
