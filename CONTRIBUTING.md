# Contributing to Honing

Thank you for helping build Honing. Contributions can include bug reports,
documentation, tests, benchmarks, code, design proposals, and reviews.

Honing is currently at the design stage: the specification in
[`system-design/`](system-design/) is the contract implementations are built
against. A change to behavior and a change to that specification are reviewed
with the same seriousness.

## Before you start

- Search existing issues and pull requests before opening a new one.
- Keep each issue and pull request focused on one problem or proposal.
- For a bug, include the smallest complete reproduction and your environment.
- When the direction or scope of a change is uncertain, open an issue before
  investing in an implementation.
- Never include credentials, private data, model-provider tokens, customer
  data, or other secrets in an issue, pull request, log, fixture, or transcript.

## Issues

Use the template that matches the work. An issue title carries exactly one
type prefix:

| Prefix | Use it for |
| --- | --- |
| `[Bug]` | Reproducible incorrect behavior |
| `[Feature]` | A new capability, algorithm, or integration |
| `[Task]` | Scoped refactoring, cleanup, documentation, CI, or project work |
| `[Experiment]` | A benchmark, reproduction, or empirical evaluation |
| `[RFC]` | A durable architecture, interface, or policy decision |

Write the rest of the title as a concise outcome, normally beginning with a
verb: `[Feature] Add expert-drop pruning method`. Do not add a second bracketed
tag for an area or status — labels carry those dimensions:

- `area: prune` — pruning methods, plans, manifests, and the standalone CLI
- `area: train` — algorithms, update policies, and training backends
- `area: data` — data providers, contracts, and dataset manifests
- `area: harness` — harness evolution and co-adaptation phases
- `area: serve` — serving, runtimes, and telemetry collection
- `area: core` — scenario lifecycle, gate, commit protocol, release chain
- `area: docs` — design specification and user documentation
- `area: ci` — automation, CI, and developer tooling

Security vulnerabilities do not belong in public issues. Follow
[SECURITY.md](SECURITY.md).

## Changes that need an RFC

Open an RFC issue before implementing a change that:

- adds a new top-level package under `honing`;
- adds or changes a plane's public protocol (`PruneMethod`, `DataProvider`,
  `UpdatePolicy`, `RolloutProvider`, `ModelSource`);
- changes a persisted format, wire contract, or the shape of `HoneRelease`;
- alters one of the five design principles in `system-design/system-design.md`; or
- establishes a project-wide policy that will constrain later work.

The issue *is* the RFC and its decision record. It must be explicitly accepted
before its implementation is treated as approved direction. Bug fixes,
documentation, tests, behavior-preserving refactors, and implementation of an
accepted RFC do not need a new RFC.

## Commit messages

Honing follows [Conventional Commits](https://www.conventionalcommits.org/).
The subject line is what a reader sees in `git log`, so it states the outcome,
not the activity.

```
<type>(<scope>): <imperative summary, lowercase, no trailing period>

<body: why the change is necessary, what it does not do, and anything a
future reader would otherwise have to reconstruct>

<footer: Closes #N, BREAKING CHANGE: ...>
```

Types:

| Type | Use it for |
| --- | --- |
| `feat` | A new capability |
| `fix` | A correction to behavior |
| `docs` | Documentation and the design specification |
| `test` | Tests only |
| `refactor` | Behavior-preserving restructuring |
| `perf` | A measurable performance change |
| `build` | Packaging, dependencies, build configuration |
| `ci` | Workflows and developer automation |
| `chore` | Routine maintenance with no source effect |

Scopes are the plane or area the change belongs to: `prune`, `train`, `data`,
`harness`, `serve`, `core`, `design` (the specification under `system-design/`),
`docs`, and `ci`. Omit the scope when a change is genuinely repository-wide.

```
feat(prune): add expert-drop analysis over router activation counts
fix(train): reject a sparse-mask policy whose selection spans pruned experts
docs(design): record the co-adaptation invariant for harness releases
```

Rules the CI check enforces:

- the subject is at most 72 characters, imperative mood, and has no trailing period;
- a breaking change carries `!` after the scope **and** a `BREAKING CHANGE:` footer;
- the body wraps at 79 columns and explains *why*, not *what*.

Keep each commit a single logical change. Rebase to clean up work-in-progress
commits before requesting review; do not merge `main` into your branch.

Bot-authored pull requests are exempt from the subject check. A dependency
bot's title is written by the tool and its wording is not configurable, for the
same reason the checker exempts the subjects git itself generates.

## Pull requests

Branch from `main` and name the branch `<type>/<short-slug>`, e.g.
`feat/expert-drop` or `docs/prune-plane-contract`. The pull request title
follows the same Conventional Commits format as a commit subject, because the
merge commit is derived from it.

Fill in every section of the pull request template. In particular:

- **Verification** lists the commands that actually ran and their real results.
  A claim that tests pass without the command and its output is not review
  evidence.
- **Compatibility** describes changes to protocols, configuration, persisted
  data, or dependencies. Write "None" when unaffected.
- **AI assistance** names any tool that made a non-trivial contribution.

A pull request needs one maintainer approval and a green CI run to merge.
Merges use squash, so the pull request title becomes the commit subject.

## AI-assisted contributions

AI-assisted work is welcome when it is directed, understood, and verified by
the human contributor. Do not submit autonomous or bulk-generated issues, pull
requests, reviews, or comments.

When an AI tool makes a non-trivial contribution:

- disclose the tool and how it was used in the pull request;
- review and understand every submitted line and factual claim;
- reproduce the problem yourself instead of trusting a generated diagnosis;
- run the relevant checks and report their actual results; and
- remove speculative fixes, unrelated cleanup, and unnecessary abstractions.

AI assistance does not lower the bar for tests, documentation, security, or
long-term maintenance.

## Set up the repository

[uv](https://docs.astral.sh/uv/) is the supported workflow.

```bash
git clone https://github.com/HectorHHZ/llm-honing.git
cd llm-honing
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install
```

The usual local checks, which are also what CI runs:

```bash
pre-commit run --all-files
pytest
```

To regenerate the architecture diagram after editing its source:

```bash
uv run --no-project --with matplotlib python system-design/honing_architecture.py
```

The script writes the PNG next to itself, so it is safe to run from any
working directory.

## Python coding style

Follow [PEP 8](https://peps.python.org/pep-0008/) and the
[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
`pyproject.toml` is the authority for mechanical style: Black formats, isort
orders imports, Ruff lints, and mypy type-checks the `honing` package. Do not
hand-format against these tools.

Beyond mechanical style, the conventions that matter most here:

- **Make contracts explicit.** A plane's public surface is a named `Protocol`
  with meaningful methods, not a bag of callables. State the invariant a class
  maintains in its docstring.
- **Annotate new and changed interfaces.** Do not reach for `Any` to avoid
  describing a type unless the boundary is genuinely dynamic.
- **Comment the *why*.** A comment that restates the code earns nothing; a
  comment recording why an ordering or a rejection is necessary earns its place.
- **Keep I/O and framework integration at the edges,** so that core behavior is
  testable with ordinary Python objects.
- **Name for readers.** Include units or representation where ambiguity could
  cause a bug: `timeout_seconds`, `keep_ratio`, `checkpoint_path`.

## Tests

Every behavior change needs a test, and every public protocol needs a contract
test. Tests carry one of the markers declared in `pyproject.toml`:

| Marker | Meaning |
| --- | --- |
| `unit` | No I/O beyond a temporary directory; runs everywhere; the default |
| `integration` | Multiple components together; may start subprocesses |
| `gpu` | Requires an accelerator; skipped in CI |
| `slow` | Takes more than roughly ten seconds |

CI runs `pytest -m "not gpu"`. Keep the unit suite fast enough that running it
before every commit stays habitual.
