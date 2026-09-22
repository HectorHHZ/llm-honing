# Contributing to Honing

Thank you for helping build Honing. Contributions can include bug reports,
documentation, tests, benchmarks, code, design proposals, and reviews.

The specification in [`system-design/`](system-design/) is the contract
implementations are built against. The core contracts and a walking skeleton
that runs a campaign across every plane with fakes exist today; each plane is
made real by replacing its fake while the skeleton keeps passing. A change to
behavior and a change to the specification are reviewed with the same
seriousness.

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
verb: `[Feature] Add expert-drop scoring over router activation counts`. Do not
add a second bracketed tag for an area or status — labels carry those
dimensions:

- `area: core` — contracts, gate, selectors, release chain
- `area: compression` — pruning scorers, selectors, and editors; quantization; plans
- `area: train` — algorithms, update policies, numerics, and training backends
- `area: data` — data providers, contracts, and dataset manifests
- `area: harness` — harness specs, episode runners, and evolution methods
- `area: campaign` — scheduling and the ordering rules every move passes
- `area: serve` — serving, runtimes, and telemetry collection
- `area: docs` — design specification and user documentation
- `area: ci` — automation, CI, and developer tooling

Security vulnerabilities do not belong in public issues. Follow
[SECURITY.md](SECURITY.md).

## Changes that need an RFC

Open an RFC issue before implementing a change that:

- adds a new top-level package under `honing`;
- adds or changes a protocol in `honing/core/protocols.py` (`ModelSource`,
  `DataProvider`, `RolloutProvider`, `Trainer`, `ImportanceScorer`,
  `PruneSelector`, `ModelEditor`, `Quantizer`, `EpisodeRunner`,
  `EvolveMethod`, `CoAdaptScheduler`, and the rest);
- changes a persisted format, a wire contract, or the shape of `HoneRelease`;
- adds a training-framework dependency, or imports one outside `honing/backends/`;
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

Scopes are the plane or area the change belongs to: `core`, `compression`,
`train`, `data`, `harness`, `campaign`, `serve`, `design` (the specification
under `system-design/`), `docs`, and `ci`. Omit the scope when a change is
genuinely repository-wide.

```
feat(compression): add expert-drop scoring over router activation counts
fix(train): reject a sparse-mask policy whose selection spans pruned experts
feat(campaign): refuse a quantize step whose scheme differs from qat
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
`feat/expert-drop` or `docs/compression-plane-contract`. The pull request title
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

Generated prose carries its own tells. [Writing style](#writing-style) lists
the ones to remove before you submit.

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

The base install and the full test suite need no GPU and no training framework.
Slime, the one training dependency, is pinned by commit in the `runtime`
dependency group and is installed only inside the supported GPU image:

```bash
uv pip install --no-deps --group runtime
```

`--no-deps` keeps the image's CUDA-specific torch, SGLang, and Megatron in place.

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
  in `honing/core/protocols.py`, not a bag of callables. State the invariant a
  class maintains in its docstring.
- **Respect the dependency boundary.** `honing/core` imports only the standard
  library, planes depend on `honing.core` and never on each other, and only
  `honing/backends/` may import a training framework. `tests/contracts/` checks
  all three.
- **Refuse early.** A combination that cannot work is a `ContractError` at
  construction or a `CampaignRuleError` before any weights are touched, never a
  failure hours into a job.
- **Annotate new and changed interfaces.** Do not reach for `Any` to avoid
  describing a type unless the boundary is genuinely dynamic.
- **Comment the *why*.** A comment that restates the code earns nothing; a
  comment recording why an ordering or a rejection is necessary earns its place.
- **Name for readers.** Include units or representation where ambiguity could
  cause a bug: `timeout_seconds`, `keep_ratio`, `checkpoint_path`. Prefer plain
  terms such as manifest, record, and metadata over coined vocabulary.

## Writing style

Prose here is read by someone deciding something: whether a bug is real,
whether a design holds, whether a pull request is safe to merge. Issue and
pull request bodies, docstrings, comments, and the documents under
[`system-design/`](system-design/) are held to the standard code is.

The checklist below is condensed from
[Humanizer](https://github.com/blader/humanizer) (MIT), an agent skill whose
pattern list comes from Wikipedia's
[Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing).
One idea explains nearly all of it. A language model writes whatever is most
likely to come next, so it makes the choice that fits the widest range of
readers and subjects; a person writes for one reader and one subject. Each
family below is a form of that default choice.

| Family | What it looks like | What to write instead |
| --- | --- | --- |
| Staging instead of stating | "It's not just X, it's Y"; a one-line closer after every section; "At its core"; "Let's dive in"; rebutting an objection nobody raised | The claim itself, once |
| Rhythm by rule | Three items where the meaning has two; several sentences opening the same way; dashes used as the universal connector | The number of items the meaning needs; periods, commas, colons, parentheses |
| Inflation and borrowed authority | "a pivotal moment"; "delve", "robust", "landscape", "underscore"; "experts believe"; "serves as" instead of "is" | The fact, the named source, the plain verb |
| Formatting by rule | Bold on every list label; Title Case headings; emoji and arrows | Sentence case, prose, bold only where it carries information |
| Leftovers | "Great question!", "I hope this helps"; "as of my last update"; a heading restated in its first sentence | Delete them; nothing here needs rewriting |

Five of these justify an edit the first time you see them: the not-X-but-Y
contrast, the one-line closer, the deep-sounding saying, the staged run-up,
and the argument with no one. The weaker signals, a dash, a stacked qualifier,
a hyphenated pair, a passive sentence, a curly quote, count only when several
share a passage, because a careful writer uses any one of them on purpose.

Two of the patterns hit this repository more than most prose:

- **Do not describe the previous version.** A docstring or comment says what
  the code does now. "This replaces the old loop that was O(n²)" belongs in
  the commit message or the release notes, not next to the function.
- **Do not invent a supporting fact.** A name, number, date, quote, citation,
  or benchmark result must come from a source you have. When a sentence needs
  a detail you do not have, ask for it or write the simpler sentence. This is
  the same rule the pull request template states for **Verification**: a claim
  that tests pass, without the command and its real output, is not evidence.

Do not sand off the writing in the process. Keep the specific detail, the
unresolved tension, the first-person choice you can explain, and the aside
that carries real information. Leave a watched phrase alone inside a
quotation, a title, a proper name, or a passage discussing the phrase rather
than using it. Removing tells is half the job; the result still has to sound
like a person wrote it.

The skill itself is worth installing if you draft with an agent. In Claude
Code:

```text
/plugin marketplace add blader/humanizer
/plugin install humanizer@humanizer
```

It answers to `/humanizer:humanizer`, rewrites a pasted passage or a named
file, changes prose only, and leaves code blocks, commands, paths, YAML
metadata, and link targets untouched. Give it a few paragraphs of your own
writing first if you want the result to sound like you.

## Tests

Every behavior change needs a test, and every public protocol needs a fake in
`tests/fakes/` and an entry in `tests/contracts/test_core_protocols.py`. A new
plane feature should keep `tests/test_walking_skeleton.py` passing; replacing a
fake with a real implementation is how a plane becomes real. Tests carry one of
the markers declared in `pyproject.toml`:

| Marker | Meaning |
| --- | --- |
| `unit` | No I/O beyond a temporary directory; runs everywhere; the default |
| `integration` | Multiple components together; may start subprocesses |
| `gpu` | Requires an accelerator; skipped in CI |
| `slow` | Takes more than roughly ten seconds |

CI runs `pytest -m "not gpu"`. Keep the unit suite fast enough that running it
before every commit stays habitual.
