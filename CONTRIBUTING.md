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
- **End-to-end** reports the CI `e2e` result and any run CI cannot do. When
  that run trains anything, either a harness evolving or model weights being
  post-trained, it also links the run's Weights & Biases (wandb) log. The log
  records every step (each iteration, or each training step) with its scores
  or losses, plus the commit and configuration the run used. Make the run
  readable by the maintainers, and put the charts a reviewer needs in the pull
  request as well, as #105 does. If the run could not log to wandb, say why
  and attach the same numbers some other way.
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

This section restates the pattern list from
[Humanizer](https://github.com/blader/humanizer) (MIT), whose patterns come
from Wikipedia's
[Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing),
maintained by WikiProject AI Cleanup. One idea explains nearly all of them. A
language model writes whatever is most likely to come next, so it makes the
choice that fits the widest range of readers and subjects; a person writes for
one reader and one subject. Every pattern below is a form of that default
choice. Cite them in review by number, as in "§1" or "§19".

Two rules follow. Every sentence you keep must add something the reader did
not already have, and a tell counts in proportion to how rarely a careful
writer would make it on purpose. The patterns are ordered by strength and
frequency: §1 to §5 justify an edit on a single sighting, and one marked
*weak alone* counts only when several tells share a passage.

### How to revise

1. **Mark the tells.** Read the whole text once and mark every pattern,
   strongest first. Look at paragraph shape as well as sentences: a contrast
   split across two sentences, three parallel examples, or the same closer
   after every section is the same tell at a larger scale.
2. **Draft the rewrite.** Keep every supported claim. Shorten, merge, split,
   and restructure freely, but do not add a fact, name, number, date, quote,
   citation, or benchmark result unless it comes from a source you have. When
   a sentence needs a detail you do not have, ask for it or write the simpler
   sentence.
3. **Check the draft.** Ask whether the rewrite added or dropped any fact,
   ranking, or simultaneity claim; the shape edits under §6, §8, and §19 drop
   those most often. Then search for the five tells that most often survive a
   rewrite: a not-X-but-Y contrast, a one-line closer, a dash, a triad, and a
   bold label.
4. **Write the final version.** State each point naturally instead of patching
   flagged phrases one at a time. If a sentence stays awkward, rewrite the
   paragraph around its main point.

### A. Staging instead of stating (§1 to §5)

The strongest and most frequent tells. Act on one sighting.

| # | Pattern | Watch for | Write instead |
| --- | --- | --- | --- |
| 1 | Not X but Y | "not just / not only / not merely X, but Y"; "it's not X, it's Y"; "X rather than Y"; the contrast split across two sentences ("This does not mean X. It means Y."); a clipped negative tail (", no guessing") | The point, stated directly. Keep a contrast only when the negative half corrects a belief the reader actually holds, or when both halves carry information |
| 2 | One-line closers and dramatic fragments | A one-sentence paragraph restating the one before it; "That is the real win."; "Read that again."; the same closer after several sections; a row of fragments ("No prior. No nostalgia."); one word in ALL CAPS or with periods between words | Cut a closer that repeats. Merge a row of fragments into one sentence with a specific claim. A short sentence is fine when it carries a new fact |
| 3 | Sayings that sound deep | "the real question is", "at its core", "what really matters", "fundamentally", "the deeper issue", "X is the Y of Z", "the language of", "the currency of", "the architecture of" | The specific claim the saying is dressing up |
| 4 | Staged run-up before the point | "Let's dive in", "let's break this down", "here's what you need to know", "without further ado", "Honestly?", "Look,", "Here's the thing", "Real talk" | Remove the run-up and make the point. "Honestly" inside a casual sentence is ordinary; the tell is the standalone opener before a routine claim |
| 5 | Arguing with no one | "This isn't (mainly) about", "I'm not saying", "To be clear", "Don't get me wrong", "Some might say... but", "A tempting approach would be", "You might think... but" | Delete the defense; if it holds a real claim, state the claim. Keep an objection the text answers in full, and an option a reader would actually weigh |

### B. Rhythm by rule (§6 to §11)

A person may do any one of these on purpose, so the weaker ones need company.

| # | Pattern | Watch for | Write instead |
| --- | --- | --- | --- |
| 6 | Forced triads | Three items in a sentence ("innovation, inspiration, and insights"); three parallel examples; three short facts followed by a lesson | The number of items the meaning needs. Merge the examples or develop the strongest one. Keep three when three are real |
| 7 | Repeated sentence openings | Several sentences in a row starting with the same subject, often a pronoun | Merge the sentences, change the subject, or begin with the action. Repetition used deliberately for rhythm is not a tell |
| 8 | Dashes as the universal connector (*weak alone*) | Em dash (—), en dash (–), spaced dash, or double hyphen (` -- `) used as a connector | A period, comma, colon, or parentheses, or a rewritten sentence. Leave dashes and hyphens inside code, inline code, commands, paths, and URLs alone |
| 9 | Stacked qualifiers (*weak alone*) | "could potentially", "might arguably", "it's also possible", "in some cases it may", "to be fair" | Only the qualifiers the source supports. Keep scope statements, safety notices, and real corrections; "perhaps" and "tends to" are human habits |
| 10 | Hyphenated pairs everywhere (*weak alone*) | third-party, cross-functional, data-driven, real-time, long-term, end-to-end, hyphenated in every position | The hyphen only where grammar needs it: "a high-quality report" but "the report is high quality" |
| 11 | Passive voice and missing subjects (*weak alone*) | "No configuration file needed"; "The results are preserved automatically" | Active voice when it makes the actor and the action clearer |

### C. Inflation and borrowed authority (§12 to §18)

The fact underneath is usually sound. Keep it and remove the dressing.

| # | Pattern | Watch for | Write instead |
| --- | --- | --- | --- |
| 12 | Overused AI words | actually, additionally, align with, bolstered, crucial, deep dive, delve, emphasizing, enduring, enhance, fostering, garner, highlight (verb), interplay, intricate, key (adjective), landscape (abstract), meticulous, pivotal, quietly, showcase, tapestry, testament, underscore (verb), valuable, vibrant | Plain words. This is the only vocabulary list; a formal word outside it is not a tell by itself. Figurative *gate* and *robust* count, but their technical senses here (the release gate, a robust estimator) do not |
| 13 | Inflated significance | "stands as a testament", "a pivotal moment", "plays a key role", "underscores its importance", "reflects a broader", "evolving landscape"; stock "Challenges and Legacy" or "Future Outlook" sections; "the future looks bright", "a step in the right direction" | The fact, without the significance. End on the last concrete fact; if the source states real plans, use those |
| 14 | Vague connection or association | "associated with", "in association with", "connected to", "in connection with", "linked to", "tied to" | The relationship the source actually gives. If the source does not say, keep the vague wording rather than inventing a role |
| 15 | Shallow -ing riders | "highlighting", "underscoring", "emphasizing", "ensuring", "reflecting", "symbolizing", "contributing to", "showcasing" bolted onto a simple fact | The fact alone. Keep the rider only when the source supports what it claims |
| 16 | Sales language | boasts, vibrant, rich (figurative), profound, exemplifies, commitment to, nestled, in the heart of, groundbreaking, renowned, diverse array, breathtaking, must-visit, stunning | What the thing is |
| 17 | Borrowed authority | "experts argue", "observers have cited", "industry reports", "some critics"; a list of prestige outlets; follower counts | The named source and what it said, or nothing. Never invent a source. A missing citation alone is not a tell; most writing is unsourced |
| 18 | Avoiding is, are, and has | serves as, stands as, functions as, operates as, marks, represents; boasts, features, offers, maintains; refers to | is, are, has |

### D. Formatting by rule (§19 to §21)

Templates and editors also produce clean formatting. The tell is decoration on
every item.

| # | Pattern | Watch for | Write instead |
| --- | --- | --- | --- |
| 19 | Bold as decoration | Bold applied with no reason; a vertical list where every item gets a bold label and a colon | No bold. Turn a labeled list into prose when the labels carry no information of their own; keep a label that names a distinct thing |
| 20 | Decorative headings | Title Case On Every Main Word; emoji or arrows (→) in headings and list items; a horizontal rule between every section; a top-level heading repeating the document's own title | Sentence case, no decoration, no rules, the title stated once |
| 21 | Curly quotation marks (*weak alone*) | Curly quotes where the writer or the target format uses straight ones | Straight quotes. Most editors auto-curl, so this is weak by itself |

### E. Leftovers from the chat and the draft (§22 to §25)

Remove these outright. Nothing here needs rewriting.

| # | Pattern | Watch for | Write instead |
| --- | --- | --- | --- |
| 22 | Chatbot residue | "Great question!", "Of course!", "Certainly!", "You're absolutely right", "I hope this helps", "Would you like...", "Should I continue?", "let me know", "here is a..." | The content, with the wrapper deleted. This is the most certain tell in the list and the easiest to miss when it wraps real content |
| 23 | Knowledge-limit disclaimers and guesses | "as of [date]", "up to my last training update", "while specific details are limited", "based on available information", "not widely documented", "it is believed that", "likely began" | What the source does or does not show, or nothing. Never present a guess as a fact |
| 24 | A heading repeated in the first sentence | A heading followed by a one-line paragraph that restates it before the real content begins | The content. Let the heading do the work |
| 25 | Writing about the previous version | "This function was added to replace the previous approach, which caused O(n²)" | What the code does now. The previous version belongs in commit messages, change logs, release notes, and migration guides |

### What this means here

Three of the patterns land on this repository harder than the rest:

- **§25 governs docstrings and comments.** A docstring says what the object
  does and what invariant it maintains now. History belongs in the commit
  message, and a rule that was reversed belongs in the RFC that reversed it.
- **§17 and step 2 are the Verification rule.** A claim that checks pass,
  without the command and its real output, is borrowed authority. The pull
  request template asks for the output for this reason.
- **§12 has a local exception.** *Gate*, *robust*, and *key* are ordinary
  technical terms in this codebase. The watch list targets the figurative use,
  not `honing/release/gate.py`.

Do not sand off the writing in the process. Keep the specific detail, the
unresolved tension, the first-person choice you can explain, and the aside
that carries real information. Leave a watched phrase alone inside a
quotation, a title, a proper name, or a passage discussing the phrase rather
than using it. Removing tells is half the job; the result still has to sound
like a person wrote it.

### Using the skill

Humanizer is also installable, so an agent can apply the list for you. In
Claude Code:

```text
/plugin marketplace add blader/humanizer
/plugin install humanizer@humanizer
```

It answers to `/humanizer:humanizer`. Paste a passage and it returns a draft,
the tells that survived, and a final version; name a file and it rewrites the
prose in place, leaving code blocks, commands, paths, YAML metadata, and link
targets untouched. Give it a few paragraphs of your own writing first if you
want the result to sound like you rather than like no one.

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
