# Honing

**Continual specialization infra: a large model that sheds its own dead weight while it learns.**

Honing is post-training infrastructure that turns a large general model into a
leaner domain-specialized one *over time*. As the model serves real traffic and
real data, it keeps the capacity that carries the work, removes what it no
longer needs, and publishes each result as a versioned, rollback-able release.
The model gets better in-domain while getting cheaper to run.

> **Status: design stage.** The specification in [`system-design/`](system-design/)
> is complete and is the contract implementations are built against. Code lands
> plane by plane; see the implementation phases in the design document.

## Start here

| Document | What it covers |
| --- | --- |
| [`system-design/README.md`](system-design/README.md) | What Honing is, the five planes, and how it compares to inference engines, RL frameworks, and Reef |
| [`system-design/system-design.md`](system-design/system-design.md) | Subsystem contracts, the coupling model, worked lifecycles, HTTP surface, and the phased plan |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Issue, commit, and pull request conventions; development setup |

![Honing system design](system-design/honing_architecture.png)

## The five planes

| Plane | Owns |
| --- | --- |
| **Model Source** | Where the initial trainable model comes from — an open checkpoint, or a large model thinned first |
| **Data Plane** | Online feedback and offline datasets, behind one batch type |
| **Train Plane** | Algorithm × gradient update policy × rollout source, as three orthogonal registries |
| **Prune Plane** | Structural capacity reduction and its audit trail; usable standalone |
| **Harness Evolve** | Composition search, run before, during, and after weight training |
| **Serve Plane** | Live traffic and artifact activation (reserved) |

The idea that keeps this small enough to build:

> **A prune step *is* an update candidate.** Evaluation, publication,
> provenance, and rollback are shared across weight training, pruning, and
> harness evolution rather than reimplemented per plane. A bad prune is a
> rejected candidate or a rolled-back release, not a new failure mode.

## Development

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install

pre-commit run --all-files
pytest
```

## License

[Apache-2.0](LICENSE)
