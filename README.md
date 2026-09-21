# Honing

**Continual specialization infra: a large model that sheds its own dead weight while it learns.**

Honing is post-training infrastructure that turns a large general model into a
leaner domain-specialized one *over time*. As the model serves real traffic and
real data, it keeps the capacity that carries the work, removes what it no
longer needs, and publishes each result as a versioned, rollback-able release.
The model gets better in-domain while getting cheaper to run.

> **Status: skeleton.** The contracts every plane is built against
> (`honing/core`), the gate and release chain (`honing/release`), and the
> campaign with its ordering rules (`honing/campaign`) are implemented. A
> walking skeleton runs a full campaign — train, prune with recovery, quantize,
> evolve the harness, roll back — with every other plane faked on a CPU. Planes
> become real one at a time; see the implementation steps in the design document.

## Start here

| Document | What it covers |
| --- | --- |
| [`system-design/README.md`](system-design/README.md) | What Honing is, the planes, its dependency on Slime, and how it compares to inference engines, RL frameworks, and Reef |
| [`system-design/system-design.md`](system-design/system-design.md) | Subsystem contracts, the coupling model, the ordering rules, worked lifecycles, and the implementation steps |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Issue, commit, and pull request conventions; development setup |

![Honing system design](system-design/honing_architecture.png)

## The planes

| Plane | Owns |
| --- | --- |
| **Core** | The value types and protocols every plane is built against; imports only the standard library |
| **Model Source** | Where the initial model comes from — an open checkpoint, or a large model compressed first |
| **Data** | Online feedback and offline datasets, behind one batch type |
| **Train** | Algorithm, update policy, numerics, rollout source, and teacher as independent choices |
| **Compression** | Pruning (score, select, edit) and quantization; usable standalone |
| **Harness Evolve** | Harness specs, episode runs, and evolution methods before, during, and after training |
| **Campaign** | Which move comes next, and the rules each move must pass before it touches weights |
| **Serve** | Live traffic and artifact activation (reserved) |

The idea that keeps this small enough to build:

> **Every step is an update candidate.** A training step, a prune, a
> quantization, and a harness change all pass the same gate and settle into the
> same release chain. A bad prune is a rejected candidate or a rolled-back
> release, not a new failure mode.

## Dependencies

[Slime](https://github.com/THUDM/slime) is the one training-framework
dependency, pinned by commit and imported only under `honing/backends/`. Reef
informs the design but is not a dependency. The reasoning, and what Slime does
and does not supply, is in [`system-design/README.md`](system-design/README.md#dependencies).

## Development

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install

pre-commit run --all-files
pytest
```

No GPU is needed for the test suite.

## License

[Apache-2.0](LICENSE)
