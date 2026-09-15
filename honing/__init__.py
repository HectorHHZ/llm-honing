"""Honing: continual specialization infra for large models.

Honing turns a large general model into a leaner domain-specialized one over
time. As the model serves real traffic and real data, the system keeps the
capacity that carries the work, removes what it no longer needs, and publishes
each result as a versioned, rollback-able release.

The design specification lives in ``system-design/`` at the repository root and
is the contract this package is built against. The planes it defines:

``prune``
    Structural capacity reduction as an independent system: pruning methods,
    declarative plans, and the manifest chain that records every cut.
``train``
    Training algorithm, gradient update policy, and rollout source as three
    orthogonal registries.
``data``
    Online feedback and offline datasets behind one batch type.
``harness``
    Composition search, run before, during, and after weight training.
``serve``
    Live traffic and artifact activation. Reserved; see phase 10 of the plan.

The one invariant that shapes the rest: a prune step *is* an update candidate,
so evaluation, publication, provenance, and rollback are shared with weight
training and harness evolution rather than reimplemented per plane.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("llm-honing")
except PackageNotFoundError:  # a source tree that was never installed
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
