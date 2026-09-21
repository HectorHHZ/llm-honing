"""Honing: continual specialization infra for large models.

Honing turns a large general model into a leaner domain-specialized one over
time. As the model serves real traffic and real data, the system keeps the
capacity that carries the work, removes what it no longer needs, and publishes
each result as a versioned, rollback-able release.

The design specification lives in ``system-design/`` at the repository root and
is the contract this package is built against. The planes it defines:

``core``
    The contracts every plane is built against: value types and protocols,
    importing nothing outside the standard library (``honing.core``).
``compression``
    Structural pruning (score, select, edit) and post-training quantization,
    with the plan each release records.
``train``
    Algorithm, update policy, numerics, rollout source, and teacher as
    independent choices. Slime runs most algorithms; DPO runs in a local trainer.
``data``
    Online feedback and offline datasets behind one batch type.
``harness``
    Harness specs, the episode runner, and pluggable evolution methods, run
    before, during, and after weight training.
``campaign``
    The scheduler that picks the next move and the ordering rules every move
    must pass (``honing.campaign``).
``serve``
    Live traffic and artifact activation. Reserved.

Every plane produces the same update candidate, so one gate and one release
chain (``honing.release``) evaluate, publish, and roll back weights, pruning,
quantization, and harness changes alike.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("llm-honing")
except PackageNotFoundError:  # a source tree that was never installed
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
