"""DiffusionGemma structured decisions served by djev, in the Serve plane.

A djev deployment (:mod:`honing.backends.djev.runtime`) runs DiffusionGemma on a
patched vLLM behind the djev API, in djev's own pinned environment, one GPU per
deployment. :class:`DjevDecisionModel` puts it behind Honing's
:class:`~honing.backends.jev.model.DecisionModel` interface, so the decision
server serves it and runners reach it through ``PolicyEndpoints``. Honing talks
to the deployment over HTTP and to its environment through subprocesses only.
"""

from honing.backends.djev.decision import DjevClient, DjevDecisionModel, DjevSettings
from honing.backends.djev.runtime import DjevRuntimeSettings, djev_runtime, verify_environment

__all__ = [
    "DjevClient",
    "DjevDecisionModel",
    "DjevRuntimeSettings",
    "DjevSettings",
    "djev_runtime",
    "verify_environment",
]
