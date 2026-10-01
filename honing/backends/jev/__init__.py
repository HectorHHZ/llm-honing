"""Jev-style decision models: typed choices with probabilities, served over HTTP.

A decision model answers typed questions about a state with a probability for
every option. :mod:`honing.backends.jev.laya` loads the Laya family and
:mod:`honing.backends.jev.kev` and :mod:`honing.backends.jev.nanojev` the Kev
and NanoJev families; :mod:`honing.backends.jev.djev` bridges a remote
DiffusionGemma structured-read server;
:mod:`honing.backends.jev.server` serves any :class:`DecisionModel` over HTTP,
and :class:`JevEndpoints` lets a runner find the server for a policy.
"""

from honing.backends.jev.endpoints import JevEndpoints
from honing.backends.jev.model import DecisionModel

__all__ = ["DecisionModel", "JevEndpoints"]
