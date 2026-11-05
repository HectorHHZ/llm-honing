"""Jev-style decision models: typed choices with probabilities, served over HTTP.

A decision model answers typed questions about a state with a probability for
every option. :mod:`honing.backends.jev.laya` loads the Laya family and
:mod:`honing.backends.jev.kev`, :mod:`honing.backends.jev.nanojev`,
:mod:`honing.backends.jev.startlux` and :mod:`honing.backends.jev.nimble` the
Kev, NanoJev, StartLux-Decision and Nimble families; :mod:`honing.backends.jev.djev` bridges a remote
DiffusionGemma structured-read server;
:mod:`honing.backends.jev.server` serves any :class:`DecisionModel` over HTTP,
and :class:`JevEndpoints` lets a runner find the server for a policy;
:mod:`honing.backends.jev.nanojev_serving` starts a NanoJev server per
checkpoint on demand, and :mod:`honing.backends.jev.direction_edit` edits a
NanoJev head along a direction read from the model (the ``direction_edit``
algorithm), with no training. :mod:`honing.backends.jev.trainer`,
:mod:`honing.backends.jev.nanojev_trainer` and :mod:`honing.backends.jev.kev_trainer`
train the Laya, NanoJev and Kev families on typed decisions, with the examples,
validation split, losses and loops of :mod:`honing.backends.jev.decision_training`.
"""

from honing.backends.jev.endpoints import JevEndpoints
from honing.backends.jev.model import DecisionModel

__all__ = ["DecisionModel", "JevEndpoints"]
