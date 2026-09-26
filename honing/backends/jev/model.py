"""The surface every decision model offers to the server and the stdio worker."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable


@runtime_checkable
class DecisionModel(Protocol):
    """A loaded decision model.

    ``metadata`` describes exactly what is loaded (revision, dtype, device), and
    is returned by the server's health route and recorded by every run.
    ``predict`` answers every question about one state; it is called by one
    thread at a time.
    """

    @property
    def metadata(self) -> Mapping[str, object]: ...

    def predict(self, parts: Mapping[str, object], questions: Mapping[str, object]) -> Mapping[str, object]: ...
