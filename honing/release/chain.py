"""The ordered releases of one campaign, and the commit record written ahead of every change.

The record is an append-only JSON-lines file. Each operation (publish, reject,
skip, rollback) is written and flushed to disk before the in-memory head moves,
so a crash between the two leaves a record describing a change that did not
take effect, never a head that points at something no record describes.

Every value in the record must be one JSON can represent exactly. A value of a
type the encoder does not know is an error, not a string: a record that quietly
stringified a set could not restore it on rollback.

Recovering a chain from its record is not implemented yet; the ordering that
makes recovery possible is.
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from honing.core import (
    CandidateKind,
    ContractError,
    HoneRelease,
    ReleaseError,
    SelectionDecision,
    UpdateCandidate,
    validate_method_state,
)


@dataclass(frozen=True)
class CurvePoint:
    """One point of the honing curve: a release's size and metrics."""

    release_id: str
    kind: CandidateKind
    parameters: int
    metrics: Mapping[str, float]


_SCALARS = (str, int, float, bool)


def _json_default(value: object) -> object:
    """Encode the few non-JSON types the record holds, and refuse everything else."""
    if isinstance(value, MappingProxyType):
        return dict(value)
    if isinstance(value, (set, frozenset)) and all(isinstance(item, _SCALARS) for item in value):
        # Sorted so the same set always writes the same line.
        return sorted(value, key=lambda item: (type(item).__name__, item))
    raise TypeError(f"the commit record cannot hold a {type(value).__name__}: {value!r}")


def _candidate_record(candidate: UpdateCandidate) -> dict[str, object]:
    return dataclasses.asdict(candidate)


class ReleaseChain:
    """Publishes, rejects, and rolls back releases, recording each operation first.

    Invariant: every change to the head is preceded by a flushed record of it.
    """

    RECORD_FILE_NAME = "commits.jsonl"

    def __init__(self, work_dir: Path) -> None:
        work_dir.mkdir(parents=True, exist_ok=True)
        self._record_path = work_dir / self.RECORD_FILE_NAME
        self._releases: dict[str, HoneRelease] = {}
        self._order: list[str] = []
        self._head: str | None = None

    @property
    def record_path(self) -> Path:
        return self._record_path

    def initialize(self, candidate: UpdateCandidate) -> HoneRelease:
        """Publish release 0, the model a campaign starts from."""
        if self._order:
            raise ReleaseError("the chain already has an initial release")
        release = HoneRelease(release_id=self._next_id(), candidate=candidate, decision=None)
        self._publish(release)
        return release

    def commit(self, candidate: UpdateCandidate, decision: SelectionDecision) -> HoneRelease:
        """Publish a selected candidate as the new head."""
        head = self.head()
        if candidate.parent_id != head.release_id:
            raise ReleaseError(
                f"candidate was built on {candidate.parent_id}, but the head is now {head.release_id}; rebuild it"
            )
        if not decision.selected:
            raise ReleaseError(f"a rejected candidate cannot be committed: {decision.reason}")
        release = HoneRelease(release_id=self._next_id(), candidate=candidate, decision=decision)
        self._publish(release)
        return release

    def record_rejection(self, candidate: UpdateCandidate, decision: SelectionDecision) -> None:
        """Record a rejected candidate. The head does not move."""
        self._append_record(
            {"op": "reject", "candidate": _candidate_record(candidate), "decision": dataclasses.asdict(decision)}
        )

    def record_skip(
        self,
        *,
        parent_id: str,
        method: str,
        reason: str,
        state: Mapping[str, object],
        episodes_spent: int,
        label: str | None = None,
    ) -> None:
        """Record a harness attempt that proposed nothing. The head does not move.

        The attempt may still have run episodes and advanced the method, so its
        state and the episodes it spent are recorded the way a rejection's are.
        """
        if parent_id != self.head().release_id:
            raise ReleaseError(f"the attempt ran on {parent_id}, but the head is now {self.head().release_id}")
        if episodes_spent < 0:
            raise ContractError(f"episodes_spent must be >= 0, got {episodes_spent}")
        validate_method_state(state)
        self._append_record(
            {
                "op": "skip",
                "parent_id": parent_id,
                "method": method,
                "label": label,
                "reason": reason,
                "episodes_spent": episodes_spent,
                "method_state": dict(state),
            }
        )

    def rollback(self, release_id: str) -> HoneRelease:
        """Make an earlier release the head again, restoring every part of it together."""
        target = self.get(release_id)
        self._append_record({"op": "rollback", "from": self.head().release_id, "to": release_id})
        self._move_head(release_id)
        return target

    def head(self) -> HoneRelease:
        if self._head is None:
            raise ReleaseError("the chain has no releases yet; initialize it first")
        return self._releases[self._head]

    def get(self, release_id: str) -> HoneRelease:
        try:
            return self._releases[release_id]
        except KeyError:
            raise ReleaseError(f"no release {release_id!r} in this chain") from None

    def releases(self) -> tuple[HoneRelease, ...]:
        """Every published release, in publication order. Rollback does not remove any."""
        return tuple(self._releases[release_id] for release_id in self._order)

    def curve(self) -> tuple[CurvePoint, ...]:
        return tuple(
            CurvePoint(
                release_id=release.release_id,
                kind=release.kind,
                parameters=release.master.parameters,
                metrics=release.metrics,
            )
            for release in self.releases()
        )

    def _publish(self, release: HoneRelease) -> None:
        self._append_record(
            {
                "op": "publish",
                "release_id": release.release_id,
                "candidate": _candidate_record(release.candidate),
                "decision": None if release.decision is None else dataclasses.asdict(release.decision),
            }
        )
        self._releases[release.release_id] = release
        self._order.append(release.release_id)
        self._move_head(release.release_id)

    def _append_record(self, record: Mapping[str, object]) -> None:
        line = json.dumps(record, sort_keys=True, default=_json_default)
        with self._record_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _move_head(self, release_id: str) -> None:
        self._head = release_id

    def _next_id(self) -> str:
        return f"r{len(self._order):04d}"
