"""Meta-Harness's search state, and the directory the proposer agent reads it from.

Upstream keeps its archive on disk: every candidate's source, a JSONL row per
evaluated candidate, a frontier file, and every trial's trajectory. Honing
publishes a method's state with each release and hands it back on rollback, so
the archive lives in JSON-native state instead, and :func:`write_archive`
renders it into upstream's layout before every proposal. A rollback therefore
rolls back the directory the proposer sees as well.

The layout written under the archive directory:

``candidates/<name>/``
    every file of the candidate's harness, plus ``harness.json``
    (``{"agent": ..., "files": {path: kind}}``) so a folder reads back as a spec.
``evolution_summary.jsonl``
    one row per scored candidate.
``frontier_val.json``
    the best candidate per task and ``_best``, the best mean, over scores taken
    under the current weights.
``traces/<name>/<task id, percent-quoted>/``
    the episode transcript copied from its location, and ``feedback.txt``.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import cast, get_args
from urllib.parse import quote

from honing.core import ContractError, HarnessFile, HarnessFileKind, HarnessSpec, require
from honing.core.state import METHOD_STATE_VERSION_KEY, method_state_version

ARCHIVE_SCHEMA_VERSION = 1

SEED_NAME = "seed"
MANIFEST = "harness.json"
CANDIDATES_DIR = "candidates"
TRACES_DIR = "traces"
SUMMARY_FILE = "evolution_summary.jsonl"
FRONTIER_FILE = "frontier_val.json"

_NAME = re.compile(r"[a-z0-9_]+")


def valid_name(name: object) -> bool:
    """A candidate name is a snake_case folder name, as upstream's ``agents/<name>.py``."""
    return isinstance(name, str) and _NAME.fullmatch(name) is not None


@dataclass(frozen=True)
class Trace:
    """Where one gate episode's transcript is, and the grader's feedback on it."""

    transcript: str | None = None
    feedback: str | None = None


@dataclass(frozen=True)
class Candidate:
    """One harness in the archive.

    ``scores`` follows the archive's task order and stays ``None`` until a gate
    scores the candidate; ``policy`` is the checksum of the weights they were
    taken under, and ``traces`` maps each task id to that gate's episode.
    """

    name: str
    parent: str | None
    files: tuple[tuple[str, HarnessFile], ...]
    hypothesis: str = ""
    changes: str = ""
    scores: tuple[float, ...] | None = None
    policy: str | None = None
    traces: Mapping[str, Trace] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require(valid_name(self.name), f"a candidate name must match [a-z0-9_]+, got {self.name!r}")
        require(all(path != MANIFEST for path, _ in self.files), f"a harness file may not be named {MANIFEST!r}")
        if self.scores is not None:
            require(
                all(0.0 <= score <= 1.0 for score in self.scores),
                f"candidate {self.name} has scores outside [0, 1]: {self.scores}",
            )
            require(self.policy is not None, f"candidate {self.name} has scores but no policy")

    def spec(self, agent: str) -> HarnessSpec:
        return HarnessSpec(agent=agent, files=self.files)

    @property
    def mean(self) -> float | None:
        """``sum / len``, as upstream averages a candidate's pass rates."""
        return None if self.scores is None else sum(self.scores) / len(self.scores)


@dataclass(frozen=True)
class Archive:
    """Every candidate proposed so far and the one being served.

    Each operation returns a new archive and leaves this one unchanged.
    ``task_ids`` is the gate's task order, fixed for the whole search so scores
    stay comparable.
    """

    task_ids: tuple[str, ...]
    served: str
    candidates: tuple[Candidate, ...]

    def __post_init__(self) -> None:
        require(len(self.task_ids) > 0, "a Meta-Harness archive needs at least one validation task")
        require(len(set(self.task_ids)) == len(self.task_ids), "archive task ids must be unique")
        seen: set[str] = set()
        for candidate in self.candidates:
            require(candidate.name not in seen, f"candidate name {candidate.name!r} is used twice")
            require(
                candidate.parent is None or candidate.parent in seen,
                f"candidate {candidate.name} must name an earlier candidate as its parent, got {candidate.parent!r}",
            )
            require(
                candidate.scores is None or len(candidate.scores) == len(self.task_ids),
                f"candidate {candidate.name} has {len(candidate.scores or ())} scores for {len(self.task_ids)} tasks",
            )
            seen.add(candidate.name)
        require(self.served in seen, f"the served candidate {self.served!r} is not in the archive")

    @classmethod
    def seed(cls, spec: HarnessSpec, task_ids: Sequence[str]) -> Archive:
        return cls(
            task_ids=tuple(task_ids),
            served=SEED_NAME,
            candidates=(Candidate(name=SEED_NAME, parent=None, files=spec.files),),
        )

    def get(self, name: str) -> Candidate:
        for candidate in self.candidates:
            if candidate.name == name:
                return candidate
        raise ContractError(f"the archive has no candidate named {name!r}")

    @property
    def served_candidate(self) -> Candidate:
        return self.get(self.served)

    def find(self, spec: HarnessSpec) -> Candidate | None:
        """The candidate whose files are ``spec``'s, if any."""
        return next(
            (candidate for candidate in self.candidates if candidate.spec(spec.agent).spec_id == spec.spec_id), None
        )

    @property
    def iteration(self) -> int:
        """The number of the next proposal: upstream counts iterations from 1, and roots are not proposals."""
        return sum(candidate.parent is not None for candidate in self.candidates) + 1

    def adopt_head(self, head: HarnessSpec) -> Archive:
        """Serve the head harness, adding it as a new root when another method produced it."""
        if self.served_candidate.spec(head.agent).spec_id == head.spec_id:
            return self
        known = self.find(head)
        if known is not None:
            return replace(self, served=known.name)
        name = "head_" + head.spec_id.removeprefix("h-")[:8]
        root = Candidate(name=name, parent=None, files=head.files)
        return replace(self, served=name, candidates=(*self.candidates, root))

    def add(self, candidate: Candidate) -> Archive:
        require(candidate.scores is None, "a candidate joins the archive unscored")
        return replace(self, candidates=(*self.candidates, candidate))

    def score(self, name: str, scores: Sequence[float], policy: str, traces: Mapping[str, Trace]) -> Archive:
        scored = replace(self.get(name), scores=tuple(scores), policy=policy, traces=dict(traces))
        return replace(
            self, candidates=tuple(scored if candidate.name == name else candidate for candidate in self.candidates)
        )

    def with_served(self, name: str) -> Archive:
        self.get(name)
        return replace(self, served=name)

    def to_state(self) -> dict[str, object]:
        return {
            METHOD_STATE_VERSION_KEY: ARCHIVE_SCHEMA_VERSION,
            "task_ids": list(self.task_ids),
            "served": self.served,
            "candidates": [
                {
                    "name": candidate.name,
                    "parent": candidate.parent,
                    "files": [[path, file.kind, file.content] for path, file in candidate.files],
                    "hypothesis": candidate.hypothesis,
                    "changes": candidate.changes,
                    "scores": None if candidate.scores is None else list(candidate.scores),
                    "policy": candidate.policy,
                    "traces": {
                        task_id: {"transcript": trace.transcript, "feedback": trace.feedback}
                        for task_id, trace in candidate.traces.items()
                    },
                }
                for candidate in self.candidates
            ],
        }

    @classmethod
    def from_state(cls, state: Mapping[str, object]) -> Archive:
        version = method_state_version(state)
        require(
            version == ARCHIVE_SCHEMA_VERSION,
            f"Meta-Harness reads archive schema {ARCHIVE_SCHEMA_VERSION}, got {version}",
        )
        try:
            rows = cast(list[dict[str, object]], state["candidates"])
            candidates = tuple(_candidate_from_row(row) for row in rows)
            return cls(
                task_ids=tuple(cast(list[str], state["task_ids"])),
                served=str(state["served"]),
                candidates=candidates,
            )
        except (KeyError, TypeError) as error:
            raise ContractError(f"Meta-Harness state is malformed: {error!r}") from error


def _candidate_from_row(row: Mapping[str, object]) -> Candidate:
    files = tuple(
        (str(path), HarnessFile(kind=cast(HarnessFileKind, kind), content=str(content)))
        for path, kind, content in cast(list[list[str]], row["files"])
    )
    scores = row["scores"]
    traces = cast(dict[str, dict[str, str | None]], row["traces"])
    parent = row["parent"]
    policy = row["policy"]
    return Candidate(
        name=str(row["name"]),
        parent=None if parent is None else str(parent),
        files=files,
        hypothesis=str(row["hypothesis"]),
        changes=str(row["changes"]),
        scores=None if scores is None else tuple(float(score) for score in cast(list[float], scores)),
        policy=None if policy is None else str(policy),
        traces={
            task_id: Trace(transcript=trace["transcript"], feedback=trace["feedback"])
            for task_id, trace in traces.items()
        },
    )


# ------------------------------------------------------------ the directory the proposer reads


def write_archive(archive: Archive, directory: Path, *, agent: str, policy: str) -> None:
    """Replace ``directory`` with the archive in upstream's layout.

    ``policy`` is the checksum of the current weights: a score taken under
    other weights is marked stale in the summary and left out of the frontier.
    """
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    for candidate in archive.candidates:
        write_candidate(candidate.spec(agent), directory / CANDIDATES_DIR / candidate.name)
        for task_id, trace in candidate.traces.items():
            _write_trace(trace, directory / TRACES_DIR / candidate.name / quote(task_id, safe=""))

    current = [c for c in archive.candidates if c.scores is not None and c.policy == policy]
    rows = [
        {
            "name": candidate.name,
            "parent": candidate.parent,
            "served": candidate.name == archive.served,
            "stale": candidate.policy != policy,
            "mean": candidate.mean,
            "per_task": dict(zip(archive.task_ids, candidate.scores, strict=True)),
            "hypothesis": candidate.hypothesis,
            "changes": candidate.changes,
        }
        for candidate in archive.candidates
        if candidate.scores is not None
    ]
    (directory / SUMMARY_FILE).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (directory / FRONTIER_FILE).write_text(json.dumps(frontier(archive.task_ids, current), indent=2), encoding="utf-8")


def frontier(task_ids: Sequence[str], candidates: Sequence[Candidate]) -> dict[str, object]:
    """Upstream's ``update_frontier`` over ``candidates`` in order: a later one replaces only on a strict gain."""
    result: dict[str, object] = {}
    best: dict[str, object] | None = None
    for candidate in candidates:
        assert candidate.scores is not None
        for task_id, score in zip(task_ids, candidate.scores, strict=True):
            current = cast(dict[str, object] | None, result.get(task_id))
            if current is None or score > cast(float, current["score"]):
                result[task_id] = {"best_candidate": candidate.name, "score": score}
        mean = cast(float, candidate.mean)
        if best is None or mean > cast(float, best["mean"]):
            best = {"candidate": candidate.name, "mean": mean}
    if best is not None:
        result["_best"] = best
    return result


def write_candidate(spec: HarnessSpec, directory: Path) -> None:
    """Write a harness's files and its ``harness.json`` manifest into ``directory``."""
    directory.mkdir(parents=True)
    for path, file in spec.files:
        target = directory / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(file.content, encoding="utf-8")
    manifest = {"agent": spec.agent, "files": {path: file.kind for path, file in spec.files}}
    (directory / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def read_candidate(directory: Path) -> HarnessSpec:
    """Read a candidate folder back into a spec. Raises ``ValueError`` for a folder that is not one."""
    manifest = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        raise ValueError(f"{MANIFEST} must be an object with a 'files' object")
    kinds = manifest["files"]
    present = {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()} - {MANIFEST}
    if present != set(kinds):
        raise ValueError(f"{MANIFEST} names {sorted(kinds)} but the folder holds {sorted(present)}")
    for kind in kinds.values():
        if kind not in get_args(HarnessFileKind):
            raise ValueError(f"{kind!r} is not a harness file kind")
    files = tuple(
        (path, HarnessFile(kind=kind, content=(directory / path).read_text(encoding="utf-8")))
        for path, kind in sorted(kinds.items())
    )
    return HarnessSpec(agent=str(manifest.get("agent", "")), files=files)


def _write_trace(trace: Trace, directory: Path) -> None:
    directory.mkdir(parents=True)
    if trace.feedback is not None:
        (directory / "feedback.txt").write_text(trace.feedback, encoding="utf-8")
    source = None if trace.transcript is None else Path(trace.transcript)
    try:
        if source is not None and source.is_dir():
            shutil.copytree(source, directory / "transcript")
            return
        if source is not None and source.is_file():
            shutil.copyfile(source, directory / f"transcript{source.suffix}")
            return
    except OSError:
        pass
    (directory / "transcript.unavailable").write_text(f"{trace.transcript}\n", encoding="utf-8")
