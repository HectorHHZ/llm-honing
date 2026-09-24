"""The harness as data: the files an agent reads, the episodes it runs, what a method proposes.

A :class:`HarnessSpec` is a set of files for one agent: instructions, skills,
commands, settings, hooks, or a program for an agent whose loop is itself
code. The paths and kinds are chosen so each maps onto the node kinds Reef's
harness layer renders, which keeps a Reef-backed runner possible later without
changing this type.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal, get_args

from honing.core.errors import require, require_choice

HarnessFileKind = Literal["instructions", "skill", "command", "settings", "hook", "program"]


@dataclass(frozen=True)
class HarnessFile:
    """One file an agent reads at startup."""

    kind: HarnessFileKind
    content: str

    def __post_init__(self) -> None:
        require_choice(self.kind, get_args(HarnessFileKind), "HarnessFile.kind")


def _validate_path(path: str) -> None:
    require(bool(path), "a harness file needs a path")
    posix = PurePosixPath(path)
    require(not posix.is_absolute(), f"harness paths are relative to the agent's root, got {path!r}")
    require(".." not in posix.parts, f"harness paths may not leave the agent's root, got {path!r}")


@dataclass(frozen=True)
class HarnessSpec:
    """The complete harness for one agent. Equal content means an equal ``spec_id``."""

    agent: str
    files: tuple[tuple[str, HarnessFile], ...] = ()

    def __post_init__(self) -> None:
        require(bool(self.agent), "HarnessSpec.agent must name an agent")
        paths = [path for path, _ in self.files]
        for path in paths:
            _validate_path(path)
        require(len(set(paths)) == len(paths), "a harness spec names each path once")

    @property
    def spec_id(self) -> str:
        """A content address: the same agent and files always give the same id."""
        canonical = json.dumps(
            {
                "agent": self.agent,
                "files": [[path, file.kind, file.content] for path, file in sorted(self.files, key=lambda f: f[0])],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return "h-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    @property
    def kinds(self) -> frozenset[HarnessFileKind]:
        return frozenset(file.kind for _, file in self.files)

    def get(self, path: str) -> HarnessFile | None:
        return next((file for existing, file in self.files if existing == path), None)

    def with_file(self, path: str, file: HarnessFile) -> HarnessSpec:
        """A copy with ``path`` set to ``file``, replacing it in place if it exists."""
        _validate_path(path)
        if self.get(path) is None:
            return HarnessSpec(agent=self.agent, files=(*self.files, (path, file)))
        files = tuple((existing, file if existing == path else current) for existing, current in self.files)
        return HarnessSpec(agent=self.agent, files=files)


@dataclass(frozen=True)
class Task:
    """One task an episode runs."""

    task_id: str
    prompt: str

    def __post_init__(self) -> None:
        require(bool(self.task_id), "Task.task_id must be set")


@dataclass(frozen=True)
class EpisodeOutcome:
    """What one episode produced, before anyone judged it.

    A runner only runs the agent; grading is task-specific and belongs to a
    :class:`~honing.core.protocols.TaskGrader`. ``transcript_location`` points
    at the agent's own readable transcript, and ``workspace_location`` at the
    files it left behind, which a grader may inspect.
    """

    task_id: str
    spec_id: str
    policy_checksum: str
    exit_code: int
    final_output: str = ""
    transcript_location: str | None = None
    workspace_location: str | None = None

    def __post_init__(self) -> None:
        require(bool(self.task_id), "EpisodeOutcome.task_id must be set")


@dataclass(frozen=True)
class Grade:
    """A grader's verdict on one episode.

    ``score`` is in ``[0, 1]`` for every grader, because selectors compare
    candidate and incumbent task by task and need one scale to do it.
    ``feedback`` is the text a reflective method reads.
    """

    score: float
    feedback: str | None = None

    def __post_init__(self) -> None:
        require(
            math.isfinite(self.score) and 0.0 <= self.score <= 1.0,
            f"a grade's score must be in [0, 1], got {self.score}",
        )


@dataclass(frozen=True)
class EpisodeRecord:
    """One graded episode: the outcome a runner produced and the grade a grader gave it.

    ``samples_location`` points at token-level training samples and is set only
    when the episode was a training rollout.
    """

    outcome: EpisodeOutcome
    grade: Grade
    samples_location: str | None = None

    @property
    def task_id(self) -> str:
        return self.outcome.task_id

    @property
    def spec_id(self) -> str:
        return self.outcome.spec_id

    @property
    def policy_checksum(self) -> str:
        return self.outcome.policy_checksum

    @property
    def score(self) -> float:
        return self.grade.score

    @property
    def feedback(self) -> str | None:
        return self.grade.feedback

    @property
    def transcript_location(self) -> str | None:
        return self.outcome.transcript_location


@dataclass(frozen=True)
class HarnessProposal:
    """A complete harness a method wants evaluated, with a short note on what changed."""

    spec: HarnessSpec
    note: str = ""


@dataclass(frozen=True)
class ProposeResult:
    """What one call to a method's ``propose`` produced.

    ``state`` is returned whether or not there is a proposal, because an attempt
    that proposes nothing has still spent episodes and advanced the method.
    ``episodes_spent`` is how many episodes the attempt ran itself, so a
    restarted campaign can enforce its budgets from the commit record.
    """

    proposal: HarnessProposal | None
    state: Mapping[str, object]
    episodes_spent: int = 0

    def __post_init__(self) -> None:
        require(self.episodes_spent >= 0, f"episodes_spent must be >= 0, got {self.episodes_spent}")


@dataclass(frozen=True)
class HarnessMoveConfig:
    """What one harness move runs with, overriding the campaign's defaults for that move only.

    ``tasks`` empty and ``episode_budget`` ``None`` mean the campaign's own.
    ``method`` names the method the move is for; ``None`` means the campaign's
    method. ``label`` is the scheduler's name for the move, such as a phase: the
    campaign records it on the candidate and never interprets it, so phases and
    their total budgets stay a scheduler concern.
    """

    tasks: tuple[Task, ...] = ()
    episode_budget: int | None = None
    method: str | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        require(
            self.episode_budget is None or self.episode_budget >= 0,
            f"episode_budget must be >= 0, got {self.episode_budget}",
        )
        require(self.method is None or bool(self.method), "HarnessMoveConfig.method must name a method or be None")
        require(self.label is None or bool(self.label), "HarnessMoveConfig.label must be a name or None")


@dataclass(frozen=True)
class RunnerSupport:
    """What an episode runner can run.

    ``local_policy_only`` records that the runner evaluates only a policy served
    by a local engine; Slime's agent adapters forward to SGLang, so a runner built
    on them cannot evaluate a harness against an external model API.
    """

    agents: frozenset[str]
    file_kinds: frozenset[HarnessFileKind]
    programmable_loop: bool
    local_policy_only: bool = True

    def __post_init__(self) -> None:
        require(len(self.agents) > 0, "a runner supports at least one agent")


@dataclass(frozen=True)
class MethodRequirements:
    """What a harness evolution method needs from the runner. ``agents`` empty means any agent."""

    file_kinds: frozenset[HarnessFileKind] = frozenset()
    programmable_loop: bool = False
    agents: frozenset[str] = frozenset()
    extra_episodes: bool = False
