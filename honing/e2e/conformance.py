"""What any harness runner must do to plug into Honing, checked against the real thing.

A harness framework (OpenHands, Reef, Claude Code, ...) enters Honing as an
:class:`~honing.core.EpisodeRunner` with its :class:`~honing.core.RunnerSupport`,
graded by a :class:`~honing.core.TaskGrader`. :func:`check_runner` runs a
runner through the cases a campaign depends on and returns one
:class:`RunnerCheck` per case, so a new integration can be tested before it
meets a campaign:

- ``support``: the runner declares what it can run, and the seed harness is runnable;
- ``episodes``: every task runs and grades, and each outcome names the task,
  harness, and policy it ran (``run_graded`` refuses anything else);
- ``delivery``: each harness file reached the agent, including one probe file
  of every kind the runner declares (needs a :class:`DeliveryProbe`);
- ``concurrent``: a gate's worth of episodes at once all finish, in separate workspaces;
- ``repeat``: a second gate in a new event loop works (a campaign runs one loop per gate);
- ``gate``: the real gate scores the seed and the probe harness on the same tasks.

``scripts/e2e/check_runner.py`` runs these on the runner and grader a pipeline
config names.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, get_args, runtime_checkable

from honing.campaign import rules
from honing.core import (
    CampaignRuleError,
    EpisodeOutcome,
    EpisodeRecord,
    EpisodeRunner,
    HarnessFile,
    HarnessFileKind,
    HarnessSpec,
    HoneRelease,
    ModelSnapshot,
    PolicyRef,
    RunnerSupport,
    Task,
    TaskGrader,
    UpdateCandidate,
    run_graded,
)
from honing.release import EpisodeEvaluator, Gate, WinLossSelector


@runtime_checkable
class DeliveryProbe(Protocol):
    """Tells, after an episode, which of the harness's files reached the agent."""

    def delivered(self, outcome: EpisodeOutcome, spec: HarnessSpec) -> Mapping[str, bool]: ...


@dataclass(frozen=True)
class RunnerCheck:
    """One case: whether it passed, and what was seen."""

    name: str
    passed: bool
    detail: str


def probe_harness(seed: HarnessSpec, support: RunnerSupport) -> HarnessSpec:
    """The seed plus one file of every kind the runner declares, each with content no other file has.

    A settings file holds a JSON object, as agent settings files are JSON, so a runner
    that checks its settings can take the probe.
    """
    spec = seed
    for kind in sorted(support.file_kinds):
        if kind in get_args(HarnessFileKind):
            content = f"honing conformance probe: a {kind} file\n"
            if kind == "settings":
                content = json.dumps({"honingConformanceProbe": "a settings file"}) + "\n"
            spec = spec.with_file(f"honing-probe/{kind}.md", HarnessFile(kind=kind, content=content))
    return spec


def _episodes(
    runner: EpisodeRunner, grader: TaskGrader, spec: HarnessSpec, tasks: Sequence[Task], policy: PolicyRef
) -> list[EpisodeRecord]:
    async def all_of() -> list[EpisodeRecord]:
        return list(await asyncio.gather(*(run_graded(runner, grader, spec, task, policy) for task in tasks)))

    return asyncio.run(all_of())


def check_runner(
    runner: EpisodeRunner,
    grader: TaskGrader,
    *,
    seed: HarnessSpec,
    tasks: Sequence[Task],
    snapshot: ModelSnapshot,
    probe: DeliveryProbe | None = None,
    concurrent: int = 4,
) -> list[RunnerCheck]:
    """Run every case; a case that raises fails with the error as its detail, and later cases still run."""
    checks: list[RunnerCheck] = []
    policy = PolicyRef(checkpoint=snapshot.checkpoint, architecture=snapshot.architecture, variant="master")

    def case(name: str, body) -> object:  # type: ignore[no-untyped-def]
        try:
            passed, detail = body()
        except Exception as error:  # a conformance case reports, rather than stops on, a runner's failure
            passed, detail = False, f"{type(error).__name__}: {error}"
        checks.append(RunnerCheck(name, passed, detail))
        return passed

    def support() -> tuple[bool, str]:
        declared = runner.support
        if not isinstance(declared, RunnerSupport):
            return False, f"support is a {type(declared).__name__}, not a RunnerSupport"
        rules.spec_runnable(seed, declared)
        return True, f"agents {sorted(declared.agents)}, file kinds {sorted(declared.file_kinds)}"

    if not case("support", support):
        return checks
    probed = probe_harness(seed, runner.support)
    records: list[EpisodeRecord] = []

    def episodes() -> tuple[bool, str]:
        records.extend(_episodes(runner, grader, probed, tasks, policy))
        scores = [record.score for record in records]
        return True, f"{len(records)} episodes graded, scores {min(scores):.2f} to {max(scores):.2f}"

    case("episodes", episodes)

    def delivery() -> tuple[bool, str]:
        if probe is None:
            return False, "no delivery probe was given; pass one that reads where this framework puts the files"
        if not records:
            return False, "no episode finished to inspect"
        missing = sorted(
            {path for record in records for path, ok in probe.delivered(record.outcome, probed).items() if not ok}
        )
        expected = {path for path, _ in probed.files}
        unreported = sorted(expected - set(probe.delivered(records[0].outcome, probed)))
        if missing or unreported:
            return False, f"not delivered: {missing + unreported}"
        return (
            True,
            f"all {len(expected)} files reached the agent in every episode, one per kind {sorted(probed.kinds)}",
        )

    case("delivery", delivery)

    def concurrently() -> tuple[bool, str]:
        # The same task each time: a runner that finds its scenario by task id can run it, and a
        # workspace named only after the task would collide.
        many = [tasks[0]] * concurrent
        done = _episodes(runner, grader, probed, many, policy)
        places = {record.outcome.workspace_location for record in done}
        if None not in places and len(places) != len(done):
            return False, f"{len(done)} concurrent episodes shared {len(places)} workspaces"
        return True, f"{len(done)} episodes at once finished"

    case("concurrent", concurrently)

    def repeat() -> tuple[bool, str]:
        again = _episodes(runner, grader, probed, tasks[:1], policy)
        return True, f"a second event loop ran {len(again)} episode"

    case("repeat", repeat)

    def gate() -> tuple[bool, str]:
        incumbent = HoneRelease("r0000", UpdateCandidate("init", None, snapshot, harness=seed), None)
        candidate = UpdateCandidate("harness", "r0000", snapshot, harness=probed)
        evaluation, decision = Gate(EpisodeEvaluator(runner, grader, tasks), WinLossSelector()).judge(
            candidate, incumbent
        )
        if len(evaluation.records) != 2 * len(tasks):
            return False, f"the gate recorded {len(evaluation.records)} episodes for {len(tasks)} tasks"
        return True, f"the gate scored both harnesses on {len(tasks)} tasks: {decision.outcome} ({decision.reason})"

    case("gate", gate)
    return checks


def refusal(spec: HarnessSpec, support: RunnerSupport) -> str | None:
    """Why a campaign would refuse ``spec`` on this runner before running it, or None."""
    try:
        rules.spec_runnable(spec, support)
    except CampaignRuleError as error:
        return str(error)
    return None
