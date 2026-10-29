"""Meta-Harness as a Honing evolution method: a coding agent proposes from the full history on disk.

One call to :meth:`MetaHarnessMethod.propose` is one iteration of upstream's
loop (``reference_examples/terminal_bench_2/meta_harness.py``): render the
archive into a directory, run the proposer agent there with the Meta-Harness
skill, read back the candidate it wrote and its ``pending_eval.json``, and
smoke-test it on one task. The campaign's gate is upstream's benchmark run:
:class:`MetaHarnessSelector` applies upstream's ``update_frontier`` rule, and
:meth:`MetaHarnessMethod.record` stores the scores and trajectories the next
proposer reads, whatever the verdict.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping, Sequence
from importlib import resources
from pathlib import Path

from honing.core import (
    CampaignRuleError,
    ContractError,
    EpisodeRecord,
    EvaluationResult,
    EvolveContext,
    HarnessProposal,
    HarnessSpec,
    MethodRequirements,
    ProposerAgent,
    ProposeResult,
    SelectionDecision,
    UpdateCandidate,
)
from honing.harness.methods.meta_harness.archive import (
    ARCHIVE_SCHEMA_VERSION,
    CANDIDATES_DIR,
    FRONTIER_FILE,
    SUMMARY_FILE,
    TRACES_DIR,
    Archive,
    Candidate,
    Trace,
    read_candidate,
    valid_name,
    write_archive,
)

_LOGGER = logging.getLogger(__name__)

ARCHIVE_DIR = "archive"
PENDING_FILE = "pending_eval.json"


def load_skill() -> str:
    """The skill the proposer follows, shipped beside this module."""
    return resources.files(__package__).joinpath("SKILL.md").read_text(encoding="utf-8")


class MetaHarnessSelector:
    """Serve a candidate whose mean strictly beats the served candidate's recorded mean.

    This is upstream's ``update_frontier``: ``_best`` moves only when a
    candidate's average is strictly higher, so a tie keeps the served
    candidate. The recorded mean is compared, not the gate's fresh run of the
    incumbent, because upstream scores each candidate once. Before the served
    candidate has scores under the gate's weights (the first gate, or the first
    after the weights changed), the gate's incumbent run is its score.
    """

    name = "meta_harness"

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        archive = Archive.from_state(candidate.method_state)
        _check_tasks(archive, evaluation)
        served = archive.served_candidate
        recorded = served.mean if served.policy == _policy_of(evaluation) else None
        served_mean = _mean(evaluation.incumbent_scores) if recorded is None else recorded
        candidate_mean = _mean(evaluation.candidate_scores)
        selected = candidate_mean > served_mean
        return SelectionDecision(
            outcome="select" if selected else "reject",
            policy=self.name,
            reason=(
                f"mean {candidate_mean:.4g} against {served_mean:.4g} for served candidate {served.name};"
                " a tie keeps the served candidate"
            ),
            metrics={
                "meta_harness_candidates": float(len(archive.candidates)),
                "meta_harness_candidate_mean": candidate_mean,
                "meta_harness_served_mean": served_mean,
            },
        )


class MetaHarnessMethod:
    """Meta-Harness over a whole harness, with ``proposer`` as upstream's Claude Code session.

    ``work_dir / "archive"`` is rewritten from the state before every proposal,
    so it is the method's own directory. ``validation_task_ids`` are the gate's
    tasks, in the gate's order; they stay fixed so recorded scores stay
    comparable.

    Before the first proposal under a set of weights, the served candidate is
    scored on the validation tasks, as upstream's Phase 0 scores its
    baselines, so the first proposer already has trajectories to read. That
    takes one episode per validation task from the move's ``episode_budget``,
    and is skipped when the budget or the move's tasks cannot cover it; the
    gate's incumbent run then scores the served candidate instead. The smoke
    test runs on the move's first task with one more episode, and a move with
    no budget left skips it, as upstream's ``--skip-smoke`` does.
    """

    name = "meta_harness"
    state_schema_version = ARCHIVE_SCHEMA_VERSION

    def __init__(self, proposer: ProposerAgent, work_dir: Path, validation_task_ids: Sequence[str]) -> None:
        self.requires = MethodRequirements()
        self.selector = MetaHarnessSelector()
        self.proposer = proposer
        self.work_dir = work_dir
        self.validation_task_ids = tuple(validation_task_ids)
        self.skill = load_skill()

    def initial_state(self, seed: HarnessSpec) -> Mapping[str, object]:
        return Archive.seed(seed, self.validation_task_ids).to_state()

    def propose(self, context: EvolveContext) -> ProposeResult:
        """One upstream iteration. ``asyncio.run`` drives the episodes, so call it outside a running loop."""
        archive = Archive.from_state(context.state).adopt_head(context.current)
        archive, spent = _baseline(archive, context)
        directory = self.work_dir / ARCHIVE_DIR
        write_archive(archive, directory, agent=context.current.agent, policy=context.policy.checkpoint.checksum)
        exit_code = self.proposer.run(_prompt(archive, directory), directory, instructions=self.skill)
        try:
            candidate = _read_proposal(archive, directory, exit_code, context.current.agent)
        except (ValueError, OSError) as error:
            _LOGGER.warning("Meta-Harness iteration %d proposed nothing: %s", archive.iteration, error)
            return ProposeResult(proposal=None, state=archive.to_state(), episodes_spent=spent)

        spec = candidate.spec(context.current.agent)
        if context.episode_budget - spent >= 1 and context.tasks:
            spent += 1
            failure = _smoke_test(context, spec)
            if failure is not None:
                _LOGGER.warning("Meta-Harness candidate %s failed its smoke test: %s", candidate.name, failure)
                return ProposeResult(proposal=None, state=archive.to_state(), episodes_spent=spent)

        note = f"{candidate.name}: {candidate.hypothesis}"
        return ProposeResult(
            proposal=HarnessProposal(spec=spec, note=note),
            state=archive.add(candidate).to_state(),
            episodes_spent=spent,
        )

    def record(
        self,
        state: Mapping[str, object],
        proposal: HarnessProposal,
        evaluation: EvaluationResult,
        decision: SelectionDecision,
    ) -> Mapping[str, object]:
        """Store the gate's scores and trajectories for the candidate, and for the served one if it has none yet."""
        archive = Archive.from_state(state)
        newest = archive.candidates[-1]
        if newest.scores is not None or newest.spec(proposal.spec.agent).spec_id != proposal.spec.spec_id:
            raise ContractError(
                "the proposal is not Meta-Harness's newest, unscored candidate; record the state propose returned"
            )
        _check_tasks(archive, evaluation)
        policy = _policy_of(evaluation)
        candidate_records = [r for r in evaluation.records if r.spec_id == proposal.spec.spec_id]
        incumbent_records = [r for r in evaluation.records if r.spec_id != proposal.spec.spec_id]
        archive = archive.score(newest.name, evaluation.candidate_scores, policy, _traces(candidate_records))
        if archive.served_candidate.policy != policy:
            archive = archive.score(archive.served, evaluation.incumbent_scores, policy, _traces(incumbent_records))
        if decision.selected:
            archive = archive.with_served(newest.name)
        return archive.to_state()


def _prompt(archive: Archive, directory: Path) -> str:
    """Upstream's ``render_task_prompt``: where everything is, and where to write the result."""
    return (
        f"Run iteration {archive.iteration} of the harness evolution loop. "
        f"The served harness is `{CANDIDATES_DIR}/{archive.served}/`; any candidate may be the parent.\n\n"
        f"## Evaluation: {len(archive.task_ids)} tasks\n\n"
        f"## Run directories\n"
        f"All results for this run are under `{directory}/`.\n"
        f"- `{directory / SUMMARY_FILE}` — past results\n"
        f"- `{directory / FRONTIER_FILE}` — frontier\n"
        f"- `{directory / TRACES_DIR}/` — evaluation trajectories and feedback\n"
        f"- Write the new candidate to `{directory / CANDIDATES_DIR}/<name>/`\n"
        f"- Write pending_eval.json to: `{directory / PENDING_FILE}`"
    )


def _read_proposal(archive: Archive, directory: Path, exit_code: int, agent: str) -> Candidate:
    """The candidate the proposer wrote. Raises ``ValueError`` or ``OSError`` when there is none to evaluate."""
    if exit_code != 0:
        raise ValueError(f"the proposer exited with code {exit_code}")
    pending = json.loads((directory / PENDING_FILE).read_text(encoding="utf-8"))
    entries = pending.get("candidates") if isinstance(pending, dict) else None
    if not isinstance(entries, list) or len(entries) != 1 or not isinstance(entries[0], dict):
        raise ValueError(f"{PENDING_FILE} must list exactly one candidate")
    entry = entries[0]
    name, parent = entry.get("name"), entry.get("parent")
    if not valid_name(name) or any(candidate.name == name for candidate in archive.candidates):
        raise ValueError(f"the candidate name {name!r} is invalid or already used")
    if not isinstance(parent, str) or all(candidate.name != parent for candidate in archive.candidates):
        raise ValueError(f"the parent {parent!r} is not a candidate")
    spec = read_candidate(directory / CANDIDATES_DIR / str(name))
    if spec.agent != agent:
        raise ValueError(f"the candidate is for agent {spec.agent!r}, not {agent!r}")
    duplicate = archive.find(spec)
    if duplicate is not None:
        raise ValueError(f"the candidate repeats candidate {duplicate.name}")
    return Candidate(
        name=str(name),
        parent=parent,
        files=spec.files,
        hypothesis=str(entry.get("hypothesis", "")),
        changes=str(entry.get("changes", "")),
    )


def _baseline(archive: Archive, context: EvolveContext) -> tuple[Archive, int]:
    """Score the served candidate on the validation tasks when it has no scores under these weights.

    Returns the archive and the episodes spent. Skipped, spending nothing, when
    the move's tasks do not include every validation task or its budget cannot
    run them all.
    """
    checksum = context.policy.checkpoint.checksum
    served = archive.served_candidate
    tasks = {task.task_id: task for task in context.tasks}
    if served.policy == checksum or context.episode_budget < len(archive.task_ids):
        return archive, 0
    if any(task_id not in tasks for task_id in archive.task_ids):
        return archive, 0
    spec = served.spec(context.current.agent)

    async def run_all() -> list[EpisodeRecord]:
        runs = (context.episode(spec, tasks[task_id], context.policy) for task_id in archive.task_ids)
        return list(await asyncio.gather(*runs))

    records = asyncio.run(run_all())
    archive = archive.score(served.name, [record.score for record in records], checksum, _traces(records))
    return archive, len(records)


def _smoke_test(context: EvolveContext, spec: HarnessSpec) -> str | None:
    """Upstream's smoke test: one episode on one task must run without error. Returns why it failed, if it did."""
    try:
        record = asyncio.run(context.episode(spec, context.tasks[0], context.policy))
    except CampaignRuleError as error:
        return str(error)
    if record.outcome.exit_code != 0:
        return f"the episode exited with code {record.outcome.exit_code}"
    return None


def _check_tasks(archive: Archive, evaluation: EvaluationResult) -> None:
    if evaluation.task_ids != archive.task_ids:
        raise ContractError(
            f"the gate scored tasks {list(evaluation.task_ids)}, but Meta-Harness evaluates on"
            f" {list(archive.task_ids)}; give the gate Meta-Harness's validation tasks, in order"
        )


def _policy_of(evaluation: EvaluationResult) -> str:
    """The checksum of the weights the gate ran under, from its episode records."""
    checksums = {record.policy_checksum for record in evaluation.records}
    if len(checksums) != 1:
        raise ContractError(
            "Meta-Harness needs the gate's episode records, all under one set of weights;"
            f" got checksums {sorted(checksums)}"
        )
    return checksums.pop()


def _traces(records: Sequence[EpisodeRecord]) -> dict[str, Trace]:
    return {record.task_id: Trace(record.transcript_location, record.feedback) for record in records}


def _mean(scores: Sequence[float]) -> float:
    """``sum / len``, as upstream averages pass rates."""
    return sum(scores) / len(scores)
