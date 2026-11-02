"""Meta-Harness as a Honing evolution method: a coding agent proposes from the full history on disk.

One call to :meth:`MetaHarnessMethod.propose` is one iteration of upstream's
loop (``reference_examples/terminal_bench_2/meta_harness.py``): render the
archive into a directory, run the proposer agent there with the Meta-Harness
skill, read back the candidate it wrote and its ``pending_eval.json``,
smoke-test it on one task, and score it on the search tasks. Its scores and
trajectories join the archive the next proposer reads, and it is proposed only
when its search mean strictly beats the served candidate's, upstream's
``update_frontier`` rule.

The search tasks are the move's tasks (``EvolveContext.tasks``). The gate
judges a proposal on held-out tasks, as upstream's text-classification example
holds out its test split: :class:`MetaHarnessSelector` refuses a gate on any
search task, and the gate's scores and trajectories never enter the archive.
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
    Task,
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
    """Serve a candidate whose mean on the gate's held-out tasks strictly beats the served harness's.

    The rule is upstream's ``update_frontier``: ``_best`` moves only on a
    strictly higher mean, so a tie keeps the served harness. Both means come
    from the gate's own run, which scores the candidate and the served harness
    on the same tasks under the same weights, so the bar is never a score the
    served harness drew when it won its own gate. A gate on any task the
    proposer searches on is refused: the proposer has read those trajectories.
    """

    name = "meta_harness"

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        archive = Archive.from_state(candidate.method_state)
        shared = sorted(set(evaluation.task_ids) & set(archive.task_ids))
        if shared:
            raise ContractError(
                f"the gate scored tasks {shared}, which Meta-Harness searches on and its proposer reads;"
                " gate it on held-out tasks"
            )
        candidate_mean = _mean(evaluation.candidate_scores)
        served_mean = _mean(evaluation.incumbent_scores)
        selected = candidate_mean > served_mean
        return SelectionDecision(
            outcome="select" if selected else "reject",
            policy=self.name,
            reason=(
                f"held-out mean {candidate_mean:.4g} against {served_mean:.4g} for served candidate"
                f" {archive.served}; a tie keeps the served candidate"
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
    so it is the method's own directory. The search tasks are the first move's
    tasks, in its order; they stay fixed so scores stay comparable, and a later
    move must offer every one of them.

    A move spends, from its ``episode_budget``: one episode per search task to
    score the served candidate when it has no scores under the current weights
    (upstream's Phase 0, so the first proposer already has trajectories to
    read); one per search task to score the candidate; and one more for the
    smoke test, which a move without room for it skips, as upstream's
    ``--skip-smoke`` does. A move whose budget cannot score the candidate does
    not run the proposer.
    """

    name = "meta_harness"
    state_schema_version = ARCHIVE_SCHEMA_VERSION

    def __init__(self, proposer: ProposerAgent, work_dir: Path) -> None:
        self.requires = MethodRequirements()
        self.selector = MetaHarnessSelector()
        self.proposer = proposer
        self.work_dir = work_dir
        self.skill = load_skill()

    def initial_state(self, seed: HarnessSpec) -> Mapping[str, object]:
        return Archive.seed(seed).to_state()

    def propose(self, context: EvolveContext) -> ProposeResult:
        """One upstream iteration. ``asyncio.run`` drives the episodes, so call it outside a running loop."""
        tasks = {task.task_id: task for task in context.tasks}
        archive = Archive.from_state(context.state).adopt_head(context.current)
        archive = archive.searching([task.task_id for task in context.tasks])
        search = [tasks[task_id] for task_id in archive.task_ids]
        checksum = context.policy.checkpoint.checksum
        served = archive.served_candidate
        needed = len(search) * (1 if served.policy == checksum else 2)
        if context.episode_budget < needed:
            _LOGGER.warning(
                "Meta-Harness needs %d episodes to score on %d search tasks; the move allows %d",
                needed,
                len(search),
                context.episode_budget,
            )
            return ProposeResult(proposal=None, state=archive.to_state())

        spent = 0
        if served.policy != checksum:
            records = asyncio.run(_gather(context, served.spec(context.current.agent), search))
            archive = archive.score(served.name, [r.score for r in records], checksum, _traces(records))
            spent += len(records)

        directory = self.work_dir / ARCHIVE_DIR
        write_archive(archive, directory, agent=context.current.agent, policy=checksum)
        exit_code = self.proposer.run(_prompt(archive, directory), directory, instructions=self.skill)
        try:
            candidate = _read_proposal(archive, directory, exit_code, context.current.agent)
        except (ValueError, OSError) as error:
            _LOGGER.warning("Meta-Harness iteration %d proposed nothing: %s", archive.iteration, error)
            return ProposeResult(proposal=None, state=archive.to_state(), episodes_spent=spent)

        spec = candidate.spec(context.current.agent)
        if context.episode_budget - spent - len(search) >= 1:
            spent += 1
            failure = _smoke_test(context, spec, search[0])
            if failure is not None:
                _LOGGER.warning("Meta-Harness candidate %s failed its smoke test: %s", candidate.name, failure)
                return ProposeResult(proposal=None, state=archive.to_state(), episodes_spent=spent)

        records, refused = _run_candidate(context, spec, search)
        spent += len(records)
        if refused is not None:
            _LOGGER.warning("the runner refused Meta-Harness candidate %s: %s", candidate.name, refused)
            return ProposeResult(proposal=None, state=archive.to_state(), episodes_spent=spent)
        archive = archive.add(candidate).score(candidate.name, [r.score for r in records], checksum, _traces(records))
        served_mean, mean = archive.served_candidate.mean, archive.get(candidate.name).mean
        assert served_mean is not None and mean is not None
        if not mean > served_mean:
            _LOGGER.info(
                "Meta-Harness candidate %s scored %.4g on the search tasks, not above %.4g for %s",
                candidate.name,
                mean,
                served_mean,
                archive.served,
            )
            return ProposeResult(proposal=None, state=archive.to_state(), episodes_spent=spent)

        note = f"{candidate.name}: {candidate.hypothesis}; search mean {served_mean:.4g} -> {mean:.4g}"
        return ProposeResult(
            proposal=HarnessProposal(spec=spec, note=note), state=archive.to_state(), episodes_spent=spent
        )

    def record(
        self,
        state: Mapping[str, object],
        proposal: HarnessProposal,
        evaluation: EvaluationResult,
        decision: SelectionDecision,
    ) -> Mapping[str, object]:
        """Serve the proposed candidate if the gate selected it. The gate's held-out run stays out of the archive."""
        archive = Archive.from_state(state)
        newest = archive.candidates[-1]
        if newest.spec(proposal.spec.agent).spec_id != proposal.spec.spec_id:
            raise ContractError(
                "the proposal is not Meta-Harness's newest candidate; record the state propose returned"
            )
        return (archive.with_served(newest.name) if decision.selected else archive).to_state()


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
    folder = directory / CANDIDATES_DIR / str(name)
    # The proposer could swap ``candidates/`` itself for a link; read_candidate checks only inside the folder.
    if folder.resolve().parent != (directory / CANDIDATES_DIR).resolve() or (directory / CANDIDATES_DIR).is_symlink():
        raise ValueError(f"{folder} is not inside the archive")
    spec = read_candidate(folder)
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


async def _gather(context: EvolveContext, spec: HarnessSpec, tasks: Sequence[Task]) -> list[EpisodeRecord]:
    return list(await asyncio.gather(*(context.episode(spec, task, context.policy) for task in tasks)))


def _smoke_test(context: EvolveContext, spec: HarnessSpec, task: Task) -> str | None:
    """Upstream's smoke test: one episode on one task must run without error. Returns why it failed, if it did."""
    try:
        record = asyncio.run(context.episode(spec, task, context.policy))
    except CampaignRuleError as error:
        return str(error)
    if record.outcome.exit_code != 0:
        return f"the episode exited with code {record.outcome.exit_code}"
    return None


def _run_candidate(
    context: EvolveContext, spec: HarnessSpec, tasks: Sequence[Task]
) -> tuple[list[EpisodeRecord], CampaignRuleError | None]:
    """Score the candidate on every search task. A runner's refusal is no proposal; any other error propagates."""

    async def run_all() -> list[EpisodeRecord | BaseException]:
        runs = (context.episode(spec, task, context.policy) for task in tasks)
        return list(await asyncio.gather(*runs, return_exceptions=True))

    records: list[EpisodeRecord] = []
    refused: CampaignRuleError | None = None
    for result in asyncio.run(run_all()):
        if isinstance(result, EpisodeRecord):
            records.append(result)
        elif isinstance(result, CampaignRuleError):
            refused = refused or result
        else:
            raise result
    return records, refused


def _traces(records: Sequence[EpisodeRecord]) -> dict[str, Trace]:
    return {record.task_id: Trace(record.transcript_location, record.feedback) for record in records}


def _mean(scores: Sequence[float]) -> float:
    """``sum / len``, as upstream averages pass rates."""
    return sum(scores) / len(scores)
