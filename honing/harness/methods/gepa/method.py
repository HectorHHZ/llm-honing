"""GEPA as a Honing evolution method: reflective proposal, then selection on the gate's validation run.

One call to :meth:`GEPAMethod.propose` is one iteration of upstream GEPA's
``ReflectiveMutationProposer``: sample a parent from the archive, score it on a
training minibatch, reflect on one of its components with a chat model, score
the child on the same minibatch, and propose the child only if its minibatch sum
strictly beats the parent's. The full validation run is the campaign's gate:
:class:`GEPASelector` decides on its per-task scores, and
:meth:`GEPAMethod.record` stores them in the archive whatever the verdict.

Upstream is used for the pure pieces only: the prompt renderer and output
extractor of ``InstructionProposalSignature``. The loop stays here, because the
campaign owns iteration, budgets and state.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from honing.core import (
    CampaignRuleError,
    ChatMessage,
    ChatModel,
    ContractError,
    EpisodeRecord,
    EvaluationResult,
    EvolveContext,
    HarnessFile,
    HarnessFileKind,
    HarnessProposal,
    HarnessSpec,
    MethodRequirements,
    ProposeResult,
    SelectionDecision,
    Task,
    UpdateCandidate,
    require,
)
from honing.harness.methods.gepa.archive import ARCHIVE_SCHEMA_VERSION, Archive, ArchiveCandidate

_LOGGER = logging.getLogger(__name__)

# A reflection reply is prose, so only prose files are rewritten; settings,
# hooks, commands and programs are never touched.
REWRITABLE_KINDS: frozenset[HarnessFileKind] = frozenset({"instructions", "skill"})


@dataclass(frozen=True)
class GEPAConfig:
    """GEPA's knobs, named as in upstream ``gepa.optimize`` where one exists.

    ``reflection_prompt_template`` is one template for every component, or a
    template per component path; a component with none uses upstream's default.
    A template must contain ``<curr_param>`` and ``<side_info>``.
    """

    max_metric_calls: int
    minibatch_size: int = 3
    perfect_score: float = 1.0
    skip_perfect_score: bool = True
    reflection_prompt_template: str | Mapping[str, str] | None = None
    reflection_max_tokens: int = 4096
    reflection_temperature: float = 1.0
    rng_seed: int = 0

    def __post_init__(self) -> None:
        require(self.max_metric_calls >= 0, f"max_metric_calls must be >= 0, got {self.max_metric_calls}")
        require(self.minibatch_size >= 1, f"minibatch_size must be >= 1, got {self.minibatch_size}")
        require(
            self.reflection_max_tokens >= 1, f"reflection_max_tokens must be >= 1, got {self.reflection_max_tokens}"
        )
        require(
            self.reflection_temperature >= 0.0,
            f"reflection_temperature must be >= 0, got {self.reflection_temperature}",
        )

    def template_for(self, path: str) -> str | None:
        """The template for one component, resolved as upstream's ``StatelessReflectionLM`` does."""
        if isinstance(self.reflection_prompt_template, Mapping):
            return self.reflection_prompt_template.get(path)
        return self.reflection_prompt_template


class GEPASelector:
    """Serve a GEPA candidate when its validation mean strictly beats the served candidate's stored mean.

    Upstream GEPA has no gate. It adds every child it scores on the validation
    set, and its best candidate is the one with the highest mean, the earliest
    on a tie (``FullEvaluationPolicy.get_best_program``). Selecting on a strict
    improvement over the served candidate keeps the served pointer on that same
    candidate. The stored mean is compared, not the gate's fresh run of the
    incumbent, because upstream never scores a candidate twice. Before the seed
    has scores, the gate's incumbent side is the seed's validation run.

    Only an evaluation of the archive's validation tasks, with episode records
    under the weights the archive's scores were taken under, is accepted. A
    duel's relative scores fail the first check.
    """

    name = "gepa"

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        archive = Archive.from_state(candidate.method_state)
        if evaluation.task_ids != archive.task_ids:
            raise ContractError(
                f"the gate scored tasks {list(evaluation.task_ids)}, but GEPA validates on"
                f" {list(archive.task_ids)}; give the gate GEPA's validation tasks, in order"
            )
        stored = archive.candidates[archive.served].validation_scores
        served_mean = mean(evaluation.incumbent_scores if stored is None else stored)
        candidate_mean = mean(evaluation.candidate_scores)
        selected = candidate_mean > served_mean
        # Settled here only for the metrics; record settles the same way.
        settled = settle(archive, evaluation, selected)
        on_a_front = set().union(*settled.fronts().values())
        return SelectionDecision(
            outcome="select" if selected else "reject",
            policy=self.name,
            reason=(
                f"validation mean {candidate_mean:.4g} against {served_mean:.4g} for served"
                f" candidate {archive.served}; a tie keeps the served candidate"
            ),
            metrics={
                "gepa_archive_size": float(len(settled.candidates)),
                "gepa_front_size": float(len(on_a_front)),
                "gepa_metric_calls": float(settled.metric_calls),
                "gepa_candidate_mean": candidate_mean,
                "gepa_served_mean": served_mean,
            },
        )


class GEPAMethod:
    """GEPA over a harness's ``instructions`` and ``skill`` files, keyed by path.

    ``validation_task_ids`` are the gate's tasks, in the gate's order: the
    archive stores validation scores in that order. The minibatches are drawn
    from ``EvolveContext.tasks``, which a move can set to a training split the
    gate never sees. After a move changes the weights, the search starts over
    from the served harness (:meth:`Archive.restart_from_served`).
    """

    name = "gepa"
    state_schema_version = ARCHIVE_SCHEMA_VERSION

    def __init__(self, chat_model: ChatModel, config: GEPAConfig, validation_task_ids: Sequence[str]) -> None:
        # Only ``instructions`` is required of the runner: a runner without ``skill``
        # files (RA2's) must still accept GEPA. The seed decides which kinds are rewritten.
        self.requires = MethodRequirements(file_kinds=frozenset({"instructions"}))
        self.selector = GEPASelector()
        self.chat_model = chat_model
        self.config = config
        self.validation_task_ids = tuple(validation_task_ids)
        _require_gepa()
        from gepa.strategies.instruction_proposal import InstructionProposalSignature

        templates = config.reflection_prompt_template
        for template in templates.values() if isinstance(templates, Mapping) else [templates]:
            InstructionProposalSignature.validate_prompt_template(template)

    def initial_state(self, seed: HarnessSpec) -> Mapping[str, object]:
        components = {path: file.content for path, file in seed.files if file.kind in REWRITABLE_KINDS}
        require(len(components) > 0, "GEPA needs at least one instructions or skill file in the seed harness")
        return Archive.seed(components, self.validation_task_ids, self.config.rng_seed).to_state()

    def propose(self, context: EvolveContext) -> ProposeResult:
        """One reflective mutation. ``asyncio.run`` drives the episodes, so call it outside a running loop."""
        config = self.config
        archive = Archive.from_state(context.state)
        self._check_components(archive, context)
        # A train, prune or quantize move changed the weights since the archive was
        # scored. Scores under other weights are not comparable, so the search starts
        # over from the served harness before a parent is drawn from them; the next
        # gate scores that harness under the new weights.
        checksum = context.policy.checkpoint.checksum
        if archive.validation_policy is not None and archive.validation_policy != checksum:
            archive = archive.restart_from_served()
        # Upstream checks the budget before each iteration starts (engine.py:731), so an
        # iteration may overshoot it; nothing is drawn or run once it is reached.
        if archive.metric_calls >= config.max_metric_calls:
            return ProposeResult(proposal=None, state=archive.to_state())
        # Checked before any draw, with the worst case, so a retry under a larger
        # budget repeats the same parent and minibatch.
        if 2 * config.minibatch_size > context.episode_budget:
            return ProposeResult(proposal=None, state=archive.to_state())

        archive, parent_index = archive.sample_parent()
        tasks = {task.task_id: task for task in context.tasks}
        require(len(tasks) == len(context.tasks), "the move's task ids must be unique")
        archive, batch_ids = archive.next_minibatch([task.task_id for task in context.tasks], config.minibatch_size)
        batch = [tasks[task_id] for task_id in batch_ids]

        parent = archive.candidates[parent_index]
        parent_spec = _spec_of(context.current, parent)
        parent_records, ran = self._parent_records(context, parent_spec, batch)
        archive = archive.charge(ran)
        spent = ran
        before = [record.score for record in parent_records]
        if config.skip_perfect_score and all(score >= config.perfect_score for score in before):
            return ProposeResult(proposal=None, state=archive.to_state(), episodes_spent=spent)

        archive, path = archive.next_component(parent_index)
        current = dict(parent.components)[path]
        text = self._reflect(current, path, batch, parent_records)
        # Two deliberate differences from upstream, which evaluates the child in
        # both cases (reflective_mutation.py:472 skips only an empty dict of texts):
        # an empty extraction, and a reply that repeats the current text. The
        # second child would be the parent itself, so running it spends a
        # minibatch to learn nothing, and a noisy runner could even accept it.
        if not text or text == current:
            return ProposeResult(proposal=None, state=archive.to_state(), episodes_spent=spent)

        child_components = {**dict(parent.components), path: text}
        child_spec = _spec_of(context.current, ArchiveCandidate(tuple(sorted(child_components.items())), parent_index))
        child_records, ran, refused = _run_child(context, child_spec, batch)
        archive = archive.charge(ran)
        spent += ran
        if refused is not None:
            return ProposeResult(proposal=None, state=archive.to_state(), episodes_spent=spent)
        after = [record.score for record in child_records]
        if not improves(before, after):
            return ProposeResult(proposal=None, state=archive.to_state(), episodes_spent=spent)

        # The parent's cursor has already advanced, so the child starts after the
        # component just rewritten, as upstream's GEPAState does.
        archive, child_index = archive.add(child_components, parent_index, after, batch_ids)
        note = (
            f"GEPA candidate {child_index}: rewrote {path} of candidate {parent_index};"
            f" minibatch sum {sum(before):.4g} -> {sum(after):.4g}"
        )
        return ProposeResult(
            proposal=HarnessProposal(spec=child_spec, note=note), state=archive.to_state(), episodes_spent=spent
        )

    def record(
        self,
        state: Mapping[str, object],
        proposal: HarnessProposal,
        evaluation: EvaluationResult,
        decision: SelectionDecision,
    ) -> Mapping[str, object]:
        """Store the gate's scores for the proposed child, selected or not, and serve it if it was selected.

        A rejected child stays in the archive and joins every task front it
        leads or ties, so it can still be a parent. The served pointer follows
        the gate's decision, because it must name the harness the chain published.
        """
        archive = Archive.from_state(state)
        if _spec_of(proposal.spec, archive.candidates[-1]).spec_id != proposal.spec.spec_id:
            raise ContractError(
                "the proposal is not GEPA's newest candidate; record the state propose returned with it"
            )
        return settle(archive, evaluation, decision.selected).to_state()

    def _check_components(self, archive: Archive, context: EvolveContext) -> None:
        """Every component must still be a rewritable file the runner declares it can run."""
        for path in archive.candidates[0].paths:
            file = context.current.get(path)
            if file is None or file.kind not in REWRITABLE_KINDS:
                raise ContractError(
                    f"GEPA component {path!r} is not an instructions or skill file in the head harness"
                )
            if file.kind not in context.runner.support.file_kinds:
                raise ContractError(f"GEPA component {path!r} is a {file.kind} file, which the runner cannot run")

    def _parent_records(
        self, context: EvolveContext, spec: HarnessSpec, batch: Sequence[Task]
    ) -> tuple[list[EpisodeRecord], int]:
        """Score the parent on the minibatch, reusing a record the campaign already holds for the same episode.

        A record is reusable when it ran this spec under this policy on the same
        task. Upstream always reruns the parent; a reused record costs no
        episode, so it is not a metric call either.
        """
        checksum = context.policy.checkpoint.checksum
        cached = {
            record.task_id: record
            for record in context.records
            if record.spec_id == spec.spec_id and record.policy_checksum == checksum
        }
        missing = [task for task in batch if task.task_id not in cached]
        fresh = iter(asyncio.run(_gather(context, spec, missing)))
        return [cached[task.task_id] if task.task_id in cached else next(fresh) for task in batch], len(missing)

    def _reflect(self, current: str, path: str, batch: Sequence[Task], records: Sequence[EpisodeRecord]) -> str:
        """Render upstream's reflection prompt, ask the chat model, and extract the new text ("" if none)."""
        _require_gepa()
        from gepa.strategies.instruction_proposal import InstructionProposalSignature

        dataset = [
            {
                "Inputs": task.prompt,
                "Generated Outputs": record.outcome.final_output,
                "Feedback": record.feedback or "",
            }
            for task, record in zip(batch, records, strict=True)
        ]
        prompt = InstructionProposalSignature.prompt_renderer(
            {
                "current_instruction_doc": current,
                "dataset_with_feedback": dataset,
                "prompt_template": self.config.template_for(path),
            }
        )
        if not isinstance(prompt, str):
            # Upstream returns chat messages only when a record holds an image; Honing's never do.
            raise ContractError("the reflection prompt rendered as multimodal messages; expected text")
        try:
            reply = self.chat_model.complete(
                [ChatMessage(role="user", content=prompt)],
                max_tokens=self.config.reflection_max_tokens,
                temperature=self.config.reflection_temperature,
            )
        except Exception:
            # Upstream isolates a failed reflection and proposes nothing for it
            # (reflective_mutation.py:203-224); the parent episodes stay charged.
            # Logged with the traceback so a bug in a chat client is not silent.
            _LOGGER.exception("GEPA reflection on %s failed; proposing nothing this iteration", path)
            return ""
        return extract_text(reply.text)


def settle(archive: Archive, evaluation: EvaluationResult, selected: bool) -> Archive:
    """The archive after the gate's validation run of its newest candidate, charged as upstream charges it.

    The seed takes its scores from the first gate's incumbent side, and both
    runs cost one metric call per task: upstream charges the seed's full
    validation (state.py:704) and each child's (engine.py:282). A later
    incumbent run is the gate's, not GEPA's, and is not charged.
    """
    child = len(archive.candidates) - 1
    require(
        child >= 1 and archive.candidates[child].validation_scores is None,
        "the gate settles GEPA's newest candidate, which must not be validated yet",
    )
    archive = archive.under_policy(_policy_of(evaluation))
    tasks = len(evaluation.task_ids)
    if archive.candidates[archive.served].validation_scores is None:
        # The served pointer moves only to a validated candidate, so this is the seed's first gate.
        require(archive.served == 0, f"served candidate {archive.served} has no validation scores")
        archive = archive.validate(0, evaluation.task_ids, evaluation.incumbent_scores).charge(tasks)
    archive = archive.validate(child, evaluation.task_ids, evaluation.candidate_scores).charge(tasks)
    return archive.with_served(child) if selected else archive


def mean(scores: Sequence[float]) -> float:
    """``sum / len``, as upstream averages validation scores (state.py:444), not ``statistics.fmean``.

    The two can round differently, and on a tie that decides which candidate is served.
    """
    return sum(scores) / len(scores)


def _policy_of(evaluation: EvaluationResult) -> str:
    """The checksum of the weights the gate ran under, from its episode records."""
    checksums = {record.policy_checksum for record in evaluation.records}
    if len(checksums) != 1:
        raise ContractError(
            f"GEPA needs the gate's episode records, all under one set of weights; got checksums {sorted(checksums)}"
        )
    return checksums.pop()


def improves(before: Sequence[float], after: Sequence[float]) -> bool:
    """Upstream's default ``StrictImprovementAcceptance``: the child's minibatch sum strictly beats the parent's."""
    return sum(after) > sum(before)


def extract_text(reply: str) -> str:
    """The new component text in a reply, as upstream's ``StatelessReflectionLM`` extracts it (stripped first)."""
    _require_gepa()
    from gepa.strategies.instruction_proposal import InstructionProposalSignature

    return str(InstructionProposalSignature.output_extractor(reply.strip())["new_instruction"])


def _require_gepa() -> None:
    try:
        import gepa  # noqa: F401
    except ImportError as error:
        raise ImportError("GEPA reflection needs the 'gepa' extra: pip install 'llm-honing[gepa]'") from error


def _spec_of(current: HarnessSpec, candidate: ArchiveCandidate) -> HarnessSpec:
    """The head harness with the candidate's component texts, each file keeping its kind."""
    spec = current
    for path, text in candidate.components:
        file = current.get(path)
        require(file is not None, f"component {path!r} is missing from the head harness")
        assert file is not None
        spec = spec.with_file(path, HarnessFile(kind=file.kind, content=text))
    return spec


async def _gather(context: EvolveContext, spec: HarnessSpec, tasks: Sequence[Task]) -> list[EpisodeRecord]:
    return list(await asyncio.gather(*(context.episode(spec, task, context.policy) for task in tasks)))


def _run_child(
    context: EvolveContext, spec: HarnessSpec, batch: Sequence[Task]
) -> tuple[list[EpisodeRecord], int, CampaignRuleError | None]:
    """Score the child. A runner may refuse a harness with ``CampaignRuleError``; that is no proposal, not a crash.

    Only the episodes that produced a record are counted. Any other error,
    such as a ``ContractError`` for a broken environment, propagates.
    """

    async def run_all() -> list[EpisodeRecord | BaseException]:
        runs = (context.episode(spec, task, context.policy) for task in batch)
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
    return records, len(records), refused
