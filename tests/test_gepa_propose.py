"""GEPA's reflective proposal (#24): one iteration of upstream's reflective mutation, run on the fakes."""

from __future__ import annotations

import asyncio
import json
import logging
import random
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from gepa.core.data_loader import ensure_loader
from gepa.gepa_utils import select_program_candidate_from_pareto_front
from gepa.proposer.base import CandidateProposal
from gepa.proposer.reflective_mutation.reflection_lm import StatelessReflectionLM
from gepa.strategies.acceptance import StrictImprovementAcceptance
from gepa.strategies.batch_sampler import EpochShuffledBatchSampler
from gepa.utils.stop_condition import MaxMetricCallsStopper

from honing.core import (
    CampaignRuleError,
    ChatMessage,
    ChatReply,
    ContractError,
    EpisodeOutcome,
    EpisodeRecord,
    EvolveContext,
    HarnessFile,
    HarnessSpec,
    PolicyRef,
    ProposeResult,
    Task,
    run_graded,
)
from honing.core.state import validate_method_state
from honing.harness.methods.gepa import Archive, GEPAConfig, GEPAMethod, extract_text, improves
from tests.fakes.harness import FakeChatModel, FakeEpisodeRunner, FakeTaskGrader
from tests.fakes.model import FakeModelSource, FakeWorld

INSTRUCTIONS = "CLAUDE.md"
SKILL = "skills/tests.md"
SETTINGS = "settings.json"
SETTINGS_TEXT = '{"model": "settings-marker"}'
SEED = HarnessSpec(
    agent="claude_code",
    files=(
        (INSTRUCTIONS, HarnessFile(kind="instructions", content="Solve the task.")),
        (SKILL, HarnessFile(kind="skill", content="Run the tests.")),
        (SETTINGS, HarnessFile(kind="settings", content=SETTINGS_TEXT)),
    ),
)
TRAIN = tuple(Task(task_id=f"t{index}", prompt=f"Solve problem {index}.") for index in range(4))
VALIDATION = ("v0", "v1")
IMPROVING = "```\nSolve the task. Verify the result before answering.\n```"
NOT_IMPROVING = "```\nSolve the task carefully.\n```"
FEEDBACK = "answered without checking the result"


class Fixture:
    def __init__(self, replies: Sequence[str] = (), quality: float = 0.5, **config: object) -> None:
        self.world = FakeWorld()
        snapshot = FakeModelSource(self.world, quality=quality).materialize(Path("unused"))
        self.policy = PolicyRef(snapshot.checkpoint, snapshot.architecture, "master")
        self.runner = FakeEpisodeRunner(self.world, file_kinds=frozenset({"instructions", "skill", "settings"}))
        self.grader = FakeTaskGrader()
        self.chat = FakeChatModel(replies)
        self.method = GEPAMethod(self.chat, GEPAConfig(**{"max_metric_calls": 100, **config}), VALIDATION)  # type: ignore[arg-type]

    def context(
        self,
        state: Mapping[str, object],
        records: tuple[EpisodeRecord, ...] = (),
        episode_budget: int = 100,
        tasks: tuple[Task, ...] = TRAIN,
    ) -> EvolveContext:
        return EvolveContext(
            current=SEED,
            policy=self.policy,
            tasks=tasks,
            records=records,
            state=state,
            runner=self.runner,
            grader=self.grader,
            episode_budget=episode_budget,
        )

    def propose(self, state: Mapping[str, object] | None = None, **context: object) -> ProposeResult:
        state = self.method.initial_state(SEED) if state is None else state
        return self.method.propose(self.context(state, **context))  # type: ignore[arg-type]


def _changed_paths(spec: HarnessSpec) -> set[str]:
    return {path for path, file in spec.files if SEED.get(path) != file}


def _archive(result: ProposeResult) -> Archive:
    return Archive.from_state(result.state)


# ------------------------------------------------------------ #24 acceptance


@pytest.mark.unit
def test_an_improving_reply_changes_only_the_chosen_component() -> None:
    fixture = Fixture([IMPROVING])

    result = fixture.propose()

    assert result.proposal is not None
    assert _changed_paths(result.proposal.spec) == {INSTRUCTIONS}
    assert result.proposal.spec.get(INSTRUCTIONS) == HarnessFile(
        kind="instructions", content="Solve the task. Verify the result before answering."
    )
    archive = _archive(result)
    assert len(archive.candidates) == 2
    assert archive.candidates[1].parent == 0
    assert archive.candidates[1].minibatch_scores == (0.55, 0.55, 0.55)


@pytest.mark.unit
def test_a_proposed_child_records_the_task_ids_of_its_minibatch() -> None:
    fixture = Fixture([IMPROVING], minibatch_size=3)
    drawn: list[str] = []
    run = fixture.runner.run

    async def recording_run(spec: HarnessSpec, task: Task, policy: PolicyRef) -> EpisodeOutcome:
        if spec.get(INSTRUCTIONS) != SEED.get(INSTRUCTIONS):
            drawn.append(task.task_id)
        return await run(spec, task, policy)

    fixture.runner.run = recording_run  # type: ignore[method-assign]
    result = fixture.propose()

    child = _archive(result).candidates[1]
    assert child.minibatch_task_ids == _archive(result).train_order[:3]
    assert sorted(child.minibatch_task_ids) == sorted(drawn)
    assert len(child.minibatch_task_ids) == len(child.minibatch_scores) == 3


@pytest.mark.unit
def test_a_non_improving_reply_proposes_nothing_but_charges_both_minibatches() -> None:
    fixture = Fixture([NOT_IMPROVING], minibatch_size=3)

    result = fixture.propose()

    assert result.proposal is None
    assert _archive(result).metric_calls == 6
    assert result.episodes_spent == 6
    assert len(_archive(result).candidates) == 1


@pytest.mark.unit
@pytest.mark.parametrize("reply", ["```\n```", "", "   \n  "])
def test_an_empty_extraction_proposes_nothing_and_runs_no_child(reply: str) -> None:
    fixture = Fixture([reply], minibatch_size=3)

    result = fixture.propose()

    assert result.proposal is None
    assert fixture.runner.runs == 3
    assert (_archive(result).metric_calls, result.episodes_spent) == (3, 3)


@pytest.mark.unit
def test_a_reply_repeating_the_current_text_proposes_nothing_and_runs_no_child() -> None:
    # A deliberate difference from upstream, which would score the identical child.
    fixture = Fixture(["```\nSolve the task.\n```"], minibatch_size=3)

    result = fixture.propose()

    assert result.proposal is None
    assert len(fixture.chat.conversations) == 1
    assert fixture.runner.runs == 3
    assert (_archive(result).metric_calls, result.episodes_spent) == (3, 3)
    assert len(_archive(result).candidates) == 1


@pytest.mark.unit
def test_an_exhausted_budget_proposes_nothing_and_runs_nothing() -> None:
    fixture = Fixture([IMPROVING], max_metric_calls=6)
    state = Archive.from_state(fixture.method.initial_state(SEED)).charge(6).to_state()

    result = fixture.propose(state)

    assert (result.proposal, result.episodes_spent, dict(result.state)) == (None, 0, state)
    assert (fixture.runner.runs, fixture.grader.grades, fixture.chat.conversations) == (0, 0, [])


@pytest.mark.unit
def test_settings_are_never_rewritten_or_shown_to_the_chat_model() -> None:
    replies = [IMPROVING, "```\nRun the tests twice.\n```"] * 3
    fixture = Fixture(replies)
    state = fixture.method.initial_state(SEED)
    proposals = []

    for _ in range(6):
        result = fixture.propose(state)
        state = result.state
        if result.proposal is not None:
            proposals.append(result.proposal)

    assert proposals
    assert all(proposal.spec.get(SETTINGS) == SEED.get(SETTINGS) for proposal in proposals)
    assert all(SETTINGS not in dict(candidate.components) for candidate in Archive.from_state(state).candidates)
    assert len(fixture.chat.conversations) == 6
    assert not any(SETTINGS_TEXT in message.content for talk in fixture.chat.conversations for message in talk)


@pytest.mark.unit
def test_metric_calls_match_upstream_parent_plus_child_minibatch_per_attempt() -> None:
    # Upstream charges len(minibatch) for the parent and again for the child
    # (reflective_mutation.py:328-332 and 570-574), and never reuses a record.
    fixture = Fixture([IMPROVING, NOT_IMPROVING, IMPROVING, NOT_IMPROVING], minibatch_size=3)
    state = fixture.method.initial_state(SEED)
    spent = 0

    for _ in range(4):
        result = fixture.propose(state)
        state, spent = result.state, spent + result.episodes_spent

    assert Archive.from_state(state).metric_calls == spent == fixture.runner.runs == 4 * (3 + 3)


# ------------------------------------------------------------ comparisons with upstream


@pytest.mark.unit
@pytest.mark.parametrize(
    ("train_size", "minibatch_size", "score_rows"),
    [
        (4, 3, [(0.5, 0.2), (0.4, 0.6), (0.5, 0.6)]),
        (5, 2, [(0.1, 0.9), (0.9, 0.1)]),
        (3, 3, [(0.3, 0.3)]),
        (2, 3, [(0.2, 0.8), (0.8, 0.2), (0.5, 0.5)]),
        (7, 3, []),
    ],
)
def test_minibatches_match_upstream_epoch_shuffled_sampler(
    train_size: int, minibatch_size: int, score_rows: list[tuple[float, float]]
) -> None:
    # Upstream draws the parent, then the minibatch, from one shared RNG (api.py:304, 309, 351).
    train_ids = [f"t{index}" for index in range(train_size)]
    archive = Archive.seed({INSTRUCTIONS: "seed"}, VALIDATION, rng_seed=7)
    for index, scores in enumerate(score_rows):
        if index > 0:
            archive, _ = archive.add({INSTRUCTIONS: f"variant {index}"}, parent=0)
        archive = archive.validate(index, VALIDATION, scores)
    rng = random.Random(7)
    sampler = EpochShuffledBatchSampler(minibatch_size, rng=rng)
    loader = ensure_loader(list(range(train_size)))

    for iteration in range(12):
        fronts, means = archive.fronts(), archive.mean_scores()
        archive, parent = archive.sample_parent()
        archive, batch = archive.next_minibatch(train_ids, minibatch_size)
        upstream_parent = select_program_candidate_from_pareto_front(fronts, means, rng) if fronts else 0
        upstream_batch = sampler.next_minibatch_ids(loader, SimpleNamespace(i=iteration))
        assert (parent, list(batch)) == (upstream_parent, [f"t{index}" for index in upstream_batch])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("before", "after"),
    [
        ([0.5, 0.5], [0.5, 0.5]),
        ([0.5, 0.5], [0.5, 0.6]),
        ([0.5, 0.6], [0.5, 0.5]),
        ([0.0, 1.0], [1.0, 0.0]),
        ([0.3], [0.1, 0.2]),
        ([0.1, 0.2], [0.3]),
        ([], []),
    ],
)
def test_the_accept_rule_matches_upstream_strict_improvement(before: list[float], after: list[float]) -> None:
    proposal = CandidateProposal(
        candidate={}, parent_program_ids=[0], subsample_scores_before=before, subsample_scores_after=after
    )

    assert improves(before, after) == StrictImprovementAcceptance().should_accept(proposal, None)


@pytest.mark.unit
@pytest.mark.parametrize(
    "template",
    [
        None,
        "Rewrite this:\n<curr_param>\nGiven:\n<side_info>\nAnswer in ``` blocks.",
        {INSTRUCTIONS: "For the instructions file:\n<curr_param>\n<side_info>"},
        {SKILL: "Only for the skill:\n<curr_param>\n<side_info>"},
    ],
)
def test_the_reflection_prompt_and_new_text_match_upstream(template: str | dict[str, str] | None) -> None:
    fixture = Fixture([IMPROVING], reflection_prompt_template=template)

    result = fixture.propose()

    batch = [task for task_id in _archive(result).train_order[:3] for task in TRAIN if task.task_id == task_id]
    records = [
        {"Inputs": task.prompt, "Generated Outputs": "quality=0.5 verified=0", "Feedback": FEEDBACK} for task in batch
    ]
    sent: list[object] = []
    upstream = StatelessReflectionLM(
        lm=lambda prompt: sent.append(prompt) or IMPROVING, reflection_prompt_template=template
    )
    proposal, _ = upstream.reflect(
        {INSTRUCTIONS: "Solve the task.", SKILL: "Run the tests."}, {INSTRUCTIONS: records}, [INSTRUCTIONS]
    )
    assert fixture.chat.conversations == [(ChatMessage(role="user", content=str(sent[0])),)]
    assert result.proposal is not None
    assert result.proposal.spec.get(INSTRUCTIONS) == HarnessFile("instructions", proposal.new_texts[INSTRUCTIONS])


@pytest.mark.unit
@pytest.mark.parametrize(
    "reply",
    [
        "```\nnew text\n```",
        "```markdown\nnew text\n```",
        "Here it is:\n```\nnew text\n```\nDone.",
        "  ```\nunclosed block",
        "unopened block\n```  ",
        "no fences at all",
        "```a``` then ```b```",
        "\n\n```\n  padded  \n```\n\n",
        "```",
        "",
    ],
)
def test_text_extraction_matches_upstream(reply: str) -> None:
    upstream = StatelessReflectionLM(lm=lambda prompt: reply)
    record = {"Inputs": "x", "Generated Outputs": "y", "Feedback": "z"}
    proposal, _ = upstream.reflect({INSTRUCTIONS: "old"}, {INSTRUCTIONS: [record]}, [INSTRUCTIONS])

    assert extract_text(reply) == proposal.new_texts[INSTRUCTIONS]


@pytest.mark.unit
@pytest.mark.parametrize("metric_calls", [9, 10, 11])
def test_the_budget_stop_matches_upstream_max_metric_calls_stopper(metric_calls: int) -> None:
    fixture = Fixture([NOT_IMPROVING], max_metric_calls=10)
    state = Archive.from_state(fixture.method.initial_state(SEED)).charge(metric_calls).to_state()

    fixture.propose(state)

    stopped = fixture.runner.runs == 0
    assert stopped == MaxMetricCallsStopper(10)(SimpleNamespace(total_num_evals=metric_calls))


@pytest.mark.unit
@pytest.mark.parametrize(("perfect_score", "skips"), [(0.5, True), (0.4, True), (0.5000001, False)])
def test_the_perfect_score_skip_follows_upstream(perfect_score: float, skips: bool) -> None:
    # Written from upstream's rule, which has no callable of its own (reflective_mutation.py:394-399):
    #   self.skip_perfect_score and self.perfect_score is not None
    #   and all(s is not None and s >= self.perfect_score for s in eval_curr.scores)
    fixture = Fixture([NOT_IMPROVING], perfect_score=perfect_score)

    result = fixture.propose()

    assert (fixture.chat.conversations == []) == skips
    if skips:
        assert result.proposal is None
        assert result.episodes_spent == 3
        # Upstream skips before the component selector runs, so the cursor has not moved.
        assert _archive(result).candidates[0].component_cursor == 0


@pytest.mark.unit
def test_skip_perfect_score_off_reflects_even_on_a_perfect_minibatch() -> None:
    fixture = Fixture([NOT_IMPROVING], quality=1.0, skip_perfect_score=False)

    fixture.propose()

    assert len(fixture.chat.conversations) == 1


@pytest.mark.unit
def test_a_child_starts_at_its_parents_advanced_cursor() -> None:
    # Written from upstream: the selector advances the parent's cursor
    # (component_selector.py:19-23), and a new program then takes the max of
    # its parents' cursors (state.py:539-543).
    fixture = Fixture([IMPROVING])

    archive = _archive(fixture.propose())

    assert archive.candidates[0].component_cursor == 1
    assert archive.candidates[1].component_cursor == 1


# ------------------------------------------------------------ Honing-specific behavior


@pytest.mark.unit
def test_the_state_is_json_native_and_restores_the_same_draws() -> None:
    fixture = Fixture([NOT_IMPROVING] * 4)
    state = fixture.method.initial_state(SEED)
    for _ in range(2):
        state = fixture.propose(state).state
    restored = json.loads(json.dumps(state))
    validate_method_state(state)

    first = Fixture([NOT_IMPROVING] * 2).propose(state).state
    second = Fixture([NOT_IMPROVING] * 2).propose(restored).state

    assert first == second


@pytest.mark.unit
def test_changed_train_ids_reshuffle_even_at_the_same_size() -> None:
    # A deliberate difference: upstream reshuffles only when the size changes (batch_sampler.py:89).
    archive = Archive.seed({INSTRUCTIONS: "seed"}, VALIDATION, rng_seed=0)
    archive, _ = archive.next_minibatch(["a", "b", "c", "d"], 2)

    reshuffled, batch = archive.next_minibatch(["e", "f", "g", "h"], 2)

    assert reshuffled.train_ids == ("e", "f", "g", "h")
    assert set(batch) <= {"e", "f", "g", "h"}
    assert reshuffled.rng_state != archive.rng_state


@pytest.mark.unit
def test_a_record_the_campaign_holds_for_the_parent_is_reused_not_rerun() -> None:
    fixture = Fixture([IMPROVING])

    async def gate_records() -> list[EpisodeRecord]:
        return [await run_graded(fixture.runner, fixture.grader, SEED, task, fixture.policy) for task in TRAIN]

    records = tuple(asyncio.run(gate_records()))
    # Same spec and tasks under other weights: not the same episode, so not reusable.
    other_policy = tuple(
        replace(record, outcome=replace(record.outcome, policy_checksum="other")) for record in records
    )
    fixture.runner.runs = 0

    result = fixture.propose(records=(*other_policy, *records))

    assert fixture.runner.runs == 3
    assert (result.episodes_spent, _archive(result).metric_calls) == (3, 3)


@pytest.mark.unit
def test_too_small_an_episode_budget_proposes_nothing_before_drawing() -> None:
    fixture = Fixture([IMPROVING], minibatch_size=3)
    state = fixture.method.initial_state(SEED)

    result = fixture.propose(state, episode_budget=5)

    assert (result.proposal, result.episodes_spent, dict(result.state)) == (None, 0, state)
    assert fixture.runner.runs == 0


class RefusingRunner(FakeEpisodeRunner):
    """Refuses, as RA2's question check does, any harness whose instructions contain ``marker``."""

    def __init__(self, world: FakeWorld, error: Exception, marker: str = "Verify") -> None:
        super().__init__(world, file_kinds=frozenset({"instructions", "skill", "settings"}))
        self.error, self.marker = error, marker

    async def run(self, spec: HarnessSpec, task: Task, policy: PolicyRef) -> EpisodeOutcome:
        instructions = spec.get(INSTRUCTIONS)
        if instructions is not None and self.marker in instructions.content:
            raise self.error
        return await super().run(spec, task, policy)


@pytest.mark.unit
def test_a_runner_refusing_the_child_proposes_nothing_and_charges_what_ran() -> None:
    fixture = Fixture([IMPROVING])
    fixture.runner = RefusingRunner(fixture.world, CampaignRuleError("questions_fit_model", "too long"))

    result = fixture.propose()

    assert result.proposal is None
    assert (fixture.runner.runs, result.episodes_spent, _archive(result).metric_calls) == (3, 3, 3)


class FailingChatModel(FakeChatModel):
    def complete(self, messages: Sequence[ChatMessage], *, max_tokens: int, temperature: float) -> ChatReply:
        super().complete(messages, max_tokens=max_tokens, temperature=temperature)
        raise RuntimeError("client bug")


@pytest.mark.unit
def test_a_failing_chat_model_proposes_nothing_and_logs_the_traceback(caplog: pytest.LogCaptureFixture) -> None:
    fixture = Fixture()
    fixture.method.chat_model = FailingChatModel([IMPROVING])

    with caplog.at_level(logging.ERROR, logger="honing.harness.methods.gepa.method"):
        result = fixture.propose()

    assert result.proposal is None
    assert (fixture.runner.runs, result.episodes_spent) == (3, 3)
    [log] = caplog.records
    assert log.levelno == logging.ERROR
    assert INSTRUCTIONS in log.getMessage()
    assert log.exc_info is not None and isinstance(log.exc_info[1], RuntimeError)
    assert "client bug" in caplog.text and "Traceback" in caplog.text


@pytest.mark.unit
def test_a_contract_error_on_the_child_propagates() -> None:
    fixture = Fixture([IMPROVING])
    fixture.runner = RefusingRunner(fixture.world, ContractError("the match crashed"))

    with pytest.raises(ContractError, match="the match crashed"):
        fixture.propose()


@pytest.mark.unit
def test_a_component_the_runner_cannot_run_is_refused() -> None:
    fixture = Fixture([IMPROVING])
    fixture.runner = FakeEpisodeRunner(fixture.world, file_kinds=frozenset({"instructions", "settings"}))

    with pytest.raises(ContractError, match="skill file, which the runner cannot run"):
        fixture.propose()


@pytest.mark.unit
def test_only_instructions_and_skill_files_are_components_and_only_instructions_is_required() -> None:
    fixture = Fixture()

    archive = Archive.from_state(fixture.method.initial_state(SEED))

    assert archive.candidates[0].paths == (INSTRUCTIONS, SKILL)
    assert archive.task_ids == VALIDATION
    assert fixture.method.requires.file_kinds == frozenset({"instructions"})
    with pytest.raises(ContractError, match="at least one instructions or skill file"):
        fixture.method.initial_state(HarnessSpec(agent="claude_code", files=((SETTINGS, SEED.files[2][1]),)))


@pytest.mark.unit
def test_a_template_without_its_placeholders_is_refused() -> None:
    with pytest.raises(ValueError, match="<side_info>"):
        GEPAMethod(
            FakeChatModel([]), GEPAConfig(max_metric_calls=1, reflection_prompt_template="<curr_param>"), VALIDATION
        )


@pytest.mark.unit
def test_selection_and_recording_wait_for_issue_25() -> None:
    method = Fixture().method

    with pytest.raises(NotImplementedError, match="#25"):
        method.selector.decide(None, None)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="#25"):
        method.record({}, None, None, None)  # type: ignore[arg-type]
