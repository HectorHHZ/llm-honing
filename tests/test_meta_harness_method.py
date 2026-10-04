"""Meta-Harness proposal, selection, and recording, with the proposer agent replaced by a script that edits files."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from honing.core import (
    CampaignRuleError,
    ContractError,
    EpisodeOutcome,
    EpisodeRecord,
    EvaluationResult,
    EvolveContext,
    Grade,
    HarnessFile,
    HarnessMoveConfig,
    HarnessProposal,
    HarnessSpec,
    Move,
    PolicyRef,
    SelectionDecision,
    Task,
    UpdateCandidate,
)
from honing.harness.methods.meta_harness import Archive, Candidate, MetaHarnessMethod, load_skill
from tests.fakes.builder import TASKS, build_campaign
from tests.fakes.harness import INSTRUCTIONS_PATH, SEED_HARNESS, FakeEpisodeRunner, FakeProposerAgent, FakeTaskGrader
from tests.fakes.model import FakeModelSource, FakeWorld

TASK_IDS = tuple(task.task_id for task in TASKS)
WORLD = FakeWorld()
SNAPSHOT = FakeModelSource(WORLD).materialize(Path("unused"))
CHECKSUM = SNAPSHOT.checkpoint.checksum
POLICY = PolicyRef(SNAPSHOT.checkpoint, SNAPSHOT.architecture, "master")

Script = Callable[[Path], int]


def _candidate(name: str, text: str, *, parent: str = "seed", pending: object = None) -> Script:
    """What the skill asks for: copy the parent's folder, edit it, and write pending_eval.json."""

    def script(workdir: Path) -> int:
        folder = workdir / "candidates" / name
        shutil.copytree(workdir / "candidates" / parent, folder)
        (folder / INSTRUCTIONS_PATH).write_text(text)
        entry = {"name": name, "parent": parent, "hypothesis": f"{name} helps", "changes": f"wrote {text!r}"}
        body = {"iteration": 1, "candidates": [entry]} if pending is None else pending
        (workdir / "pending_eval.json").write_text(json.dumps(body))
        return 0

    return script


def _in_turn(*scripts: Script) -> Script:
    queue = list(scripts)
    return lambda workdir: queue.pop(0)(workdir)


def _method(tmp_path: Path, script: Script | None = None) -> tuple[MetaHarnessMethod, FakeProposerAgent]:
    proposer = FakeProposerAgent(script)
    return MetaHarnessMethod(proposer, tmp_path / "mh", TASK_IDS), proposer


def _context(
    state: Mapping[str, object],
    *,
    current: HarnessSpec = SEED_HARNESS,
    runner: FakeEpisodeRunner | None = None,
    budget: int = 1,
) -> EvolveContext:
    return EvolveContext(
        current=current,
        policy=POLICY,
        tasks=TASKS,
        records=(),
        state=state,
        runner=runner if runner is not None else FakeEpisodeRunner(WORLD),
        grader=FakeTaskGrader(),
        episode_budget=budget,
    )


def _evaluation(
    candidate: Sequence[float],
    incumbent: Sequence[float],
    *,
    candidate_spec: str = "h-candidate",
    task_ids: Sequence[str] = TASK_IDS,
    checksum: str = CHECKSUM,
) -> EvaluationResult:
    def record(spec_id: str, task_id: str, score: float) -> EpisodeRecord:
        outcome = EpisodeOutcome(
            task_id=task_id,
            spec_id=spec_id,
            policy_checksum=checksum,
            exit_code=0,
            transcript_location=f"/runs/{spec_id}/{task_id}.jsonl",
        )
        return EpisodeRecord(outcome=outcome, grade=Grade(score=score, feedback=f"{spec_id} on {task_id}"))

    records = tuple(
        record(spec_id, task_id, score)
        for spec_id, scores in ((candidate_spec, candidate), ("h-incumbent", incumbent))
        for task_id, score in zip(task_ids, scores, strict=True)
    )
    return EvaluationResult(
        task_ids=tuple(task_ids),
        candidate_scores=tuple(candidate),
        incumbent_scores=tuple(incumbent),
        evaluated_variant="master",
        records=records,
    )


def _proposed(
    tmp_path: Path, text: str = "Verify."
) -> tuple[MetaHarnessMethod, Mapping[str, object], HarnessProposal]:
    method, _ = _method(tmp_path, _candidate("child", text))
    result = method.propose(_context(method.initial_state(SEED_HARNESS), budget=0))
    assert result.proposal is not None
    return method, result.state, result.proposal


def _judge(
    method: MetaHarnessMethod, state: Mapping[str, object], proposal: HarnessProposal, evaluation: EvaluationResult
) -> tuple[SelectionDecision, Archive]:
    """Decide on and record one proposal, as the campaign does (campaign.py:323-327)."""
    candidate = UpdateCandidate(
        kind="harness", parent_id="r0000", master=SNAPSHOT, harness=proposal.spec, method_state=state
    )
    decision = method.selector.decide(candidate, evaluation)
    return decision, Archive.from_state(method.record(state, proposal, evaluation, decision))


# ------------------------------------------------------------ propose


@pytest.mark.unit
def test_propose_runs_the_agent_on_the_archive_and_proposes_its_candidate(tmp_path: Path) -> None:
    runner = FakeEpisodeRunner(WORLD)
    method, proposer = _method(tmp_path, _candidate("verify_first", "Verify the result."))

    result = method.propose(_context(method.initial_state(SEED_HARNESS), runner=runner))

    assert result.proposal is not None
    assert result.proposal.spec.get(INSTRUCTIONS_PATH) == HarnessFile("instructions", "Verify the result.")
    assert result.proposal.note == "verify_first: verify_first helps"
    archive = Archive.from_state(result.state)
    child = archive.candidates[-1]
    assert (child.name, child.parent, child.scores) == ("verify_first", "seed", None)
    assert archive.served == "seed"
    # The smoke test: one episode on the move's first task.
    assert result.episodes_spent == 1 and runner.runs == 1
    ((prompt, workdir, instructions),) = proposer.calls
    assert workdir == tmp_path / "mh" / "archive"
    assert instructions == load_skill() and "pending_eval.json" in instructions
    assert "iteration 1" in prompt and "`candidates/seed/`" in prompt
    assert f"Write pending_eval.json to: `{workdir / 'pending_eval.json'}`" in prompt


def _write(path: str, content: str) -> Script:
    def script(workdir: Path) -> int:
        (workdir / path).write_text(content)
        return 0

    return script


def _then(first: Script, second: Script) -> Script:
    def script(workdir: Path) -> int:
        first(workdir)
        return second(workdir)

    return script


NO_PROPOSAL: dict[str, Script] = {
    "nonzero exit": _then(_candidate("child", "Verify."), lambda _: 1),
    "no pending file": lambda _: 0,
    "malformed pending file": _then(_candidate("child", "Verify."), _write("pending_eval.json", "{not json")),
    "two candidates": _candidate(
        "child", "Verify.", pending={"candidates": [{"name": "child", "parent": "seed"}] * 2}
    ),
    "existing name": _candidate("seed2", "Verify.", pending={"candidates": [{"name": "seed", "parent": "seed"}]}),
    "invalid name": _candidate("child", "Verify.", pending={"candidates": [{"name": "Child!", "parent": "seed"}]}),
    "unknown parent": _candidate("child", "Verify.", pending={"candidates": [{"name": "child", "parent": "nobody"}]}),
    "file missing from the manifest": _then(_candidate("child", "Verify."), _write("candidates/child/x.md", "x")),
    "another agent": _then(
        _candidate("child", "Verify."),
        _write("candidates/child/harness.json", '{"agent": "codex", "files": {"CLAUDE.md": "instructions"}}'),
    ),
    "a copy of an existing candidate": _candidate("child", "Solve the task.\n"),
}


@pytest.mark.unit
@pytest.mark.parametrize("case", sorted(NO_PROPOSAL))
def test_an_unusable_agent_result_proposes_nothing_and_leaves_the_archive(tmp_path: Path, case: str) -> None:
    runner = FakeEpisodeRunner(WORLD)
    method, _ = _method(tmp_path, NO_PROPOSAL[case])
    state = method.initial_state(SEED_HARNESS)

    result = method.propose(_context(state, runner=runner))

    assert result.proposal is None
    assert result.state == state
    assert result.episodes_spent == 0 and runner.runs == 0


class CrashingRunner(FakeEpisodeRunner):
    async def run(self, spec: HarnessSpec, task: Task, policy: PolicyRef) -> EpisodeOutcome:
        return replace(await super().run(spec, task, policy), exit_code=1)


class RefusingRunner(FakeEpisodeRunner):
    async def run(self, spec: HarnessSpec, task: Task, policy: PolicyRef) -> EpisodeOutcome:
        self.runs += 1
        raise CampaignRuleError("spec_runnable", "the runner cannot run this harness")


@pytest.mark.unit
@pytest.mark.parametrize("runner_type", [CrashingRunner, RefusingRunner])
def test_a_candidate_that_fails_its_smoke_test_is_not_proposed(
    tmp_path: Path, runner_type: type[FakeEpisodeRunner]
) -> None:
    runner = runner_type(WORLD)
    method, _ = _method(tmp_path, _candidate("child", "Verify."))
    state = method.initial_state(SEED_HARNESS)

    result = method.propose(_context(state, runner=runner))

    assert result.proposal is None and result.state == state
    assert result.episodes_spent == 1 and runner.runs == 1


@pytest.mark.unit
def test_a_move_with_no_episode_budget_skips_the_smoke_test(tmp_path: Path) -> None:
    runner = CrashingRunner(WORLD)
    method, _ = _method(tmp_path, _candidate("child", "Verify."))

    result = method.propose(_context(method.initial_state(SEED_HARNESS), runner=runner, budget=0))

    assert result.proposal is not None
    assert result.episodes_spent == 0 and runner.runs == 0


@pytest.mark.unit
def test_the_served_candidate_is_scored_before_the_first_proposal(tmp_path: Path) -> None:
    def script(workdir: Path) -> int:
        # Upstream's Phase 0: the first proposer already reads the baseline's results and trajectories.
        rows = [json.loads(line) for line in (workdir / "evolution_summary.jsonl").read_text().splitlines()]
        assert [(row["name"], row["served"]) for row in rows] == [("seed", True)]
        assert (workdir / "traces/seed/sum/feedback.txt").read_text() == "answered without checking the result"
        return _candidate("child", "Verify.")(workdir)

    runner = FakeEpisodeRunner(WORLD)
    method, _ = _method(tmp_path, script)

    result = method.propose(_context(method.initial_state(SEED_HARNESS), runner=runner, budget=len(TASKS) + 1))

    assert result.proposal is not None
    seed = Archive.from_state(result.state).get("seed")
    assert seed.policy == CHECKSUM and seed.scores is not None and len(seed.scores) == len(TASKS)
    assert set(seed.traces) == set(TASK_IDS)
    # One episode per validation task, then the smoke test.
    assert result.episodes_spent == len(TASKS) + 1 == runner.runs


@pytest.mark.unit
def test_the_baseline_runs_once_per_set_of_weights_and_survives_a_failed_proposal(tmp_path: Path) -> None:
    runner = FakeEpisodeRunner(WORLD)
    method, _ = _method(tmp_path, _in_turn(lambda _: 1, _candidate("child", "Verify.")))

    first = method.propose(_context(method.initial_state(SEED_HARNESS), runner=runner, budget=len(TASKS) + 1))
    second = method.propose(_context(first.state, runner=runner, budget=len(TASKS) + 1))

    assert first.proposal is None and first.episodes_spent == len(TASKS)
    assert Archive.from_state(first.state).get("seed").policy == CHECKSUM
    assert second.proposal is not None and second.episodes_spent == 1
    assert runner.runs == len(TASKS) + 1


@pytest.mark.unit
def test_the_bar_after_a_baseline_is_the_baselines_mean(tmp_path: Path) -> None:
    method, _ = _method(tmp_path, _candidate("child", "Verify."))
    result = method.propose(_context(method.initial_state(SEED_HARNESS), budget=len(TASKS) + 1))
    assert result.proposal is not None
    baseline = Archive.from_state(result.state).get("seed")
    assert baseline.mean is not None
    evaluation = _evaluation((1.0,) * 3, (0.0,) * 3, candidate_spec=result.proposal.spec.spec_id)

    decision, archive = _judge(method, result.state, result.proposal, evaluation)

    assert decision.metrics["meta_harness_served_mean"] == pytest.approx(baseline.mean)
    # The gate's incumbent rerun does not replace the baseline it was measured against.
    assert archive.get("seed").scores == baseline.scores


@pytest.mark.unit
def test_a_head_another_method_wrote_is_adopted_and_served(tmp_path: Path) -> None:
    head = SEED_HARNESS.with_file(INSTRUCTIONS_PATH, HarnessFile("instructions", "Edited elsewhere."))
    root = "head_" + head.spec_id[2:10]
    method, proposer = _method(tmp_path, lambda _: 1)

    result = method.propose(_context(method.initial_state(SEED_HARNESS), current=head))

    archive = Archive.from_state(result.state)
    assert archive.served == root and archive.served_candidate.spec(head.agent) == head
    ((prompt, workdir, _),) = proposer.calls
    assert f"`candidates/{root}/`" in prompt
    assert (workdir / "candidates" / root / INSTRUCTIONS_PATH).read_text() == "Edited elsewhere."


# ------------------------------------------------------------ select and record


@pytest.mark.unit
def test_a_strictly_higher_mean_is_selected_and_served(tmp_path: Path) -> None:
    method, state, proposal = _proposed(tmp_path)
    evaluation = _evaluation((1.0, 0.5, 0.5), (0.5, 0.5, 0.5), candidate_spec=proposal.spec.spec_id)

    decision, archive = _judge(method, state, proposal, evaluation)

    assert decision.selected and decision.policy == "meta_harness"
    assert decision.metrics["meta_harness_candidate_mean"] == pytest.approx(2 / 3)
    assert decision.metrics["meta_harness_served_mean"] == 0.5
    assert archive.served == "child"
    child, seed = archive.get("child"), archive.get("seed")
    assert child.scores == (1.0, 0.5, 0.5) and child.policy == CHECKSUM
    assert child.traces["sum"].transcript == f"/runs/{proposal.spec.spec_id}/sum.jsonl"
    assert child.traces["sum"].feedback == f"{proposal.spec.spec_id} on sum"
    # The first gate's incumbent run is the seed's score, as upstream's baseline run is.
    assert seed.scores == (0.5, 0.5, 0.5) and seed.policy == CHECKSUM
    assert seed.traces["parse"].transcript == "/runs/h-incumbent/parse.jsonl"


@pytest.mark.unit
def test_a_tie_is_rejected_and_the_candidate_stays_in_the_archive(tmp_path: Path) -> None:
    method, state, proposal = _proposed(tmp_path)
    evaluation = _evaluation((1.0, 0.0, 0.5), (0.5, 0.5, 0.5), candidate_spec=proposal.spec.spec_id)

    decision, archive = _judge(method, state, proposal, evaluation)

    assert not decision.selected
    assert archive.served == "seed"
    assert archive.get("child").scores == (1.0, 0.0, 0.5)


@pytest.mark.unit
def test_the_bar_is_the_recorded_mean_not_the_gates_rerun(tmp_path: Path) -> None:
    method, state, proposal = _proposed(tmp_path)
    _, archive = _judge(
        method, state, proposal, _evaluation((1.0, 1.0, 0.5), (0.5,) * 3, candidate_spec=proposal.spec.spec_id)
    )
    files = ((INSTRUCTIONS_PATH, HarnessFile("instructions", "Again.")),)
    second = archive.add(Candidate("grandchild", "child", files))
    spec = second.candidates[-1].spec("claude_code")

    # The incumbent reruns low this time; the recorded 0.8333 is still the bar.
    decision, archive = _judge(
        method,
        second.to_state(),
        HarnessProposal(spec),
        _evaluation((0.75,) * 3, (0.0,) * 3, candidate_spec=spec.spec_id),
    )

    assert not decision.selected
    assert decision.metrics["meta_harness_served_mean"] == pytest.approx(2.5 / 3)
    assert archive.get("child").scores == (1.0, 1.0, 0.5)


@pytest.mark.unit
def test_after_new_weights_the_bar_is_the_incumbents_run_under_them(tmp_path: Path) -> None:
    method, state, proposal = _proposed(tmp_path)
    _, archive = _judge(
        method, state, proposal, _evaluation((1.0, 1.0, 1.0), (0.5,) * 3, candidate_spec=proposal.spec.spec_id)
    )
    files = ((INSTRUCTIONS_PATH, HarnessFile("instructions", "Again.")),)
    second = archive.add(Candidate("grandchild", "child", files))
    spec = second.candidates[-1].spec("claude_code")
    evaluation = _evaluation((0.75,) * 3, (0.5,) * 3, candidate_spec=spec.spec_id, checksum="ck-retrained")

    decision, archive = _judge(method, second.to_state(), HarnessProposal(spec), evaluation)

    assert decision.selected
    assert decision.metrics["meta_harness_served_mean"] == 0.5
    assert archive.served == "grandchild"
    assert archive.get("child").scores == (0.5, 0.5, 0.5) and archive.get("child").policy == "ck-retrained"
    # The seed keeps its old scores, which the next archive marks stale.
    assert archive.get("seed").policy == CHECKSUM


@pytest.mark.unit
def test_a_gate_on_other_tasks_is_refused(tmp_path: Path) -> None:
    method, state, proposal = _proposed(tmp_path)
    evaluation = _evaluation((1.0, 1.0), (0.5, 0.5), candidate_spec=proposal.spec.spec_id, task_ids=("sum", "sort"))

    with pytest.raises(ContractError, match="validation tasks"):
        _judge(method, state, proposal, evaluation)


@pytest.mark.unit
def test_recording_a_proposal_that_is_not_the_newest_candidate_is_refused(tmp_path: Path) -> None:
    method, state, _ = _proposed(tmp_path)
    other = HarnessProposal(SEED_HARNESS.with_file(INSTRUCTIONS_PATH, HarnessFile("instructions", "Other.")))
    decision = SelectionDecision(outcome="reject", policy="meta_harness", reason="")

    with pytest.raises(ContractError, match="newest"):
        method.record(state, other, _evaluation((0.5,) * 3, (0.5,) * 3), decision)


# ------------------------------------------------------------ end to end


@pytest.mark.unit
def test_the_campaign_serves_a_better_candidate_and_the_next_proposer_reads_its_history(tmp_path: Path) -> None:
    def second(workdir: Path) -> int:
        # The proposer sees the first iteration's results before writing the next candidate.
        rows = [json.loads(line) for line in (workdir / "evolution_summary.jsonl").read_text().splitlines()]
        assert [(row["name"], row["served"]) for row in rows] == [("seed", False), ("verify", True)]
        assert json.loads((workdir / "frontier_val.json").read_text())["_best"]["candidate"] == "verify"
        feedback = (workdir / "traces/seed/sum/feedback.txt").read_text()
        assert feedback == "answered without checking the result"
        return _candidate("terse", "Answer briefly.", parent="verify")(workdir)

    proposer = FakeProposerAgent(_in_turn(_candidate("verify", "Verify the result before answering."), second))
    method = MetaHarnessMethod(proposer, tmp_path / "mh", TASK_IDS)
    fixture = build_campaign(tmp_path, method=method)
    move = Move(kind="harness", harness=HarnessMoveConfig(episode_budget=1))

    selected = fixture.campaign.advance(move)
    rejected = fixture.campaign.advance(move)

    assert selected.decision is not None and selected.decision.selected
    assert selected.release is not None and selected.release.harness is not None
    assert "Verify" in selected.release.harness.files[0][1].content
    assert rejected.decision is not None and not rejected.decision.selected
    assert len(proposer.calls) == 2
