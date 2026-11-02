"""Meta-Harness proposal, selection, and recording, with the proposer agent replaced by a script that edits files."""

from __future__ import annotations

import json
import os
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
from honing.harness.methods.meta_harness import Archive, MetaHarnessMethod, load_skill
from tests.fakes.builder import TASKS, build_campaign
from tests.fakes.harness import INSTRUCTIONS_PATH, SEED_HARNESS, FakeEpisodeRunner, FakeProposerAgent, FakeTaskGrader
from tests.fakes.model import FakeModelSource, FakeWorld

# The method searches on these; the gate judges on TASKS, held out from them.
SEARCH = tuple(Task(task_id=f"train-{task.task_id}", prompt=task.prompt) for task in TASKS)
SEARCH_IDS = tuple(task.task_id for task in SEARCH)
HELD_OUT_IDS = tuple(task.task_id for task in TASKS)
WORLD = FakeWorld()
SNAPSHOT = FakeModelSource(WORLD).materialize(Path("unused"))
CHECKSUM = SNAPSHOT.checkpoint.checksum
POLICY = PolicyRef(SNAPSHOT.checkpoint, SNAPSHOT.architecture, "master")
# Phase 0, the candidate's search run, and the smoke test.
FULL = 2 * len(SEARCH) + 1

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
    return MetaHarnessMethod(proposer, tmp_path / "mh"), proposer


def _context(
    state: Mapping[str, object],
    *,
    current: HarnessSpec = SEED_HARNESS,
    runner: FakeEpisodeRunner | None = None,
    budget: int = FULL,
    tasks: Sequence[Task] = SEARCH,
    policy: PolicyRef = POLICY,
) -> EvolveContext:
    return EvolveContext(
        current=current,
        policy=policy,
        tasks=tuple(tasks),
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
    task_ids: Sequence[str] = HELD_OUT_IDS,
) -> EvaluationResult:
    def record(spec_id: str, task_id: str, score: float) -> EpisodeRecord:
        outcome = EpisodeOutcome(
            task_id=task_id,
            spec_id=spec_id,
            policy_checksum=CHECKSUM,
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


def _proposed(tmp_path: Path) -> tuple[MetaHarnessMethod, Mapping[str, object], HarnessProposal]:
    method, _ = _method(tmp_path, _candidate("child", "Verify."))
    result = method.propose(_context(method.initial_state(SEED_HARNESS)))
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
def test_propose_scores_the_baseline_and_the_candidate_on_the_search_tasks(tmp_path: Path) -> None:
    def script(workdir: Path) -> int:
        # Upstream's Phase 0: the first proposer already reads the baseline's results and trajectories.
        rows = [json.loads(line) for line in (workdir / "evolution_summary.jsonl").read_text().splitlines()]
        assert [(row["name"], row["served"]) for row in rows] == [("seed", True)]
        assert set(rows[0]["per_task"]) == set(SEARCH_IDS)
        feedback = (workdir / "traces/seed/train-sum/feedback.txt").read_text()
        assert feedback == "answered without checking the result"
        return _candidate("verify_first", "Verify the result.")(workdir)

    runner = FakeEpisodeRunner(WORLD)
    method, proposer = _method(tmp_path, script)

    result = method.propose(_context(method.initial_state(SEED_HARNESS), runner=runner))

    assert result.proposal is not None
    assert result.proposal.spec.get(INSTRUCTIONS_PATH) == HarnessFile("instructions", "Verify the result.")
    assert result.proposal.note == "verify_first: verify_first helps; search mean 0.5 -> 0.55"
    archive = Archive.from_state(result.state)
    assert archive.task_ids == SEARCH_IDS and archive.served == "seed"
    seed, child = archive.get("seed"), archive.get("verify_first")
    assert seed.policy == child.policy == CHECKSUM
    assert child.parent == "seed" and child.scores == (0.55,) * 3 and set(child.traces) == set(SEARCH_IDS)
    # Phase 0, the smoke test, then the candidate's search run.
    assert result.episodes_spent == FULL == runner.runs
    ((prompt, workdir, instructions),) = proposer.calls
    assert workdir == tmp_path / "mh" / "archive"
    assert instructions == load_skill() and "pending_eval.json" in instructions
    assert "iteration 1" in prompt and "`candidates/seed/`" in prompt
    assert f"Write pending_eval.json to: `{workdir / 'pending_eval.json'}`" in prompt


@pytest.mark.unit
def test_a_candidate_that_does_not_beat_the_served_one_on_the_search_tasks_is_kept_but_not_proposed(
    tmp_path: Path,
) -> None:
    method, _ = _method(tmp_path, _candidate("terse", "Answer briefly."))

    result = method.propose(_context(method.initial_state(SEED_HARNESS)))

    assert result.proposal is None and result.episodes_spent == FULL
    # The tie stays in the archive, scored, for the next proposer to read.
    archive = Archive.from_state(result.state)
    assert archive.served == "seed" and archive.get("terse").scores == (0.5,) * 3


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


def _link_to(outside: Path, *, hard: bool = False) -> Script:
    """Replace the candidate's instructions with a link to a file outside the archive, such as ground truth."""

    def script(workdir: Path) -> int:
        target = workdir / "candidates/child" / INSTRUCTIONS_PATH
        target.unlink()
        (os.link if hard else os.symlink)(outside, target)
        return 0

    return script


def _linked_candidates_dir(outside: Path) -> Script:
    def script(workdir: Path) -> int:
        shutil.copytree(workdir / "candidates", outside / "candidates")
        (outside / "candidates/seed" / INSTRUCTIONS_PATH).write_text("ground truth")
        shutil.copytree(outside / "candidates/seed", outside / "candidates/child")
        shutil.rmtree(workdir / "candidates")
        (workdir / "candidates").symlink_to(outside / "candidates")
        entry = {"name": "child", "parent": "seed"}
        (workdir / "pending_eval.json").write_text(json.dumps({"candidates": [entry]}))
        return 0

    return script


NO_PROPOSAL: dict[str, Callable[[Path], Script]] = {
    "nonzero exit": lambda _: _then(_candidate("child", "Verify."), lambda _: 1),
    "no pending file": lambda _: lambda _: 0,
    "malformed pending file": lambda _: _then(
        _candidate("child", "Verify."), _write("pending_eval.json", "{not json")
    ),
    "two candidates": lambda _: _candidate(
        "child", "Verify.", pending={"candidates": [{"name": "child", "parent": "seed"}] * 2}
    ),
    "existing name": lambda _: _candidate(
        "seed2", "Verify.", pending={"candidates": [{"name": "seed", "parent": "seed"}]}
    ),
    "invalid name": lambda _: _candidate(
        "child", "Verify.", pending={"candidates": [{"name": "Child!", "parent": "seed"}]}
    ),
    "unknown parent": lambda _: _candidate(
        "child", "Verify.", pending={"candidates": [{"name": "child", "parent": "nobody"}]}
    ),
    "file missing from the manifest": lambda _: _then(
        _candidate("child", "Verify."), _write("candidates/child/x.md", "x")
    ),
    "another agent": lambda _: _then(
        _candidate("child", "Verify."),
        _write("candidates/child/harness.json", '{"agent": "codex", "files": {"CLAUDE.md": "instructions"}}'),
    ),
    "a copy of an existing candidate": lambda _: _candidate("child", "Solve the task.\n"),
    "a symbolic link out of the archive": lambda outside: _then(
        _candidate("child", "Verify."), _link_to(outside / "groundtruth.txt")
    ),
    "a hard link out of the archive": lambda outside: _then(
        _candidate("child", "Verify."), _link_to(outside / "groundtruth.txt", hard=True)
    ),
    "a linked candidates folder": _linked_candidates_dir,
}


@pytest.mark.unit
@pytest.mark.parametrize("case", sorted(NO_PROPOSAL))
def test_an_unusable_agent_result_proposes_nothing(tmp_path: Path, case: str) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "groundtruth.txt").write_text("the answers")
    runner = FakeEpisodeRunner(WORLD)
    method, _ = _method(tmp_path, NO_PROPOSAL[case](outside))

    result = method.propose(_context(method.initial_state(SEED_HARNESS), runner=runner))

    assert result.proposal is None
    # Only the baseline ran, and it stays in the archive.
    archive = Archive.from_state(result.state)
    assert [candidate.name for candidate in archive.candidates] == ["seed"]
    assert archive.get("seed").policy == CHECKSUM
    assert result.episodes_spent == len(SEARCH) == runner.runs


class CrashingRunner(FakeEpisodeRunner):
    """Runs the seed, and crashes every other harness."""

    async def run(self, spec: HarnessSpec, task: Task, policy: PolicyRef) -> EpisodeOutcome:
        outcome = await super().run(spec, task, policy)
        return outcome if spec == SEED_HARNESS else replace(outcome, exit_code=1)


class RefusingRunner(FakeEpisodeRunner):
    """Runs the seed, and refuses every other harness."""

    async def run(self, spec: HarnessSpec, task: Task, policy: PolicyRef) -> EpisodeOutcome:
        if spec == SEED_HARNESS:
            return await super().run(spec, task, policy)
        self.runs += 1
        raise CampaignRuleError("spec_runnable", "the runner cannot run this harness")


@pytest.mark.unit
@pytest.mark.parametrize("runner_type", [CrashingRunner, RefusingRunner])
def test_a_candidate_that_fails_its_smoke_test_is_not_proposed_or_kept(
    tmp_path: Path, runner_type: type[FakeEpisodeRunner]
) -> None:
    runner = runner_type(WORLD)
    method, _ = _method(tmp_path, _candidate("child", "Verify."))

    result = method.propose(_context(method.initial_state(SEED_HARNESS), runner=runner))

    assert result.proposal is None
    assert [candidate.name for candidate in Archive.from_state(result.state).candidates] == ["seed"]
    assert result.episodes_spent == len(SEARCH) + 1 == runner.runs


@pytest.mark.unit
def test_a_move_without_room_for_the_smoke_test_skips_it(tmp_path: Path) -> None:
    runner = FakeEpisodeRunner(WORLD)
    method, _ = _method(tmp_path, _candidate("child", "Verify."))

    result = method.propose(_context(method.initial_state(SEED_HARNESS), runner=runner, budget=FULL - 1))

    assert result.proposal is not None
    assert result.episodes_spent == 2 * len(SEARCH) == runner.runs


@pytest.mark.unit
def test_a_refusal_in_the_search_run_proposes_nothing(tmp_path: Path) -> None:
    runner = RefusingRunner(WORLD)
    method, _ = _method(tmp_path, _candidate("child", "Verify."))

    result = method.propose(_context(method.initial_state(SEED_HARNESS), runner=runner, budget=FULL - 1))

    assert result.proposal is None
    assert [candidate.name for candidate in Archive.from_state(result.state).candidates] == ["seed"]


@pytest.mark.unit
@pytest.mark.parametrize("budget", [0, 2 * len(SEARCH) - 1])
def test_a_move_that_cannot_score_the_candidate_does_not_run_the_proposer(tmp_path: Path, budget: int) -> None:
    runner = FakeEpisodeRunner(WORLD)
    method, proposer = _method(tmp_path, _candidate("child", "Verify."))

    result = method.propose(_context(method.initial_state(SEED_HARNESS), runner=runner, budget=budget))

    assert result.proposal is None and result.episodes_spent == 0 == runner.runs
    assert proposer.calls == []


@pytest.mark.unit
def test_the_baseline_runs_once_per_set_of_weights_and_survives_a_failed_proposal(tmp_path: Path) -> None:
    runner = FakeEpisodeRunner(WORLD)
    method, _ = _method(tmp_path, _in_turn(lambda _: 1, _candidate("child", "Verify.")))

    first = method.propose(_context(method.initial_state(SEED_HARNESS), runner=runner))
    # With the baseline already scored, one run of the search tasks and the smoke test fit.
    second = method.propose(_context(first.state, runner=runner, budget=len(SEARCH) + 1))

    assert first.proposal is None and first.episodes_spent == len(SEARCH)
    assert second.proposal is not None and second.episodes_spent == len(SEARCH) + 1
    assert runner.runs == 2 * len(SEARCH) + 1


@pytest.mark.unit
def test_new_weights_rescore_the_served_candidate_and_mark_old_scores_stale(tmp_path: Path) -> None:
    world = FakeWorld(quality=dict(WORLD.quality), created=WORLD.created)
    retrained = PolicyRef(world.checkpoint(0.6), SNAPSHOT.architecture, "master")

    def second(workdir: Path) -> int:
        rows = {row["name"]: row for row in map(json.loads, (workdir / "evolution_summary.jsonl").open())}
        assert rows["seed"]["stale"] and not rows["child"]["stale"] and rows["child"]["served"]
        assert rows["child"]["mean"] == pytest.approx(0.65)
        return 1

    method, state, proposal = _proposed(tmp_path)
    evaluation = _evaluation((1.0,) * 3, (0.5,) * 3, candidate_spec=proposal.spec.spec_id)
    _, archive = _judge(method, state, proposal, evaluation)
    method.proposer = FakeProposerAgent(second)
    runner = FakeEpisodeRunner(world)

    result = method.propose(_context(archive.to_state(), current=proposal.spec, runner=runner, policy=retrained))

    assert result.episodes_spent == len(SEARCH) == runner.runs
    assert Archive.from_state(result.state).get("child").policy == retrained.checkpoint.checksum


@pytest.mark.unit
def test_the_search_tasks_are_fixed_by_the_first_move(tmp_path: Path) -> None:
    method, _ = _method(tmp_path, lambda _: 1)
    first = method.propose(_context(method.initial_state(SEED_HARNESS)))

    # A move may offer more tasks, but not fewer.
    wider = method.propose(_context(first.state, tasks=(*SEARCH, Task("train-extra", "More."))))
    assert Archive.from_state(wider.state).task_ids == SEARCH_IDS
    with pytest.raises(ContractError, match="lacks"):
        method.propose(_context(first.state, tasks=SEARCH[:2]))


@pytest.mark.unit
def test_a_head_another_method_wrote_is_adopted_and_served(tmp_path: Path) -> None:
    head = SEED_HARNESS.with_file(INSTRUCTIONS_PATH, HarnessFile("instructions", "Edited elsewhere."))
    root = "head_" + head.spec_id[2:10]
    method, proposer = _method(tmp_path, lambda _: 1)

    result = method.propose(_context(method.initial_state(SEED_HARNESS), current=head))

    archive = Archive.from_state(result.state)
    assert archive.served == root and archive.served_candidate.spec(head.agent) == head
    assert archive.served_candidate.scores is not None
    ((prompt, workdir, _),) = proposer.calls
    assert f"`candidates/{root}/`" in prompt
    assert (workdir / "candidates" / root / INSTRUCTIONS_PATH).read_text() == "Edited elsewhere."


# ------------------------------------------------------------ select and record


@pytest.mark.unit
def test_a_strictly_higher_held_out_mean_is_selected_and_served(tmp_path: Path) -> None:
    method, state, proposal = _proposed(tmp_path)
    evaluation = _evaluation((1.0, 0.5, 0.5), (0.5, 0.5, 0.5), candidate_spec=proposal.spec.spec_id)

    decision, archive = _judge(method, state, proposal, evaluation)

    assert decision.selected and decision.policy == "meta_harness"
    assert decision.metrics["meta_harness_candidate_mean"] == pytest.approx(2 / 3)
    assert decision.metrics["meta_harness_served_mean"] == 0.5
    assert archive.served == "child"
    # The gate's held-out run never enters the archive: the proposer reads only search scores and traces.
    assert archive == replace(Archive.from_state(state), served="child")
    assert all(set(candidate.traces) <= set(SEARCH_IDS) for candidate in archive.candidates)


@pytest.mark.unit
def test_a_tie_is_rejected_and_the_served_candidate_stays(tmp_path: Path) -> None:
    method, state, proposal = _proposed(tmp_path)
    evaluation = _evaluation((1.0, 0.0, 0.5), (0.5, 0.5, 0.5), candidate_spec=proposal.spec.spec_id)

    decision, archive = _judge(method, state, proposal, evaluation)

    assert not decision.selected
    assert archive.served == "seed" and archive == Archive.from_state(state)


@pytest.mark.unit
def test_the_bar_is_the_gates_run_of_the_served_harness_not_its_search_score(tmp_path: Path) -> None:
    method, state, proposal = _proposed(tmp_path)
    assert Archive.from_state(state).served_candidate.mean == 0.5
    # The served harness reruns low on the held-out tasks; that run is the bar, not its 0.5 on the search tasks.
    evaluation = _evaluation((0.25,) * 3, (0.0,) * 3, candidate_spec=proposal.spec.spec_id)

    decision, _ = _judge(method, state, proposal, evaluation)

    assert decision.selected and decision.metrics["meta_harness_served_mean"] == 0.0


@pytest.mark.unit
def test_a_gate_on_a_search_task_is_refused(tmp_path: Path) -> None:
    method, state, proposal = _proposed(tmp_path)
    evaluation = _evaluation(
        (1.0, 1.0), (0.5, 0.5), candidate_spec=proposal.spec.spec_id, task_ids=("sum", "train-sort")
    )

    with pytest.raises(ContractError, match=r"\['train-sort'\].*held-out"):
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
        # The proposer sees the first iteration's search results, and none of the gate's held-out tasks.
        rows = [json.loads(line) for line in (workdir / "evolution_summary.jsonl").read_text().splitlines()]
        assert [(row["name"], row["served"]) for row in rows] == [("seed", False), ("verify", True)]
        assert json.loads((workdir / "frontier_val.json").read_text())["_best"]["candidate"] == "verify"
        assert {path.name for path in (workdir / "traces/verify").iterdir()} == set(SEARCH_IDS)
        return _candidate("terse", "Answer briefly.", parent="verify")(workdir)

    proposer = FakeProposerAgent(_in_turn(_candidate("verify", "Verify the result before answering."), second))
    method = MetaHarnessMethod(proposer, tmp_path / "mh")
    fixture = build_campaign(tmp_path, method=method)
    move = Move(kind="harness", harness=HarnessMoveConfig(tasks=SEARCH, episode_budget=FULL))

    selected = fixture.campaign.advance(move)
    tied = fixture.campaign.advance(move)

    assert selected.decision is not None and selected.decision.selected
    assert selected.release is not None and selected.release.harness is not None
    assert "Verify" in selected.release.harness.files[0][1].content
    # The second candidate ties on the search tasks, so it never reaches the gate.
    assert tied.decision is None and tied.release is None
    assert len(proposer.calls) == 2
