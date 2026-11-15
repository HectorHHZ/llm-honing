"""Decision traces (#168): the tank loader, the compact renderer, and the GEPA and Meta-Harness adapters.

The fixtures are two recorded tank rounds, played by ``match.mjs`` with the
dry run's stand-in model on a 10-second clock, and kept byte for byte as the
match wrote them. ``s3114-blue-destroyed`` ends with the agent shot down after
two missed shots, and ``s3138-red-opponent-self-hit`` with the opponent hit by
its own shell. Each has its rendering at the default cap next to it as
``<name>.txt``. Set ``HONING_UPDATE_GOLDEN=1`` to write those again.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections import Counter
from itertools import pairwise
from pathlib import Path

import pytest

from honing.backends.tank.traces import (
    OUTPUTS,
    SEATS,
    SHELL_LIFETIME_SECONDS,
    TankTraceSource,
    load_run,
)
from honing.core import (
    ChatMessage,
    ContractError,
    DecisionQuestion,
    DecisionSample,
    DecisionTrace,
    EpisodeOutcome,
    EpisodeRecord,
    EvolveContext,
    Grade,
    PolicyRef,
    Task,
    TraceEvent,
)
from honing.harness.methods.meta_harness import Archive, MetaHarnessMethod
from honing.harness.traces import CUT_MARK, DEFAULT_MAX_TOKENS, estimate_tokens, render_traces
from tests.fakes.harness import (
    SEED_HARNESS,
    FakeChatModel,
    FakeEpisodeRunner,
    FakeProposerAgent,
    FakeTaskGrader,
    FakeTraceSource,
)
from tests.fakes.model import FakeModelSource, FakeWorld
from tests.fakes.tank import chat

REPO = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures/decision_traces"
DESTROYED = "s3114-blue-destroyed"
SELF_HIT = "s3138-red-opponent-self-hit"
RUNS = (DESTROYED, SELF_HIT)
KEY_EVENTS = {DESTROYED: {"death": 1, "missed_shot": 2}, SELF_HIT: {"self_hit": 1}}
RENDERED_VERIFY = re.compile(
    r"^  verify: check [01]\.\d{3}, finish [01]\.\d{3}; chose (?:check|finish)$", re.MULTILINE
)


def _copy(tmp_path: Path, name: str) -> Path:
    return Path(shutil.copytree(FIXTURES / name, tmp_path / name))


def _lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _blocks(text: str) -> dict[int, str]:
    """Each shown decision's lines, by step."""
    return {int(block.split(" ", 1)[0]): block for block in text.split("\ndecision ")[1:]}


def _policy(world: FakeWorld) -> PolicyRef:
    snapshot = FakeModelSource(world).materialize(Path("unused"))
    return PolicyRef(snapshot.checkpoint, snapshot.architecture, "master")


def _trace(step: int, state: str = "Me at 1. Enemy at 5.", events: tuple[TraceEvent, ...] = ()) -> DecisionTrace:
    question = DecisionQuestion("choice", "Fire?", (("fire", "Fire."), ("hold", "Hold.")))
    decision = DecisionSample(
        parts=(("state", (state,)),),
        questions=(("weapon", question),),
        chosen=(("weapon", "hold"),),
        behavior_probs=(("weapon", (("fire", 0.25), ("hold", 0.75))),),
    )
    return DecisionTrace("e", step, float(step), decision, events=events)


# ------------------------------------------------------------ the loader


@pytest.mark.unit
@pytest.mark.parametrize("name", RUNS)
def test_the_loader_reads_every_decision_and_marks_every_key_event(name: str) -> None:
    run = FIXTURES / name
    seat = json.loads((run / "manifest.json").read_text())["seat"]
    events = _lines(run / "events.jsonl")
    end = next(float(str(e["time"])) for e in events if e["type"] == "round")

    traces = load_run(run)

    assert [trace.step for trace in traces] == list(range(len(_lines(run / "decisions.jsonl"))))
    assert {trace.episode_id for trace in traces} == {name}
    keys = Counter(event.key for trace in traces for event in trace.events if event.key)
    assert keys == KEY_EVENTS[name]
    kills = [e for e in events if e["type"] == "kill"]
    assert keys["death"] + keys["self_hit"] == len(kills)
    assert keys["self_hit"] == sum(1 for e in kills if e.get("self"))
    me = SEATS.index(seat)
    missed = [e for e in events if e["type"] == "shot" and e["player"] == me]
    assert keys["missed_shot"] == sum(1 for e in missed if float(str(e["time"])) + SHELL_LIFETIME_SECONDS < end)
    # Each event follows the decision before it and comes before the next one.
    for trace, after in pairwise(traces):
        assert all(trace.time_seconds - 0.0005 <= e.time_seconds <= after.time_seconds + 0.0005 for e in trace.events)


@pytest.mark.unit
def test_the_shell_lifetime_matches_the_engines() -> None:
    engine = (REPO / "integrations/tank-war/web/game/engine.js").read_text(encoding="utf-8")
    match = re.search(r"SHELL_LIFETIME = ([\d.]+)", engine)
    assert match is not None and float(match.group(1)) == SHELL_LIFETIME_SECONDS


@pytest.mark.unit
@pytest.mark.parametrize("name", OUTPUTS)
def test_a_changed_output_is_refused(tmp_path: Path, name: str) -> None:
    run = _copy(tmp_path, DESTROYED)
    path = run / name
    path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(ContractError, match=f"{re.escape(name)} does not match the SHA-256"):
        load_run(run)


@pytest.mark.unit
def test_a_missing_output_is_refused(tmp_path: Path) -> None:
    run = _copy(tmp_path, DESTROYED)
    (run / "events.jsonl").unlink()

    with pytest.raises(ContractError, match=r"events\.jsonl is missing"):
        load_run(run)


@pytest.mark.unit
@pytest.mark.parametrize("drop", [None, "result.json"])
def test_a_manifest_without_every_outputs_hash_is_refused(tmp_path: Path, drop: str | None) -> None:
    run = _copy(tmp_path, DESTROYED)
    manifest = json.loads((run / "manifest.json").read_text())
    if drop is None:
        del manifest["outputs"]
    else:
        del manifest["outputs"][drop]
    (run / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ContractError, match="records no SHA-256"):
        load_run(run)


@pytest.mark.unit
def test_the_trace_source_reads_an_episodes_run_with_its_grade_and_harness() -> None:
    outcome = EpisodeOutcome("tank-val", "h-1", "c", exit_code=0, workspace_location=str(FIXTURES / SELF_HIT))
    record = EpisodeRecord(outcome=outcome, grade=Grade(score=1.0, feedback="won"))

    traces = TankTraceSource().traces(record, release_id="r0002")

    assert traces == load_run(FIXTURES / SELF_HIT, grade=record.grade, release_id="r0002", harness_id="h-1")
    assert {(t.release_id, t.harness_id, t.grade) for t in traces} == {("r0002", "h-1", record.grade)}
    assert traces[0].outcome == "the red tank won (hit) after 5.85 s"


@pytest.mark.unit
@pytest.mark.parametrize(
    "outcome",
    [
        EpisodeOutcome("t", "h", "c", exit_code=1, workspace_location="/nowhere"),
        EpisodeOutcome("t", "h", "c", exit_code=0, workspace_location="/nowhere", failure="match timed out"),
        EpisodeOutcome("t", "h", "c", exit_code=0),
    ],
    ids=["nonzero exit", "failure", "no workspace"],
)
def test_an_episode_that_left_no_finished_run_has_no_traces(outcome: EpisodeOutcome) -> None:
    assert TankTraceSource().traces(EpisodeRecord(outcome=outcome, grade=Grade(score=0.0))) == ()


# ------------------------------------------------------------ the renderer


@pytest.mark.unit
@pytest.mark.parametrize("name", RUNS)
def test_the_rendering_matches_its_golden_text(name: str) -> None:
    text = render_traces(load_run(FIXTURES / name))
    golden = FIXTURES / f"{name}.txt"
    if os.environ.get("HONING_UPDATE_GOLDEN") == "1":
        golden.write_text(text, encoding="utf-8")

    assert text == golden.read_text(encoding="utf-8")
    assert render_traces(load_run(FIXTURES / name)) == text
    assert estimate_tokens(text) <= DEFAULT_MAX_TOKENS


@pytest.mark.unit
@pytest.mark.parametrize("name", RUNS)
def test_every_death_and_self_hit_is_shown_under_the_decision_it_followed(name: str) -> None:
    traces = load_run(FIXTURES / name)
    for max_tokens in (300, 1000, DEFAULT_MAX_TOKENS):
        blocks = _blocks(render_traces(traces, max_tokens=max_tokens))
        for trace in traces:
            for event in trace.events:
                if event.key in ("death", "self_hit"):
                    label = event.key.replace("_", "-")
                    assert f", {label}: {event.text}" in blocks[trace.step]


@pytest.mark.unit
@pytest.mark.parametrize("name", RUNS)
def test_the_text_never_exceeds_the_cap(name: str) -> None:
    traces = load_run(FIXTURES / name)
    for max_tokens in (8, 50, 200, 250, 400, 700, 1000, 2000, 4000, 8000, 20000):
        text = render_traces(traces, max_tokens=max_tokens)
        assert estimate_tokens(text) <= max_tokens


@pytest.mark.unit
def test_over_the_cap_missed_shots_leave_first_then_state_fields_then_the_text_is_cut() -> None:
    traces = load_run(FIXTURES / DESTROYED)

    full = render_traces(traces, max_tokens=100_000)
    default = render_traces(traces)
    deaths_only = render_traces(traces, max_tokens=2000)
    no_fields = render_traces(traces, max_tokens=400)
    cut = render_traces(traces, max_tokens=100)

    # Windows of 4 decisions before and 2 after each key event: steps 7 to 17 and 53 to 57.
    assert sorted(_blocks(full)) == [*range(7, 18), *range(53, 58)]
    assert {11, 15, 57} <= set(_blocks(default)) < set(_blocks(full))
    assert set(_blocks(deaths_only)) <= set(range(53, 58)) and 57 in _blocks(deaths_only)
    assert "State fields are left out" in no_fields and "fields changed" not in no_fields
    assert len(_blocks(no_fields)) > 1
    assert cut.endswith(CUT_MARK + "\n") and estimate_tokens(cut) == 100


@pytest.mark.unit
def test_a_decision_lists_only_the_fields_that_changed_since_the_last_one_with_the_same_questions() -> None:
    traces = [_trace(0), _trace(1, "Me at 2. Enemy at 5."), _trace(2, "Me at 2. Enemy at 5.")]

    blocks = _blocks(render_traces(traces))

    assert "state, all 2 fields:\n    Me at 1.\n    Enemy at 5." in blocks[0]
    assert "state, 1 of 2 fields changed:\n    Me at 2.\n" in blocks[1]
    assert "state, 0 of 2 fields changed\n" in blocks[2]
    assert "  weapon: fire 0.250, hold 0.750; chose hold" in blocks[2]


@pytest.mark.unit
def test_an_episode_without_key_events_shows_its_last_decisions() -> None:
    traces = [_trace(step) for step in range(9)]
    traces.append(_trace(9, events=(TraceEvent(9.5, "the round ended (timeout), no winner"),)))

    text = render_traces(traces)

    assert "Key events: none.\nShown: 5 decisions at the end of the episode." in text
    assert "(5 decisions not shown, 0.000 s to 4.000 s)" in text
    assert sorted(_blocks(text)) == [5, 6, 7, 8, 9]
    assert "  then 9.500 s: the round ended (timeout), no winner" in text


@pytest.mark.unit
def test_an_empty_episode_renders_as_such_and_mixed_or_unordered_steps_are_refused() -> None:
    assert render_traces([]) == "No decisions were recorded in this episode.\n"
    with pytest.raises(ContractError):
        render_traces([_trace(1), _trace(0)])
    with pytest.raises(ContractError):
        render_traces([_trace(0), DecisionTrace("other", 1, 1.0, _trace(1).decision)])
    with pytest.raises(ContractError):
        render_traces([_trace(0)], max_tokens=5)


# ------------------------------------------------------------ the adapters


@pytest.mark.unit
def test_gepa_reflects_on_the_rendered_decisions_of_its_minibatch() -> None:
    pytest.importorskip("gepa", reason="requires the optional 'gepa' extra")
    from honing.harness.methods.gepa import GEPAConfig, GEPAMethod

    world = FakeWorld()
    tasks = tuple(Task(task_id=f"t{index}", prompt=f"Solve problem {index}.") for index in range(3))
    replies = ["```\nSolve the task. Verify the result before answering.\n```"]
    runs = {}
    for traces in (None, FakeTraceSource()):
        model = FakeChatModel(replies)
        method = GEPAMethod(model, GEPAConfig(max_metric_calls=100, minibatch_size=3), ("v0",), traces=traces)
        context = EvolveContext(
            current=SEED_HARNESS,
            policy=_policy(world),
            tasks=tasks,
            records=(),
            state=method.initial_state(SEED_HARNESS),
            runner=FakeEpisodeRunner(world),
            grader=FakeTaskGrader(),
            episode_budget=100,
        )
        method.propose(context)
        runs[traces is None] = "\n".join(message.content for message in model.conversations[0])

    with_traces, without = runs[False], runs[True]
    assert len(RENDERED_VERIFY.findall(with_traces)) == 3
    assert sorted(re.findall(r"^Episode (t\d): 1 decision, harness h-", with_traces, re.MULTILINE)) == [
        "t0",
        "t1",
        "t2",
    ]
    assert RENDERED_VERIFY.search(without) is None and "verified=0" in without


@pytest.mark.unit
def test_meta_harness_writes_each_search_episodes_rendered_decisions_for_the_proposer(tmp_path: Path) -> None:
    world = FakeWorld()
    tasks = (Task(task_id="search a", prompt="Add."), Task(task_id="search-b", prompt="Sort."))
    method = MetaHarnessMethod(FakeProposerAgent(), tmp_path / "mh", traces=FakeTraceSource())
    context = EvolveContext(
        current=SEED_HARNESS,
        policy=_policy(world),
        tasks=tasks,
        records=(),
        state=method.initial_state(SEED_HARNESS),
        runner=FakeEpisodeRunner(world),
        grader=FakeTaskGrader(),
        episode_budget=5,
    )

    result = method.propose(context)

    archive = Archive.from_state(result.state)
    assert Archive.from_state(json.loads(json.dumps(result.state))) == archive
    for folder, task in zip(("search%20a", "search-b"), tasks, strict=True):
        written = (tmp_path / "mh/archive/traces/seed" / folder / "decisions.txt").read_text(encoding="utf-8")
        assert written == archive.get("seed").traces[task.task_id].decisions
        assert RENDERED_VERIFY.search(written) is not None


@pytest.mark.unit
def test_the_dry_runs_chat_model_refuses_a_prompt_without_rendered_decisions() -> None:
    rendered = render_traces(load_run(FIXTURES / DESTROYED))
    model = chat(None, replies=("first", "second"), expect_decisions=True)

    assert model.complete([ChatMessage("user", f"# Example 1\n{rendered}")], max_tokens=10, temperature=0.0)
    with pytest.raises(ContractError, match="does not show each round's decisions"):
        model.complete([ChatMessage("user", "# Example 1\nblue lost")], max_tokens=10, temperature=0.0)
