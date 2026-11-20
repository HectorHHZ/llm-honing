"""Decision traces (#168): the tank loader, the compact renderer, and the GEPA and Meta-Harness adapters.

The fixtures are four recorded tank rounds, played by ``match.mjs`` against
the planner with the dry run's stand-in model (``ForecastModel`` in
``tests/fakes/tank.py`` at quality 0.7) and kept byte for byte as the match
wrote them. On a 10-second clock, ``s3114-blue-destroyed`` ends with the agent
shot down after five missed shots, and ``s3138-red-opponent-self-hit`` with the
opponent hit by its own shell. Both were recorded before the engine named the
shell on each shot and kill. On the suite's 30-second clock,
``s3104-blue-opponent-destroyed`` ends with the agent's sixth shell hitting the
opponent after three later shots, and ``s3102-red-timeout`` at the time limit
after twenty missed shots. Those two run to 0.3 and 1.5 MB, so each is kept
packed as ``<name>.tar.gz``. Each round has its rendering at the default cap
next to it as ``<name>.txt``. Set ``HONING_UPDATE_GOLDEN=1`` to write those
again.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tarfile
from collections import Counter
from itertools import pairwise
from pathlib import Path

import pytest

from honing.backends.tank.traces import OUTPUTS, TankTraceSource, load_run
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
from honing.harness.methods.meta_harness.archive import write_archive
from honing.harness.traces import (
    CUT_MARK,
    DEFAULT_AFTER_SECONDS,
    DEFAULT_BEFORE_SECONDS,
    DEFAULT_MAX_TOKENS,
    estimate_tokens,
    render_traces,
)
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

FIXTURES = Path(__file__).parent / "fixtures/decision_traces"
DESTROYED = "s3114-blue-destroyed"
SELF_HIT = "s3138-red-opponent-self-hit"
KILL = "s3104-blue-opponent-destroyed"
TIMEOUT = "s3102-red-timeout"
RUNS = (DESTROYED, SELF_HIT, KILL, TIMEOUT)
KEY_EVENTS = {
    DESTROYED: {"agent destroyed": 1, "missed shot": 5},
    SELF_HIT: {"opponent self-hit": 1, "missed shot": 5},
    KILL: {"opponent destroyed": 1, "missed shot": 8},
    TIMEOUT: {"missed shot": 20},
}
CAPS = (300, 1000, 2000, DEFAULT_MAX_TOKENS, 20000)
RENDERED_VERIFY = re.compile(
    r"^  verify: check [01]\.\d{3}, finish [01]\.\d{3}; chose (?:check|finish)$", re.MULTILINE
)
GAP = re.compile(r"^\(\d+ decisions? not shown, .*\)$", re.MULTILINE)


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Every fixture round's run directory, the packed ones unpacked."""
    unpacked = tmp_path_factory.mktemp("runs")
    for packed in sorted(FIXTURES.glob("*.tar.gz")):
        with tarfile.open(packed) as archive:
            archive.extractall(unpacked, filter="data")
    return {name: FIXTURES / name if (FIXTURES / name).is_dir() else unpacked / name for name in RUNS}


def _copy(tmp_path: Path, run: Path) -> Path:
    return Path(shutil.copytree(run, tmp_path / run.name))


def _rehash(run: Path) -> None:
    """Record the run's outputs as they are now, as match.mjs does when a round ends."""
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["outputs"] = {name: hashlib.sha256((run / name).read_bytes()).hexdigest() for name in OUTPUTS}
    (run / "manifest.json").write_text(json.dumps(manifest))


def _lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _blocks(text: str) -> dict[int, str]:
    """Each shown decision's lines, by step."""
    return {int(block.split(" ", 1)[0]): block for block in GAP.sub("", text).split("\ndecision ")[1:]}


def _windows(text: str) -> list[list[int]]:
    """The shown decisions' steps, grouped into windows: a window ends where a hidden run begins."""
    windows = [
        [int(step) for step in re.findall(r"^decision (\d+) at", chunk, re.MULTILINE)] for chunk in GAP.split(text)
    ]
    return [window for window in windows if window]


def _fields(trace: DecisionTrace) -> dict[str, list[str]]:
    """The state's fields by section: each sentence of each line, once, in order."""
    return {
        section: list(
            dict.fromkeys(
                field
                for line in lines
                for row in line.split("\n")
                for field in re.split(r"(?<=\.)\s+", row.strip())
                if field
            )
        )
        for section, lines in trace.decision.parts
    }


def _policy(world: FakeWorld) -> PolicyRef:
    snapshot = FakeModelSource(world).materialize(Path("unused"))
    return PolicyRef(snapshot.checkpoint, snapshot.architecture, "master")


def _trace(
    step: int,
    state: str = "Me at 1. Enemy at 5.",
    events: tuple[TraceEvent, ...] = (),
    question: str = "weapon",
    applied: str | None = None,
    time: float | None = None,
) -> DecisionTrace:
    """A decision asked at ``step`` seconds, or at ``time``."""
    decision = DecisionSample(
        parts=(("state", (state,)),),
        questions=((question, DecisionQuestion("choice", "Fire?", (("fire", "Fire."), ("hold", "Hold.")))),),
        chosen=((question, "hold"),),
        behavior_probs=((question, (("fire", 0.25), ("hold", 0.75))),),
    )
    return DecisionTrace("e", step, float(step) if time is None else time, decision, events=events, applied=applied)


def _turns(count: int, events: dict[int, TraceEvent]) -> list[DecisionTrace]:
    """``count`` decisions, two per 0.2 s turn as plans-v1 asks them: the motion, then the weapon."""
    return [
        _trace(
            step,
            state=f"Me at {step}.",
            events=(events[step],) if step in events else (),
            question=("motion", "weapon")[step % 2],
            time=0.2 * (step // 2),
        )
        for step in range(count)
    ]


# ------------------------------------------------------------ the loader


@pytest.mark.unit
@pytest.mark.parametrize("name", RUNS)
def test_the_loader_reads_every_decision_and_marks_every_key_event(name: str, runs: dict[str, Path]) -> None:
    run = runs[name]
    events = _lines(run / "events.jsonl")

    traces = load_run(run)

    assert [trace.step for trace in traces] == list(range(len(_lines(run / "decisions.jsonl"))))
    assert {trace.episode_id for trace in traces} == {name}
    keys = Counter(event.label for trace in traces for event in trace.events if event.key)
    assert keys == KEY_EVENTS[name]
    assert sum(n for label, n in keys.items() if label != "missed shot") == sum(e["type"] == "kill" for e in events)
    # Only the agent's own death has priority 0.
    priorities = {(e.label, e.priority) for trace in traces for e in trace.events if e.key}
    assert all((priority == 0) == str(label).startswith("agent ") for label, priority in priorities)
    # Each event follows the decision before it and comes before the next one.
    for trace, after in pairwise(traces):
        assert all(trace.time_seconds - 0.0005 <= e.time_seconds <= after.time_seconds + 0.0005 for e in trace.events)


@pytest.mark.unit
@pytest.mark.parametrize("name", RUNS)
def test_missed_shots_are_the_agents_shots_less_its_hits_in_the_rounds_result(
    name: str, runs: dict[str, Path]
) -> None:
    result = json.loads((runs[name] / "result.json").read_text())
    seat = result["seat"]
    hits = result["enemy_hits"].count(seat) + result["self_hits"].count(seat)

    traces = load_run(runs[name])

    missed = [e for trace in traces for e in trace.events if e.label == "missed shot"]
    assert len(missed) == result["shots"][seat] - hits
    assert {e.text for e in missed} == {f"{seat} fired"}


@pytest.mark.unit
def test_the_hit_is_the_shell_the_kill_names_even_when_later_shots_were_fired(runs: dict[str, Path]) -> None:
    events = _lines(runs[KILL] / "events.jsonl")
    kill = next(e for e in events if e["type"] == "kill")
    shots = [e for e in events if e["type"] == "shot" and e["player"] == kill["player"]]
    assert shots[-1]["shell"] != kill["shell"]

    traces = load_run(runs[KILL])

    fired = [e for trace in traces for e in trace.events if e.text == "blue fired"]
    assert [e.label is None for e in fired] == [shot["shell"] == kill["shell"] for shot in shots]


@pytest.mark.unit
def test_a_kill_by_the_agents_shell_that_names_no_shell_is_refused(tmp_path: Path, runs: dict[str, Path]) -> None:
    run = _copy(tmp_path, runs[KILL])
    events = _lines(run / "events.jsonl")
    for event in events:
        event.pop("shell", None)
    (run / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    _rehash(run)

    with pytest.raises(ContractError, match="names no shell"):
        load_run(run)


@pytest.mark.unit
@pytest.mark.parametrize("name", OUTPUTS)
def test_a_changed_output_is_refused(tmp_path: Path, name: str) -> None:
    run = _copy(tmp_path, FIXTURES / DESTROYED)
    path = run / name
    path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(ContractError, match=f"{re.escape(name)} does not match the SHA-256"):
        load_run(run)


@pytest.mark.unit
def test_a_missing_output_is_refused(tmp_path: Path) -> None:
    run = _copy(tmp_path, FIXTURES / DESTROYED)
    (run / "events.jsonl").unlink()

    with pytest.raises(ContractError, match=r"events\.jsonl is missing"):
        load_run(run)


@pytest.mark.unit
@pytest.mark.parametrize("drop", [None, "result.json"])
def test_a_manifest_without_every_outputs_hash_is_refused(tmp_path: Path, drop: str | None) -> None:
    run = _copy(tmp_path, FIXTURES / DESTROYED)
    manifest = json.loads((run / "manifest.json").read_text())
    if drop is None:
        del manifest["outputs"]
    else:
        del manifest["outputs"][drop]
    (run / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ContractError, match="records no SHA-256"):
        load_run(run)


@pytest.mark.unit
def test_the_trace_source_reads_an_episodes_run_with_its_grade_harness_and_policy() -> None:
    outcome = EpisodeOutcome("tank-val", "h-1", "c", exit_code=0, workspace_location=str(FIXTURES / SELF_HIT))
    record = EpisodeRecord(outcome=outcome, grade=Grade(score=1.0, feedback="won"))

    traces = TankTraceSource().traces(record, release_id="r0002")

    assert traces == load_run(
        FIXTURES / SELF_HIT, grade=record.grade, release_id="r0002", harness_id="h-1", policy_checksum="c"
    )
    assert {(t.release_id, t.harness_id, t.grade, t.policy_checksum) for t in traces} == {
        ("r0002", "h-1", record.grade, "c")
    }
    assert traces[0].outcome == "the red tank won (hit) after 5.85 s"


@pytest.mark.unit
def test_the_header_names_the_release_or_without_one_the_policy_checksum() -> None:
    checksum = hashlib.sha256(b"weights").hexdigest()
    outcome = EpisodeOutcome("tank-val", "h-1", checksum, exit_code=0, workspace_location=str(FIXTURES / SELF_HIT))
    record = EpisodeRecord(outcome=outcome, grade=Grade(score=1.0))

    # GEPA and Meta-Harness pass no release id: EvolveContext carries none.
    unreleased = render_traces(TankTraceSource().traces(record))
    released = render_traces(TankTraceSource().traces(record, release_id="r0002"))

    assert unreleased.startswith(f"Episode {SELF_HIT}: 56 decisions, policy checksum {checksum[:12]}, harness h-1.\n")
    assert released.startswith(f"Episode {SELF_HIT}: 56 decisions, release r0002, harness h-1.\n")
    assert render_traces(TankTraceSource().traces(record)) == unreleased


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
def test_the_rendering_matches_its_golden_text(name: str, runs: dict[str, Path]) -> None:
    text = render_traces(load_run(runs[name]))
    golden = FIXTURES / f"{name}.txt"
    if os.environ.get("HONING_UPDATE_GOLDEN") == "1":
        golden.write_text(text, encoding="utf-8")

    assert text == golden.read_text(encoding="utf-8")
    assert render_traces(load_run(runs[name])) == text
    assert estimate_tokens(text) <= DEFAULT_MAX_TOKENS


@pytest.mark.unit
@pytest.mark.parametrize("name", RUNS)
def test_the_first_decision_of_every_window_shows_the_whole_state(name: str, runs: dict[str, Path]) -> None:
    traces = load_run(runs[name])
    for max_tokens in CAPS:
        text = render_traces(traces, max_tokens=max_tokens)
        if text.endswith(CUT_MARK + "\n"):
            continue
        blocks, fields = _blocks(text), "State fields are left out" not in text
        for window in _windows(text):
            block, trace = blocks[window[0]] + "\n", traces[window[0]]
            if fields:
                for section, items in _fields(trace).items():
                    if items:
                        listed = "\n".join(f"    {field}" for field in items)
                        assert (
                            f"\n  {section}, all {len(items)} field{'s' if len(items) > 1 else ''}:\n{listed}\n"
                            in block
                        )
            for key, value in (("constraints", trace.constraints), ("applied", trace.applied)):
                if value:
                    assert f"\n  {key}: {value}\n" in block


@pytest.mark.unit
@pytest.mark.parametrize("name", RUNS)
def test_every_priority_0_event_is_shown_under_the_decision_it_followed(name: str, runs: dict[str, Path]) -> None:
    traces = load_run(runs[name])
    for max_tokens in (300, 1000, DEFAULT_MAX_TOKENS):
        blocks = _blocks(render_traces(traces, max_tokens=max_tokens))
        for trace in traces:
            for event in trace.events:
                if event.priority == 0:
                    assert f", {event.label}: {event.text}" in blocks[trace.step]


@pytest.mark.unit
@pytest.mark.parametrize("name", RUNS)
def test_the_text_never_exceeds_the_cap(name: str, runs: dict[str, Path]) -> None:
    traces = load_run(runs[name])
    for max_tokens in (8, 50, 200, 250, 400, 700, 1000, 2000, 4000, 8000, 20000):
        text = render_traces(traces, max_tokens=max_tokens)
        assert estimate_tokens(text) <= max_tokens


@pytest.mark.unit
def test_over_the_cap_missed_shots_leave_first_then_state_fields_then_the_text_is_cut() -> None:
    traces = load_run(FIXTURES / DESTROYED)

    full = render_traces(traces, max_tokens=100_000)
    default = render_traces(traces)
    two = render_traces(traces, max_tokens=3000)
    deaths_only = render_traces(traces, max_tokens=2000)
    no_fields = render_traces(traces, max_tokens=1000)
    cut = render_traces(traces, max_tokens=100)

    # Windows reach 2 s before each key event and 0.5 s after it. The five missed shots' merge into 0 to 31,
    # and the death at 6.000 s opens its window at decision 38, the first asked from 4.000 s.
    assert _windows(full) == [list(range(0, 32)), list(range(38, 58))]
    assert traces[37].time_seconds < 4.0 <= traces[38].time_seconds
    # Every missed shot's decision leaves before the death's, and the death's window loses its earliest first.
    assert sorted(_blocks(default)) == [*range(53, 58)]
    # The motion asked with the weapon decision the death followed is the last to leave before it.
    assert sorted(_blocks(two)) == [56, 57] and traces[56].time_seconds == traces[57].time_seconds
    assert sorted(_blocks(deaths_only)) == [57]
    assert "State fields are left out" in no_fields
    assert re.search(r"^  \w+, (?:all|\d+ of) ", no_fields, re.MULTILINE) is None
    assert len(_blocks(no_fields)) > 1
    assert cut.endswith(CUT_MARK + "\n") and estimate_tokens(cut) == 100


@pytest.mark.unit
def test_without_a_priority_0_event_the_decision_that_would_leave_last_stays() -> None:
    traces = load_run(FIXTURES / SELF_HIT)
    opponent_hit = next(trace.step for trace in traces for event in trace.events if event.label == "opponent self-hit")

    for max_tokens in (300, 1000, 2000):
        assert f"\ndecision {opponent_hit} at " in render_traces(traces, max_tokens=max_tokens)


@pytest.mark.unit
def test_a_later_decision_lists_what_changed_and_what_is_gone_since_the_one_shown_before() -> None:
    traces = [
        _trace(0, "Me at 1. Enemy shell incoming."),
        _trace(1, "Me at 1."),
        _trace(2, "Me at 2."),
        _trace(3, "Me at 2."),
    ]

    blocks = _blocks(render_traces(traces, before_seconds=3.0))

    assert "  state, all 2 fields:\n    Me at 1.\n    Enemy shell incoming.\n" in blocks[0]
    assert "  state, 0 of 1 field changed, 1 gone:\n    gone: Enemy shell incoming.\n" in blocks[1]
    assert "  state, 1 of 1 field changed:\n    Me at 2.\n  weapon" in blocks[2]
    assert "  state, 0 of 1 field changed\n" in blocks[3]
    assert "  weapon: fire 0.250, hold 0.750; chose hold" in blocks[3]


@pytest.mark.unit
def test_a_window_after_hidden_decisions_starts_with_the_whole_state_and_controller() -> None:
    missed = TraceEvent(2.5, "blue fired", "missed shot", 1)
    destroyed = TraceEvent(9.5, "red destroyed blue", "agent destroyed", 0)
    traces = [
        _trace(step, events=(missed,) * (step == 2) + (destroyed,) * (step == 9), applied='{"drive":"stop"}')
        for step in range(12)
    ]
    traces[10] = _trace(10, question="motion", applied='{"drive":"stop"}')

    text = render_traces(traces, before_seconds=1.5, after_seconds=1.0)

    assert _windows(text) == [[1, 2, 3], [8, 9, 10]]
    blocks = _blocks(text)
    # Decision 8 has the state and controller decision 3 had, but the decisions between them are hidden.
    for step in (1, 8):
        assert "  state, all 2 fields:\n    Me at 1.\n    Enemy at 5.\n" in blocks[step]
        assert '  applied: {"drive":"stop"}' in blocks[step]
    assert "  state, 0 of 2 fields changed\n" in blocks[9] and "applied" not in blocks[9]
    # Decision 10 is the first in its window with its questions.
    assert "  state, all 2 fields:" in blocks[10]


@pytest.mark.unit
def test_key_events_are_counted_by_label_in_priority_order() -> None:
    traces = [
        _trace(0),
        _trace(1, events=(TraceEvent(1.5, "blue fired", "missed shot", 1),)),
        _trace(2, events=(TraceEvent(2.5, "the episode timed out", "timeout", 2),)),
        _trace(3, events=(TraceEvent(3.5, "blue fell", "fall", 0), TraceEvent(3.6, "blue fired", "missed shot", 1))),
    ]

    text = render_traces(traces)

    assert "\nKey events: fall (1), missed shot (2), timeout (1).\n" in text
    assert "  then 3.500 s, fall: blue fell\n" in text


@pytest.mark.unit
def test_windows_reach_back_and_forward_in_episode_time_whatever_the_decisions_per_turn() -> None:
    # A missed shot follows the weapon decision at 0.4 s, and the death the one at 2.8 s.
    missed = TraceEvent(0.45, "blue fired", "missed shot", 1)
    destroyed = TraceEvent(2.85, "red destroyed blue", "agent destroyed", 0)
    traces = _turns(30, {5: missed, 29: destroyed})

    default = render_traces(traces, max_tokens=100_000)
    narrow = render_traces(traces, max_tokens=100_000, before_seconds=1.0, after_seconds=0.3)

    assert (DEFAULT_BEFORE_SECONDS, DEFAULT_AFTER_SECONDS) == (2.0, 0.5)
    # 2 s before the death is from 0.85 s, which decision 10 at 1.0 s opens, and 0.5 s after the missed shot
    # is to 0.95 s, which decision 9 at 0.8 s closes. The two windows touch, so they merge.
    assert _windows(default) == [list(range(30))]
    # 1 s before the death is from 1.85 s (decision 20 at 2.0 s on), and 0.3 s after the missed shot to 0.75 s.
    assert _windows(narrow) == [list(range(8)), list(range(20, 30))]


@pytest.mark.unit
def test_over_the_cap_a_window_keeps_the_decisions_nearest_in_time_to_the_one_its_event_followed() -> None:
    traces = _turns(30, {29: TraceEvent(2.85, "red destroyed blue", "agent destroyed", 0)})
    whole = len(_blocks(render_traces(traces, max_tokens=100_000)))

    shown = {
        tuple(sorted(_blocks(text)))
        for text in (render_traces(traces, max_tokens=cap) for cap in range(50, 3000, 5))
        if "State fields are left out" not in text
    }

    # Whatever the cap, what stays is the run of decisions up to the death, losing the earliest first.
    assert shown == {tuple(range(30 - count, 30)) for count in range(1, whole + 1)}


@pytest.mark.unit
def test_an_episode_without_key_events_shows_its_last_decisions() -> None:
    traces = [_trace(step) for step in range(9)]
    traces.append(_trace(9, events=(TraceEvent(9.5, "the round ended (timeout), no winner"),)))

    text = render_traces(traces)

    # The last decision was asked at 9 s, and the 2 s before it reach back to decision 7.
    assert "Key events: none.\nShown: 3 decisions at the end of the episode." in text
    assert "(7 decisions not shown, 0.000 s to 6.000 s)" in text
    assert sorted(_blocks(text)) == [7, 8, 9]
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
    with pytest.raises(ContractError, match="in time order"):
        render_traces([_trace(0, time=1.0), _trace(1, time=0.5)])
    for seconds in (-0.1, math.inf, math.nan):
        with pytest.raises(ContractError, match="finite and >= 0"):
            render_traces([_trace(0)], before_seconds=seconds)
        with pytest.raises(ContractError, match="finite and >= 0"):
            render_traces([_trace(0)], after_seconds=seconds)


# ------------------------------------------------------------ the adapters


@pytest.mark.unit
def test_gepa_reflects_on_the_rendered_decisions_of_its_minibatch() -> None:
    pytest.importorskip("gepa", reason="requires the optional 'gepa' extra")
    from honing.harness.methods.gepa import GEPAConfig, GEPAMethod

    world = FakeWorld()
    tasks = tuple(Task(task_id=f"t{index}", prompt=f"Solve problem {index}.") for index in range(3))
    replies = ["```\nSolve the task. Verify the result before answering.\n```"]
    runs = {}
    policy = _policy(world)
    for traces in (None, FakeTraceSource()):
        model = FakeChatModel(replies)
        method = GEPAMethod(model, GEPAConfig(max_metric_calls=100, minibatch_size=3), ("v0",), traces=traces)
        context = EvolveContext(
            current=SEED_HARNESS,
            policy=policy,
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
    # GEPA knows no release id, so each header names the policy by its checksum.
    checksum = policy.checkpoint.checksum[:12]
    headers = re.findall(r"^Episode (t\d): 1 decision, policy checksum (\S+), harness h-", with_traces, re.MULTILINE)
    assert sorted(headers) == [("t0", checksum), ("t1", checksum), ("t2", checksum)]
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
    # The state keeps where each rendering is, as it does a transcript, and never the text.
    assert RENDERED_VERIFY.search(json.dumps(result.state)) is None
    folders = ("search%20a", "search-b")
    for folder, task in zip(folders, tasks, strict=True):
        rendered = archive.get("seed").traces[task.task_id].decisions
        assert rendered is not None and Path(rendered).parent == tmp_path / "mh/decisions"
        written = (tmp_path / "mh/archive/traces/seed" / folder / "decisions.txt").read_text(encoding="utf-8")
        assert written == Path(rendered).read_text(encoding="utf-8")
        assert RENDERED_VERIFY.search(written) is not None

    shutil.rmtree(tmp_path / "mh/decisions")
    write_archive(archive, tmp_path / "again", agent=SEED_HARNESS.agent, policy="ck")
    for folder, task in zip(folders, tasks, strict=True):
        unavailable = (tmp_path / "again/traces/seed" / folder / "decisions.unavailable").read_text(encoding="utf-8")
        assert unavailable == f"{archive.get('seed').traces[task.task_id].decisions}\n"


@pytest.mark.unit
def test_the_dry_runs_chat_model_refuses_a_prompt_without_rendered_decisions() -> None:
    rendered = render_traces(load_run(FIXTURES / DESTROYED))
    model = chat(None, replies=("first", "second"), expect_decisions=True)

    assert model.complete([ChatMessage("user", f"# Example 1\n{rendered}")], max_tokens=10, temperature=0.0)
    with pytest.raises(ContractError, match="does not show each round's decisions"):
        model.complete([ChatMessage("user", "# Example 1\nblue lost")], max_tokens=10, temperature=0.0)
