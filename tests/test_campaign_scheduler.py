"""What a scheduler reads and keeps: the history of every move, and its own state in the commit record."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from honing.campaign import ScriptedScheduler, restore_scheduler
from honing.core import CampaignRuleError, ContractError, HarnessMoveConfig, Move
from honing.release import GATE_METRIC_NAMES, read_scheduler_state
from tests.fakes.builder import build_campaign
from tests.fakes.scheduler import HistoryScheduler

TRAIN, HARNESS = Move(kind="train"), Move(kind="harness", harness=HarnessMoveConfig(label="verify"))


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.unit
def test_the_history_holds_every_move_with_its_result_and_gate_metrics(tmp_path: Path) -> None:
    fixture = build_campaign(tmp_path, trainer_gain=-0.1)  # training makes the model worse, so the gate rejects it

    fixture.campaign.run(ScriptedScheduler([TRAIN, HARNESS, HARNESS]))

    rejected, published, skipped = fixture.campaign.state().history
    assert [(r.step, r.move, r.result) for r in (rejected, published, skipped)] == [
        (0, TRAIN, "rejected"),
        (1, HARNESS, "published"),
        (2, HARNESS, "skipped"),
    ]
    assert (rejected.parent_id, published.parent_id, skipped.parent_id) == ("r0000", "r0000", "r0001")
    assert (rejected.release_id, published.release_id, skipped.release_id) == (None, "r0001", None)
    assert (published.label, skipped.label, rejected.label) == ("verify", "verify", None)
    assert skipped.reason == "append_instruction proposed nothing" and skipped.gate_metrics == {}
    # The history carries the same gate numbers the commit record holds for each candidate.
    _, reject_line, publish_line, _ = _records(fixture.campaign.chain.record_path)
    for record, line in ((rejected, reject_line), (published, publish_line)):
        assert set(record.gate_metrics) == set(GATE_METRIC_NAMES)
        metrics = line["candidate"]["metrics"]  # type: ignore[index]
        assert {name: metrics[name] for name in GATE_METRIC_NAMES} == record.gate_metrics
        assert record.reason == line["decision"]["reason"]  # type: ignore[index]
    assert fixture.campaign.state().step == 3


@pytest.mark.unit
def test_the_gate_metrics_are_computed_once_per_gated_move(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import honing.campaign.campaign as campaign_module

    calls: list[object] = []
    computed = campaign_module.gate_metrics

    def counted(evaluation: object) -> Mapping[str, float]:
        calls.append(evaluation)
        return computed(evaluation)  # type: ignore[arg-type]

    monkeypatch.setattr(campaign_module, "gate_metrics", counted)
    fixture = build_campaign(tmp_path)
    fixture.campaign.run(ScriptedScheduler([TRAIN, TRAIN]))

    gated = [record for record in fixture.campaign.state().history if record.result != "skipped"]
    assert len(calls) == len(gated) == 2
    lines = [json.loads(line) for line in fixture.campaign.chain.record_path.read_text().splitlines()[1:]]
    for record, line in zip(gated, lines, strict=True):
        assert {name: line["candidate"]["metrics"][name] for name in record.gate_metrics} == record.gate_metrics


@pytest.mark.unit
def test_a_scheduler_sees_the_previous_moves_rejection_and_its_gate_metrics(tmp_path: Path) -> None:
    fixture = build_campaign(tmp_path, trainer_gain=-0.1)
    scheduler = HistoryScheduler([TRAIN, HARNESS])

    fixture.campaign.run(scheduler)

    first, second, last = scheduler.seen
    assert first.history == () and first.step == 0
    (previous,) = second.history
    assert previous.result == "rejected" and previous.move == TRAIN and second.step == 1
    assert previous.gate_metrics["gate_candidate_mean"] < previous.gate_metrics["gate_incumbent_mean"]
    assert previous.gate_metrics["gate_difference_high"] < 0
    # The rejected candidate never became a release, so the releases alone would not show it.
    assert [release.release_id for release in second.releases] == ["r0000"]
    assert [record.result for record in last.history] == ["rejected", "published"]
    assert scheduler.rejected == (0,)


@pytest.mark.unit
def test_every_line_after_the_schedulers_first_choice_carries_its_state(tmp_path: Path) -> None:
    fixture = build_campaign(tmp_path, trainer_gain=-0.1)
    scheduler = HistoryScheduler([TRAIN, HARNESS, HARNESS])

    fixture.campaign.run(scheduler)

    init, *lines = _records(fixture.campaign.chain.record_path)
    assert "scheduler_state" not in init and "scheduler" not in init
    assert [line["op"] for line in lines] == ["reject", "publish", "skip"]
    assert {line["scheduler"] for line in lines} == {"history"}
    # Each line holds the state after the choice it records. The rejection at step 0 is read on the next choice.
    assert [line["scheduler_state"] for line in lines] == [
        {"schema_version": 1, "next": 1, "rejected": []},
        {"schema_version": 1, "next": 2, "rejected": [0]},
        {"schema_version": 1, "next": 3, "rejected": [0]},
    ]
    assert scheduler.export_state() == lines[-1]["scheduler_state"]


@pytest.mark.unit
def test_the_scheduler_state_survives_a_rejection_and_a_rollback(tmp_path: Path) -> None:
    fixture = build_campaign(tmp_path)
    campaign = fixture.campaign
    scheduler = HistoryScheduler([TRAIN, TRAIN, Move(kind="harness")])
    campaign.step(scheduler)  # r0001
    fixture.trainer.gain = -0.5
    rejected = campaign.step(scheduler)
    assert rejected is not None and rejected.release is None
    after_rejection = dict(scheduler.export_state())

    campaign.rollback("r0000")

    *_, reject_line, rollback_line = _records(campaign.chain.record_path)
    assert reject_line["scheduler_state"] == rollback_line["scheduler_state"] == after_rejection
    assert scheduler.export_state() == after_rejection
    # The scheduler goes on from its place in the script, not from where it was when r0000 was the head.
    outcome = campaign.step(scheduler)
    assert outcome is not None and outcome.move.kind == "harness"
    assert outcome.release is not None and outcome.release.parent_id == "r0000"
    assert scheduler.rejected == (1,)


@pytest.mark.unit
def test_a_restarted_scheduler_continues_from_the_last_recorded_state(tmp_path: Path) -> None:
    moves = [TRAIN, TRAIN, HARNESS, TRAIN]
    fixture = build_campaign(tmp_path / "first", trainer_gain=-0.1)
    fixture.campaign.run(HistoryScheduler(moves), max_steps=3)
    record_path = fixture.campaign.chain.record_path

    # A new process: a fresh scheduler and a fresh campaign. Recovering the chain itself is not implemented yet.
    restarted = HistoryScheduler(moves)
    assert restore_scheduler(restarted, record_path)
    assert restarted.export_state() == read_scheduler_state(record_path)[1]  # type: ignore[index]
    campaign = build_campaign(tmp_path / "second").campaign
    (outcome,) = campaign.run(restarted)

    assert outcome.move == TRAIN  # the fourth move, the one the first run never reached
    assert restarted.rejected == (0, 1)  # remembered from the first run, though this campaign's history is new
    assert campaign.state().history[0].step == 0


@pytest.mark.unit
def test_a_scripted_scheduler_restores_its_place_in_the_script(tmp_path: Path) -> None:
    moves = [TRAIN, HARNESS]
    fixture = build_campaign(tmp_path)
    fixture.campaign.run(ScriptedScheduler(moves), max_steps=1)

    restarted = ScriptedScheduler(moves)
    assert restore_scheduler(restarted, fixture.campaign.chain.record_path)

    state = fixture.campaign.state()
    assert restarted.next_move(state) == HARNESS
    assert restarted.next_move(state) == Move(kind="stop")
    with pytest.raises(ContractError, match="a script of 2 moves, but this one has 3"):
        restore_scheduler(ScriptedScheduler([*moves, TRAIN]), fixture.campaign.chain.record_path)


@pytest.mark.unit
@pytest.mark.parametrize(
    "other",
    [
        [Move(kind="prune"), Move(kind="quantize")],  # the case in #211: same length, other kinds
        [HARNESS, TRAIN],  # same moves, other order
        [TRAIN, Move(kind="harness", harness=HarnessMoveConfig(label="another"))],  # same kinds, another label
    ],
)
def test_a_scripted_scheduler_refuses_another_script_of_the_same_length(tmp_path: Path, other: list[Move]) -> None:
    fixture = build_campaign(tmp_path)
    fixture.campaign.run(ScriptedScheduler([TRAIN, HARNESS]), max_steps=1)
    restarted = ScriptedScheduler(other)

    with pytest.raises(ContractError, match="another script of the same length"):
        restore_scheduler(restarted, fixture.campaign.chain.record_path)
    assert restarted.export_state()["next"] == 0


@pytest.mark.unit
def test_a_record_without_scheduler_state_restores_nothing(tmp_path: Path) -> None:
    fixture = build_campaign(tmp_path)
    fixture.campaign.advance(TRAIN)  # moved by hand, with no scheduler
    scheduler = ScriptedScheduler([TRAIN])

    assert read_scheduler_state(fixture.campaign.chain.record_path) is None
    assert not restore_scheduler(scheduler, fixture.campaign.chain.record_path)
    assert not restore_scheduler(scheduler, tmp_path / "missing.jsonl")
    assert scheduler.export_state()["next"] == 0


@pytest.mark.unit
def test_a_line_cut_short_by_a_crash_is_not_read(tmp_path: Path) -> None:
    fixture = build_campaign(tmp_path)
    fixture.campaign.run(ScriptedScheduler([TRAIN, TRAIN]), max_steps=1)
    path = fixture.campaign.chain.record_path
    complete = read_scheduler_state(path)

    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"op": "publish", "scheduler": "scripted", "scheduler_state": {"next"')

    assert complete is not None and complete[0] == "scripted"
    assert read_scheduler_state(path) == complete
    assert {key: complete[1][key] for key in ("schema_version", "next", "moves")} == {
        "schema_version": 2,
        "next": 1,
        "moves": 2,
    }


@pytest.mark.unit
def test_state_from_another_scheduler_or_schema_is_refused_before_it_is_read(tmp_path: Path) -> None:
    fixture = build_campaign(tmp_path)
    fixture.campaign.run(ScriptedScheduler([TRAIN]))
    path = fixture.campaign.chain.record_path

    with pytest.raises(CampaignRuleError, match="written by scheduler 'scripted', but this campaign runs 'history'"):
        restore_scheduler(HistoryScheduler([TRAIN]), path)
    newer = ScriptedScheduler([TRAIN])
    newer.state_schema_version = 3  # type: ignore[misc]
    with pytest.raises(CampaignRuleError, match="reads scheduler state schema version 3, but the state is version 2"):
        restore_scheduler(newer, path)
    assert newer.export_state()["next"] == 0


class _BadStateScheduler(ScriptedScheduler):
    state_schema_version = 1

    def __init__(self, state: Mapping[str, object]) -> None:
        super().__init__([TRAIN])
        self._state = state

    def export_state(self) -> Mapping[str, object]:
        return self._state


@pytest.mark.unit
@pytest.mark.parametrize(
    ("state", "error", "message"),
    [
        (
            {"schema_version": 2},
            CampaignRuleError,
            "reads scheduler state schema version 1, but the state is version 2",
        ),
        ({"next": 1}, ContractError, "scheduler state must carry an integer 'schema_version'"),
        ({"schema_version": 1, "seen": {"a"}}, ContractError, "scheduler state must be JSON-native"),
    ],
)
def test_a_scheduler_state_that_could_not_be_restored_is_refused_before_the_move(
    tmp_path: Path, state: Mapping[str, object], error: type[Exception], message: str
) -> None:
    fixture = build_campaign(tmp_path)

    with pytest.raises(error, match=message):
        fixture.campaign.step(_BadStateScheduler(state))

    assert fixture.trainer.jobs == [] and len(_records(fixture.campaign.chain.record_path)) == 1


class _LiveStateScheduler(ScriptedScheduler):
    """Exports its live state, nested list included, so the caller keeps a handle on what it handed over."""

    name = "live"
    state_schema_version = 1

    def __init__(self) -> None:
        super().__init__([TRAIN])
        self.live: dict[str, object] = {"schema_version": 1, "rejected": [0]}

    def export_state(self) -> Mapping[str, object]:
        return self.live


@pytest.mark.unit
def test_nothing_a_caller_holds_can_change_what_a_later_line_records(tmp_path: Path) -> None:
    fixture = build_campaign(tmp_path)
    campaign = fixture.campaign
    scheduler = _LiveStateScheduler()
    campaign.step(scheduler)  # r0001
    assert not hasattr(campaign.state(), "scheduler_state") and not hasattr(campaign.chain, "scheduler_state")

    # The review's example on the one object a caller still holds: a nested change, then a rollback.
    rejected = scheduler.live["rejected"]
    assert isinstance(rejected, list)
    rejected.append(99)
    campaign.rollback("r0000")

    *_, publish_line, rollback_line = _records(campaign.chain.record_path)
    assert (publish_line["op"], rollback_line["op"]) == ("publish", "rollback")
    written = {"schema_version": 1, "rejected": [0]}
    assert publish_line["scheduler_state"] == rollback_line["scheduler_state"] == written
