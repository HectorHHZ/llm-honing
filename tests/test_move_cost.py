"""Every move's cost (#167): episodes by purpose, GPU-seconds, chat tokens and dollars, wall clock, and the budget."""

from __future__ import annotations

import asyncio
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from honing.campaign import ScriptedScheduler
from honing.campaign.cost import CampaignBudget, ChatRate, CostMeter, MeteredChatModel, budget_from_config
from honing.core import (
    ChatMessage,
    ChatModel,
    ChatReply,
    ContractError,
    EvaluationResult,
    EvolveContext,
    HarnessProposal,
    HarnessSpec,
    Move,
    MoveCost,
    ProposeResult,
    SelectionDecision,
    TrainJobSpec,
    charging,
)
from honing.e2e.components import PipelineContext, build_component
from honing.e2e.pipeline import main as pipeline_main
from honing.e2e.tracking import step_values
from tests.fakes.builder import build_campaign
from tests.fakes.harness import AppendInstructionMethod, FakeChatModel

REPO = Path(__file__).resolve().parents[1]
ALL_PLANES = REPO / "configs/e2e/all_planes.json"

sys.path.insert(0, str(REPO / "scripts/e2e"))
from check_campaign import problems  # noqa: E402

GPU_SECONDS_PER_JOB = 15.0  # the fake trainer's 1.5 GPU-seconds a step, times the 10-step default jobs
GATE_EPISODES = 6  # three gate tasks, on both sides


def _records(work: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (work / "releases/commits.jsonl").read_text().splitlines()]


def _episodes(gate: int = 0, method: int = 0, rollout: int = 0, demonstration: int = 0) -> dict[str, int]:
    return {"gate": gate, "method": method, "rollout": rollout, "demonstration": demonstration}


class ChattyMethod:
    """Adds the verify instruction like :class:`AppendInstructionMethod`, after one chat call and one episode."""

    name = "append_instruction"
    state_schema_version = 1

    def __init__(self, chat: ChatModel) -> None:
        self._inner = AppendInstructionMethod()
        self._chat = chat
        self.requires = self._inner.requires
        self.selector = self._inner.selector

    def initial_state(self, seed: HarnessSpec) -> Mapping[str, object]:
        return self._inner.initial_state(seed)

    def propose(self, context: EvolveContext) -> ProposeResult:
        self._chat.complete([ChatMessage("user", "propose")], max_tokens=8, temperature=0.0)
        asyncio.run(context.episode(context.current, context.tasks[0], context.policy))
        return replace(self._inner.propose(context), episodes_spent=1)

    def record(
        self,
        state: Mapping[str, object],
        proposal: HarnessProposal,
        evaluation: EvaluationResult,
        decision: SelectionDecision,
    ) -> Mapping[str, object]:
        return self._inner.record(state, proposal, evaluation, decision)


class NamedChat:
    """A chat model with a name and no price, as a local server is."""

    model = "local/qwen"

    def complete(self, messages: Sequence[ChatMessage], *, max_tokens: int, temperature: float) -> ChatReply:
        return ChatReply(text="ok", input_tokens=1_000_000, output_tokens=500_000, cost_usd=None)


# ---------------------------------------------------------------- the record


@pytest.mark.unit
def test_a_cost_counts_every_purpose_and_sums_with_unknowns_kept_unknown() -> None:
    known = MoveCost(episodes=_episodes(gate=2), gpu_seconds=3.0, input_tokens=5, api_usd=0.5, usd=1.0)
    unknown = MoveCost(episodes=_episodes(method=1), gpu_seconds=None, output_tokens=7, api_usd=None, usd=None)

    total = known.plus(unknown)

    assert dict(total.episodes) == _episodes(gate=2, method=1)
    assert (total.input_tokens, total.output_tokens) == (5, 7)
    assert total.gpu_seconds is None and total.api_usd is None and total.usd is None
    assert dict(known.plus(known).episodes) == _episodes(gate=4) and known.plus(known).usd == 2.0
    assert MoveCost.from_record(json.loads(json.dumps(unknown.to_record()))) == unknown


@pytest.mark.unit
def test_a_cost_puts_only_its_known_parts_in_the_release_metrics() -> None:
    metrics = MoveCost(episodes=_episodes(rollout=4), gpu_seconds=None, api_usd=0.25, usd=None).metrics()

    assert metrics["cost_episodes_rollout"] == 4.0 and metrics["cost_api_usd"] == 0.25
    assert "cost_gpu_seconds" not in metrics and "cost_usd" not in metrics


@pytest.mark.unit
@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"episodes": {"gate": 1}}, "exactly"),
        ({"episodes": _episodes(gate=-1)}, "gate episodes"),
        ({"input_tokens": True}, "input_tokens"),
        ({"usd": -0.1}, "usd must be"),
        ({"gpu_seconds": math.nan}, "gpu_seconds must be"),
    ],
)
def test_a_cost_refuses_values_that_could_not_be_spent(change: dict[str, Any], message: str) -> None:
    with pytest.raises(ContractError, match=message):
        MoveCost(**change)


@pytest.mark.unit
def test_a_cost_record_with_missing_or_unknown_keys_is_refused() -> None:
    record = MoveCost().to_record()
    with pytest.raises(ContractError, match="keys"):
        MoveCost.from_record({**record, "extra": 1})
    with pytest.raises(ContractError, match="keys"):
        MoveCost.from_record({k: v for k, v in record.items() if k != "usd"})


# ---------------------------------------------------------------- the meter


@pytest.mark.unit
def test_an_episode_outside_every_purpose_is_refused() -> None:
    meter = CostMeter(None)
    with pytest.raises(ContractError, match="outside"):
        meter.charge_episode()


@pytest.mark.unit
def test_the_meter_prices_unpriced_replies_from_the_budget_and_keeps_the_rest_unknown() -> None:
    budget = CampaignBudget("usd", 10.0, gpu_hour_usd=1.0, chat_rates={"local/qwen": ChatRate(2.0, 4.0)})
    meter = CostMeter(budget)
    meter.charge_chat("local/qwen", 1_000_000, 500_000, None)
    meter.charge_chat("local/qwen", 10, 10, 0.5)  # the client's own price is kept over the rate
    assert meter.close().api_usd == pytest.approx(2.0 + 2.0 + 0.5)

    meter.charge_chat("elsewhere", 1, 1, None)
    cost = meter.close()
    assert cost.api_usd is None and cost.usd is None and meter.unpriced_models == ("elsewhere",)


@pytest.mark.unit
def test_gpu_time_is_priced_at_the_budgets_rate_and_unknown_without_one() -> None:
    priced = CostMeter(CampaignBudget("usd", 10.0, gpu_hour_usd=3.6))
    priced.charge_gpu_seconds(100.0)
    assert priced.close().usd == pytest.approx(0.1)

    no_rate = CostMeter(None)
    no_rate.charge_gpu_seconds(100.0)
    assert no_rate.close().gpu_seconds == 100.0 and no_rate.close().usd is None

    unreported = CostMeter(CampaignBudget("usd", 10.0, gpu_hour_usd=3.6))
    unreported.charge_gpu_seconds(None)
    unreported.charge_gpu_seconds(5.0)
    assert unreported.close().gpu_seconds is None and unreported.close().usd is None


@pytest.mark.unit
def test_a_metered_chat_model_charges_each_call_to_the_installed_sink() -> None:
    chat = MeteredChatModel(FakeChatModel(["one two"], cost_per_call=0.125))
    meter = CostMeter(None)
    with charging(meter):
        chat.complete([ChatMessage("user", "a b c")], max_tokens=4, temperature=0.0)
    chat_outside = MeteredChatModel(FakeChatModel(["x"], cost_per_call=9.0))
    chat_outside.complete([ChatMessage("user", "x")], max_tokens=4, temperature=0.0)  # no sink: charged nowhere

    cost = meter.close()
    assert (cost.input_tokens, cost.output_tokens, cost.api_usd) == (3, 2, 0.125)


# ---------------------------------------------------------------- the budget config


@pytest.mark.unit
@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"unit": "usd", "limit": 1.0}, "gpu_hour_usd"),
        ({"unit": "usd", "limit": 1.0, "gpu_hour_usd": 1.0, "rate": 2}, "unknown budget keys"),
        ({"unit": "tokens", "limit": 1.0}, "unit"),
        ({"unit": "wall_seconds", "limit": 0}, "limit"),
        ({"unit": "wall_seconds"}, "a unit and a limit"),
        ({"unit": "usd", "limit": "5", "gpu_hour_usd": 1.0}, "must be a number"),
        ({"unit": "usd", "limit": 1, "gpu_hour_usd": 1, "chat_usd_per_million_tokens": {"m": {"input": 1}}}, "input"),
    ],
)
def test_a_bad_budget_is_refused(spec: dict[str, Any], message: str) -> None:
    with pytest.raises(ContractError, match=message):
        budget_from_config(spec)


@pytest.mark.unit
def test_a_usd_budget_refuses_a_chat_model_it_cannot_price_when_the_config_is_built(tmp_path: Path) -> None:
    budget = budget_from_config({"unit": "usd", "limit": 1.0, "gpu_hour_usd": 1.0})
    context = PipelineContext(root=REPO, work_dir=tmp_path, shared={"budget": budget})

    with pytest.raises(ContractError, match="returns no price"):
        build_component({"factory": "tests.fakes.tank:chat"}, context, "chat", ChatModel)
    priced = build_component(
        {"factory": "tests.fakes.tank:chat", "args": {"cost_per_call": 0.01}}, context, "chat", ChatModel
    )
    assert isinstance(priced, MeteredChatModel)

    with pytest.raises(ContractError, match="local/qwen"):
        assert budget is not None
        budget.check_chat_model(MeteredChatModel(NamedChat()))
    rated = CampaignBudget("usd", 1.0, gpu_hour_usd=1.0, chat_rates={"local/qwen": ChatRate(1.0, 1.0)})
    rated.check_chat_model(MeteredChatModel(NamedChat()))
    CampaignBudget("wall_seconds", 1.0).check_chat_model(MeteredChatModel(NamedChat()))


# ---------------------------------------------------------------- the campaign


@pytest.mark.unit
def test_every_move_kind_records_its_cost_in_the_chain_and_the_release_metrics(tmp_path: Path) -> None:
    fixture = build_campaign(tmp_path)
    campaign = fixture.campaign
    outcomes = [campaign.advance(Move(kind=kind)) for kind in ("train", "prune", "quantize", "harness", "harness")]

    expected = [
        ("publish", _episodes(gate=GATE_EPISODES), GPU_SECONDS_PER_JOB),
        ("publish", _episodes(gate=GATE_EPISODES), GPU_SECONDS_PER_JOB),  # the distill recovery
        ("publish", _episodes(gate=GATE_EPISODES), 0.0),
        ("publish", _episodes(gate=GATE_EPISODES), 0.0),
        ("skip", _episodes(), 0.0),
    ]
    records = _records(tmp_path)[1:]
    assert [r["op"] for r in records] == [op for op, _, _ in expected]
    for outcome, record, (_, episodes, gpu_seconds) in zip(outcomes, records, expected, strict=True):
        assert outcome.cost is not None
        assert record["cost"] == outcome.cost.to_record()
        assert record["cost"]["episodes"] == episodes and record["cost"]["gpu_seconds"] == gpu_seconds
        if record["op"] == "publish":
            assert record["candidate"]["metrics"]["cost_gpu_seconds"] == gpu_seconds
            assert record["candidate"]["metrics"]["cost_episodes_gate"] == GATE_EPISODES
    # No budget, so no GPU-hour price: GPU time is known but its dollars are not.
    assert outcomes[0].cost.usd is None and outcomes[2].cost.usd == 0.0  # type: ignore[union-attr]
    assert campaign.spent.episodes["gate"] == 4 * GATE_EPISODES


@pytest.mark.unit
def test_a_rejected_move_is_charged(tmp_path: Path) -> None:
    fixture = build_campaign(tmp_path, trainer_gain=-0.1)

    outcome = fixture.campaign.advance(Move(kind="train"))

    assert outcome.release is None
    reject = _records(tmp_path)[-1]
    assert reject["op"] == "reject"
    assert reject["cost"]["episodes"] == _episodes(gate=GATE_EPISODES)
    assert reject["cost"]["gpu_seconds"] == GPU_SECONDS_PER_JOB
    assert reject["candidate"]["metrics"]["cost_gpu_seconds"] == GPU_SECONDS_PER_JOB


@pytest.mark.unit
def test_chat_dollars_and_method_episodes_reach_the_move_including_a_skipped_one(tmp_path: Path) -> None:
    chat = MeteredChatModel(FakeChatModel(["first", "second"], cost_per_call=0.25))
    fixture = build_campaign(tmp_path, method=ChattyMethod(chat))  # type: ignore[arg-type]

    published = fixture.campaign.advance(Move(kind="harness"))
    skipped = fixture.campaign.advance(Move(kind="harness"))

    assert published.release is not None and skipped.skipped_reason is not None
    publish, skip = _records(tmp_path)[1:]
    assert publish["cost"]["episodes"] == _episodes(gate=GATE_EPISODES, method=1)
    assert skip["cost"]["episodes"] == _episodes(method=1)
    for record in (publish, skip):
        assert record["cost"]["api_usd"] == 0.25 and record["cost"]["usd"] == 0.25
        assert (record["cost"]["input_tokens"], record["cost"]["output_tokens"]) == (1, 1)
    assert skip["episodes_spent"] == skip["cost"]["episodes"]["method"]


@pytest.mark.unit
def test_a_budget_stops_admitting_moves_once_spent_and_reports_the_overshoot(tmp_path: Path) -> None:
    # One GPU-second costs one dollar, so each 10-step train move costs 15 against a limit of 20.
    budget = CampaignBudget("usd", 20.0, gpu_hour_usd=3600.0)
    fixture = build_campaign(tmp_path, budget=budget)
    moves = [Move(kind="train", train=TrainJobSpec(algorithm="sft", steps=10)) for _ in range(4)]

    outcomes = fixture.campaign.run(ScriptedScheduler(moves))

    assert len(outcomes) == 2  # the second move started at 15 < 20 and ended at 30
    assert fixture.campaign.budget_report() == {
        "unit": "usd",
        "limit": 20.0,
        "spent": 30.0,
        "overshoot": 10.0,
        "stopped": "limit",
        "reason": "spent 30.0 of the 20.0 usd budget",
    }
    assert not fixture.campaign.admits_move()


@pytest.mark.unit
def test_a_spend_that_reaches_the_limit_exactly_admits_no_more_moves(tmp_path: Path) -> None:
    fixture = build_campaign(tmp_path, budget=CampaignBudget("usd", 15.0, gpu_hour_usd=3600.0))

    outcomes = fixture.campaign.run(ScriptedScheduler([Move(kind="train")] * 3))

    assert len(outcomes) == 1
    assert fixture.campaign.budget_report() == {
        "unit": "usd",
        "limit": 15.0,
        "spent": 15.0,
        "overshoot": 0.0,
        "stopped": "limit",
        "reason": "spent 15.0 of the 15.0 usd budget",
    }


@pytest.mark.unit
def test_a_wall_clock_budget_stops_after_the_move_that_spends_it(tmp_path: Path) -> None:
    fixture = build_campaign(tmp_path, budget=CampaignBudget("wall_seconds", 1e-9))

    outcomes = fixture.campaign.run(ScriptedScheduler([Move(kind="train")] * 3))

    report = fixture.campaign.budget_report()
    assert len(outcomes) == 1 and report is not None and report["stopped"] == "limit"
    assert report["spent"] == outcomes[0].cost.wall_seconds  # type: ignore[union-attr]


@pytest.mark.unit
def test_unknown_gpu_dollars_under_a_usd_budget_are_recorded_and_stop_the_campaign(tmp_path: Path) -> None:
    fixture = build_campaign(tmp_path, budget=CampaignBudget("usd", 20.0, gpu_hour_usd=1.0), gpu_seconds_per_step=None)

    outcomes = fixture.campaign.run(ScriptedScheduler([Move(kind="train")] * 3))

    assert len(outcomes) == 1
    record = _records(tmp_path)[-1]
    assert record["cost"]["gpu_seconds"] is None and record["cost"]["usd"] is None
    assert "cost_usd" not in record["candidate"]["metrics"]
    report = fixture.campaign.budget_report()
    assert report is not None and report["stopped"] == "unknown_cost" and report["spent"] is None
    assert "FakeTrainer reported no gpu_seconds" in str(report["reason"])
    assert not fixture.campaign.admits_move()


class UnpricedChat(NamedChat):
    """Claims to price its replies, then returns one without a price, as a misconfigured client might."""

    def prices_replies(self) -> bool:
        return True


@pytest.mark.unit
def test_an_unpriced_chat_reply_names_the_model_and_the_config_key_that_would_price_it(tmp_path: Path) -> None:
    budget = CampaignBudget("usd", 20.0, gpu_hour_usd=1.0)
    budget.check_chat_model(MeteredChatModel(UnpricedChat()))  # passes the config-time check
    fixture = build_campaign(tmp_path, budget=budget, method=ChattyMethod(MeteredChatModel(UnpricedChat())))  # type: ignore[arg-type]

    outcomes = fixture.campaign.run(ScriptedScheduler([Move(kind="harness"), Move(kind="train")]))

    assert [outcome.move.kind for outcome in outcomes] == ["harness"]
    assert _records(tmp_path)[-1]["cost"]["api_usd"] is None
    report = fixture.campaign.budget_report()
    assert report is not None and report["stopped"] == "unknown_cost"
    assert "budget.chat_usd_per_million_tokens[local/qwen]" in str(report["reason"])


# ---------------------------------------------------------------- the pipeline and the checker


def _all_planes(tmp_path: Path, exit_code: int = 0, **changes: Any) -> Path:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({**json.loads(ALL_PLANES.read_text()), **changes}))
    work = tmp_path / "out"
    assert pipeline_main(["--config", str(config), "--root", str(REPO), "--work-dir", str(work)]) == exit_code
    return work


@pytest.mark.unit
def test_every_move_of_the_all_planes_run_has_a_cost_that_sums_to_the_summary(tmp_path: Path) -> None:
    work = _all_planes(tmp_path)

    summary = json.loads((work / "summary.json").read_text())
    moves = [s for s in summary["steps"] if s["move"] != "rollback"]
    assert all(s["cost"] is not None for s in moves)
    assert [s["cost"]["gpu_seconds"] for s in moves] == [15.0, 15.0, 0.0, 15.0, 0.0, 0.0, 0.0, 15.0]
    assert [s["cost"]["episodes"]["gate"] for s in moves] == [6, 6, 6, 6, 6, 6, 0, 6]
    assert summary["cost_total"]["usd"] == pytest.approx(60.0 / 3600 * 2.5)
    assert summary["budget"]["stopped"] is None
    assert problems(work, []) == []


@pytest.mark.unit
def test_a_budget_limited_pipeline_stops_and_still_checks_out(tmp_path: Path) -> None:
    work = _all_planes(tmp_path, budget={"unit": "usd", "limit": 1.0, "gpu_hour_usd": 3600.0}, expect=[])

    summary = json.loads((work / "summary.json").read_text())
    assert [s["move"] for s in summary["steps"]] == ["train"]
    assert summary["budget"] == {
        "unit": "usd",
        "limit": 1.0,
        "spent": 15.0,
        "overshoot": 14.0,
        "stopped": "limit",
        "reason": "spent 15.0 of the 1.0 usd budget",
    }
    assert problems(work, []) == []


@pytest.mark.unit
def test_a_pipeline_whose_spend_cannot_be_priced_stops_writes_its_summary_and_fails(tmp_path: Path) -> None:
    config = json.loads(ALL_PLANES.read_text())
    trainer = {**config["components"]["trainer"], "args": {"gain": 0.1, "gpu_seconds_per_step": None}}
    work = _all_planes(tmp_path, 1, components={**config["components"], "trainer": trainer})

    summary = json.loads((work / "summary.json").read_text())
    assert [s["move"] for s in summary["steps"]] == ["train"]
    assert summary["budget"]["stopped"] == "unknown_cost" and summary["budget"]["spent"] is None
    assert summary["cost_total"]["usd"] is None
    # Only a stop at the limit excuses the steps it cut short; an unpriced stop fails on them too.
    assert summary["failures"] == [
        f"stopped: {summary['budget']['reason']}",
        f"1 steps ran, but expect lists {len(config['expect'])}",
        "the run ended before schedule step 9 (rollback to r0001) ran",
    ]
    assert [p for p in problems(work, []) if "could not be priced" in p]


@pytest.mark.unit
@pytest.mark.parametrize("module", ["honing.backends.tank.e2e", "honing.backends.ra2.e2e"])
def test_a_game_campaign_writes_its_summary_and_fails_when_its_spend_cannot_be_priced(
    module: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import importlib
    import types

    e2e = importlib.import_module(module)
    campaign = types.SimpleNamespace(run=lambda scheduler: ())
    run = types.SimpleNamespace(
        campaign=campaign, moves=(), context=types.SimpleNamespace(work_dir=tmp_path), scheduler=ScriptedScheduler(())
    )
    stop = {"stopped": "unknown_cost", "reason": "chat model m returned no price"}
    summary = {"head": "r0000", "steps": [], "releases": "", "runs": "", "budget": stop}
    monkeypatch.setattr(e2e, "build", lambda config, root, work_dir: run)
    monkeypatch.setattr(e2e, "summarize", lambda outcomes, built: summary)
    config = tmp_path / "config.json"
    config.write_text("{}")

    assert e2e.main(["--config", str(config)]) == 1
    assert json.loads((tmp_path / "summary.json").read_text())["budget"] == stop
    assert "chat model m returned no price" in capsys.readouterr().err

    summary["budget"] = {"stopped": "limit", "reason": "spent 2.0 of the 1.0 usd budget"}
    assert e2e.main(["--config", str(config)]) == 0


def _rewrite(work: Path, change_record: Any = None, change_summary: Any = None) -> None:
    records = _records(work)
    if change_record is not None:
        for record in records:
            change_record(record)
    (work / "releases/commits.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    summary = json.loads((work / "summary.json").read_text())
    if change_summary is not None:
        change_summary(summary)
    (work / "summary.json").write_text(json.dumps(summary))


@pytest.mark.unit
def test_the_checker_reads_a_run_written_before_costs_and_catches_costs_that_do_not_add_up(tmp_path: Path) -> None:
    work = _all_planes(tmp_path)

    def drop_total(summary: dict[str, Any]) -> None:
        summary["cost_total"]["episodes"]["gate"] += 1

    _rewrite(work, change_summary=drop_total)
    assert [p for p in problems(work, []) if "episodes sum to" in p]

    def spend_more(summary: dict[str, Any]) -> None:
        summary["cost_total"]["episodes"]["gate"] -= 1
        summary["budget"]["spent"] = 1.0

    _rewrite(work, change_summary=spend_more)
    assert [p for p in problems(work, []) if "budget reports" in p]

    def only_first(record: dict[str, Any]) -> None:
        if record.get("release_id") != "r0001":
            record.pop("cost", None)

    _rewrite(work, change_record=only_first)
    assert [p for p in problems(work, []) if "carry no cost" in p]

    def old(summary: dict[str, Any]) -> None:
        for key in ("cost_total", "budget"):
            summary.pop(key)
        for step in summary["steps"]:
            step.pop("cost", None)

    _rewrite(work, change_record=lambda record: record.pop("cost", None), change_summary=old)
    assert problems(work, []) == []


@pytest.mark.unit
def test_a_steps_cost_is_logged_as_cost_and_an_unknown_part_is_not_logged() -> None:
    step = {"move": "train", "cost": MoveCost(gpu_seconds=None, usd=None, api_usd=0.5).to_record()}
    record = {"op": "publish", "candidate": {"metrics": {"cost_api_usd": 0.5, "gate_tasks": 3.0}}}

    values = step_values(step, [record])

    assert values["cost/api_usd"] == 0.5 and values["cost/episodes_gate"] == 0.0
    assert "cost/usd" not in values and "cost/gpu_seconds" not in values
    assert "candidate/cost_api_usd" not in values and values["candidate/gate_tasks"] == 3.0
