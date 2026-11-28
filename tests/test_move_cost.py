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

from honing.backends.llm_gateway import GatewayEndpoints, GatewayModel
from honing.campaign import ScriptedScheduler
from honing.campaign import cost as cost_module
from honing.campaign.cost import (
    CampaignBudget,
    ChatRate,
    CostMeter,
    MeteredChatModel,
    MeteredPolicyEndpoints,
    budget_from_config,
    unmetered,
)
from honing.core import (
    ChatMessage,
    ChatModel,
    ChatReply,
    ContractError,
    DataContract,
    EpisodeOutcome,
    EvaluationResult,
    EvolveContext,
    EvolveMethod,
    HarnessProposal,
    HarnessSpec,
    HoneRelease,
    Move,
    MoveCost,
    PolicyEndpoint,
    PolicyEndpoints,
    PolicyRef,
    ProposeResult,
    SelectionDecision,
    Task,
    TrainingBatch,
    TrainJobSpec,
    UpdateCandidate,
    charging,
)
from honing.e2e.components import PipelineContext, build_component
from honing.e2e.pipeline import _stopped_at_limit
from honing.e2e.pipeline import main as pipeline_main
from honing.e2e.tracking import step_values
from honing.release import EpisodeEvaluator, PairedIntervalSelector, read_scheduler_state
from tests.fakes.builder import TASKS, build_campaign
from tests.fakes.data import FakeRolloutProvider, ListDatasetProvider, token_batch
from tests.fakes.harness import AppendInstructionMethod, FakeChatModel, FakeEpisodeRunner, FakeTaskGrader
from tests.fakes.scheduler import HistoryScheduler

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
    training = [s["cost"]["gpu_seconds"] - s["cost"]["serving_gpu_seconds"] for s in moves]
    assert training == pytest.approx([15.0, 15.0, 0.0, 15.0, 0.0, 0.0, 0.0, 15.0])
    assert [s["cost"]["episodes"]["gate"] for s in moves] == [6, 6, 6, 6, 6, 6, 0, 6]
    # The runner resolves the stand-in server (1 GPU) through the pipeline's metered endpoints, so every move
    # that played an episode is charged serving time, and the skipped one, which played none, is not.
    served = [sum(s["cost"]["episodes"].values()) > 0 for s in moves]
    assert [s["cost"]["serving_gpu_seconds"] > 0 for s in moves] == served
    total = summary["cost_total"]
    assert total["usd"] == pytest.approx(total["gpu_seconds"] / 3600 * 2.5)
    assert summary["budget"]["stopped"] is None
    assert problems(work, []) == []


@pytest.mark.unit
def test_a_budget_limited_pipeline_stops_and_still_checks_out(tmp_path: Path) -> None:
    # The config's expect lists 9 steps and ends on a rollback. A stop at the limit cuts both short, and is no failure.
    work = _all_planes(tmp_path, budget={"unit": "usd", "limit": 1.0, "gpu_hour_usd": 3600.0})

    summary = json.loads((work / "summary.json").read_text())
    assert [s["move"] for s in summary["steps"]] == ["train"]
    assert summary["failures"] == []
    # 15 GPU-seconds of training plus the gate's serving time, at a dollar a GPU-second.
    spent = summary["steps"][0]["cost"]["usd"]
    assert spent == pytest.approx(15.0 + summary["steps"][0]["cost"]["serving_gpu_seconds"]) and spent > 15.0
    assert summary["budget"] == {
        "unit": "usd",
        "limit": 1.0,
        "spent": spent,
        "overshoot": spent - 1.0,
        "stopped": "limit",
        "reason": f"spent {spent} of the 1.0 usd budget",
    }
    assert problems(work, []) == []


@pytest.mark.unit
def test_no_rollback_runs_after_the_budget_stops_the_pipeline(tmp_path: Path) -> None:
    # The first train move spends the whole budget, so the rollback after it, which costs nothing, does not run.
    schedule = [{"kind": "train", "algorithm": "sft", "steps": 10}, {"kind": "rollback", "to": "r0000"}]
    work = _all_planes(
        tmp_path,
        budget={"unit": "usd", "limit": 1.0, "gpu_hour_usd": 3600.0},
        schedule=schedule,
        expect=["published", "restored"],
    )

    summary = json.loads((work / "summary.json").read_text())
    assert [s["move"] for s in summary["steps"]] == ["train"]
    assert summary["head"] == "r0001" and summary["failures"] == []
    assert [line["op"] for line in _records(work)] == ["publish", "publish"]


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
        campaign=campaign,
        moves=(),
        context=types.SimpleNamespace(work_dir=tmp_path, components={"grader": None}),
        scheduler=ScriptedScheduler(()),
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

    def hide_serving(summary: dict[str, Any]) -> None:
        summary["cost_total"]["episodes"]["gate"] -= 1
        summary["cost_total"]["serving_gpu_seconds"] += 1.0

    _rewrite(work, change_summary=hide_serving)
    assert [p for p in problems(work, []) if "serving_gpu_seconds sum to" in p]

    def restore_serving(summary: dict[str, Any]) -> None:
        summary["cost_total"]["serving_gpu_seconds"] -= 1.0

    _rewrite(work, change_summary=restore_serving)
    assert problems(work, []) == []

    def spend_more(summary: dict[str, Any]) -> None:
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


# ---------------------------------------------------------------- the scheduler slot and the gate's looks


class FreshTaskBlocks:
    """A gate evaluator that plays ``block`` new tasks at every call, so a sequential gate never replays one."""

    def __init__(self, runner: FakeEpisodeRunner, grader: FakeTaskGrader, block: int) -> None:
        self._runner, self._grader, self._block = runner, grader, block
        self.calls = 0

    def evaluate(self, candidate: UpdateCandidate, incumbent: HoneRelease) -> EvaluationResult:
        tasks = tuple(Task(task_id=f"block{self.calls}-{index}", prompt="Answer.") for index in range(self._block))
        self.calls += 1
        return EpisodeEvaluator(self._runner, self._grader, tasks).evaluate(candidate, incumbent)


def _cost_of(record: Mapping[str, Any]) -> MoveCost:
    return MoveCost.from_record(record["cost"])


@pytest.mark.unit
@pytest.mark.parametrize(("gain", "blocks", "stopped_early"), [(0.2, 2, 1.0), (0.0, 3, 0.0)])
def test_a_sequential_gate_is_charged_for_every_block_it_played(
    gain: float, blocks: int, stopped_early: float, tmp_path: Path
) -> None:
    # Every task gains the trainer's gain. A gain of 0.2 clears the second of three looks, on 40 tasks, so the gate
    # stops a block early; a gain of 0 clears none, and the gate plays all three.
    evaluators: list[FreshTaskBlocks] = []

    def gate_evaluator(runner: FakeEpisodeRunner, grader: FakeTaskGrader) -> FreshTaskBlocks:
        evaluators.append(FreshTaskBlocks(runner, grader, block=20))
        return evaluators[-1]

    fixture = build_campaign(
        tmp_path, trainer_gain=gain, gate_evaluator=gate_evaluator, default_selector=PairedIntervalSelector(looks=3)
    )
    outcome = fixture.campaign.advance(Move(kind="train"))

    assert outcome.decision is not None and outcome.decision.metrics["stopped_early"] == stopped_early
    assert evaluators[0].calls == blocks
    assert outcome.cost is not None and dict(outcome.cost.episodes) == _episodes(gate=blocks * 20 * 2)
    # The charge agrees with the episodes the runner actually ran, every block on both sides.
    assert fixture.runner.runs == blocks * 20 * 2
    assert _records(tmp_path)[-1]["cost"]["episodes"]["gate"] == blocks * 20 * 2


@pytest.mark.unit
@pytest.mark.parametrize("bonus", [0.5, 0.0])
def test_a_methods_own_validation_run_is_charged_to_the_method_and_the_gate_only_to_the_gate(
    bonus: float, tmp_path: Path
) -> None:
    gate_tasks = tuple(Task(task_id=f"gate{index}", prompt="Answer.") for index in range(12))
    fixture = build_campaign(
        tmp_path,
        make_runner=lambda world: FakeEpisodeRunner(world, instruction_bonus=bonus),
        gate_tasks=gate_tasks,
        harness_selector=PairedIntervalSelector(),
        method_tasks=TASKS[:2],
    )
    outcome = fixture.campaign.advance(Move(kind="harness"))

    record = _records(tmp_path)[-1]
    method_episodes = int(record["candidate"]["metrics"]["method_episodes"])
    # With a bonus the method's win-loss selects on its 2 tasks and the gate plays its 12; with none, the method
    # rejects and the gate never runs. Either way the method's validation run counts as the method's.
    gate = 2 * len(gate_tasks) if bonus else 0
    assert outcome.decision is not None and (outcome.decision.policy == "win_loss+paired_interval") == bool(bonus)
    assert record["cost"]["episodes"] == _episodes(gate=gate, method=method_episodes + 2 * 2)
    assert fixture.runner.runs == method_episodes + 2 * 2 + gate
    assert outcome.evaluation is not None and (len(outcome.evaluation.records) == gate) == bool(bonus)


@pytest.mark.unit
def test_every_step_record_carries_the_cost_its_commit_line_holds(tmp_path: Path) -> None:
    # A trainer that reports no GPU time makes the rejected train move's GPU-seconds unknown: None, never 0.
    fixture = build_campaign(tmp_path, trainer_gain=-0.1, gpu_seconds_per_step=None)
    scheduler = HistoryScheduler([Move(kind="train"), Move(kind="harness"), Move(kind="harness")])

    outcomes = fixture.campaign.run(scheduler)

    history = fixture.campaign.state().history
    assert [record.result for record in history] == ["rejected", "published", "skipped"]
    lines = _records(tmp_path)[1:]
    assert [line["op"] for line in lines] == ["reject", "publish", "skip"]
    assert [record.cost for record in history] == [_cost_of(line) for line in lines]
    assert [record.cost for record in history] == [outcome.cost for outcome in outcomes]
    assert history[0].cost is not None and history[0].cost.gpu_seconds is None and history[0].cost.usd is None
    assert history[2].cost is not None and history[2].cost.episodes["gate"] == 0
    # The scheduler reads the same costs at its last choice, the stop.
    assert [record.cost for record in scheduler.seen[-1].history] == [record.cost for record in history]
    assert [len(state.history) for state in scheduler.seen] == [0, 1, 2, 3]
    with pytest.raises(ContractError, match="MoveCost or None"):
        replace(history[0], cost=history[0].cost.to_record())  # type: ignore[arg-type]


@pytest.mark.unit
def test_a_budget_stop_asks_the_scheduler_for_no_move_and_records_no_choice(tmp_path: Path) -> None:
    # Each train move costs 15 against a limit of 20, so the third is never admitted.
    fixture = build_campaign(tmp_path, budget=CampaignBudget("usd", 20.0, gpu_hour_usd=3600.0))
    scheduler = HistoryScheduler([Move(kind="train")] * 4)

    outcomes = fixture.campaign.run(scheduler)

    assert len(outcomes) == 2 and len(scheduler.seen) == 2
    assert fixture.campaign.step(scheduler) is None and len(scheduler.seen) == 2
    report = fixture.campaign.budget_report()
    assert report is not None and report["stopped"] == "limit"
    # The last line was written after the scheduler's second choice; no line names a third.
    assert _records(tmp_path)[-1]["scheduler_state"]["next"] == 2
    assert [record.cost for record in fixture.campaign.state().history] == [o.cost for o in outcomes]


@pytest.mark.unit
def test_only_a_stop_right_after_the_first_move_to_reach_the_limit_counts_as_a_stop_at_the_limit() -> None:
    budget = CampaignBudget("usd", 20.0, gpu_hour_usd=3600.0)
    report = {"stopped": "limit"}

    def train(gpu_seconds: float) -> MoveCost:
        return MoveCost(gpu_seconds=gpu_seconds, usd=gpu_seconds)

    assert _stopped_at_limit(budget, report, [train(15.0), train(15.0)])
    # The spend had reached the limit before the last move, so that move ran unadmitted.
    assert not _stopped_at_limit(budget, report, [train(25.0), train(15.0)])
    assert not _stopped_at_limit(budget, report, [train(15.0)])  # never reached the limit
    assert not _stopped_at_limit(budget, {"stopped": "unknown_cost"}, [train(15.0), train(15.0)])
    assert not _stopped_at_limit(None, report, [train(15.0), train(15.0)])
    assert not _stopped_at_limit(budget, report, [])


@pytest.mark.unit
def test_the_checker_accepts_lines_with_scheduler_state_and_cost_and_old_lines_with_neither(tmp_path: Path) -> None:
    work = _all_planes(tmp_path)
    lines = _records(work)
    assert all("scheduler_state" in line for line in lines[1:])
    assert all("cost" in line for line in lines[1:] if line["op"] != "rollback")
    assert problems(work, []) == []

    def strip_record(record: dict[str, Any]) -> None:
        for key in ("cost", "scheduler", "scheduler_state"):
            record.pop(key, None)

    def strip_summary(summary: dict[str, Any]) -> None:
        for key in ("cost_total", "budget"):
            summary.pop(key)
        for step in summary["steps"]:
            step.pop("cost", None)

    _rewrite(work, strip_record, strip_summary)
    assert problems(work, []) == []
    assert read_scheduler_state(work / "releases/commits.jsonl") is None


# ---------------------------------------------------------------- serving GPU time


class FakeClock:
    """Stands in for the meters' clock; the served fakes below move it as if their work took time."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr(cost_module, "_now", fake)
    return fake


class FakeServers:
    """One server per checkpoint, each holding ``gpus`` GPUs (``None``: it does not say)."""

    def __init__(self, gpus: int | None = 2) -> None:
        self.gpus = gpus
        self.resolved = 0

    def resolve(self, policy: PolicyRef) -> PolicyEndpoint:
        self.resolved += 1
        checksum = policy.checkpoint.checksum
        return PolicyEndpoint(f"http://server-{checksum[:8]}", "fake", checksum[:8], gpus=self.gpus)


def _served_runner(endpoints: PolicyEndpoints, clock: FakeClock, seconds: float = 10.0) -> Any:
    class ServedRunner(FakeEpisodeRunner):
        """Asks the endpoints for the policy's server, as a real runner does, and plays for ``seconds``."""

        async def run(self, spec: HarnessSpec, task: Task, policy: PolicyRef) -> EpisodeOutcome:
            endpoints.resolve(policy)
            clock.now += seconds
            return await super().run(spec, task, policy)

    return ServedRunner


class ServedRollouts(FakeRolloutProvider):
    """Plays the head against its server for 30 seconds to collect the batch."""

    def __init__(self, endpoints: PolicyEndpoints, clock: FakeClock) -> None:
        super().__init__()
        self._endpoints = endpoints
        self._clock = clock

    def collect(self, tasks: Sequence[Task], policy: PolicyRef, *args: Any, **kwargs: Any) -> TrainingBatch:
        self._endpoints.resolve(policy)
        self._clock.now += 30.0
        return super().collect(tasks, policy, *args, **kwargs)


class ScriptedTeacher(ListDatasetProvider):
    """Demonstrations from a scripted teacher: 100 seconds of play, and no policy server asked for."""

    def __init__(self, clock: FakeClock) -> None:
        super().__init__([token_batch()])
        self._clock = clock

    def next_batch(self, contract: DataContract) -> TrainingBatch | None:
        self._clock.now += 100.0
        return super().next_batch(contract)


def _served_campaign(tmp_path: Path, clock: FakeClock, servers: FakeServers, **kwargs: Any) -> Any:
    metered = MeteredPolicyEndpoints(servers)  # type: ignore[arg-type]
    return build_campaign(tmp_path, make_runner=_served_runner(metered, clock), **kwargs), metered


@pytest.mark.unit
def test_a_gate_is_charged_its_wall_time_times_both_servers_gpus_each_counted_once(
    tmp_path: Path, clock: FakeClock
) -> None:
    servers = FakeServers(gpus=2)
    fixture, _ = _served_campaign(tmp_path, clock, servers, data=ScriptedTeacher(clock))

    outcome = fixture.campaign.advance(Move(kind="train"))

    # Six gate episodes of 10 s each: a 60 s window in which the candidate's and the incumbent's servers, 2 GPUs
    # each, were resolved three times apiece. Charged once per server: 60 * (2 + 2), not 60 * 6 * 2.
    assert servers.resolved == GATE_EPISODES
    cost = outcome.cost
    assert cost.serving_gpu_seconds == 60.0 * 4
    assert cost.gpu_seconds == GPU_SECONDS_PER_JOB + 60.0 * 4
    # The scripted teacher's 100 s asked for no server, so the demonstrations cost no serving time.
    assert cost.wall_seconds == 100.0 + 60.0
    assert _records(tmp_path)[-1]["cost"]["serving_gpu_seconds"] == 240.0
    assert outcome.release.metrics["cost_serving_gpu_seconds"] == 240.0


@pytest.mark.unit
def test_a_rollout_window_is_charged_for_the_head_server(tmp_path: Path, clock: FakeClock) -> None:
    servers = FakeServers(gpus=2)
    metered = MeteredPolicyEndpoints(servers)  # type: ignore[arg-type]
    fixture = build_campaign(
        tmp_path, rollout=ServedRollouts(metered, clock), make_runner=_served_runner(metered, clock)  # type: ignore[arg-type]
    )

    outcome = fixture.campaign.advance(Move(kind="train", train=TrainJobSpec(algorithm="grpo", rollout="harness")))

    # The rollout: 30 s on the head's server (2 GPUs); the gate: 60 s on two servers (4 GPUs).
    assert outcome.cost.serving_gpu_seconds == 30.0 * 2 + 60.0 * 4


@pytest.mark.unit
def test_the_methods_episodes_are_charged_serving_time_and_a_shared_server_once(
    tmp_path: Path, clock: FakeClock
) -> None:
    servers = FakeServers(gpus=2)
    chat = MeteredChatModel(FakeChatModel(["first"], cost_per_call=0.0))
    fixture, _ = _served_campaign(tmp_path, clock, servers, method=ChattyMethod(chat))

    outcome = fixture.campaign.advance(Move(kind="harness"))

    # The method's one episode: 10 s on the head's server. A harness move keeps the weights, so the gate's six
    # episodes, candidate and incumbent alike, share that one server: 60 s * 2 GPUs.
    assert outcome.cost.episodes["method"] == 1
    assert outcome.cost.serving_gpu_seconds == 10.0 * 2 + 60.0 * 2


@pytest.mark.unit
def test_a_server_that_does_not_report_its_gpus_makes_the_dollars_unknown_and_stops_a_usd_run(
    tmp_path: Path, clock: FakeClock
) -> None:
    fixture, _ = _served_campaign(
        tmp_path, clock, FakeServers(gpus=None), budget=CampaignBudget("usd", 1e6, gpu_hour_usd=1.0)
    )

    outcomes = fixture.campaign.run(ScriptedScheduler([Move(kind="train")] * 3))

    assert len(outcomes) == 1
    cost = _records(tmp_path)[-1]["cost"]
    assert cost["serving_gpu_seconds"] is None and cost["gpu_seconds"] is None and cost["usd"] is None
    report = fixture.campaign.budget_report()
    assert report is not None and report["stopped"] == "unknown_cost"
    assert "reported no GPUs" in str(report["reason"]) and "PolicyEndpoint.gpus" in str(report["reason"])
    assert "FakeTrainer" not in str(report["reason"])  # the trainer did report
    assert not fixture.campaign.admits_move()


@pytest.mark.unit
def test_a_remote_api_endpoint_costs_no_serving_time(tmp_path: Path, clock: FakeClock) -> None:
    fixture, _ = _served_campaign(
        tmp_path, clock, FakeServers(gpus=0), budget=CampaignBudget("usd", 1e6, gpu_hour_usd=1.0)
    )

    outcome = fixture.campaign.advance(Move(kind="harness"))

    assert outcome.cost.serving_gpu_seconds == 0.0 and outcome.cost.usd == 0.0


@pytest.mark.unit
def test_the_meter_refuses_an_endpoint_outside_every_window_and_windows_inside_each_other() -> None:
    meter = CostMeter(None)
    endpoint = PolicyEndpoint("http://server", "fake", "m", gpus=1)
    with pytest.raises(ContractError, match="outside"):
        meter.charge_endpoint(endpoint)
    with meter.purpose("gate"), pytest.raises(ContractError, match="do not nest"), meter.purpose("method"):
        pass


@pytest.mark.unit
def test_a_window_that_raises_is_still_charged(clock: FakeClock) -> None:
    meter = CostMeter(None)
    with pytest.raises(RuntimeError), meter.purpose("gate"):
        meter.charge_endpoint(PolicyEndpoint("http://server", "fake", "m", gpus=3))
        clock.now += 5.0
        raise RuntimeError("the gate failed halfway")
    assert meter.close().serving_gpu_seconds == 15.0


@pytest.mark.unit
def test_metered_endpoints_charge_the_installed_sink_and_pass_everything_else_through(clock: FakeClock) -> None:
    model = GatewayModel(base_url="https://api.example/v1", model="m", api_key_env="KEY")
    gateway = GatewayEndpoints(model)
    metered = MeteredPolicyEndpoints(gateway)
    snapshot = model.snapshot()
    policy = PolicyRef(checkpoint=snapshot.checkpoint, architecture=snapshot.architecture, variant="master")
    meter = CostMeter(None)
    with charging(meter), meter.purpose("gate"):
        endpoint = metered.resolve(policy)
        clock.now += 50.0
    metered.resolve(policy)  # no sink installed: charged nowhere

    assert endpoint.gpus == 0  # a gateway is a remote API, priced by its tokens
    assert meter.close().serving_gpu_seconds == 0.0
    assert metered.api_key_env(policy) == "KEY" and unmetered(metered) is gateway and unmetered(gateway) is gateway


@pytest.mark.unit
def test_a_resolve_on_resolve_endpoints_thread_is_charged_to_the_move_in_progress(clock: FakeClock) -> None:
    from honing.harness.endpoints import resolve_endpoint

    servers = FakeServers(gpus=2)
    metered = MeteredPolicyEndpoints(servers)  # type: ignore[arg-type]
    snapshot = GatewayModel(base_url="https://api.example/v1", model="m").snapshot()
    policy = PolicyRef(checkpoint=snapshot.checkpoint, architecture=snapshot.architecture, variant="master")
    meter = CostMeter(None)
    with charging(meter), meter.purpose("gate"):
        # A runner resolves through resolve_endpoint, on a thread of its own (#154), as the real runners do.
        endpoint = asyncio.run(resolve_endpoint(metered, policy))
        clock.now += 10.0

    assert servers.resolved == 1 and endpoint.gpus == 2
    assert meter.close().serving_gpu_seconds == 10.0 * 2


@pytest.mark.unit
def test_the_gepa_live_policy_is_a_hosted_api_still_priced_behind_metered_endpoints() -> None:
    from honing.e2e.gepa_live import OpenAIEndpoints, live_options

    hosted = OpenAIEndpoints("https://api.openai.com/v1", "gpt-4.1-mini")
    config = {"model": {"provider": "openai", "name": "gpt-4.1-mini"}}

    assert hosted.resolve(None).gpus == 0  # type: ignore[arg-type]  # priced by its tokens
    plain = live_options(config, hosted)
    assert plain.policy_price_model is not None
    assert live_options(config, MeteredPolicyEndpoints(hosted)) == plain  # type: ignore[arg-type]


@pytest.mark.unit
def test_every_pipeline_endpoints_component_is_metered(tmp_path: Path) -> None:
    context = PipelineContext(root=REPO, work_dir=tmp_path)
    built = build_component({"factory": "tests.fakes.pipeline:endpoints"}, context, "endpoints", PolicyEndpoints)
    assert isinstance(built, MeteredPolicyEndpoints)
    again = build_component({"factory": "tests.fakes.pipeline:endpoints"}, context, "endpoints", PolicyEndpoints)
    assert not isinstance(unmetered(again), MeteredPolicyEndpoints)  # wrapped once, never twice


@pytest.mark.unit
@pytest.mark.parametrize(("unit", "refused"), [("usd", True), ("wall_seconds", False)])
def test_a_usd_budget_refuses_claude_code_as_the_meta_harness_proposer(
    tmp_path: Path, unit: str, refused: bool
) -> None:
    budget = CampaignBudget(unit, 10.0, gpu_hour_usd=1.0)  # type: ignore[arg-type]
    context = PipelineContext(root=REPO, work_dir=tmp_path, shared={"budget": budget})
    proposer = {"factory": "honing.e2e.meta_harness:claude_code", "args": {"model": "sonnet"}}
    spec = {"factory": "honing.e2e.meta_harness:method", "args": {"proposer": proposer}}
    if refused:
        with pytest.raises(ContractError, match="ClaudeCodeProposer"):
            build_component(spec, context, "method", EvolveMethod)
    else:
        assert isinstance(build_component(spec, context, "method", EvolveMethod), EvolveMethod)
