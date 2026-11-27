"""What the decision-model trainers share (honing.backends.jev.decision_training): the examples, the
validation split, and the sft and grpo loops, run on CPU with a small linear model in place of a decision
model. The loop tests need torch; the CI job that installs CPU torch runs them."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from typing import Any

import pytest

from honing.backends.jev.decision_training import (
    decision_examples,
    deduplicated_split,
    group_advantages,
    laya_question,
    policy_gradient_epochs,
    policy_gradient_record,
    record_start,
    supervised_epochs,
)
from honing.core import ContractError, DataContract, DecisionQuestion, DecisionSample, TrainingBatch

QUESTION = DecisionQuestion("choice", "Next?", (("a", "Wait"), ("b", "Build"), ("c", "Attack")))


def decision(
    group: str | None, chosen: str = "a", reward: float | None = None, behavior: float | None = None
) -> DecisionSample:
    probs = None
    if behavior is not None:
        rest = (1.0 - behavior) / 2
        probs = (("q", tuple((key, behavior if key == chosen else rest) for key in ("a", "b", "c"))),)
    return DecisionSample(
        parts=(("essential", (f"state of {group}",)),),
        questions=(("q", QUESTION),),
        chosen=(("q", chosen),),
        behavior_probs=probs,
        reward=reward,
        group=group,
    )


def pack(parts: dict[str, list[str]], questions: dict[str, dict[str, Any]]) -> tuple[Any, Any, Any]:
    text = " ".join(line for lines in parts.values() for line in lines)
    return text, {key: {"inputTokens": len(text)} for key in questions}, "long" in text


def encode(state: Any, questions: dict[str, dict[str, Any]], chosen: dict[str, str]) -> Iterator[tuple[str, Any]]:
    for key, question in questions.items():
        yield key, {"ids": [len(state)], "markers": [0, 1, 2], "label": list(question["criteria"]).index(chosen[key])}


@pytest.mark.unit
def test_validation_groups_are_held_out_and_packing_is_audited() -> None:
    decisions = [decision("m:7:0"), decision("m:101:0", "b"), decision("m:7:1 long", "c")]
    items, validation, packing = decision_examples(
        decisions, pack=pack, encode=encode, validation_groups=r"^[^:]+:101:"
    )
    assert [x["label"] for x in items] == [0, 2] and [x["label"] for x in validation] == [1]
    assert packing == {"maxInputTokens": len("state of m:7:1 long"), "trimmedRequests": 1}
    assert "advantage" not in items[0]


@pytest.mark.unit
def test_grpo_examples_skip_decisions_without_signal_and_carry_advantage_and_behavior() -> None:
    decisions = [
        decision("g", "a", 1.0, 0.5),
        decision("g", "b", 0.0),
        decision("h", "a", 1.0),
        decision("h", "a", 1.0),
    ]
    advantages = group_advantages(decisions)
    items, validation, _ = decision_examples(
        decisions, pack=pack, encode=encode, validation_groups="g", advantages=advantages
    )
    assert validation == [] and [(x["advantage"], x["behavior"]) for x in items] == [(1.0, 0.5), (-1.0, None)]
    with pytest.raises(ContractError, match="nothing to learn"):
        decision_examples(decisions[2:], pack=pack, encode=encode, validation_groups="g", advantages=[0.0, 0.0])


@pytest.mark.unit
def test_a_split_needs_training_examples_and_enough_novel_validation() -> None:
    a = {"ids": [1], "markers": [0], "label": 0}
    b = {"ids": [2], "markers": [0], "label": 1}
    train, novel, audit = deduplicated_split([a, dict(a)], [b], 1)
    assert (len(train), len(novel), audit["uniqueTraining"]) == (1, 1, 1)
    with pytest.raises(ContractError, match="need at least 2 novel validation examples after deduplication, got 0"):
        deduplicated_split([a], [dict(a)], 2)


# ------------------------------------------------------------------ the loops, on CPU


@pytest.fixture
def torch() -> Any:
    return pytest.importorskip("torch")


def toy(torch: Any) -> Any:
    torch.manual_seed(0)
    return torch.nn.Linear(4, 3)


def vectors(n: int, offset: int = 0) -> list[dict[str, Any]]:
    """Examples whose label is the index of the largest of their first three features."""
    rows = []
    for i in range(offset, offset + n):
        label = i % 3
        x = [0.1 * ((i * 7) % 5), 0.0, 0.0, 1.0]
        x[label] += 2.0
        rows.append({"x": x, "ids": [i], "markers": [0, 1, 2], "label": label})
    return rows


def forward_of(torch: Any, model: Any, aux: bool = False) -> Any:
    def forward(chunk: Sequence[dict[str, Any]]) -> tuple[Any, Any, Any]:
        logits = model(torch.tensor([x["x"] for x in chunk]))
        labels = torch.tensor([x["label"] for x in chunk])
        return logits.float(), labels, (logits * 3.0 if aux else None)

    return forward


class CountingSchedule:
    steps = 0

    def step(self) -> None:
        self.steps += 1


@pytest.mark.unit
def test_sft_learns_keeps_the_best_validation_and_steps_the_schedule(torch: Any) -> None:
    model = toy(torch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    items, validation = vectors(30), vectors(9, offset=100)
    saves: list[int] = []
    schedule = CountingSchedule()

    run = supervised_epochs(
        torch, model, optimizer, items, validation, forward=forward_of(torch, model),
        epochs=6, batch_size=8, seed=7, save=lambda: saves.append(len(saves)), schedule=schedule,
    )  # fmt: skip

    losses = [entry["validationLoss"] for entry in run.history]
    improvements = sum(1 for i, loss in enumerate(losses) if loss < min([float("inf"), *losses[:i]]))
    assert run.best == min(losses) < run.initial[0] and len(saves) == improvements
    assert schedule.steps == 6 * 4  # four chunks of at most 8 in each of six epochs
    assert run.history[-1]["validationAccuracy"] == 1.0
    fields, metrics = run.record(items, validation, {"novelValidation": 9})
    assert fields["trainExamples"] == 30 and metrics["validation_loss"] == run.best


@pytest.mark.unit
def test_an_auxiliary_output_stays_in_the_graph_with_weight_zero(torch: Any) -> None:
    weights = []
    for aux in (False, True):
        model = toy(torch)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
        supervised_epochs(
            torch, model, optimizer, vectors(12), vectors(3, offset=50), forward=forward_of(torch, model, aux),
            epochs=2, batch_size=4, seed=1, save=lambda: None,
        )  # fmt: skip
        weights.append(model.weight.detach().clone())
    assert torch.equal(weights[0], weights[1])


@pytest.mark.unit
def test_the_first_grpo_step_starts_at_ratio_one_and_moves_toward_the_advantage(torch: Any) -> None:
    model = toy(torch)
    favored = [{**row, "label": 0, "advantage": 1.0, "behavior": 0.5} for row in vectors(4)]
    examples = favored + [{**row, "label": 1, "advantage": -1.0, "behavior": None} for row in vectors(4)]
    forward = forward_of(torch, model)

    gap = record_start(torch, model, examples, forward=forward, batch_size=16)
    before = torch.tensor([x["old"] for x in favored]).exp()
    assert gap == pytest.approx(float((before - 0.5).abs().mean()))

    one_step = policy_gradient_epochs(
        torch, model, torch.optim.SGD(model.parameters(), lr=0.0), examples, forward=forward,
        epochs=1, batch_size=16, seed=3, clip_ratio=0.2,
    )  # fmt: skip
    assert one_step[0]["ratio"] == pytest.approx(1.0) and one_step[0]["kl"] == pytest.approx(0.0, abs=1e-7)
    assert one_step[0]["clipped"] == 0.0 and one_step[0]["loss"] == pytest.approx(0.0, abs=1e-7)  # mean advantage 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    history = policy_gradient_epochs(
        torch, model, optimizer, examples, forward=forward, epochs=5, batch_size=4, seed=3, clip_ratio=0.2
    )
    with torch.no_grad():
        after = torch.softmax(model(torch.tensor([x["x"] for x in favored])), dim=-1)[:, 0]
    assert (after > before).all() and len(history) == 5 and 0.0 <= history[-1]["clipped"] <= 1.0


@pytest.mark.unit
def test_a_grpo_record_counts_decisions_without_signal() -> None:
    decisions = (decision("g", "a", 1.0), decision("g", "b", 0.0), decision("h", "a", 1.0))
    batch = TrainingBatch("b", DataContract("typed_decisions"), decisions=decisions)
    history = [{"epoch": 1, "loss": -0.5, "clipped": 0.25, "kl": 0.01, "ratio": 1.0}]
    fields, metrics = policy_gradient_record(batch, group_advantages(decisions), [{}, {}], history, 0.1)
    assert (fields["decisionsWithoutSignal"], fields["examples"], fields["meanBehaviorGap"]) == (1, 2, 0.1)
    assert metrics == {
        "policy_loss": -0.5, "approx_kl": 0.01, "clipped_fraction": 0.25, "examples": 2.0,
        "decisions_without_signal": 1.0,
    }  # fmt: skip


# ------------------------------------------------------------------ what the split and the loops refuse


@pytest.mark.unit
def test_one_input_with_two_labels_is_refused_within_a_split_and_across_splits() -> None:
    a = {"ids": [1], "markers": [0], "label": 0}
    with pytest.raises(ContractError, match="Conflicting labels for the same complete model input"):
        deduplicated_split([a, {**a, "label": 1}], [], 0)
    with pytest.raises(ContractError, match="Conflicting labels across episode splits"):
        deduplicated_split([a], [{**a, "label": 2}], 0)
    with pytest.raises(ContractError, match="choice questions only, not score"):
        laya_question(DecisionQuestion("score", "How good?", (("1", "Poor"), ("5", "Great"))))


@pytest.mark.unit
def test_a_nonfinite_training_loss_stops_the_run_before_the_optimizer_steps(torch: Any) -> None:
    model = toy(torch)
    before = model.weight.detach().clone()
    forward = forward_of(torch, model)

    def poisoned(chunk: Sequence[dict[str, Any]]) -> tuple[Any, Any, Any]:
        logits, labels, _ = forward(chunk)
        return logits * float("nan"), labels, None

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(ContractError, match="nonfinite training loss"):
        supervised_epochs(
            torch, model, optimizer, vectors(4), vectors(2, offset=9), forward=poisoned,
            epochs=1, batch_size=4, seed=0, save=lambda: None,
        )  # fmt: skip
    assert torch.equal(model.weight, before)


@pytest.mark.unit
def test_only_an_epoch_that_improves_validation_saves(torch: Any) -> None:
    model = toy(torch)
    saves: list[int] = []
    run = supervised_epochs(
        torch, model, torch.optim.SGD(model.parameters(), lr=0.0), vectors(6), vectors(3, offset=20),
        forward=forward_of(torch, model), epochs=3, batch_size=4, seed=0, save=lambda: saves.append(1),
    )  # fmt: skip
    assert len(saves) == 1  # the first epoch beats infinity; later epochs only tie
    assert run.best == run.initial[0] == run.history[2]["validationLoss"]
    assert [entry["trainLoss"] for entry in run.history] == pytest.approx([run.history[0]["trainLoss"]] * 3)


# ------------------------------------------------------------------ the clipped surrogate, exactly


def started(torch: Any, model: Any, advantage: float, ratio: float) -> list[dict[str, Any]]:
    """Examples whose starting log-probability puts the current weights at ``ratio`` times it."""
    rows = [{**row, "advantage": advantage, "behavior": None} for row in vectors(4)]
    record_start(torch, model, rows, forward=forward_of(torch, model), batch_size=8)
    for row in rows:
        row["old"] -= math.log(ratio)
    return rows


@pytest.mark.unit
def test_a_ratio_past_the_clip_with_a_positive_advantage_gives_no_gradient(torch: Any) -> None:
    model = toy(torch)
    before = model.weight.detach().clone()
    examples = started(torch, model, advantage=1.0, ratio=1.5)

    (epoch,) = policy_gradient_epochs(
        torch, model, torch.optim.SGD(model.parameters(), lr=0.5), examples, forward=forward_of(torch, model),
        epochs=1, batch_size=8, seed=0, clip_ratio=0.2,
    )  # fmt: skip

    assert torch.equal(model.weight, before)  # the surrogate is the constant 1.2 * A
    assert epoch["loss"] == pytest.approx(-1.2) and epoch["ratio"] == pytest.approx(1.5)
    assert epoch["clipped"] == 1.0 and epoch["kl"] == pytest.approx(-math.log(1.5))


@pytest.mark.unit
def test_a_ratio_past_the_clip_with_a_negative_advantage_still_pushes_the_choice_down(torch: Any) -> None:
    model = toy(torch)
    examples = started(torch, model, advantage=-1.0, ratio=1.5)
    forward = forward_of(torch, model)
    with torch.no_grad():
        logits, labels, _ = forward(examples)
        start = torch.log_softmax(logits, -1).gather(1, labels[:, None]).mean()

    (epoch,) = policy_gradient_epochs(
        torch, model, torch.optim.SGD(model.parameters(), lr=0.5), examples, forward=forward,
        epochs=1, batch_size=8, seed=0, clip_ratio=0.2,
    )  # fmt: skip

    assert epoch["loss"] == pytest.approx(1.5)  # min(1.5 * -1, 1.2 * -1): the unclipped term
    with torch.no_grad():
        logits, labels, _ = forward(examples)
        assert torch.log_softmax(logits, -1).gather(1, labels[:, None]).mean() < start


@pytest.mark.unit
def test_policy_gradient_keeps_the_auxiliary_output_steps_the_schedule_and_refuses_a_nonfinite_loss(
    torch: Any,
) -> None:
    model = toy(torch)
    examples = started(torch, model, advantage=1.0, ratio=1.0)
    schedule = CountingSchedule()
    history = policy_gradient_epochs(
        torch, model, torch.optim.SGD(model.parameters(), lr=0.1), examples, forward=forward_of(torch, model, True),
        epochs=2, batch_size=3, seed=0, clip_ratio=0.2, schedule=schedule,
    )  # fmt: skip
    assert schedule.steps == 2 * 2 and [entry["epoch"] for entry in history] == [1, 2]

    def poisoned(chunk: Sequence[dict[str, Any]]) -> tuple[Any, Any, Any]:
        logits, labels, _ = forward_of(torch, model)(chunk)
        return logits, labels, logits * float("inf")

    with pytest.raises(ContractError, match="nonfinite policy-gradient loss"):
        policy_gradient_epochs(
            torch, model, torch.optim.SGD(model.parameters(), lr=0.1), examples, forward=poisoned,
            epochs=1, batch_size=4, seed=0, clip_ratio=0.2,
        )  # fmt: skip


@pytest.mark.unit
def test_record_start_reports_no_gap_when_the_rollout_recorded_no_probabilities(torch: Any) -> None:
    model = toy(torch)
    rows = [{**row, "advantage": 1.0, "behavior": None} for row in vectors(5)]
    assert record_start(torch, model, rows, forward=forward_of(torch, model), batch_size=2) is None
    with torch.no_grad():
        expected = torch.log_softmax(model(torch.tensor([x["x"] for x in rows])), -1)
    assert [x["old"] for x in rows] == pytest.approx([float(expected[i, x["label"]]) for i, x in enumerate(rows)])
