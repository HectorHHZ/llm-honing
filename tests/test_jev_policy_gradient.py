"""Policy gradient for decision models (#57): group advantages, rollouts from RA2 decision logs, and the
trainer's refusals. No GPU."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from honing.backends.jev.policy_gradient import group_advantages
from honing.backends.ra2 import RA2RolloutProvider, decisions_from_log, load_suite
from honing.backends.ra2.rollouts import ROLLOUT_CONTRACT
from honing.campaign import rules
from honing.core import ContractError, DecisionQuestion, DecisionSample, TrainJobSpec
from tests.test_jev_training import _trainer
from tests.test_ra2_backend import SPEC, SUITE_PATH, _endpoints, _grader, _policy, _runner

QUESTION = DecisionQuestion("choice", "Next?", (("A", "Wait"), ("B", "Power plant")))


def sample(reward: float, group: str) -> DecisionSample:
    return DecisionSample(
        parts=(("essential", ("s",)),), questions=(("q", QUESTION),), chosen=(("q", "B"),), reward=reward, group=group
    )


@pytest.mark.unit
def test_advantage_is_the_reward_against_its_group_in_standard_deviations() -> None:
    advantages = group_advantages([sample(1.0, "g"), sample(0.0, "g"), sample(0.5, "h"), sample(1.0, "h")])
    assert advantages == pytest.approx([1.0, -1.0, -1.0, 1.0])


@pytest.mark.unit
def test_a_group_whose_rewards_agree_teaches_nothing() -> None:
    assert group_advantages([sample(1.0, "g"), sample(1.0, "g"), sample(0.0, "solo")]) == [0.0, 0.0, 0.0]


@pytest.mark.unit
def test_advantages_need_a_group_and_a_reward() -> None:
    with pytest.raises(ContractError, match="group and a reward"):
        group_advantages([DecisionSample((), (("q", QUESTION),), (("q", "A"),))])


def _log(run: Path, *, packed: bool = False) -> None:
    stage = {
        "contextParts": {"essential": ["Tick 3."], "logs": ["t1: queued"]},
        "questions": {"g1": {"type": "choice", "instructions": "Next?", "criteria": {"A": "Wait", "B": "Power"}}},
        "answers": {"g1": {"choice": "B", "probabilities": {"A": 0.25, "B": 0.75}}},
    }
    rows = [
        {
            "decisionMode": "staged",
            "stages": [stage, {**stage, "answers": {"g1": {"choice": "A", "probabilities": {"A": 0.6, "B": 0.4}}}}],
        },
        {"decisionMode": "staged", "stages": [{"contextParts": {}, "questions": {}, "answers": {}}]},
    ]
    text = "".join(json.dumps(r) + "\n" for r in rows)
    run.mkdir(parents=True)
    if packed:
        with gzip.open(run / "decisions.jsonl.gz", "wt") as handle:
            handle.write(text)
    else:
        (run / "decisions.jsonl").write_text(text)


@pytest.mark.unit
@pytest.mark.parametrize("packed", [False, True])
def test_each_model_call_in_a_decision_log_is_one_decision(tmp_path: Path, packed: bool) -> None:
    _log(tmp_path / "run", packed=packed)
    decisions = decisions_from_log(tmp_path / "run", reward=0.5, group="task-1", release_id="r0002", harness_id="h-1")
    assert len(decisions) == 2  # the stage with no questions is not a decision
    first, second = decisions
    assert dict(first.chosen) == {"g1": "B"} and dict(second.chosen) == {"g1": "A"}
    assert dict(first.behavior_probs)["g1"] == (("A", 0.25), ("B", 0.75))  # type: ignore[arg-type]
    assert dict(first.parts)["logs"] == ("t1: queued",)
    assert (first.reward, first.group, first.source_release, first.source_harness) == (0.5, "task-1", "r0002", "h-1")


@pytest.mark.unit
def test_a_run_without_a_decision_log_is_refused(tmp_path: Path) -> None:
    (tmp_path / "run").mkdir()
    with pytest.raises(ContractError, match="no decision log"):
        decisions_from_log(tmp_path / "run", reward=1.0, group="g", release_id=None, harness_id=None)


@pytest.mark.unit
def test_rollouts_play_each_task_and_label_decisions_with_score_group_and_release(tmp_path: Path) -> None:
    provider = RA2RolloutProvider(_runner(tmp_path, _endpoints(a="http://strong:1")), _grader())
    tasks = load_suite(SUITE_PATH).tasks("train", limit=2)

    batch = provider.collect(tasks, _policy("a"), SPEC, "r0003", samples_per_task=2)

    assert batch.contract == ROLLOUT_CONTRACT and len(batch.decisions) == 4
    assert {d.group for d in batch.decisions} == {t.task_id for t in tasks}
    assert all(d.reward == 1.0 for d in batch.decisions)
    assert batch.source_releases == frozenset({"r0003"}) and batch.source_harnesses == frozenset({SPEC.spec_id})
    again = provider.collect(tasks, _policy("a"), SPEC, "r0003", samples_per_task=2)
    assert again.batch_id != batch.batch_id


@pytest.mark.unit
def test_rollouts_need_a_harness(tmp_path: Path) -> None:
    provider = RA2RolloutProvider(_runner(tmp_path, _endpoints(a="http://strong:1")), _grader())
    with pytest.raises(ContractError, match="play under a harness"):
        provider.collect(load_suite(SUITE_PATH).tasks("train", limit=1), _policy("a"), None, "r0003", 1)


@pytest.mark.unit
def test_the_trainer_runs_grpo_on_decisions_and_the_rules_require_rollouts(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)
    assert trainer.support.support_for("grpo") == "native"
    rules.algorithm_requirements_met(
        TrainJobSpec("grpo", rollout="harness", contract="typed_decisions"), trainer.support
    )
    with pytest.raises(rules.CampaignRuleError, match="needs a rollout source"):
        rules.algorithm_requirements_met(TrainJobSpec("grpo", contract="typed_decisions"), trainer.support)
