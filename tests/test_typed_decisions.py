"""The typed-decision data contract (#55): decision samples, the jobs that train on them, and the rules around them."""

from __future__ import annotations

from dataclasses import replace

import pytest

from honing.campaign import rules
from honing.core import (
    ALGORITHM_REQUIREMENTS,
    LANGUAGE_MODEL_CONTRACTS,
    BackendSupport,
    CampaignRuleError,
    ContractError,
    DataContract,
    DecisionQuestion,
    DecisionSample,
    HoneRelease,
    RecoverySpec,
    TrainingBatch,
    TrainJobSpec,
    UpdateCandidate,
    requirements_for,
)
from tests.fakes.data import token_batch
from tests.fakes.harness import SEED_HARNESS
from tests.fakes.model import SLIME_LIKE, FakeModelSource, FakeWorld

BUILD = DecisionQuestion(
    kind="choice",
    instructions="What should be built next?",
    options=(("power", "Power plant"), ("refinery", "Ore refinery"), ("wait", "Wait")),
)


def decision(**overrides: object) -> DecisionSample:
    fields: dict[str, object] = {
        "parts": (("essential", ("Tick 30. 5000 credits. No power plant.",)),),
        "questions": (("build", BUILD),),
        "chosen": (("build", "power"),),
    }
    fields.update(overrides)
    return DecisionSample(**fields)  # type: ignore[arg-type]


def rollout_decision(group: str = "mp03t4-red", release: str | None = None) -> DecisionSample:
    return decision(
        behavior_probs=(("build", (("power", 0.6), ("refinery", 0.15), ("wait", 0.25))),),
        reward=1.0,
        group=group,
        source_release=release,
    )


JEV_TRAINER = BackendSupport(
    name="local-jev",
    architectures="uniform",
    train_numerics=frozenset({"bf16"}),
    weight_formats=frozenset({"hf_safetensors"}),
    algorithms=(("sft", "native"), ("grpo", "native")),
    contracts=frozenset({"typed_decisions"}),
)


# ------------------------------------------------------------------ samples and batches


@pytest.mark.unit
def test_a_supervised_decision_needs_only_state_questions_and_choices() -> None:
    sample = decision()
    assert sample.behavior_probs is None and sample.reward is None
    assert BUILD.option_keys == ("power", "refinery", "wait")


@pytest.mark.unit
def test_a_decision_batch_carries_decisions_and_names_their_releases() -> None:
    batch = TrainingBatch(
        "b",
        DataContract("typed_decisions"),
        decisions=(rollout_decision(release="r0003"), decision()),
    )
    assert batch.source_releases == frozenset({"r0003"})


@pytest.mark.unit
def test_a_policy_gradient_batch_needs_rewards_probabilities_and_groups() -> None:
    contract = requirements_for(TrainJobSpec("grpo", rollout="harness", contract="typed_decisions")).data_contract()
    assert (contract.requires_reward, contract.requires_logprobs, contract.requires_group) == (True, True, True)
    TrainingBatch("b", contract, decisions=(rollout_decision(),))
    for missing in ("reward", "behavior_probs", "group"):
        with pytest.raises(ContractError):
            TrainingBatch("b", contract, decisions=(replace(rollout_decision(), **{missing: None}),))


# ------------------------------------------------------------------ jobs and requirements


@pytest.mark.unit
def test_sft_on_decisions_runs_locally_and_needs_no_rollout() -> None:
    requirements = requirements_for(TrainJobSpec("sft", contract="typed_decisions"))
    assert (requirements.contract, requirements.executor, requirements.requires_rollout) == (
        "typed_decisions",
        "local",
        False,
    )


@pytest.mark.unit
def test_a_job_that_names_no_contract_keeps_its_algorithm_s_default() -> None:
    assert requirements_for(TrainJobSpec("sft")) is ALGORITHM_REQUIREMENTS["sft"]
    assert requirements_for(TrainJobSpec("sft", contract="token_sequences")) is ALGORITHM_REQUIREMENTS["sft"]


@pytest.mark.unit
def test_recovery_after_a_prune_still_trains_the_language_model() -> None:
    job = RecoverySpec("sft", steps=5).as_job(TrainJobSpec("sft").numerics)
    assert job.contract is None and requirements_for(job).contract == "token_sequences"


# ------------------------------------------------------------------ campaign rules


@pytest.mark.unit
def test_a_language_model_backend_refuses_a_decision_job() -> None:
    assert SLIME_LIKE.contracts == LANGUAGE_MODEL_CONTRACTS
    with pytest.raises(CampaignRuleError, match="cannot train on typed_decisions data"):
        rules.algorithm_requirements_met(TrainJobSpec("sft", contract="typed_decisions"), SLIME_LIKE)


@pytest.mark.unit
def test_a_decision_backend_accepts_decision_jobs_and_refuses_language_model_ones() -> None:
    rules.algorithm_requirements_met(TrainJobSpec("sft", contract="typed_decisions"), JEV_TRAINER)
    rules.algorithm_requirements_met(TrainJobSpec("grpo", rollout="harness", contract="typed_decisions"), JEV_TRAINER)
    with pytest.raises(CampaignRuleError, match="cannot train on token_sequences"):
        rules.algorithm_requirements_met(TrainJobSpec("sft"), JEV_TRAINER)


@pytest.mark.unit
def test_a_decision_job_takes_a_decision_batch_and_nothing_else() -> None:
    job = TrainJobSpec("sft", contract="typed_decisions")
    rules.batch_meets_contract(job, TrainingBatch("b", DataContract("typed_decisions"), decisions=(decision(),)))
    with pytest.raises(CampaignRuleError, match="needs a typed_decisions batch"):
        rules.batch_meets_contract(job, token_batch())


@pytest.mark.unit
def test_decisions_collected_under_another_release_are_refused() -> None:
    snapshot = FakeModelSource(FakeWorld()).materialize(__import__("pathlib").Path("unused"))
    head = HoneRelease("r0001", UpdateCandidate("init", None, snapshot, harness=SEED_HARNESS), None)
    batch = TrainingBatch("b", DataContract("typed_decisions"), decisions=(rollout_decision(release="r0000"),))
    with pytest.raises(CampaignRuleError, match="batch_from_one_release"):
        rules.batch_from_one_release(batch, head)
