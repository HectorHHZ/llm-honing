"""Each campaign rule, with one case it allows and one it refuses."""

from __future__ import annotations

from dataclasses import replace

import pytest

from honing.campaign import rules
from honing.core import (
    BackendSupport,
    CampaignRuleError,
    CheckpointRef,
    CompressionStep,
    CostDelta,
    DataContract,
    Expert,
    HarnessFile,
    HarnessSpec,
    HoneRelease,
    ImportanceMap,
    MethodRequirements,
    ModelSnapshot,
    NumericsPolicy,
    ParamMask,
    PolicySample,
    PrunePlan,
    RecoverySpec,
    ServingWeights,
    TeacherRef,
    TrainingBatch,
    TrainJobSpec,
    UpdateCandidate,
)
from tests.fakes.builder import INT4_G32, INT4_G128
from tests.fakes.data import preference_batch, token_batch
from tests.fakes.harness import SEED_HARNESS
from tests.fakes.model import SLIME_LIKE, parameters_of, small_moe

ARCH = small_moe()
UNEVEN = replace(ARCH, layers=(ARCH.layers[0], replace(ARCH.layers[1], num_experts=6), *ARCH.layers[2:]))
FEWER_EXPERTS = replace(ARCH, layers=tuple(replace(shape, num_experts=6) for shape in ARCH.layers))

LOCAL_DPO = BackendSupport(
    name="local-preference",
    architectures="uniform",
    train_numerics=frozenset({"bf16"}),
    weight_formats=frozenset({"hf_safetensors"}),
    algorithms=(("dpo", "native"),),
)


def checkpoint(checksum: str, weight_format: str = "hf_safetensors") -> CheckpointRef:
    return CheckpointRef(location=f"fake://{checksum}", weight_format=weight_format, checksum=checksum)


def snapshot(checksum: str = "a", weight_format: str = "hf_safetensors") -> ModelSnapshot:
    return ModelSnapshot(checkpoint(checksum, weight_format), ARCH, parameters_of(ARCH))


def head_release() -> HoneRelease:
    candidate = UpdateCandidate(kind="init", parent_id=None, master=snapshot(), harness=SEED_HARNESS)
    return HoneRelease(release_id="r0000", candidate=candidate, decision=None)


def refused(rule: str) -> pytest.RaisesContext:
    return pytest.raises(CampaignRuleError, match=f"^{rule}:")


def shape_changing_plan() -> PrunePlan:
    return PrunePlan(
        removals=(Expert(0, 7), Expert(0, 6)),
        before=ARCH,
        after=FEWER_EXPERTS,
        importance=ImportanceMap(signal="telemetry", scores=()),
        estimated=CostDelta(parameters=parameters_of(FEWER_EXPERTS) - parameters_of(ARCH)),
    )


# ---------------------------------------------------------------- architecture and formats


@pytest.mark.unit
def test_a_uniform_architecture_is_allowed() -> None:
    rules.architecture_supported(FEWER_EXPERTS, SLIME_LIKE)


@pytest.mark.unit
def test_an_uneven_architecture_is_refused_by_a_uniform_backend() -> None:
    with refused("architecture_supported"):
        rules.architecture_supported(UNEVEN, SLIME_LIKE)


@pytest.mark.unit
def test_an_uneven_architecture_is_allowed_by_a_heterogeneous_backend() -> None:
    rules.architecture_supported(UNEVEN, replace(SLIME_LIKE, architectures="heterogeneous"))


@pytest.mark.unit
def test_a_torch_dist_master_is_refused() -> None:
    candidate = UpdateCandidate(kind="train", parent_id="r0000", master=snapshot("b", "megatron_torch_dist"))
    rules.master_is_hf_safetensors(replace(candidate, master=snapshot("b")))
    with refused("master_is_hf_safetensors"):
        rules.master_is_hf_safetensors(candidate)


@pytest.mark.unit
def test_serving_weights_must_come_from_the_master_they_ship_with() -> None:
    master = snapshot("b")
    own = ServingWeights(checkpoint=checkpoint("q"), scheme=INT4_G128, source=master.checkpoint)
    stale = replace(own, source=checkpoint("a"))
    rules.serving_built_from_master(UpdateCandidate(kind="train", parent_id="r0000", master=master, serving=own))
    with refused("serving_built_from_master"):
        rules.serving_built_from_master(UpdateCandidate(kind="train", parent_id="r0000", master=master, serving=stale))


# ---------------------------------------------------------------- numerics and compression


@pytest.mark.unit
def test_a_qat_master_exports_only_its_own_scheme() -> None:
    trained = NumericsPolicy(mode="fake_quant_qat", scheme=INT4_G128)
    rules.qat_scheme_matches_export(trained, INT4_G128)
    rules.qat_scheme_matches_export(NumericsPolicy(), INT4_G32)
    with refused("qat_scheme_matches_export"):
        rules.qat_scheme_matches_export(trained, INT4_G32)


@pytest.mark.unit
def test_an_in_process_teacher_is_refused_after_a_shape_change() -> None:
    in_process = TeacherRef(kind="lineage", ref="parent", placement="in_process")
    engine = replace(in_process, placement="engine")
    plan = shape_changing_plan()
    rules.recovery_teacher_fits_new_shape(
        CompressionStep(transform=plan, recovery=RecoverySpec(algorithm="distill", steps=1, teacher=engine))
    )
    with refused("recovery_teacher_fits_new_shape"):
        rules.recovery_teacher_fits_new_shape(
            CompressionStep(transform=plan, recovery=RecoverySpec(algorithm="distill", steps=1, teacher=in_process))
        )


@pytest.mark.unit
def test_an_in_process_teacher_is_allowed_when_the_shape_is_unchanged() -> None:
    mask_only = PrunePlan(
        removals=(ParamMask(param="layers.0.mlp.weight", mask_location="fake://mask"),),
        before=ARCH,
        after=ARCH,
        importance=ImportanceMap(signal="weights", scores=()),
        estimated=CostDelta(parameters=0),
    )
    teacher = TeacherRef(kind="lineage", ref="parent", placement="in_process")
    rules.recovery_teacher_fits_new_shape(
        CompressionStep(transform=mask_only, recovery=RecoverySpec(algorithm="distill", steps=1, teacher=teacher))
    )


# ---------------------------------------------------------------- training jobs and batches


@pytest.mark.unit
def test_grpo_without_a_rollout_source_is_refused() -> None:
    rules.algorithm_requirements_met(TrainJobSpec(algorithm="grpo", rollout="direct"), SLIME_LIKE)
    with refused("algorithm_requirements_met"):
        rules.algorithm_requirements_met(TrainJobSpec(algorithm="grpo"), SLIME_LIKE)


@pytest.mark.unit
def test_dpo_runs_on_the_local_trainer_and_not_on_slime() -> None:
    dpo = TrainJobSpec(algorithm="dpo")
    rules.algorithm_requirements_met(dpo, LOCAL_DPO)
    with refused("algorithm_requirements_met"):
        rules.algorithm_requirements_met(dpo, SLIME_LIKE)


@pytest.mark.unit
def test_distillation_without_a_teacher_is_refused() -> None:
    with refused("algorithm_requirements_met"):
        rules.algorithm_requirements_met(TrainJobSpec(algorithm="distill"), SLIME_LIKE)


@pytest.mark.unit
def test_a_precision_the_backend_cannot_train_in_is_refused() -> None:
    with refused("algorithm_requirements_met"):
        rules.algorithm_requirements_met(TrainJobSpec(algorithm="dpo", numerics=NumericsPolicy(mode="fp8")), LOCAL_DPO)


@pytest.mark.unit
def test_a_batch_must_match_the_algorithm_contract() -> None:
    sft = TrainJobSpec(algorithm="sft")
    rules.batch_meets_contract(sft, token_batch())
    with refused("batch_meets_contract"):
        rules.batch_meets_contract(sft, preference_batch())
    with refused("batch_meets_contract"):
        rules.batch_meets_contract(sft, None)


def rollout_batch(source_release: str | None, source_harness: str | None) -> TrainingBatch:
    sample = PolicySample(
        tokens=(1, 2),
        loss_mask=(0, 1),
        reward=1.0,
        logprobs=(0.0, -0.1),
        source_release=source_release,
        source_harness=source_harness,
    )
    return TrainingBatch(
        batch_id="rollout",
        contract=DataContract(kind="on_policy_samples", requires_reward=True, requires_logprobs=True),
        samples=(sample,),
    )


@pytest.mark.unit
def test_a_batch_from_the_head_is_allowed_and_offline_data_always_is() -> None:
    head = head_release()
    rules.batch_from_one_release(rollout_batch("r0000", SEED_HARNESS.spec_id), head)
    rules.batch_from_one_release(token_batch(), head)


@pytest.mark.unit
def test_a_batch_from_another_release_or_harness_is_refused() -> None:
    head = head_release()
    with refused("batch_from_one_release"):
        rules.batch_from_one_release(rollout_batch("r0007", SEED_HARNESS.spec_id), head)
    other = SEED_HARNESS.with_file("CLAUDE.md", HarnessFile(kind="instructions", content="Be brief.\n"))
    with refused("batch_from_one_release"):
        rules.batch_from_one_release(rollout_batch("r0000", other.spec_id), head)


# ---------------------------------------------------------------- harness


@pytest.mark.unit
def test_a_method_needing_a_programmable_loop_is_refused_by_fixed_loop_agents() -> None:
    from tests.fakes.harness import FakeEpisodeRunner
    from tests.fakes.model import FakeWorld

    support = FakeEpisodeRunner(FakeWorld()).support
    rules.method_supported_by_runner(MethodRequirements(file_kinds=frozenset({"instructions"})), support)
    with refused("method_supported_by_runner"):
        rules.method_supported_by_runner(MethodRequirements(programmable_loop=True), support)
    with refused("method_supported_by_runner"):
        rules.method_supported_by_runner(MethodRequirements(file_kinds=frozenset({"program"})), support)


@pytest.mark.unit
def test_a_spec_for_an_agent_the_runner_lacks_is_refused() -> None:
    from tests.fakes.harness import FakeEpisodeRunner
    from tests.fakes.model import FakeWorld

    support = FakeEpisodeRunner(FakeWorld()).support
    rules.spec_runnable(SEED_HARNESS, support)
    with refused("spec_runnable"):
        rules.spec_runnable(HarnessSpec(agent="codex", files=SEED_HARNESS.files), support)
