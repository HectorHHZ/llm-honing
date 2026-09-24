"""The walking skeleton: one campaign that crosses every plane with fakes.

This is the proof that the planes fit together before any of them is real.
Each later change replaces one fake with a real implementation and must keep
these tests passing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from honing.campaign import ScriptedScheduler
from honing.core import CampaignRuleError, ContractError, Move, NumericsPolicy, TrainJobSpec
from tests.fakes.builder import INT4_G32, INT4_G128, build_campaign
from tests.fakes.compression import UnevenExpertSelector
from tests.fakes.data import FakeRolloutProvider


def _ops(record_path: Path) -> list[str]:
    return [json.loads(line)["op"] for line in record_path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.unit
def test_a_campaign_crosses_every_plane(tmp_path: Path) -> None:
    fixture = build_campaign(tmp_path)
    campaign = fixture.campaign
    moves = [Move(kind="train"), Move(kind="prune"), Move(kind="quantize"), Move(kind="harness")]

    outcomes = campaign.run(ScriptedScheduler(moves))

    releases = campaign.chain.releases()
    assert [release.kind for release in releases] == ["init", "train", "prune", "quantize", "harness"]
    assert all(outcome.decision is not None and outcome.decision.selected for outcome in outcomes)
    init, trained, pruned, quantized, evolved = releases

    # Compression: the cut is uniform, recorded on the release, recovered before the gate, and smaller.
    assert pruned.master.parameters < trained.master.parameters
    assert pruned.architecture.uniform
    assert [manifest.kind for manifest in pruned.transforms] == ["prune"]
    recovery_job = fixture.trainer.jobs[-1]
    assert recovery_job.algorithm == "distill"
    assert recovery_job.teacher is not None and recovery_job.teacher.placement == "engine"

    # Quantization: serving weights come from the released master, and the gate scores them.
    assert pruned.serving is None
    assert quantized.serving is not None
    assert quantized.serving.source == quantized.master.checkpoint
    assert [outcome.evaluation.evaluated_variant for outcome in outcomes if outcome.evaluation] == [
        "master",
        "master",
        "serving",
        "serving",
    ]

    # Harness: a new spec, the method's state published with it, the weights unchanged.
    assert evolved.harness != init.harness
    assert evolved.method_state == {"schema_version": 1, "proposals": 1, "accepted": 1, "attempts": 1}
    assert evolved.master == quantized.master and evolved.serving == quantized.serving

    # The honing curve has one point per release, and the size falls at the cut.
    curve = campaign.chain.curve()
    assert [point.release_id for point in curve] == [release.release_id for release in releases]
    assert curve[2].parameters < curve[1].parameters

    # Rollback restores the whole release: weights, serving weights, shape, harness, and method state.
    restored = campaign.rollback(trained.release_id)
    head = campaign.chain.head()
    assert head is restored is trained
    assert head.serving is None
    assert head.architecture == init.architecture
    assert head.harness == init.harness
    assert head.method_state == init.method_state

    assert _ops(campaign.chain.record_path) == ["publish"] * 5 + ["rollback"]


@pytest.mark.unit
def test_a_rejected_candidate_leaves_the_head_serving(tmp_path: Path) -> None:
    fixture = build_campaign(tmp_path, trainer_gain=-0.1)
    campaign = fixture.campaign
    initial = campaign.chain.head()

    (outcome,) = campaign.run(ScriptedScheduler([Move(kind="train")]))

    assert outcome.decision is not None and not outcome.decision.selected
    assert outcome.release is None
    assert campaign.chain.head() is initial
    assert _ops(campaign.chain.record_path) == ["publish", "reject"]


@pytest.mark.unit
def test_a_method_with_nothing_to_propose_is_recorded_without_moving_the_head(tmp_path: Path) -> None:
    campaign = build_campaign(tmp_path).campaign
    campaign.run(ScriptedScheduler([Move(kind="harness")]))
    head = campaign.chain.head()

    (outcome,) = campaign.run(ScriptedScheduler([Move(kind="harness")]))

    assert outcome.skipped_reason == "append_instruction proposed nothing"
    assert campaign.chain.head() is head
    assert _ops(campaign.chain.record_path)[-1] == "skip"


@pytest.mark.unit
def test_an_uneven_cut_is_refused_before_the_editor_runs(tmp_path: Path) -> None:
    fixture = build_campaign(tmp_path, prune_selector=UnevenExpertSelector())

    with pytest.raises(CampaignRuleError) as refused:
        fixture.campaign.run(ScriptedScheduler([Move(kind="prune")]))

    assert refused.value.rule == "architecture_supported"
    assert fixture.editor.applied == []
    assert fixture.trainer.jobs == []


@pytest.mark.unit
def test_an_export_scheme_other_than_the_trained_one_is_refused(tmp_path: Path) -> None:
    fixture = build_campaign(tmp_path)
    qat_g32 = TrainJobSpec(algorithm="sft", numerics=NumericsPolicy(mode="fake_quant_qat", scheme=INT4_G32))
    fixture.campaign.run(ScriptedScheduler([Move(kind="train", train=qat_g32)]))

    with pytest.raises(CampaignRuleError) as refused:
        fixture.campaign.run(ScriptedScheduler([Move(kind="quantize", scheme=INT4_G128)]))

    assert refused.value.rule == "qat_scheme_matches_export"
    assert fixture.quantizer.quantized == []


@pytest.mark.unit
def test_rollout_training_runs_on_samples_from_the_head(tmp_path: Path) -> None:
    rollout = FakeRolloutProvider()
    fixture = build_campaign(tmp_path, rollout=rollout)
    grpo = TrainJobSpec(algorithm="grpo", rollout="harness")

    (outcome,) = fixture.campaign.run(ScriptedScheduler([Move(kind="train", train=grpo)]))

    assert outcome.release is not None and outcome.release.kind == "train"
    assert rollout.collected == 1


@pytest.mark.unit
def test_a_batch_from_another_release_is_refused_before_training(tmp_path: Path) -> None:
    fixture = build_campaign(tmp_path, rollout=FakeRolloutProvider(stamp_release="r9999"))
    grpo = TrainJobSpec(algorithm="grpo", rollout="harness")

    with pytest.raises(CampaignRuleError) as refused:
        fixture.campaign.run(ScriptedScheduler([Move(kind="train", train=grpo)]))

    assert refused.value.rule == "batch_from_one_release"
    assert fixture.trainer.jobs == []


@pytest.mark.unit
def test_a_rollout_job_without_a_rollout_provider_is_refused_before_editing(tmp_path: Path) -> None:
    from honing.core import RecoverySpec, TeacherRef

    opd = RecoverySpec(algorithm="opd", steps=5, teacher=TeacherRef(kind="lineage", ref="parent"))
    fixture = build_campaign(tmp_path, recovery=opd)

    with pytest.raises(ContractError, match="no rollout provider"):
        fixture.campaign.run(ScriptedScheduler([Move(kind="prune")]))

    assert fixture.editor.applied == []


@pytest.mark.unit
def test_exporting_the_trained_scheme_costs_no_quality(tmp_path: Path) -> None:
    fixture = build_campaign(tmp_path)
    qat_g128 = TrainJobSpec(algorithm="sft", numerics=NumericsPolicy(mode="fake_quant_qat", scheme=INT4_G128))

    fixture.campaign.run(ScriptedScheduler([Move(kind="train", train=qat_g128), Move(kind="quantize")]))

    head = fixture.campaign.chain.head()
    assert head.kind == "quantize" and head.serving is not None
    assert (
        fixture.world.quality[head.serving.checkpoint.checksum]
        == fixture.world.quality[head.master.checkpoint.checksum]
    )
