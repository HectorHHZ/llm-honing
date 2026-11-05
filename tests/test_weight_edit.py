"""Real checkpoint edits, refusal paths, Trainer conformance, and campaign publication."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
st = pytest.importorskip("safetensors.torch")

from honing.backends.sglang.checkpoint import hf_snapshot  # noqa: E402
from honing.backends.weight_edit import EditPlan, TensorEdit, WeightEditTrainer, weight_edit_trainer  # noqa: E402
from honing.core import ContractError, EvaluationResult, TrainJobSpec, UpdatePolicy  # noqa: E402
from honing.core.errors import CampaignRuleError  # noqa: E402
from honing.e2e.components import PipelineContext  # noqa: E402
from honing.e2e.pipeline import build, run  # noqa: E402
from honing.e2e.trainer_conformance import check_trainer  # noqa: E402
from tests.fakes.builder import TASKS  # noqa: E402
from tests.fakes.data import token_batch  # noqa: E402
from tests.fakes.harness import SEED_HARNESS  # noqa: E402

JOB = TrainJobSpec("weight_edit")


def checkpoint(root: Path, sharded: bool = False) -> Path:
    path = root / "base"
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": 3,
                "num_attention_heads": 1,
                "intermediate_size": 6,
                "num_hidden_layers": 1,
            }
        )
    )
    (path / "tokenizer.json").write_text('{"test": true}')
    state = {
        "backbone.weight": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3),
        "head.weight": torch.ones((2, 3)),
        "head.bias": torch.tensor([1.0, 2.0], dtype=torch.float16),
        "counter": torch.tensor([7], dtype=torch.int64),
    }
    if sharded:
        mapping = {}
        for i, keys in enumerate([("backbone.weight", "counter"), ("head.weight", "head.bias")], 1):
            name = f"model-0000{i}-of-00002.safetensors"
            st.save_file({k: state[k] for k in keys}, str(path / name), metadata={"format": "pt"})
            mapping.update(dict.fromkeys(keys, name))
        (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    else:
        st.save_file(state, str(path / "model.safetensors"), metadata={"format": "pt"})
    return path


def trainer(tmp_path: Path, *edits: TensorEdit) -> WeightEditTrainer:
    return WeightEditTrainer(tmp_path / "edits", lambda base, batch: EditPlan(edits))


@pytest.mark.unit
def test_multiple_named_edits_preserve_dtypes_buffers_and_the_parent(tmp_path: Path) -> None:
    parent = checkpoint(tmp_path)
    base = hf_snapshot(parent)
    before_bytes = (parent / "model.safetensors").read_bytes()
    before = st.load_file(str(parent / "model.safetensors"))
    result = trainer(
        tmp_path,
        TensorEdit("backbone.weight", "scale", 2),
        TensorEdit("head.weight", "add", [[1, 2, 3], [4, 5, 6]]),
        TensorEdit("head.bias", "replace", [4, 5]),
    ).train(JOB, base, None)
    out = Path(result.snapshot.checkpoint.location)
    after = st.load_file(str(out / "model.safetensors"))
    assert torch.equal(after["backbone.weight"], before["backbone.weight"] * 2)
    assert torch.equal(after["head.weight"], torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]))
    assert torch.equal(after["head.bias"], torch.tensor([4, 5], dtype=torch.float16))
    assert torch.equal(after["counter"], before["counter"])
    assert all(after[k].dtype == before[k].dtype and after[k].shape == before[k].shape for k in before)
    assert (parent / "model.safetensors").read_bytes() == before_bytes
    assert (out / "tokenizer.json").read_bytes() == (parent / "tokenizer.json").read_bytes()
    assert result.snapshot.architecture == base.architecture and result.snapshot.parameters == base.parameters
    assert hf_snapshot(out) == result.snapshot
    assert result.metrics == {"entries_changed": 13.0, "tensors_edited": 3.0}
    manifest = json.loads((out.parent / "checkpoint.json").read_text())
    values = st.load_file(str(out.parent / manifest["editValues"]))
    assert float(values["backbone.weight"]) == 2
    assert manifest["batchId"] is None and manifest["base"]["checksum"] == base.checkpoint.checksum


@pytest.mark.unit
def test_sharded_checkpoint_keeps_index_metadata_and_untouched_shard(tmp_path: Path) -> None:
    from safetensors import safe_open

    parent = checkpoint(tmp_path, sharded=True)
    base = hf_snapshot(parent)
    result = trainer(tmp_path, TensorEdit("head.weight", "scale", 2)).train(JOB, base, None)
    out = Path(result.snapshot.checkpoint.location)
    for name in ["model-00001-of-00002.safetensors", "model.safetensors.index.json"]:
        assert (out / name).read_bytes() == (parent / name).read_bytes()
    with safe_open(str(out / "model-00002-of-00002.safetensors"), framework="pt") as handle:
        assert handle.metadata() == {"format": "pt"}
        assert torch.equal(handle.get_tensor("head.weight"), torch.full((2, 3), 2.0))
    assert hf_snapshot(out).checkpoint.checksum == result.snapshot.checkpoint.checksum


@pytest.mark.unit
def test_replacement_values_can_share_storage_across_parameters(tmp_path: Path) -> None:
    parent = checkpoint(tmp_path)
    weights = parent / "model.safetensors"
    state = st.load_file(str(weights))
    state["second.weight"] = state["head.weight"].clone()
    st.save_file(state, str(weights))
    base = hf_snapshot(parent)
    before = weights.read_bytes()
    shared = torch.full((2, 3), 2.0)
    result = trainer(
        tmp_path,
        TensorEdit("head.weight", "replace", shared),
        TensorEdit("second.weight", "replace", shared.view(2, 3)),
    ).train(JOB, base, None)
    output = Path(result.snapshot.checkpoint.location)
    saved = st.load_file(str(output / "model.safetensors"))
    assert torch.equal(saved["head.weight"], shared)
    assert torch.equal(saved["second.weight"], shared)
    assert weights.read_bytes() == before
    assert result.metrics["entries_changed"] == 12.0
    assert hf_snapshot(output) == result.snapshot


@pytest.mark.unit
@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (TensorEdit("missing", "scale", 2), "no parameters"),
        (TensorEdit("head.weight", "add", [1, 2, 3]), "edit shape"),
        (TensorEdit("head.weight", "scale", [2]), "one scalar"),
        (TensorEdit("head.weight", "scale", float("nan")), "finite"),
        (TensorEdit("head.bias", "scale", 1e9), "overflows"),
        (TensorEdit("counter", "scale", 2), "floating-point"),
        (TensorEdit("head.weight", "scale", 1), "did not change"),
        (TensorEdit("head.weight", "other", 2), "unknown operation"),
    ],
)
def test_invalid_edits_leave_no_candidate_and_do_not_change_base(
    tmp_path: Path, edit: TensorEdit, message: str
) -> None:
    parent = checkpoint(tmp_path)
    base = hf_snapshot(parent)
    original = (parent / "model.safetensors").read_bytes()
    with pytest.raises(ContractError, match=message):
        trainer(tmp_path, edit).train(JOB, base, None)
    assert not list((tmp_path / "edits").glob("*/checkpoint"))
    assert (parent / "model.safetensors").read_bytes() == original


@pytest.mark.unit
def test_update_policy_is_enforced_and_unexpected_batches_are_refused(tmp_path: Path) -> None:
    base = hf_snapshot(checkpoint(tmp_path))
    edit = trainer(tmp_path, TensorEdit("head.weight", "scale", 2))
    with pytest.raises(ContractError, match="excluded"):
        edit.train(
            TrainJobSpec("weight_edit", update_policy=UpdatePolicy("param_subset", trainable_patterns=("backbone",))),
            base,
            None,
        )
    with pytest.raises(CampaignRuleError, match="takes no batch"):
        edit.train(JOB, base, token_batch())
    with pytest.raises(ContractError, match="steps must be 1"):
        edit.train(TrainJobSpec("weight_edit", steps=2), base, None)
    with pytest.raises(ContractError, match="not sft"):
        edit.train(TrainJobSpec("sft"), base, token_batch())
    job = TrainJobSpec(
        "weight_edit", update_policy=UpdatePolicy("param_subset", trainable_patterns=(r"^head\.weight$",))
    )
    assert edit.train(job, base, None).snapshot.checkpoint != base.checkpoint


@pytest.mark.unit
def test_bad_base_checksum_duplicate_operations_and_index_are_refused(tmp_path: Path) -> None:
    parent = checkpoint(tmp_path, sharded=True)
    base = hf_snapshot(parent)
    operation = TensorEdit("head.weight", "scale", 2)
    with pytest.raises(ContractError, match="unique"):
        trainer(tmp_path, operation, operation).train(JOB, base, None)
    index = parent / "model.safetensors.index.json"
    record = json.loads(index.read_text())
    record["weight_map"]["head.weight"] = "model-00001-of-00002.safetensors"
    index.write_text(json.dumps(record))
    with pytest.raises(ContractError, match="index does not match"):
        trainer(tmp_path, operation).train(JOB, base, None)
    (parent / "model-00001-of-00002.safetensors").write_bytes(b"changed")
    with pytest.raises(ContractError, match="checksum"):
        trainer(tmp_path, operation).train(JOB, base, None)


class NoData:
    def next_batch(self, contract: Any) -> Any:
        raise AssertionError("static weight editing must not request training data")


@pytest.mark.unit
def test_general_editor_passes_trainer_conformance_without_data(tmp_path: Path) -> None:
    base = hf_snapshot(checkpoint(tmp_path))
    edit = trainer(tmp_path, TensorEdit("head.weight", "scale", 2))
    checks = check_trainer(edit, base=base, jobs=[JOB], data=NoData(), rollout=None, tasks=TASKS, harness=SEED_HARNESS)
    assert all(c.passed for c in checks), checks
    assert {c.name for c in checks} == {"support", "train:weight_edit", "fresh", "refuses"}


def model_source(context: PipelineContext, path: str) -> Any:
    class Source:
        def materialize(self, work_dir: Path) -> Any:
            return hf_snapshot(Path(path))

    return Source()


def data(context: PipelineContext) -> NoData:
    return NoData()


def evaluator(context: PipelineContext) -> Any:
    class Evaluate:
        def evaluate(self, candidate: Any, incumbent: Any) -> EvaluationResult:
            # The pipeline still gates the new checkpoint; stand-in scores test release wiring only.
            assert candidate.master.checkpoint.checksum != incumbent.master.checkpoint.checksum
            return EvaluationResult(("weights",), (1.0,), (0.0,), "master")

    return Evaluate()


@pytest.mark.integration
def test_json_factory_file_delta_and_campaign_publish_real_weights_without_data(tmp_path: Path) -> None:
    parent = checkpoint(tmp_path)
    st.save_file({"delta": torch.ones((2, 3))}, str(tmp_path / "delta.safetensors"))
    edits = [
        {
            "parameter": "head.weight",
            "operation": "add",
            "value_file": str(tmp_path / "delta.safetensors"),
            "value_tensor": "delta",
        }
    ]
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/e2e/all_planes.json").read_text())
    config.update(
        schedule=[
            {
                "kind": "train",
                "algorithm": "weight_edit",
                "update_policy": {"kind": "param_subset", "trainable_patterns": [r"^head\.weight$"]},
            }
        ],
        expect=["published"],
    )
    config.pop("rollback", None)
    config["components"].update(
        {
            "model_source": {"factory": "tests.test_weight_edit:model_source", "args": {"path": str(parent)}},
            "trainer": {"factory": "honing.backends.weight_edit:weight_edit_trainer", "args": {"edits": edits}},
            "data": {"factory": "tests.test_weight_edit:data"},
            "evaluator": {"factory": "tests.test_weight_edit:evaluator"},
        }
    )
    pipeline = build(config, root, tmp_path / "pipeline")
    summary = run(pipeline)
    assert summary["failures"] == [] and summary["steps"][0]["result"] == "published"
    outputs = list((tmp_path / "pipeline/weight-edits").glob("*/checkpoint/model.safetensors"))
    assert len(outputs) == 1
    assert torch.equal(st.load_file(str(outputs[0]))["head.weight"], torch.full((2, 3), 2.0))


@pytest.mark.unit
def test_factory_refuses_ambiguous_value_configuration(tmp_path: Path) -> None:
    base = hf_snapshot(checkpoint(tmp_path))
    context = PipelineContext(root=tmp_path, work_dir=tmp_path / "out")
    edit = weight_edit_trainer(
        context, [{"parameter": "head.weight", "operation": "scale", "value": 2, "value_file": "unused"}]
    )
    with pytest.raises(ContractError, match="not both"):
        edit.train(JOB, base, None)
