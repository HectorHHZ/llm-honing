# Weight modification through Trainer

`WeightEditTrainer` uses the existing `Trainer.train(job, base, batch) -> TrainResult`
interface. A campaign evaluates and publishes its candidate through the same gate
as other training jobs. Editing does not bypass release selection.

Install the optional CPU checkpoint dependencies with `pip install -e '.[weight-edit]'`.
The modifier can use another runtime to derive its plan. For example, the NanoJev
direction reader still needs its pinned source and GPU runtime.

## Operations and interfaces

| Interface | Input | Output |
| --- | --- | --- |
| Modifier callable | `ModelSnapshot`, `TrainingBatch \| None` | `EditPlan` |
| `TensorEdit` | Exact parameter name, operation, value | One tensor operation |
| `EditPlan` | Tuple of operations, optional finite metrics and JSON metadata | A complete edit proposal |
| `CheckpointLayout` | Checksum function, weight-file selector, optional serving manifest fields | Model-specific storage conventions |
| `WeightEditTrainer.train` | Job, base snapshot, optional batch | New snapshot and metrics in `TrainResult` |

Each parameter can appear once in a plan. Operations apply to stored tensors;
they do not change the model architecture, tokenizer, parameter names or shapes.

| Operation | Value | Result |
| --- | --- | --- |
| `add` | Tensor with exactly the parameter's shape | `weight + value` |
| `scale` | One scalar | `weight * value` |
| `replace` | Tensor with exactly the parameter's shape | `value` |

The writer supports FP16, BF16, FP32 and FP64 master weights. It computes in
FP32, or FP64 for an FP64 parameter, then restores the original dtype. It rejects
nonfinite values and overflow after conversion. It does not broadcast values.
Integer buffers and quantized tensors cannot be edited. Unselected tensors keep
their values and dtypes. Unchanged shards retain their original bytes.

`weight_edit` takes no batch, rollout or teacher. Use `steps=1` and omit `contract`.
The campaign does not call the data provider for this algorithm. A supplied batch
is an error. `direction_edit` continues to consume `typed_decisions`; its feature
reader returns a data-derived plan through the same writer.

`full` permits any named parameter. `param_subset` requires every operation's name
to match at least one `trainable_patterns` regular expression. The editor rejects
`lora` and `sparse_mask`. The job uses the default `bf16` numerics policy; this
does not convert stored tensors to BF16. FP8 and QAT jobs are not supported.

## Configure an existing pipeline

Replace its trainer and schedule entries with this fragment. Keep the model source,
runner, grader, gate and other components from the pipeline being used.

```json
{
  "components": {
    "trainer": {
      "factory": "honing.backends.weight_edit:weight_edit_trainer",
      "args": {
        "edits": [
          {"parameter": "model.layers.0.mlp.down_proj.weight", "operation": "scale", "value": 0.95},
          {"parameter": "lm_head.weight", "operation": "add", "value_file": "edits/delta.safetensors", "value_tensor": "delta"}
        ]
      }
    },
    "data": {"factory": "honing.backends.weight_edit:no_data"}
  },
  "schedule": [
    {
      "kind": "train",
      "algorithm": "weight_edit",
      "update_policy": {
        "kind": "param_subset",
        "trainable_patterns": ["^model\\.layers\\.0\\.mlp\\.down_proj\\.weight$", "^lm_head\\.weight$"]
      }
    }
  ]
}
```

Names must exist in the selected checkpoint. `value` can also be a JSON array.
Use either `value` or `value_file`, never both. `value_tensor` defaults to the
parameter name. File paths resolve against the pipeline root and expand environment
variables. Unknown operation fields are errors. A missing file or tensor fails the job.
Keep a real data provider if the campaign also schedules data-consuming jobs.

The default layout uses Honing's HF/SGLang checksum and supports single-file and
indexed sharded safetensors checkpoints. It preserves the shard index and file
metadata, and checks that the index matches the stored tensors.

For NanoJev, use the same operation arguments with this factory:

```json
{
  "factory": "honing.backends.jev.weight_edit:weight_edit_trainer",
  "args": {
    "pins": "integrations/ra2/pins.json",
    "model": "nanojev",
    "edits": [
      {"parameter": "scalar.weight", "operation": "replace", "value_file": "edits/scalar.safetensors"}
    ]
  }
}
```

This adapter preserves NanoJev's `best.safetensors` checksum and `trained-nanojev`
manifest. The existing decision server can load the result. It accepts other
floating-point tensors too; `scalar.weight` is only an example.

## Add a custom modifier

A pipeline factory can return a `WeightEditTrainer` with its own callable:

```python
from pathlib import Path
from honing.backends.weight_edit import EditPlan, TensorEdit, WeightEditTrainer

def trainer(context):
    def propose(base, batch):
        # Read the base or external artifacts here. Do not mutate the base.
        return EditPlan(
            edits=(TensorEdit("lm_head.weight", "scale", 0.9),),
            metadata={"method": "head-scale", "factor": 0.9},
        )
    return WeightEditTrainer(Path(context.work_dir) / "edits", propose)
```

For a model-specific checksum or serving manifest, supply `CheckpointLayout`.
Its `files(parent)` must return distinct safetensors paths within `parent`.
The default checksum must match the base model source; do not use the HF default
for a model source that hashes only `best.safetensors`. Use `nanojev_layout` for
NanoJev. A data-derived typed-decision modifier selects `algorithm="direction_edit"`.
The existing `NanoJevDirectionEditTrainer` is such an adapter; its public settings
and feature-reader injection remain available.

## Output, failures and validation

Each call uses a fresh `edit-<id>/` directory:

- `checkpoint/`: a separate copy of the model with the edits applied. No hard links
  are used. Root checksum manifests from the source are omitted because they are stale.
- `checkpoint.json`: base identity, algorithm, batch ID, update policy, model-specific
  serving fields, operations, edited tensor hashes, metadata and finite metrics.
- `edit-values.safetensors`: the values actually applied, named by parameter. Together
  with the operation records and base checkpoint, these reproduce the edit.
- `error.json`: error information if checkpoint writing fails. The incomplete
  `checkpoint/` is removed and no `TrainResult` is returned.

Validation before run creation can fail without writing an error file. Errors include
an unsupported job, a changed base checksum, duplicate or missing parameters, a bad
shape, an excluded parameter, invalid numeric values, and an edit that changes no
stored values. The output directory must be outside the base checkpoint.
Core combination errors raise `CampaignRuleError`; editor validation raises
`ContractError`. Storage and safetensors errors propagate to the caller.

`entries_changed` counts changed stored elements; `tensors_edited` counts selected
tensors. These writer metrics take precedence over modifier metrics with those names.
Run `scripts/e2e/check_trainer.py` against the pipeline to check Trainer conformance.
The CPU tests cover a non-NanoJev model, mixed dtypes, shards, refusal paths, a real
configured campaign, and NanoJev compatibility. Native-game results test the pinned
NanoJev edit; they do not establish quality gains for arbitrary weight operations.
