# Parity with #49

These scripts compare the code in Honing with the original RA2 experiment in #49, run on the same inputs. They need a GPU node with #49's branch checked out (`PR49_DIR`, its `examples/ra2` directory, prepared as in its README), the model environment (`LAYA_PYTHON`), and the pinned Laya source (`LAYA_SOURCE_DIR`).

| Script | Compares | Result on an H100 |
|---|---|---|
| `decision_server.py <out>` | #49's stdio worker, Honing's stdio worker, and Honing's HTTP server (sequentially and from 4 concurrent clients), on 24 generated requests, 5 of which force trimming | identical answers, probabilities, state text, token audits, and trimming |
| `run_training.sh <work>` | #49's `train_laya.py` run twice and `LocalJevTrainer` run once on the same generated demonstrations and seed | #49 differs from itself (CUDA kernels are not deterministic), and Honing falls within that variation |
| `run_training.sh <work> --deterministic` | the same, with PyTorch's deterministic kernels on (`deterministic_49.py` enables them and then runs #49's file unmodified) | all three weight files byte-identical |

`make_demos.py` writes six labelled matches in #49's directory format, with seed 101 held out for validation, so training parity needs no game.

## Multi-epoch numerical regression

A one-epoch parity run does not establish training stability. The generated
matches contain 584 training and 118 validation examples. With seed 7 and
batch size 8 in the generated match order, the first batch whose padded length
is 192 occurs in epoch 2, batch 41 (one-based). This is a shape covered by the
[cuDNN masked-attention backward regression](https://github.com/pytorch/pytorch/issues/196678).
Shape alone is not proof that a particular failed run hit that regression;
its attention values and selected kernel must also be checked.

`LocalJevTrainer` excludes cuDNN SDPA for the entire training call, including
validation and gradient-checkpoint recomputation. Flash, memory-efficient,
and math attention remain available. It restores the caller's backend settings
when it returns or raises. This changes CUDA numerical parity with the original
unrestricted trainer, but does not change labels, learning rates, or the loss.
The checkpoint manifest records the allowed backends.

A finite loss is not enough: gradients are checked before the optimizer update,
and a nonfinite loss or gradient reports the epoch and batch. Validation also
rejects nonfinite losses rather than silently accepting an invalid checkpoint.
No invalid batch is skipped and no NaN is replaced with zero.

To validate on an exclusively available GPU, first run the reduced attention
regression, then run at least three epochs through the public trainer. Use the
same pinned environment and model as the failing run:

```bash
# From the repository root; set these to prepared external assets.
export PR49_DIR=/path/to/pr49/examples/ra2
export LAYA_SOURCE_DIR=/path/to/pinned/laya
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
# LAYA_PYTHON must have torch, transformers, safetensors, and pytest available.
CUDA_VISIBLE_DEVICES="$GPU" "$LAYA_PYTHON" -m pytest \
  tests/test_jev_training_numerics.py -m gpu
mapfile -t matches < <("$LAYA_PYTHON" integrations/ra2/parity/make_demos.py "$WORK/demos")
"$LAYA_PYTHON" integrations/ra2/parity/train_honing.py \
  "$WORK/honing" "$GPU" 3 "${matches[@]}" --deterministic
```

The CPU tests exercise a finite forward loss with infinite/NaN derivatives and
verify that neither weights nor optimizer state are updated. They also verify
backend restoration on failure. They do **not** reproduce a CUDA kernel bug;
the CUDA regression and multi-epoch GPU run remain necessary before declaring
the reported epoch-2 failure resolved.
