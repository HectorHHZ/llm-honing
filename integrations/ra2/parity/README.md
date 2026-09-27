# Parity with #49

These scripts compare the code in Honing with the original RA2 experiment in #49, run on the same inputs. They need a GPU node with #49's branch checked out (`PR49_DIR`, its `examples/ra2` directory, prepared as in its README), the model environment (`LAYA_PYTHON`), and the pinned Laya source (`LAYA_SOURCE_DIR`).

| Script | Compares | Result on an H100 |
|---|---|---|
| `decision_server.py <out>` | #49's stdio worker, Honing's stdio worker, and Honing's HTTP server (sequentially and from 4 concurrent clients), on 24 generated requests, 5 of which force trimming | identical answers, probabilities, state text, token audits, and trimming |
| `run_training.sh <work>` | #49's `train_laya.py` run twice and `LocalJevTrainer` run once on the same generated demonstrations and seed | #49 differs from itself (CUDA kernels are not deterministic), and Honing falls within that variation |
| `run_training.sh <work> --deterministic` | the same, with PyTorch's deterministic kernels on (`deterministic_49.py` enables them and then runs #49's file unmodified) | all three weight files byte-identical |

`make_demos.py` writes six labelled matches in #49's directory format, with seed 101 held out for validation, so training parity needs no game.
