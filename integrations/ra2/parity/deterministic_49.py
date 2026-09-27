"""Run #49's train_laya.py unmodified, with PyTorch's deterministic kernels switched on first."""

import os
import runpy
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

torch.use_deterministic_algorithms(True)
script = sys.argv[1]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
sys.argv = [script, *sys.argv[2:]]
runpy.run_path(script, run_name="__main__")
