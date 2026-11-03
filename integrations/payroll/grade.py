"""Grade one attendance-payroll-audit episode with upstream's checkpoint eval. ``CommandGrader`` runs it.

It runs in a copy of the task's grader files: this script and ``groundtruth/``
(upstream's ``expected.json`` and ``example.json``), which the agent never sees.
The agent's files are under ``$HONING_WORKSPACE/work``, the OpenHands runner's
working directory. Upstream's eval reads the ground truth from
``<workspace>_logs/groundtruth`` beside the workspace, so that layout is made in
a scratch directory. Prints ``{"score": ..., "feedback": ...}``.

Usage: ``python grade.py --upstream <slm-harness-adaptation checkout>``
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

#: The OpenHands runner's working directory inside an episode directory.
AGENT_DIR = "work"
EVAL_MODULE = "src.task_evals.attendance_payroll_audit_s2l"


def load_eval(upstream: Path) -> types.ModuleType:
    """Upstream's eval module, imported with ``src`` and its subpackages as bare namespaces.

    Upstream's ``src/__init__.py`` imports every task's dependencies, so it is not run.
    """
    for name, path in (
        ("src", upstream / "src"),
        ("src.task_setups", upstream / "src/task_setups"),
        ("src.task_evals", upstream / "src/task_evals"),
    ):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module
    return importlib.import_module(EVAL_MODULE)


def grade(upstream: Path, episode: Path, groundtruth: Path) -> dict[str, object]:
    evaluate = load_eval(upstream)
    example = json.loads((groundtruth / "example.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as scratch:
        workspace = Path(scratch) / "workspace"
        workspace.symlink_to(episode / AGENT_DIR, target_is_directory=True)
        shutil.copytree(groundtruth, Path(scratch) / "workspace_logs" / "groundtruth")
        result = evaluate.run_single_instance_eval(str(workspace), example)
    return {"score": float(result["score"]), "feedback": str(result["feedback"])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--upstream", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(grade(args.upstream, Path(os.environ["HONING_WORKSPACE"]), Path("groundtruth"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
