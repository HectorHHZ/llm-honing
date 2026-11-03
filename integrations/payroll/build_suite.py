"""Write the attendance-payroll-audit tasks as a Honing task suite, from a slm-harness-adaptation checkout.

The tasks are https://github.com/malusamayo/slm-harness-adaptation's, at
``2d31e17``: each example's workspace is made by upstream's own setup
(``src/task_setups/attendance_payroll_audit_s2l.py``) and graded by upstream's
own checkpoint eval through ``grade.py``. The split is upstream GEPA's: the first
``--train`` examples in data order, then the next ``--val``.

Writes under ``--out``:

``suite.jsonl``
    the manifest (:mod:`honing.harness.tasks`). ``train`` tasks are
    Meta-Harness's search set, ``val`` tasks the gate's.
``tasks/<task id>/workspace/``
    the files the agent starts from, copied into each episode's ``work/``.
``tasks/<task id>/grader/``
    ``grade.py`` and ``groundtruth/``, which ``CommandGrader`` copies outside the
    agent's workspace. Run the agent in a container
    (``configs/e2e/payroll_meta_harness.json``), so it cannot read them.
``seed/AGENTS.md``
    upstream's ``default`` system prompt, the seed harness's instructions.

The grader command is this interpreter running ``grade.py``, so build the suite
with an environment that has upstream's eval dependencies (``dspy``,
``pandas``, ``openpyxl``); building imports the eval module to check.

Run: ``python -m integrations.payroll.build_suite --slm-repo <checkout> --out integrations/payroll/.suite``
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote

from integrations.payroll.grade import load_eval

DATA_FILE = "data/attendance_payroll_audit_s2l_extra_high.json"
SEED_PROMPT_FILE = "tasks/attendance_payroll_audit_s2l/prompts/default.md"
SETUP_MODULE = "src.task_setups.attendance_payroll_audit_s2l"
GRADE_SCRIPT = Path(__file__).with_name("grade.py")


def build(upstream: Path, out: Path, train: int, val: int, python: str = sys.executable) -> Path:
    """Write the suite under ``out``, which must not exist, and return the manifest's path."""
    upstream = upstream.resolve()
    load_eval(upstream)  # a grader that cannot import upstream would score every episode 0
    setup = importlib.import_module(SETUP_MODULE)
    examples: list[dict[str, Any]] = json.loads((upstream / DATA_FILE).read_text(encoding="utf-8"))
    chosen = [(example, "train") for example in examples[:train]]
    chosen += [(example, "val") for example in examples[train : train + val]]
    out.mkdir(parents=True)
    lines = []
    for example, split in chosen:
        folder = Path("tasks") / quote(str(example["id"]), safe="")
        workspace, grader = out / folder / "workspace", out / folder / "grader"
        workspace.mkdir(parents=True)
        with tempfile.TemporaryDirectory() as logs:
            setup.setup_workspace(str(workspace), logs, dict(example))
            shutil.copytree(Path(logs) / "groundtruth", grader / "groundtruth")
        shutil.copyfile(GRADE_SCRIPT, grader / "grade.py")
        lines.append(
            {
                "task_id": str(example["id"]),
                "prompt": example["prompt"],
                "workspace": (folder / "workspace").as_posix(),
                "grader": [python, "grade.py", "--upstream", str(upstream)],
                "split": split,
                "grader_files": (folder / "grader").as_posix(),
            }
        )
    manifest = out / "suite.jsonl"
    manifest.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    (out / "seed").mkdir()
    shutil.copyfile(upstream / SEED_PROMPT_FILE, out / "seed" / "AGENTS.md")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slm-repo", type=Path, required=True, help="checkout of slm-harness-adaptation")
    parser.add_argument("--out", type=Path, required=True, help="directory to write; must not exist")
    parser.add_argument("--train", type=int, default=10)
    parser.add_argument("--val", type=int, default=10)
    args = parser.parse_args(argv)
    manifest = build(args.slm_repo, args.out, args.train, args.val)
    print(f"wrote {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
