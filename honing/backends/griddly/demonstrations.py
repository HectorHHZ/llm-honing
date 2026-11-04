"""Audited train-split rule-teacher runs as supervised typed decisions."""

from __future__ import annotations

import json
import subprocess
import uuid
from collections.abc import Sequence
from pathlib import Path

from honing.backends.griddly.decisions import decision_samples
from honing.backends.griddly.replay import audit_run
from honing.backends.griddly.runner import runtime_python
from honing.backends.griddly.scenarios import GriddlySuite
from honing.core import ContractError, DataContract, TrainingBatch


class GriddlyDemonstrationProvider:
    """Refuse evaluation/heldout data; verify every native trajectory before yielding labels."""

    def __init__(
        self,
        runs: Sequence[Path],
        *,
        suite: GriddlySuite,
        integration_dir: Path,
        command: Sequence[str] | None = None,
    ) -> None:
        self._runs, self._suite, self._served = tuple(runs), suite, False
        integration = integration_dir.resolve()
        self._command = list(command or [runtime_python(integration), str(integration / "scripts/verify_replay.py")])

    def next_batch(self, contract: DataContract) -> TrainingBatch | None:
        if (
            self._served
            or contract.kind != "typed_decisions"
            or contract.requires_logprobs
            or contract.requires_reward
        ):
            return None
        samples = []
        for run in self._runs:
            manifest, _frames, result = audit_run(run)
            if manifest["kind"] != "teacher-demonstration" or manifest["controller"] != "teacher":
                raise ContractError("Griddly demonstrations must be explicit rule-teacher runs")
            task_ids = [
                str(record["task_id"])
                for record in self._suite.records
                if record["split"] == "train"
                and self._suite.scenario(str(record["task_id"])).record() == manifest["scenario"]
            ]
            if len(task_ids) != 1:
                raise ContractError("Griddly demonstration scenario must match exactly one train task")
            checked = subprocess.run(
                [*self._command, str(run.resolve())], capture_output=True, text=True, encoding="utf-8", check=False
            )
            try:
                report = json.loads(checked.stdout.strip().splitlines()[-1]) if checked.returncode == 0 else {}
            except (ValueError, IndexError):
                report = {}
            if report.get("verified") is not True or report.get("final_frame_hash") != result["final_frame_hash"]:
                raise ContractError("Griddly teacher trajectory failed native replay")
            decisions = [
                json.loads(line) for line in (run / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            samples.extend(decision_samples(decisions, group=task_ids[0], harness_id=manifest["spec_id"]))
        if not samples:
            return None
        self._served = True
        return TrainingBatch("griddly-teacher-" + uuid.uuid4().hex[:16], contract, decisions=tuple(samples))
