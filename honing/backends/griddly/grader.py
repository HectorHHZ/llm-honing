"""Score task success only after audit and native re-execution in the game runtime."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

from honing.backends.griddly.replay import audit_run
from honing.backends.griddly.runner import runtime_python
from honing.backends.griddly.scenarios import GriddlySuite
from honing.core import CampaignRuleError, ContractError, EpisodeOutcome, Grade, Task


class GriddlyGrader:
    def __init__(self, *, suite: GriddlySuite, integration_dir: Path, command: Sequence[str] | None = None) -> None:
        self._suite = suite
        integration = integration_dir.resolve()
        self._command = list(command or [runtime_python(integration), str(integration / "scripts/verify_replay.py")])

    async def grade(self, task: Task, outcome: EpisodeOutcome) -> Grade:
        if outcome.task_id != task.task_id or outcome.exit_code != 0 or outcome.workspace_location is None:
            raise ContractError("Griddly did not complete this task")
        run = Path(outcome.workspace_location)
        try:
            manifest, _rows, result = audit_run(run)
        except (CampaignRuleError, ValueError, KeyError, TypeError, OSError) as error:
            raise ContractError(f"invalid Griddly run: {error}") from error
        if (
            manifest["scenario"] != self._suite.scenario(task.task_id).record()
            or manifest["kind"] not in ("evaluation", "rollout")
            or manifest["controller"] != "model"
            or manifest["spec_id"] != outcome.spec_id
            or manifest.get("policy_checksum") != outcome.policy_checksum
        ):
            raise ContractError("Griddly run does not match the task, policy and harness")
        process = await asyncio.create_subprocess_exec(
            *self._command, str(run), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await process.communicate()
        except BaseException:
            if process.returncode is None:
                process.kill()
                await process.wait()
            raise
        try:
            report = json.loads(stdout.decode("utf-8").strip().splitlines()[-1]) if process.returncode == 0 else {}
        except (ValueError, IndexError):
            report = {}
        if (
            report.get("verified") is not True
            or report.get("final_frame_hash") != result["final_frame_hash"]
            or report.get("steps") != result["steps"]
        ):
            raise ContractError(
                f"native Griddly replay did not verify: {stderr.decode('utf-8', errors='replace')[-2000:]}"
            )
        return Grade(
            float(result["success"]),
            f"Ended as {result['status']} after {result['steps']} ticks. "
            f"Native reward {result['reward']} is diagnostic; only reaching the gem scores 1.",
        )
