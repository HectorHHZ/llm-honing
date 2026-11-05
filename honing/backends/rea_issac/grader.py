"""Boss-defeat score computed by replaying the recorded simulation."""

from __future__ import annotations

from pathlib import Path

from honing.backends.rea_issac.replay import audit_run
from honing.backends.rea_issac.scenarios import ReaIssacSuite
from honing.core import CampaignRuleError, ContractError, EpisodeOutcome, Grade, Task


class ReaIssacGrader:
    def __init__(self, *, suite: ReaIssacSuite) -> None:
        self._suite = suite

    async def grade(self, task: Task, outcome: EpisodeOutcome) -> Grade:
        if outcome.task_id != task.task_id or outcome.exit_code != 0 or outcome.workspace_location is None:
            raise ContractError("REA-Issac did not complete this task")
        try:
            manifest, _rows, result = audit_run(Path(outcome.workspace_location))
        except (CampaignRuleError, ValueError, KeyError, TypeError, OSError) as error:
            raise ContractError(f"invalid REA-Issac run: {error}") from error
        if (
            manifest["scenario"] != self._suite.scenario(task.task_id).record()
            or manifest["kind"] != "evaluation"
            or manifest["controller"] != "model"
            or manifest["spec_id"] != outcome.spec_id
            or manifest.get("policy_checksum") != outcome.policy_checksum
        ):
            raise ContractError("REA-Issac run does not match the task and model harness")
        return Grade(
            float(result["success"]),
            f"Ended as {result['status']} at tick {result['tick']}; "
            f"health {result['health']}, rooms cleared {result['rooms_cleared']}. "
            "Only boss defeat scores 1; survival, kills and progress are diagnostic.",
        )
