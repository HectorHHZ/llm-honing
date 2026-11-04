"""Grade explicit task goals from an audited model-controlled Craftax trajectory."""

from __future__ import annotations

from pathlib import Path

from honing.backends.craftax.replay import audit_run
from honing.backends.craftax.scenarios import CraftaxSuite
from honing.core import ContractError, EpisodeOutcome, Grade, Task


class CraftaxGrader:
    """Score completed task goals / required goals, not survival or shaped engine reward.

    Introductory tasks use crafting goals in the full game. They do not measure
    full-game victory. Random and prescribed-action demos are never model grades.
    """

    def __init__(self, *, suite: CraftaxSuite) -> None:
        self._suite = suite

    async def grade(self, task: Task, outcome: EpisodeOutcome) -> Grade:
        if outcome.task_id != task.task_id or outcome.exit_code != 0 or outcome.workspace_location is None:
            raise ContractError("Craftax did not produce a completed run for this task")
        scenario = self._suite.scenario(task.task_id)
        try:
            manifest, _rows, result = audit_run(Path(outcome.workspace_location))
        except (ValueError, KeyError, TypeError, OSError) as error:
            raise ContractError(f"invalid Craftax run: {error}") from error
        if (
            manifest["scenario"] != scenario.record()
            or manifest["kind"] != "evaluation"
            or manifest["controller"] != "model"
            or manifest["spec_id"] != outcome.spec_id
            or manifest.get("policy_checksum") != outcome.policy_checksum
        ):
            raise ContractError("Craftax run does not match the task and model harness")
        completed = sorted(set(scenario.goals) & set(result["achievements"]))
        return Grade(
            len(completed) / len(scenario.goals),
            f"Completed {len(completed)}/{len(scenario.goals)} goals: {', '.join(completed) or 'none'}. "
            f"Ended as {result['status']} after {result['steps']} steps with health {result['health']}. "
            f"Engine reward {result['reward']} is diagnostic and is not the task score.",
        )
