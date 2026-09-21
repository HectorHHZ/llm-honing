"""Harness fakes: an agent whose score is its weights' quality plus a bonus for checking its work."""

from __future__ import annotations

from collections.abc import Mapping

from honing.core import (
    EpisodeRecord,
    EvaluationResult,
    EvolveContext,
    HarnessFile,
    HarnessProposal,
    HarnessSpec,
    MethodRequirements,
    PolicyRef,
    RunnerSupport,
    SelectionDecision,
    Task,
)
from honing.release import WinLossSelector
from tests.fakes.model import FakeWorld

INSTRUCTIONS_PATH = "CLAUDE.md"

SEED_HARNESS = HarnessSpec(
    agent="claude_code", files=((INSTRUCTIONS_PATH, HarnessFile(kind="instructions", content="Solve the task.\n")),)
)


class FakeEpisodeRunner:
    def __init__(self, world: FakeWorld, instruction_bonus: float = 0.05) -> None:
        self.world = world
        self.bonus = instruction_bonus
        self.support = RunnerSupport(
            agents=frozenset({"claude_code"}), file_kinds=frozenset({"instructions", "skill"}), programmable_loop=False
        )
        self.runs = 0

    async def run(self, spec: HarnessSpec, task: Task, policy: PolicyRef) -> EpisodeRecord:
        self.runs += 1
        quality = self.world.quality[policy.checkpoint.checksum]
        verifies = any(file.kind == "instructions" and "verify" in file.content.lower() for _, file in spec.files)
        score = round(min(1.0, quality + (self.bonus if verifies else 0.0)), 6)
        return EpisodeRecord(
            task_id=task.task_id,
            spec_id=spec.spec_id,
            policy_checksum=policy.checkpoint.checksum,
            score=score,
            feedback=None if verifies else "answered without checking the result",
            transcript_location=f"fake://transcripts/{spec.spec_id}/{task.task_id}",
        )


class AppendInstructionMethod:
    """Adds a verification instruction once, then has nothing more to propose."""

    name = "append_instruction"

    def __init__(self) -> None:
        self.requires = MethodRequirements(file_kinds=frozenset({"instructions"}))
        self.selector = WinLossSelector()

    def initial_state(self, seed: HarnessSpec) -> Mapping[str, object]:
        return {"proposals": 0, "accepted": 0}

    def propose(self, context: EvolveContext) -> HarnessProposal | None:
        current = context.current.get(INSTRUCTIONS_PATH)
        text = "" if current is None else current.content
        if "verify" in text.lower():
            return None
        spec = context.current.with_file(
            INSTRUCTIONS_PATH, HarnessFile(kind="instructions", content=text + "Verify the result before answering.\n")
        )
        return HarnessProposal(spec=spec, note="ask the agent to verify its result")

    def record(
        self,
        state: Mapping[str, object],
        proposal: HarnessProposal,
        evaluation: EvaluationResult,
        decision: SelectionDecision,
    ) -> Mapping[str, object]:
        return {
            "proposals": int(str(state["proposals"])) + 1,
            "accepted": int(str(state["accepted"])) + int(decision.selected),
        }
