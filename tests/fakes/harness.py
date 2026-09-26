"""Harness fakes: an agent whose score is its weights' quality plus a bonus for checking its work."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from honing.core import (
    ChatMessage,
    ChatReply,
    ContractError,
    EpisodeOutcome,
    EvaluationResult,
    EvolveContext,
    Grade,
    HarnessFile,
    HarnessProposal,
    HarnessSpec,
    MethodRequirements,
    PolicyEndpoint,
    PolicyRef,
    ProposeResult,
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
    """Runs no agent. The final output reports the answer's quality and whether the agent verified it."""

    def __init__(self, world: FakeWorld, instruction_bonus: float = 0.05) -> None:
        self.world = world
        self.bonus = instruction_bonus
        self.support = RunnerSupport(
            agents=frozenset({"claude_code"}), file_kinds=frozenset({"instructions", "skill"}), programmable_loop=False
        )
        self.runs = 0

    async def run(self, spec: HarnessSpec, task: Task, policy: PolicyRef) -> EpisodeOutcome:
        self.runs += 1
        quality = self.world.quality[policy.checkpoint.checksum]
        verifies = any(file.kind == "instructions" and "verify" in file.content.lower() for _, file in spec.files)
        quality = round(min(1.0, quality + (self.bonus if verifies else 0.0)), 6)
        return EpisodeOutcome(
            task_id=task.task_id,
            spec_id=spec.spec_id,
            policy_checksum=policy.checkpoint.checksum,
            exit_code=0,
            final_output=f"quality={quality} verified={int(verifies)}",
            transcript_location=f"fake://transcripts/{spec.spec_id}/{task.task_id}",
        )


class FakeTaskGrader:
    """Reads the score the fake runner reported, and gives feedback when the agent did not verify."""

    def __init__(self) -> None:
        self.grades = 0

    async def grade(self, task: Task, outcome: EpisodeOutcome) -> Grade:
        self.grades += 1
        fields = dict(part.split("=", 1) for part in outcome.final_output.split())
        verified = fields["verified"] == "1"
        return Grade(
            score=float(fields["quality"]),
            feedback=None if verified else "answered without checking the result",
        )


class AppendInstructionMethod:
    """Adds a verification instruction once, then has nothing more to propose."""

    name = "append_instruction"
    state_schema_version = 1

    def __init__(self) -> None:
        self.requires = MethodRequirements(file_kinds=frozenset({"instructions"}))
        self.selector = WinLossSelector()

    def initial_state(self, seed: HarnessSpec) -> Mapping[str, object]:
        return {"schema_version": self.state_schema_version, "proposals": 0, "accepted": 0, "attempts": 0}

    def propose(self, context: EvolveContext) -> ProposeResult:
        state = {**context.state, "attempts": int(str(context.state["attempts"])) + 1}
        current = context.current.get(INSTRUCTIONS_PATH)
        text = "" if current is None else current.content
        if "verify" in text.lower():
            return ProposeResult(proposal=None, state=state)
        spec = context.current.with_file(
            INSTRUCTIONS_PATH, HarnessFile(kind="instructions", content=text + "Verify the result before answering.\n")
        )
        return ProposeResult(
            proposal=HarnessProposal(spec=spec, note="ask the agent to verify its result"), state=state
        )

    def record(
        self,
        state: Mapping[str, object],
        proposal: HarnessProposal,
        evaluation: EvaluationResult,
        decision: SelectionDecision,
    ) -> Mapping[str, object]:
        return {
            **state,
            "proposals": int(str(state["proposals"])) + 1,
            "accepted": int(str(state["accepted"])) + int(decision.selected),
        }


class FakeChatModel:
    """Returns scripted replies in order and records every conversation it was sent."""

    def __init__(self, replies: Sequence[str], cost_per_call: float | None = None) -> None:
        self._replies = list(replies)
        self._cost = cost_per_call
        self.conversations: list[tuple[ChatMessage, ...]] = []

    def complete(self, messages: Sequence[ChatMessage], *, max_tokens: int, temperature: float) -> ChatReply:
        if not self._replies:
            raise ContractError("the fake chat model has no scripted replies left")
        self.conversations.append(tuple(messages))
        text = self._replies.pop(0)
        return ChatReply(
            text=text,
            input_tokens=sum(len(message.content.split()) for message in messages),
            output_tokens=len(text.split()),
            cost_usd=self._cost,
        )


class StaticPolicyEndpoints:
    """Maps checkpoint checksums to endpoints fixed at construction."""

    def __init__(self, endpoints: Mapping[str, PolicyEndpoint]) -> None:
        self._endpoints = dict(endpoints)

    def resolve(self, policy: PolicyRef) -> PolicyEndpoint:
        try:
            return self._endpoints[policy.checkpoint.checksum]
        except KeyError:
            raise ContractError(f"no engine serves checkpoint {policy.checkpoint.checksum!r}") from None
